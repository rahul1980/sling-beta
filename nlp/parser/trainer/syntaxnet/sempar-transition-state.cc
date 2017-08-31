#include "nlp/parser/trainer/syntaxnet/sempar-transition-state.h"

#include "frame/serialization.h"
#include "nlp/parser/trainer/syntaxnet/framed-sentence.pb.h"
#include "string/strcat.h"

namespace sling {
namespace nlp {

typedef syntaxnet::ParserState SyntaxnetState;
typedef syntaxnet::ParserAction SyntaxnetAction;

SemparTransitionState::SemparTransitionState(
    const SharedResources *resources,
    const GoldTransitionGenerator *gold_transition_generator) {
  QCHECK(resources != nullptr);
  QCHECK(resources->global != nullptr);
  state_ = nullptr;
  document_ = nullptr;
  shared_ = new Shared();

  // Make a new local store for frame construction and take ownership.
  shared_->resources = resources;
  shared_->store = new Store(shared_->resources->global);
  shared_->store_owned = true;

  QCHECK(!table()->action_checks());
  shared_->gold_transition_generator = gold_transition_generator;
  allowed_.assign(table()->NumActions(), false);
}

SemparTransitionState::SemparTransitionState(
    const SharedResources *resources,
    const GoldTransitionGenerator *gold_transition_generator,
    Store *store) {
  state_ = nullptr;
  document_ = nullptr;
  shared_ = new Shared();

  // Use the passed store for frame construction.
  shared_->resources = resources;
  QCHECK(store->globals() == global());
  shared_->store = store;
  shared_->store_owned = false;

  QCHECK(!table()->action_checks());
  shared_->gold_transition_generator = gold_transition_generator;
  allowed_.assign(table()->NumActions(), false);
}

SemparTransitionState::SemparTransitionState(
    const SemparTransitionState &other) {
  state_ = (other.state_ == nullptr) ? nullptr : new ParserState(*other.state_);
  shared_ = other.shared_;
  shared_->AddRef();
  allowed_ = other.allowed_;
  step_info_ = other.step_info_;
  history_ = other.history_;
  next_gold_index_ = other.next_gold_index_;

  // TODO: Make this faster once cloning is actually used.
  document_ = nullptr;
  if (other.document_ != nullptr) {
    document_ = new Document(other.document_->top());
    document_->Update();
  }
}

SemparTransitionState::~SemparTransitionState() {
  if (shared_ != nullptr) shared_->Unref();
  delete state_;
  delete document_;
}

syntaxnet::ParserTransitionState *SemparTransitionState::Clone() const {
  LOG(FATAL) << "SemparTransitionState::Clone not implemented";
  return nullptr;
  // return new SemparTransitionState(*this);
}

void SemparTransitionState::AddParseToDocument(
    const SyntaxnetState &state,
    bool rewrite_root_labels,
    syntaxnet::Sentence *sentence) const {
  state_->AddParseToDocument(document_);
  document_->Update();
  sentence->SetExtension(FramedSentence::framing, Encode(document_->top()));
}

void SemparTransitionState::Init(SyntaxnetState *state) {
  QCHECK(shared_ != nullptr);

  // Make a new parser state.
  QCHECK(shared_->store != nullptr);
  const syntaxnet::Sentence &sentence = state->sentence();
  delete state_;
  state_ = new ParserState(shared_->store, 0, sentence.token_size());

  // Make a document.
  const string &encoded = sentence.GetExtension(FramedSentence::framing);
  document_ = new Document(Decode(shared_->store, encoded).AsFrame());
  document_->Update();

  // Clear the gold sequence. This will be lazily computed in NextGoldAction().
  shared_->gold_sequence.Clear();
  next_gold_index_ = 0;

  // Compute the set of allowed actions at the initial parser state.
  ComputeAllowed();
}

void SemparTransitionState::ComputeAllowed() {
  // Disable all actions by default.
  allowed_.assign(allowed_.size(), false);
  const ActionTable &table = shared_->resources->table;

  // If we are at the end, then STOP is the only allowed action.
  if ((state_->current() == state_->end()) || state_->done()) {
    allowed_[table.StopIndex()] = true;
    return;
  }

  // If we have taken too many actions at this token, then just advance.
  // We use a small padding on the action limit to allow for variations not
  // seen in the training corpus.
  if (step_info_.NumStepsSinceShift() > 4 + table.max_actions_per_token()) {
    allowed_[table.ShiftIndex()] = true;
    return;
  }

  // Compute the rest of the allowed actions as per the action table.
  table.Allowed(*state_, {} /* fingerprints */, &allowed_);
}

SyntaxnetAction SemparTransitionState::NextGoldAction() const {
  if (shared_->gold_sequence.actions().empty()) {
    shared_->gold_transition_generator->Generate(
        *document_,
        state_->begin(),
        state_->end(),
        &shared_->gold_sequence,
        nullptr  /* report */);
    if (state_->end() > state_->begin()) {
      QCHECK_GT(shared_->gold_sequence.actions().size(), 0);
    }
    next_gold_index_ = 0;
  }

  const ParserAction &action = shared_->gold_sequence.action(next_gold_index_);
  int index = table()->Index(action);
  LOG_IF(FATAL, index == -1) << action.ToString(shared_->store);
  gold_transitions_being_reported_ = true;

  return index;
}

void SemparTransitionState::Apply(SyntaxnetAction action_index) {
  const ParserAction &action = table()->Action(action_index);

  if (gold_transitions_being_reported_) {
    // If we are truly in training mode, then only gold actions are applicable.
    const ParserAction &expected =
        shared_->gold_sequence.action(next_gold_index_);
    if (action != expected) {
      string debug = "Given gold action != expected gold action.";
      StrAppend(&debug, "\nParser State: ", state_->DebugString());
      StrAppend(&debug,
                      "\nExpected : ", expected.ToString(shared_->store));
      StrAppend(&debug, "\nGot : ", action.ToString(shared_->store));
      LOG(FATAL) << debug;
    }

    // Since the action table only allows a large percentile of all actions,
    // it is possible that the gold action is not allowed as per the table.
    // If so, explicitly whitelist the action.
    if (!allowed_[action_index]) {
      LOG_FIRST_N(WARNING, 50) << "Forcibly enabling disallowed gold action: "
          << action.ToString(shared_->store);
      allowed_[action_index] = true;
    }
    next_gold_index_++;
  }

  CHECK(allowed_[action_index]) << "Action not allowed:"
      << action.ToString(global()) << " at state:\n"
      << state_->DebugString();

  CHECK(state_->Apply(action))
      << action.ToString(global()) << " at state:\n"
      << state_->DebugString();

  // Update history.
  history_.emplace_back(action_index);
  if (history_.size() > kMaxHistory) {
    history_.erase(history_.begin(), history_.begin() + 1);
  }

  // Update step information.
  step_info_.Update(action, *state_);

  // Compute the set of allowed actions for the resulting state.
  ComputeAllowed();
}

void SemparTransitionState::StepInformation::Update(
    const ParserAction &action,
    const ParserState &state) {
  // Note that except for SHIFT and STOP, all actions set the focus.
  bool focus_set =
      (action.type != ParserAction::SHIFT) &&
      (action.type != ParserAction::STOP);
  if (focus_set && state.AttentionSize() > 0) {
    int focus = state.Attention(0);
    if (creation_step.size() < focus + 1) {
      creation_step.resize(focus + 1);
      creation_step[focus] = steps;
    }
    if (focus_step.size() < focus + 1) focus_step.resize(focus + 1);
    focus_step[focus] = steps;
  }
  steps++;
  steps_since_shift = (action.type == ParserAction::SHIFT) ? 0 :
    (steps_since_shift + 1);
}

}  // namespace nlp
}  // namespace sling
