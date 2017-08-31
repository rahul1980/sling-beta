// Transition state for semantic parsing.

#ifndef NLP_SAFT_SLING_PARSER_NEUROSIS_TRANSITION_STATE_H_
#define NLP_SAFT_SLING_PARSER_NEUROSIS_TRANSITION_STATE_H_

#include <vector>

#include "base/logging.h"
#include "nlp/parser/action-table.h"
#include "nlp/parser/parser-action.h"
#include "nlp/parser/parser-state.h"
#include "nlp/parser/trainer/gold-transition-generator.h"
#include "nlp/parser/trainer/shared-resources.h"
#include "frame/store.h"
#include "syntaxnet/parser_state.h"
#include "syntaxnet/parser_transitions.h"

namespace sling {
namespace nlp {

// In what follows, ParserState and ParserAction refer to SLING's classes.
// The identically named types in Syntaxnet have been aliased to SyntaxnetState
// and SyntaxnetAction respectively to avoid confusion.
// Also recall that SyntaxnetAction is just a typedef for an int.
//
// This class subclasses Syntaxnet's ParserTransitionState interface for
// semantic parsing. It consists of and maintains:
// - A ParserState object, which represents the frame graph.
// - Allowed actions (as a bitmap) for the current ParserState.
// - A short history of the most recent actions applied to the state.
// - Pointers to many refcounted objects that need not be cloned when the state
//   is cloned. Some of these pointers are only valid in the training mode.
//
// In training mode, the state expects a pointer to a GoldTransitionGenerator.
// It is used to eagerly compute and cache the full gold sequence for the input.
class SemparTransitionState : public syntaxnet::ParserTransitionState {
 public:
  // Aliases to avoid confusion.
  typedef syntaxnet::ParserState SyntaxnetState;
  typedef syntaxnet::ParserAction SyntaxnetAction;

  // Constructors.
  // 'gold_transition_generator' should be nullptr when not in training mode.
  SemparTransitionState(
    const SharedResources *resources,
    const GoldTransitionGenerator *gold_transition_generator);

  // Same as above, except that any frame construction will be done in 'store'.
  SemparTransitionState(
    const SharedResources *resources,
    const GoldTransitionGenerator *gold_transition_generator,
    Store *store);

  // Copy constructor.
  SemparTransitionState(const SemparTransitionState &other);

  // Implementation of the ParserTransitionState interface.
  ~SemparTransitionState() override;

  syntaxnet::ParserTransitionState *Clone() const override;

  void Init(SyntaxnetState *state) override;

  void AddParseToDocument(
      const SyntaxnetState &state,
      bool rewrite_root_labels,
      syntaxnet::Sentence *sentence) const override;

  bool IsTokenCorrect(const SyntaxnetState &state, int index) const override {
    return true;  // unused
  }

  string ToString(const SyntaxnetState &state) const override {
    return state_->DebugString();
  }

  // Returns true if 'action' is allowed at the current state.
  bool Allowed(SyntaxnetAction action) const {
    return allowed_[action];
  }

  // Returns the index of the next gold action. Only valid when we are
  // in training mode.
  SyntaxnetAction NextGoldAction() const;

  // Applies 'action_index' to the parser state, and computes the next set of
  // allowed actions for the resulting state.
  // In training mode it is assumed that 'action_index' is a gold action for
  // the current state, and the next gold action is computed accordingly.
  void Apply(SyntaxnetAction action_index);

  // Returns whether the parser state is in a final state.
  bool Done() const {
    return state_->done();
  }

  // Returns the size of the actions history.
  int HistorySize() const { return history_.size(); }

  // Returns an action from the history, where offset = 0 corresponds to the
  // latest action.
  SyntaxnetAction History(int offset) const {
    return history_[history_.size() - 1 - offset];
  }

  // Accessors.
  const ActionTable *table() const {
    return shared_ == nullptr ? nullptr : &shared_->resources->table;
  }

  Store *store() const {
    return shared_ == nullptr ? nullptr : shared_->store;
  }

  Store *global() const {
    return shared_ == nullptr ? nullptr : shared_->resources->global;
  }

  const GoldTransitionGenerator *gold_transition_generator() const {
    return shared_ == nullptr ? nullptr : shared_->gold_transition_generator;
  }

  const ParserState *parser_state() const { return state_; }

  // Returns the index of the step which created/focused the frame at
  // position 'index' in the attention buffer.
  int CreationStep(int index) const {
    if (index < 0 || index >= state_->AttentionSize()) return -1;
    return step_info_.CreationStep(state_->Attention(index));
  }

  int FocusStep(int index) const {
    if (index < 0 || index >= state_->AttentionSize()) return -1;
    return step_info_.FocusStep(state_->Attention(index));
  }

  // Returns the number of steps taken by the state so far.
  int NumSteps() const { return step_info_.NumSteps(); }

 private:
  // Information that is shared across clones of the state, and thus only needs
  // to be refcounted instead of copied.
  struct Shared {
    // Gold transition generator. Not owned. Only used during training.
    const GoldTransitionGenerator *gold_transition_generator = nullptr;

    // Gold sequence for the token range. Only populated during training.
    GoldTransitionSequence gold_sequence;

    // Shared resources. Not owned.
    const SharedResources *resources = nullptr;

    // Local store used by the ParserState. Owned if store_owned = true.
    Store *store = nullptr;
    bool store_owned = true;

    // TODO: Guard ref_count via mutex.
    void AddRef() {
      ++ref_count;
    }

    void Unref() {
      --ref_count;
      if (ref_count == 0) {
        if (store_owned) delete store;
      }
    }

    int ref_count = 1;
  };

  // Holds frame -> step information, i.e. at which step was a frame created or
  // brought to focus.
  struct StepInformation {
    // Number of steps (i.e. actions) taken so far.
    int steps = 0;

    // Number of steps since the last shift action.
    int steps_since_shift = 0;

    // Absolute frame index -> Step at which the frame was created.
    std::vector<int> creation_step;

    // Absolute frame index -> Most recent step at which the frame was focused.
    std::vector<int> focus_step;

    // Accessor.
    int NumSteps() const { return steps; }
    int NumStepsSinceShift() const { return steps_since_shift; }

    // Updates the step information using 'action' that resulted in 'state'.
    void Update(const ParserAction &action, const ParserState &state);

    // Returns the creation/focus step for the frame at absolute index 'index'.
    int CreationStep(int index) const { return creation_step[index]; }
    int FocusStep(int index) const { return focus_step[index]; }
  };

  // Computes the set of allowed actions for the current ParserState.
  void ComputeAllowed();

  // Shared information. Refcounted.
  Shared *shared_ = nullptr;

  // ParserState. Owned.
  ParserState *state_ = nullptr;

  // Document. Owned.
  Document *document_ = nullptr;

  // Bitmap of allowed actions for 'state_'.
  std::vector<bool> allowed_;

  // Step information.
  StepInformation step_info_;

  // History of the last few actions.
  std::vector<SyntaxnetAction> history_;

  // Maximum size of the history.
  static const int kMaxHistory = 10;

  // Used for sanity checking. This is mutable for use in a const method.
  mutable bool gold_transitions_being_reported_ = false;

  // Index of the next gold action to be output. This is an index into
  // shared_->gold_sequence. Only used during training.
  mutable int next_gold_index_ = 0;
};

}  // namespace nlp
}  // namespace sling

#endif  // NLP_SAFT_SLING_PARSER_NEUROSIS_TRANSITION_STATE_H_
