// Features for semantic parsing.

#include <hash_map>
#include <unordered_map>

#include "nlp/parser/action-table.h"
#include "nlp/parser/parser-state.h"
#include "nlp/parser/trainer/shared-resources.h"
#include "nlp/parser/trainer/syntaxnet/sempar-transition-state.h"
#include "syntaxnet/parser_features.h"
#include "syntaxnet/task_context.h"

namespace sling {
namespace nlp {

using syntaxnet::FeatureValue;
using syntaxnet::FeatureVector;
using syntaxnet::NumericFeatureType;
using syntaxnet::ParserFeatureFunction;
using syntaxnet::ParserIndexFeatureFunction;
using syntaxnet::ParserLocator;
using syntaxnet::TaskContext;
using syntaxnet::WorkspaceSet;
using SyntaxnetState = syntaxnet::ParserState;

namespace {

const SemparTransitionState *State(const SyntaxnetState &state) {
  return dynamic_cast<const SemparTransitionState *>(state.transition_state());
}

}  // namespace

// Locator that returns an index into the attention buffer.
class AttentionIndexLocator : public ParserLocator<AttentionIndexLocator> {
 public:
  int GetFocus(const WorkspaceSet &workspaces,
               const SyntaxnetState &state) const {
    int offset = argument();
    int size = State(state)->parser_state()->AttentionSize();
    if (offset < 0 || offset >= size) {
      return -2;
    } else {
      return offset;
    }
  }
};

REGISTER_PARSER_FEATURE_FUNCTION("attention", AttentionIndexLocator);

// Returns the index of the step that created the frame in the attention buffer.
// Example use: attention(0).creation-step.
class FrameCreationStepFeatureFunction : public ParserIndexFeatureFunction {
 public:
  FrameCreationStepFeatureFunction() {}

  void Init(TaskContext *context) override {
    set_feature_type(new NumericFeatureType(name(), kMaxStep));
  }

  // Returns the step index (to a maximum of kMaxStep) for the focus frame.
  void Evaluate(const WorkspaceSet &workspaces,
                const SyntaxnetState &state,
                int focus,
                FeatureVector *result) const override {
    FeatureValue value;
    if (focus < 0) {
      value = -1;
    } else {
     int step = State(state)->CreationStep(focus);
     value = (step > kMaxStep - 1) ? (kMaxStep - 1) : step;
    }
    result->add(feature_type(), value);
  }

 private:
  static const int kMaxStep = 500;
};

REGISTER_PARSER_IDX_FEATURE_FUNCTION("creation-step",
                                     FrameCreationStepFeatureFunction);

// Returns the index of the step that most recently brought the frame to the
// front of the attention buffer.
// Example use: attention(0).focus-step.
class FrameFocusStepFeatureFunction : public ParserIndexFeatureFunction {
 public:
  FrameFocusStepFeatureFunction() {}

  void Init(TaskContext *context) override {
    set_feature_type(new NumericFeatureType(name(), kMaxStep));
  }

  // Returns the step index (to a maximum of kMaxStep) for the focus frame.
  void Evaluate(const WorkspaceSet &workspaces,
                const SyntaxnetState &state,
                int focus,
                FeatureVector *result) const override {
    FeatureValue value;
    if (focus < 0) {
      value = -1;
    } else {
      int step = State(state)->FocusStep(focus);
      value = (step > kMaxStep - 1) ? (kMaxStep - 1) : step;
    }
    result->add(feature_type(), value);
  }

 private:
  static const int kMaxStep = 500;
};

REGISTER_PARSER_IDX_FEATURE_FUNCTION("focus-step",
                                     FrameFocusStepFeatureFunction);

// Returns the end token of the mention that evoked the focus frame (or -1).
// Example use: attention(0).frame-end.
class FrameEndFeatureFunction : public ParserIndexFeatureFunction {
 public:
  FrameEndFeatureFunction() {}

  void Init(TaskContext *context) override {
    set_feature_type(new NumericFeatureType(name(), kMaxEnd));
  }

  // Returns the end token index (to a maximum of kMaxEnd).
  void Evaluate(const WorkspaceSet &workspaces,
                const SyntaxnetState &state,
                int focus,
                FeatureVector *result) const override {
    FeatureValue value;
    if (focus < 0) {
      value = -1;
    } else {
      const ParserState *s = State(state)->parser_state();
      int frame = s->Attention(focus);
      int end = s->FrameEvokeEnd(frame) - 1;  // inclusive
      value = (end > kMaxEnd - 1) ? (kMaxEnd  - 1) : end;
    }
    result->add(feature_type(), value);
  }

 private:
  static const int kMaxEnd = 200;
};

REGISTER_PARSER_IDX_FEATURE_FUNCTION("frame-end", FrameEndFeatureFunction);

// Returns the roles of the top few frames as: (i, r), (r, j), (i, r, j), (i, j)
// where i and j are attention indices of frames and r is a role that connects
// those frames.
class FrameRolesFeatureFunction : public ParserFeatureFunction {
 public:
  // Declare the need for the action table and commons so that we can look up
  // the subset of roles that are of interest.
  void Setup(TaskContext *context) override {
    context->GetInput("commons", "store", "encoded");
    context->GetInput("action-table", "store", "encoded");
  }

  // Reads the set of all roles seen in actions in the training data.
  // The roles are paired with the attention index of the frame.
  void Init(TaskContext *context) override {
    resources_.LoadGlobalStore(
        TaskContext::InputFile(*context->GetInput("commons")));
    resources_.LoadActionTable(
        TaskContext::InputFile(*context->GetInput("action-table")));

    // Get the set of roles that connect two frames.
    for (int i = 0; i < resources_.table.NumActions(); ++i) {
      const auto &action = resources_.table.Action(i);
      if (action.type == ParserAction::CONNECT ||
          action.type == ParserAction::EMBED ||
          action.type == ParserAction::ELABORATE) {
        if (roles_.find(action.role) == roles_.end()) {
          int index = roles_.size();
          roles_[action.role] = index;
        }
      }
    }

    // Compute the offsets for the four types of features. These are laid out
    // in this order: all (i, r) features, all (r, j) features, all (i, j)
    // features, all (i, r, j) features.
    // We restrict i, j to be < frame-limit, a feature parameter.
    frame_limit_ = GetIntParameter("frame-limit", 5);
    int combinations = frame_limit_ * roles_.size();
    outlink_offset_ = 0;
    inlink_offset_ = outlink_offset_ + combinations;
    unlabeled_link_offset_ = inlink_offset_ + combinations;
    labeled_link_offset_ = unlabeled_link_offset_ + frame_limit_ * frame_limit_;
    int size = labeled_link_offset_ + frame_limit_ * combinations + 1;

    set_feature_type(new NumericFeatureType(name(), size));
  }

  // Returns the four types of features.
  void Evaluate(const WorkspaceSet &workspaces,
                const SyntaxnetState &state,
                FeatureVector *result) const override {
    const ParserState *s = State(state)->parser_state();

    // Construct a mapping from absolute frame index -> attention index.
    std::unordered_map<int, int> frame_to_attention;
    for (int i = 0; i < frame_limit_; ++i) {
      if (i < s->AttentionSize()) {
        frame_to_attention[s->Attention(i)] = i;
      } else {
        break;
      }
    }

    // Output features.
    for (const auto &kv : frame_to_attention) {
      // Attention index of the source frame.
      int source = kv.second;
      int outlink_base = outlink_offset_ + source * roles_.size();

      // Go over each slot of the source frame.
      Handle handle = s->frame(kv.first);
      const sling::FrameDatum *frame = s->store()->GetFrame(handle);
      for (const Slot *slot = frame->begin(); slot < frame->end(); ++slot) {
        const auto &it = roles_.find(slot->name);
        if (it == roles_.end()) continue;

        int role = it->second;
        result->add(feature_type(), outlink_base + role);  // (source, role)
        if (slot->value.IsIndex()) {
          const auto &it2 = frame_to_attention.find(slot->value.AsIndex());
          if (it2 != frame_to_attention.end()) {
            // Attention index of the target frame.
            int target = it2->second;
            result->add(                             // (role, target)
                feature_type(),
                inlink_offset_ + target * roles_.size() + role);
            result->add(                             // (source, target)
                feature_type(),
                unlabeled_link_offset_ + source * frame_limit_ + target);
            result->add(                             // (source, role, target)
                feature_type(),
                labeled_link_offset_ +
                source * frame_limit_ * roles_.size() +
                target * roles_.size() + role);
          }
        }
      }
    }
  }

 private:
  // Shared resources.
  SharedResources resources_;

  // Maximum attention index considered (exclusive).
  int frame_limit_;

  // Starting offset for (source, role) features.
  int outlink_offset_;

  // Starting offset for (role, target) features.
  int inlink_offset_;

  // Starting offset for (source, target) features.
  int unlabeled_link_offset_;

  // Starting offset for (source, role, target) features.
  int labeled_link_offset_;

  // Set of roles considered.
  HandleMap<int> roles_;
};

REGISTER_PARSER_FEATURE_FUNCTION("roles", FrameRolesFeatureFunction);

}  // namespace nlp
}  // namespace sling
