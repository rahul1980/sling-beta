// Transition system for semantic parsing.

#ifndef NLP_SAFT_SLING_PARSER_NEUROSIS_TRANSITION_SYSTEM_H_
#define NLP_SAFT_SLING_PARSER_NEUROSIS_TRANSITION_SYSTEM_H_

#include "base/logging.h"
#include "nlp/parser/action-table.h"
#include "nlp/parser/parser-action.h"
#include "nlp/parser/parser-state.h"
#include "nlp/parser/trainer/gold-transition-generator.h"
#include "nlp/parser/trainer/shared-resources.h"
#include "nlp/parser/trainer/syntaxnet/sempar-transition-state.h"
#include "syntaxnet/parser_state.h"
#include "syntaxnet/parser_transitions.h"
#include "syntaxnet/task_context.h"

namespace sling {
namespace nlp {

// This class is the interface between Syntaxnet and the semantic parsing task.
// Its main job is to create SemparTransitionState and delegate all
// the work to it.
class SemparTransitionSystem : public syntaxnet::ParserTransitionSystem {
 public:
  // Rename Syntaxnet classes to avoid confusion.
  typedef syntaxnet::ParserState SyntaxnetState;
  typedef syntaxnet::ParserAction SyntaxnetAction;

  // ParserTransitionSystem implementation.
  // Setup/Init for the TransitionSystem.
  void Setup(syntaxnet::TaskContext *context) override;
  void Init(syntaxnet::TaskContext *context) override;

  // Returns the number of actions / action-types.
  // TODO: Replace 8 with ParserAction::NUM_ACTIONS.
  int NumActionTypes() const override { return 8; }
  int NumActions(int labels) const override { return table().NumActions(); }

  // Returns the next gold action for 'state'.
  SyntaxnetAction GetNextGoldAction(
      const SyntaxnetState &state) const override {
    return State(state).NextGoldAction();
  }

  // Returns whether 'action' is allowed on 'state'.
  bool IsAllowedAction(
      SyntaxnetAction action, const SyntaxnetState &state) const override {
    return State(state).Allowed(action);
  }

  // Performs 'action' on 'state'.
  void PerformActionWithoutHistory(
      SyntaxnetAction action, SyntaxnetState *state) const override {
    if (action == table().ShiftIndex()) {
      // Explicitly register SHIFT with SyntaxnetState so that the latter can
      // update its current input pointer, and can report features correctly.
      DCHECK(!state->EndOfInput());
      state->Advance();
    }
    MutableState(state)->Apply(action);
  }

  // Returns whether the transition system is deterministic.
  bool IsDeterministicState(const SyntaxnetState &state) const override {
    return false;
  }

  // Returns whether 'state' is final or not.
  bool IsFinalState(const SyntaxnetState &state) const override {
    return State(state).Done();
  }

  // Returns a string representation of an action.
  string ActionAsString(
      SyntaxnetAction action, const SyntaxnetState &state) const override {
    return table().Action(action).ToString(resources_.global);
  }

  // Creates a new TransitionState. Ignores the 'training' parameter.
  syntaxnet::ParserTransitionState *NewTransitionState(
      bool training) const override {
    return new SemparTransitionState(&resources_, &gold_transition_generator_);
  }

  // Whether we should back off to the highest scoring allowed transition.
  bool BackOffToBestAllowableTransition() const override { return true; }

  // Returns the default action.
  SyntaxnetAction GetDefaultAction(const SyntaxnetState &state) const override {
    return state.EndOfInput() ? table().StopIndex() : table().ShiftIndex();
  }

  // Debugging methods.
  bool SupportsActionMetaData() const override { return false; }

  int ParentIndex(const SyntaxnetState &state,
                  const SyntaxnetAction &action) const override {
    return -1;
  }

  // Methods specific to this class.
  int NumActions() const { return NumActions(0); }

  // Const and mutable accessors to the transition state inside 'state'.
  const SemparTransitionState &State(const SyntaxnetState &state) const {
    return *dynamic_cast<const SemparTransitionState *>(state.transition_state());
  }

  SemparTransitionState *MutableState(SyntaxnetState *state) const {
    return dynamic_cast<SemparTransitionState *>(
        state->mutable_transition_state());
  }

  // A special version of NewTransitionState that passes a pre-existing store
  // to the newly built state.
  syntaxnet::ParserTransitionState *NewTransitionState(
      bool training, Store *store) const {
    return new SemparTransitionState(
        &resources_, &gold_transition_generator_, store);
  }

  // Accessors.
  Store *global() const { return resources_.global; }
  const ActionTable &table() const { return resources_.table; }
  const GoldTransitionGenerator &gold_transition_generator() const {
    return gold_transition_generator_;
  }

 private:
  // Shared resources.
  SharedResources resources_;

  // Gold sequence generator used only during training.
  GoldTransitionGenerator gold_transition_generator_;
};

}  // namespace nlp
}  // namespace sling

#endif  // NLP_SAFT_SLING_PARSER_NEUROSIS_TRANSITION_SYSTEM_H_
