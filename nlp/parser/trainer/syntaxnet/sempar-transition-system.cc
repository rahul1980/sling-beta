// Copyright 2016 Google Inc. All Rights Reserved.
// Author: grahul@google.com (Rahul Gupta)

#include "nlp/parser/trainer/syntaxnet/sempar-transition-system.h"

namespace sling {
namespace nlp {

using syntaxnet::TaskContext;
typedef syntaxnet::ParserState SyntaxnetState;
typedef syntaxnet::ParserAction SyntaxnetAction;

void SemparTransitionSystem::Setup(TaskContext *context) {
  // Specify the need for a common store and the action table.
  context->GetInput("commons", "store", "encoded");
  context->GetInput("action-table", "store", "encoded");
}

void SemparTransitionSystem::Init(TaskContext *context) {
  resources_.LoadGlobalStore(
      TaskContext::InputFile(*context->GetInput("commons")));
  resources_.LoadActionTable(
      TaskContext::InputFile(*context->GetInput("action-table")));
  gold_transition_generator_.Init(global());
}

// TODO: Following short-cut doesn't compile.
REGISTER_TRANSITION_SYSTEM("sempar", SemparTransitionSystem);
//REGISTER_SYNTAXNET_CLASS_COMPONENT(syntaxnet::ParserTransitionSystem,
//                                   "sempar",
//                                   SemparTransitionSystem);

}  // namespace nlp
}  // namespace sling
