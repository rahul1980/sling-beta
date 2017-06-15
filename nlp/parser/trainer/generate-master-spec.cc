// Copyright 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Utility tool for generating a fully populated master spec.
// In particular, it creates the action table, all resources needed by the
// features, computes the feature domain sizes and uses all this to output
// a full MasterSpec.
//
// Input arguments:
// - Path to a commons store.
// - File pattern of training documents.
// - Name of output directory.
//
// Sample usage:
//   bazel-bin/nlp/parser/trainer/generate-master-spec
//       --documents='/tmp/documents.*'
//       --commons=/tmp/common_store.encoded
//       --output_dir='/tmp/out'

#include <string>
#include <vector>

#include "base/flags.h"
#include "base/init.h"
#include "base/logging.h"
#include "base/macros.h"
#include "dragnn/protos/spec.pb.h"
#include "file/file.h"
#include "frame/object.h"
#include "frame/serialization.h"
#include "frame/store.h"
#include "nlp/parser/trainer/action-table-generator.h"
#include "nlp/parser/trainer/feature.h"
#include "nlp/parser/trainer/shared-resources.h"
#include "string/strcat.h"
#include "syntaxnet/affix.h"
#include "syntaxnet/task_context.h"
#include "syntaxnet/task_spec.pb.h"
#include "syntaxnet/term_frequency_map.h"
#include "syntaxnet/utils.h"
#include "tensorflow/core/lib/strings/str_util.h"

using sling::File;
using sling::FileDecoder;
using sling::Object;
using sling::Store;
using sling::nlp::ActionTable;
using sling::nlp::ActionTableGenerator;
using sling::nlp::SemparFeatureExtractor;
using sling::nlp::SharedResources;
using sling::nlp::Document;

using syntaxnet::dragnn::ComponentSpec;
using syntaxnet::dragnn::MasterSpec;
using syntaxnet::dragnn::RegisteredModuleSpec;

DEFINE_string(documents, "", "File pattern of training documents.");
DEFINE_string(commons, "", "Path to common store.");
DEFINE_string(output_dir, "/tmp/sempar_out", "Output directory.");

// Various options for generating the action table, lexicons, spec.
constexpr int kActionTableCoveragePercentile = 99;
constexpr bool kActionTableFromPerSentence = true;
constexpr int kLexiconMaxPrefixLength = 3;
constexpr int kLexiconMaxSuffixLength = 3;

// Workspace for various artifacts that are used/created by this tool.
struct Artifacts {
  SharedResources resources;

  std::vector<string> train_files;  // all training documents
  string commons_filename;          // full path to commons
  string action_table_filename;     // full path of generated action table
  MasterSpec spec;                  // generated master spec
  string spec_file;                 // path to the master spec

  // Lexicon name -> Full path to the generated lexicon.
  std::unordered_map<string, string> lexicon_paths;

  Store *global() { return resources.global; }
  const ActionTable &table() { return resources.table; }
};

// Returns the full output path for 'basename'.
string FullName(const string &basename) {
  string s = FLAGS_output_dir;
  CHECK(!s.empty());
  if (s.back() == '/') s.pop_back();
  return sling::StrCat(s, "/", basename);
}

void OutputActionTable(Artifacts *artifacts) {
  ActionTableGenerator generator(artifacts->global());
  generator.set_coverage_percentile(kActionTableCoveragePercentile);
  generator.set_per_sentence(kActionTableFromPerSentence);

  LOG(INFO) << "Processing " << artifacts->train_files.size() << " documents..";
  int count = 0;
  for (const string &file : artifacts->train_files) {
    Store local(artifacts->global());
    FileDecoder decoder(&local, file);
    Object top = decoder.Decode();
    if (top.invalid()) continue;

    count++;
    Document document(top.AsFrame());
    generator.Add(document);
    if (count % 100 == 1) LOG(INFO) << count << " documents processed.";
  }
  LOG(INFO) << "Processed " << count << " documents.";

  string table_file = FullName("table");
  string summary_file = FullName("table.summary");
  string unknown_file = FullName("table.unknown_symbols");
  generator.Save(table_file, summary_file, unknown_file);
  artifacts->action_table_filename = table_file;

  LOG(INFO) << "Wrote action table to " << table_file
            << ", " << summary_file << ", " << unknown_file;
  artifacts->resources.LoadActionTable(table_file);
}

void WriteAffixTable(const syntaxnet::AffixTable &affixes,
                     const string &output_file) {
  syntaxnet::ProtoRecordWriter writer(output_file);
  affixes.Write(&writer);
}

/*
void OutputLexicons(Artifacts *artifacts) {
  using syntaxnet::TermFrequencyMap;
  using syntaxnet::AffixTable;

  // Term frequency maps to be populated by the corpus.
  TermFrequencyMap words;
  TermFrequencyMap lcwords;

  // Affix tables to be populated by the corpus.
  AffixTable prefixes(AffixTable::PREFIX, kLexiconMaxPrefixLength);
  AffixTable suffixes(AffixTable::SUFFIX, kLexiconMaxSuffixLength);

  // Make a pass over the corpus.
  int64 num_tokens = 0;
  int64 num_documents = 0;
  for (const string &file : artifacts->train_files) {
    Store local(artifacts->global());
    FileDecoder decoder(&local, file);
    Object top = decoder.Decode();
    if (top.invalid()) continue;

    num_documents++;
    Document document(top.AsFrame());

    // Gather token information.
    for (int t = 0; t < document.num_tokens(); ++t) {
      // Get token and lowercased word.
      const auto &token = document.token(t);
      string word = token.text();
      syntaxnet::utils::NormalizeDigits(&word);
      string lcword = tensorflow::str_util::Lowercase(word);

      // Make sure the token does not contain a newline.
      CHECK(lcword.find('\n') == string::npos);

      // Increment frequencies (only for terms that exist).
      if (!word.empty() && !HasSpaces(word)) words.Increment(word);
      if (!lcword.empty() && !HasSpaces(lcword)) lcwords.Increment(lcword);

      // Add prefixes/suffixes for the current word.
      prefixes.AddAffixesForWord(word.c_str(), word.size());
      suffixes.AddAffixesForWord(word.c_str(), word.size());

      // Update the number of processed tokens.
      ++num_tokens;
    }
  }
  LOG(INFO) << "Term maps collected over " << num_tokens << " tokens from "
            << num_documents << " documents";

  // Write mappings to disk.
  words.Save(FullName("word-map"));
  lcwords.Save(FullName("lcword-map"));

  // Write affixes to disk.
  WriteAffixTable(prefixes, FullName("prefix-table"));
  WriteAffixTable(suffixes, FullName("suffix-table"));

  artifacts->lexicon_paths["word-map"] = FullName("word-map");
  artifacts->lexicon_paths["lcword-map"] = FullName("lcword-map");
  artifacts->lexicon_paths["prefix-table"] = FullName("prefix-table");
  artifacts->lexicon_paths["suffix-table"] = FullName("suffix-table");
  LOG(INFO) << "Wrote term maps and affix tables.";
} */

syntaxnet::dragnn::ComponentSpec *AddComponent(
    const string &name,
    const string &backend,
    const string &network_unit,
    const string &transition_system,
    Artifacts *artifacts) {
  auto *c = artifacts->spec.add_component();
  c->set_name(name);
  c->mutable_backend()->set_registered_name(backend);
  c->mutable_network_unit()->set_registered_name(network_unit);
  c->mutable_transition_system()->set_registered_name(transition_system);
  c->mutable_component_builder()->set_registered_name(
      "DynamicComponentBuilder");
  return c;
}

void SetParam(RegisteredModuleSpec *spec,
              const string &key,
              const string &value) {
  (*spec->mutable_parameters())[key] = value;
}

void AddFixedFeature(ComponentSpec *component,
                     const string &name,
                     const string &fml,
                     int embedding_dim) {
  auto *f = component->add_fixed_feature();
  f->set_name(name);
  f->set_fml(fml);
  f->set_embedding_dim(embedding_dim);
  f->set_predicate_map("hashed");
}

void AddLinkedFeature(ComponentSpec *component,
                      const string &name,
                      const string &fml_pattern,
                      int fml_arg_max,
                      int embedding_dim,
                      const string &source,
                      const string &translator) {
  // Replace "XX" with i, for every i in [0, fml_arg_max).
  // e.g. constant(XX) would yield constant(0), constant(1) etc.
  string fml;
  size_t i = fml_pattern.find("XX");
  if (i == string::npos) {
    fml = fml_pattern;
  } else {
    for (int j = 0; j < fml_arg_max; ++j) {
      if (j > 0) sling::StrAppend(&fml, " ");
      sling::StrAppend(
          &fml, fml_pattern.substr(0, i), j, fml_pattern.substr(i + 2));
    }
  }
  auto *f = component->add_linked_feature();
  f->set_name(name);
  f->set_fml(fml);
  f->set_embedding_dim(embedding_dim);
  f->set_source_component(source);
  f->set_source_translator(translator);
  f->set_source_layer("layer_0");
  std::vector<string> parts = syntaxnet::utils::Split(fml, ' ');
  f->set_size(parts.size());
}

void AddLinkedFeature(ComponentSpec *component,
                      const string &name,
                      const string &fml_pattern,
                      int fml_arg_max,
                      int embedding_dim,
                      const string &source) {
  AddLinkedFeature(component, name, fml_pattern, fml_arg_max,
                   embedding_dim, source, "identity");
}

void AddResource(ComponentSpec *spec,
                 const string &name,
                 const string &file_pattern,
                 const string &format,
                 const string &record) {
  auto *r = spec->add_resource();
  r->set_name(name);
  auto *part = r->add_part();
  part->set_file_pattern(file_pattern);
  part->set_file_format(format);
  part->set_record_format(record);
}

// Finds and fills the domain size of all fixed features.
// This is done by calling ParserEmbeddingFeatureExtractor::Init(),
// which in turn requires us to create the requisite task context.
void TrainFeatures(Artifacts *artifacts, ComponentSpec *spec) {
  using sling::StrAppend;
  using sling::StrCat;

  SemparFeatureExtractor fixed_feature_extractor;
  for (const auto &fixed_channel : spec->fixed_feature()) {
    fixed_feature_extractor.AddChannel(fixed_channel);
  }

  // Note: We are NOT copying spec->transition_system().parameters() over to
  // the features. Therefore any parameters for the features should be
  // specified in the FML itself.

  fixed_feature_extractor.Train(artifacts->train_files,
                                FLAGS_output_dir,
                                true /* fill vocabulary sizes */,
                                &artifacts->resources,
                                spec);

  SemparFeatureExtractor linked_feature_extractor;
  for (const auto &linked_channel : spec->linked_feature()) {
    linked_feature_extractor.AddChannel(linked_channel);
  }

  linked_feature_extractor.Train(
      artifacts->train_files,
      FLAGS_output_dir,
      false /* linked features don't need vocab sizes */,
      &artifacts->resources,
      spec);
}

void OutputMasterSpec(Artifacts *artifacts) {
  // Left to right LSTM.
  auto *lr_lstm = AddComponent(
      "lr_lstm", "SemparComponent", "LSTMNetwork", "shift-only", artifacts);
  SetParam(lr_lstm->mutable_transition_system(), "left_to_right", "true");
  SetParam(lr_lstm->mutable_network_unit(), "hidden_layer_sizes", "256");
  lr_lstm->set_num_actions(1);
  AddFixedFeature(lr_lstm, "words", "word", 32);
  AddFixedFeature(lr_lstm, "suffix", "suffix(length=2)", 16);
  AddFixedFeature(
      lr_lstm, "shape",
      "digit hyphen punctuation quote capitalization", 8);

  // Right to left LSTM.
  auto *rl_lstm = artifacts->spec.add_component();
  *rl_lstm = *lr_lstm;
  rl_lstm->set_name("rl_lstm");
  SetParam(rl_lstm->mutable_transition_system(), "left_to_right", "false");

  // Feed forward unit.
  auto *ff = AddComponent(
      "ff", "SemparComponent", "FeedForwardNetwork", "sempar", artifacts);
  ff->set_num_actions(artifacts->table().NumActions());
  //AddFixedFeature(ff, "roles", "roles", 16);
  AddLinkedFeature(
      ff, "frame-creation-steps", "frame-creation(XX)", 5, 64, "ff");
  AddLinkedFeature(
      ff, "frame-focus-steps", "frame-focus(XX)", 5, 64, "ff");
  AddLinkedFeature(
      ff, "frame-end-lr", "frame-end(XX)", 5, 32, "lr_lstm");
  AddLinkedFeature(
      ff, "frame-end-rl", "frame-end(XX)", 5, 32, "rl_lstm",
      "reverse_token");
  AddLinkedFeature(ff, "history", "constant(XX)", 4, 64, "ff");
  AddLinkedFeature(ff, "lr", "current-token", -1, 32, "lr_lstm");
  AddLinkedFeature(
      ff, "rl", "current-token", -1, 32, "rl_lstm", "reverse-token");

  // Add any resources required by the feed forward unit's features.
  AddResource(ff, "commons", artifacts->commons_filename, "store", "encoded");
  AddResource(
      ff, "action-table", artifacts->action_table_filename, "store", "encoded");

  // Fill vocabulary sizes and feature sizes. Recall that this will also
  // add any lexicons as resources (e.g. needed by input.word, input.suffix).
  TrainFeatures(artifacts, lr_lstm);
  TrainFeatures(artifacts, rl_lstm);
  TrainFeatures(artifacts, ff);

  // Dump the master spec.
  string spec_file = FullName("master_spec");
  CHECK_OK(File::WriteContents(spec_file, artifacts->spec.DebugString()));
  artifacts->spec_file = spec_file;
  LOG(INFO) << "Wrote master spec to " << spec_file;
}

int main(int argc, char **argv) {
  sling::InitProgram(&argc, &argv);

  CHECK(!FLAGS_documents.empty()) << "No documents specified.";
  CHECK(!FLAGS_commons.empty()) << "No commons specified.";
  CHECK(!FLAGS_output_dir.empty()) << "No output_dir specified.";

  if (!File::Exists(FLAGS_output_dir)) {
    CHECK_OK(File::Mkdir(FLAGS_output_dir));
  }

  Artifacts artifacts;
  artifacts.commons_filename = FLAGS_commons;
  artifacts.resources.LoadGlobalStore(FLAGS_commons);

  // Get a list of all training files.
  CHECK_OK(File::Match(FLAGS_documents, &artifacts.train_files));

  // Dump action table.
  OutputActionTable(&artifacts);

  // Make feature lexicons.
  //OutputLexicons(&artifacts);

  // Make master spec.
  OutputMasterSpec(&artifacts);

  return 0;
}