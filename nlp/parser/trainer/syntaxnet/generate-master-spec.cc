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
#include "nlp/document/document.h"
#include "nlp/document/document-source.h"
#include "nlp/parser/trainer/action-table-generator.h"
#include "nlp/parser/trainer/shared-resources.h"
#include "nlp/parser/trainer/syntaxnet/framed-sentence.pb.h"
#include "string/strcat.h"
#include "syntaxnet/affix.h"
#include "syntaxnet/dictionary.pb.h"
#include "syntaxnet/embedding_feature_extractor.h"
#include "syntaxnet/parser_transitions.h"
#include "syntaxnet/proto_io.h"
#include "syntaxnet/task_context.h"
#include "syntaxnet/term_frequency_map.h"
#include "syntaxnet/utils.h"
#include "tensorflow/core/lib/strings/str_util.h"


using sling::File;
using sling::FileDecoder;
using sling::Frame;
using sling::Store;
using sling::StrAppend;
using sling::StrCat;
using sling::nlp::ActionTable;
using sling::nlp::ActionTableGenerator;
using sling::nlp::SharedResources;
using sling::nlp::Document;
using sling::nlp::DocumentSource;

using syntaxnet::ProtoRecordWriter;
using syntaxnet::TermFrequencyMap;
using syntaxnet::AffixTable;

using syntaxnet::dragnn::ComponentSpec;
using syntaxnet::dragnn::MasterSpec;

using syntaxnet::utils::Join;
using syntaxnet::utils::Split;

DEFINE_string(documents, "", "File pattern of training documents.");
DEFINE_string(commons, "", "Path to common store.");
DEFINE_string(output_dir, "/tmp/sempar_out", "Output directory.");
DEFINE_int32(word_embeddings_dim, 32, "Word embeddings dimensionality.");
DEFINE_string(word_embeddings,
              "/usr/local/google/home/grahul/sempar_ontonotes/"
              "word2vec-embedding-bi-true-32.tf.recordio",
              "Pretrained word embeddings TF recordio. Should have a "
              "dimensionality of FLAGS_word_embeddings_dim.");
DEFINE_bool(oov_lstm_features, true,
            "Whether fallback features (shape, suffix etc) should "
            "be used in the LSTMs");
DEFINE_string(allowed_words_file, "", "File of allowed words");

// Various options for generating the action table, lexicons, spec.
constexpr int kActionTableCoveragePercentile = 99;
constexpr bool kActionTableFromPerSentence = true;
constexpr int kLexiconMaxPrefixLength = 3;
constexpr int kLexiconMaxSuffixLength = 3;

// Workspace for various artifacts that are used/created by this tool.
struct Artifacts {
  SharedResources resources;

  DocumentSource *train_corpus = nullptr;  // training corpus
  string commons_filename;                 // path to commons
  string action_table_filename;            // path of generated action table

  // Filenames of generated lexical resources.
  string prefix_table;
  string suffix_table;
  string word_map;
  string label_map;

  MasterSpec spec;                         // generated master spec
  string spec_file;                        // path to the master spec

  ~Artifacts() { delete train_corpus; }
  Store *global() { return resources.global; }
  const ActionTable &table() { return resources.table; }
};

string FileName(const string &file) {
  return StrCat(FLAGS_output_dir, "/", file);
}

void OutputActionTable(Artifacts *artifacts) {
  ActionTableGenerator generator(artifacts->global());
  generator.set_coverage_percentile(kActionTableCoveragePercentile);
  generator.set_per_sentence(kActionTableFromPerSentence);

  artifacts->train_corpus->Rewind();
  int count = 0;
  while (true) {
    Store store(artifacts->global());
    Document *document = artifacts->train_corpus->Next(&store);
    if (document == nullptr) break;

    count++;
    generator.Add(*document);
    if (count % 10000 == 1) LOG(INFO) << count << " documents processed.";
    delete document;
  }
  LOG(INFO) << "Processed " << count << " documents.";

  string table_file = FileName("table");
  string summary_file = FileName("table.summary");
  string unknown_file = FileName("table.unknown_symbols");
  generator.Save(table_file, summary_file, unknown_file);
  artifacts->action_table_filename = table_file;

  LOG(INFO) << "Wrote action table to " << table_file
            << ", " << summary_file << ", " << unknown_file;
  artifacts->resources.LoadActionTable(table_file);
}

// Returns true if the word contains spaces.
bool HasSpaces(const string &word) {
  for (char c : word) {
    if (c == ' ') return true;
  }
  return false;
}

// Writes an affix table to 'output_file'.
void WriteAffixTable(const AffixTable &affixes, const string &output_file) {
  ProtoRecordWriter writer(output_file);
  affixes.Write(&writer);
}

void OutputResources(Artifacts *artifacts) {
  // SyntaxNetComponent uses a mandatory label-map file. Making a dummy one.
  artifacts->label_map = FileName("label-map");
  CHECK(File::WriteContents(artifacts->label_map, "0"));
  LOG(INFO) << "Wrote dummy label-map to " << artifacts->label_map;

  std::unordered_set<string> allowed_words;
  if (!FLAGS_allowed_words_file.empty()) {
    string contents;
    CHECK(File::ReadContents(FLAGS_allowed_words_file, &contents));
    std::vector<string> words = Split(contents, '\n');
    for (const string &w : words) {
      if (!w.empty()) allowed_words.insert(w);
    }
  }
  LOG(INFO) << "Read " << allowed_words.size() << " allowed words";

  // Term frequency maps to be populated by the corpus.
  TermFrequencyMap words;

  // Affix tables to be populated by the corpus.
  AffixTable prefixes(AffixTable::PREFIX, kLexiconMaxPrefixLength);
  AffixTable suffixes(AffixTable::SUFFIX, kLexiconMaxSuffixLength);

  int count = 0;
  artifacts->train_corpus->Rewind();
  while (true) {
    Store store(artifacts->global());
    Document *document = artifacts->train_corpus->Next(&store);
    if (document == nullptr) break;

    for (int t = 0; t < document->num_tokens(); ++t) {
      // Get token and lowercased word.
      const auto &token = document->token(t);
      string word = token.text();
      syntaxnet::utils::NormalizeDigits(&word);

      // Increment frequencies (only for terms that exist).
      if (allowed_words.empty() || allowed_words.count(word) > 0) {
        if (!word.empty() && !HasSpaces(word)) words.Increment(word);
      }

      // Add prefixes/suffixes for the current word.
      prefixes.AddAffixesForWord(word.c_str(), word.size());
      suffixes.AddAffixesForWord(word.c_str(), word.size());
    }

    count++;
    if (count % 10000 == 1) {
      LOG(INFO) << count << " documents processsed while building lexicons";
    }
    delete document;
  }

  // Write affixes to disk.
  artifacts->prefix_table = FileName("prefix-table");
  artifacts->suffix_table = FileName("suffix-table");
  WriteAffixTable(prefixes, artifacts->prefix_table);
  WriteAffixTable(suffixes, artifacts->suffix_table);

  // Write mappings to disk.
  artifacts->word_map = FileName("word-map");
  words.Save(artifacts->word_map);

  LOG(INFO) << count << " documents processsed while building lexicons";
}

void CheckWordEmbeddingsDimensionality() {
  if (FLAGS_word_embeddings.empty()) return;

  syntaxnet::ProtoRecordReader reader(FLAGS_word_embeddings);
  syntaxnet::TokenEmbedding embedding;
  CHECK_EQ(reader.Read(&embedding), tensorflow::Status::OK());
  int size = embedding.vector().values_size();
  CHECK_EQ(size, FLAGS_word_embeddings_dim)
      << "Pretrained embeddings have dim=" << size
      << ", whereas word embeddings have dim=" << FLAGS_word_embeddings_dim;
}

string MakeFML(const string &locator,
               const std::vector<string> &features,
               int start,
               int end) {
  string kPattern = "XX";
  string output;
  for (const string &f : features) {
    string full;
    StrAppend(&full, locator.empty() ? "" : (locator + "."), f);
    size_t i = full.find(kPattern);
    if (i == string::npos) {
      StrAppend(&output, output.empty() ? "" : " ", full);
    } else {
      for (int j = start; j < end; ++j) {
        string copy = full;
        copy.replace(i, kPattern.size(), StrCat(j));
        StrAppend(&output, output.empty() ? "" : " ", copy);
      }
    }
  }

  return output;
}

void SpecifyResource(ComponentSpec *s, const string &name, const string &file) {
  for (auto &resource : *s->mutable_resource()) {
    if (resource.name() == name) {
      resource.clear_part();
      resource.add_part()->set_file_pattern(file);
    }
  }
}

void OutputMasterSpec(Artifacts *artifacts) {
  CheckWordEmbeddingsDimensionality();

  string lstm_spec_str =
      "transition_system { "
      "  registered_name: 'shift-only' "
      "  parameters { key: 'left_to_right'  value: 'true' } "
      "} "
      "resource { name: 'word-map' } "
      "resource { name: 'label-map' } "
      "resource { name: 'suffix-table' } "
      "fixed_feature { "
      "  name: 'words' "
      "  fml: 'input.word' "
      "  embedding_dim: 32 " +
      (FLAGS_word_embeddings.empty() ? "" :
       ("pretrained_embedding_matrix { part { file_pattern: '" +
        FLAGS_word_embeddings + "' } } " )) +
      "} " +
      (!FLAGS_oov_lstm_features ? "" :
      "fixed_feature { "
      "  name: 'suffix' "
      "  fml: 'input.suffix(length=3)' "
      "  embedding_dim: 16 "
      "} "
      "fixed_feature { "
      "  name: 'shape' "
      "  fml: 'input.digit input.hyphen input.punctuation-amount"
      " input.quote input.capitalization' "
      "  embedding_dim: 8 "
      "} ") +
      "network_unit { "
      "  registered_name: 'LSTMNetwork' "
      "  parameters { key: 'hidden_layer_sizes' value: '256' } "
      "} ";
  ComponentSpec lr_lstm;
  CHECK(TextFormat::ParseFromString(lstm_spec_str, &lr_lstm));
  lr_lstm.set_name("lr_lstm");

  ComponentSpec rl_lstm = lr_lstm;
  rl_lstm.set_name("rl_lstm");
  (*rl_lstm.mutable_transition_system()->mutable_parameters())["left_to_right"]
      = "false";

  string ff_spec_str =
      "name: 'ff' "
      "transition_system { "
      "  registered_name: 'sempar' "
      "  parameters { key: 'left_to_right' value: 'true' } "
      "} "
      "resource { name: 'commons' } "
      "resource { name: 'action-table' }"
      "resource { name: 'label-map' }"
      "fixed_feature { "
      "  name: 'roles' "
      "  fml: 'roles(frame-limit=5)' "
      "  embedding_dim: 16 "
      "  size: 1 "
      "} "
      "linked_feature { "
      "  name: 'lr' fml: 'input.focus' embedding_dim: 32 size: 1 "
      "  source_component: 'lr_lstm' "
      "  source_translator: 'identity' "
      "  source_layer: 'layer_0' "
      "} "
      "linked_feature { "
      "  name: 'rl' fml: 'input.focus' embedding_dim: 32 size: 1 "
      "  source_component: 'rl_lstm' "
      "  source_translator: 'reverse-token' "
      "  source_layer: 'layer_0' "
      "} "
      "linked_feature { "
      "  name: 'frame-end-lr' "
      "  fml: 'attention(0).frame-end attention(1).frame-end"
      " attention(2).frame-end attention(3).frame-end attention(4).frame-end' "
      "  embedding_dim: 32 "
      "  source_component: 'lr_lstm' "
      "  source_translator: 'identity' "
      "  source_layer: 'layer_0' "
      "} "
      "linked_feature { "
      "  name: 'frame-end-rl' "
      "  fml: 'attention(0).frame-end attention(1).frame-end"
      " attention(2).frame-end attention(3).frame-end attention(4).frame-end' "
      "  embedding_dim: 32 "
      "  source_component: 'rl_lstm' "
      "  source_translator: 'reverse-token' "
      "  source_layer: 'layer_0' "
      "} "
      "linked_feature { "
      "  name: 'history' "
      "  fml: 'constant(value=0) constant(value=1) constant(value=2)"
      " constant(value=3)' "
      "  embedding_dim: 64 "
      "  source_component: 'ff' "
      "  source_translator: 'history' "
      "  source_layer: 'layer_0' "
      "} "
      "linked_feature { "
      "  name: 'frame-focus-steps' "
      "  fml: 'attention(0).focus-step attention(1).focus-step"
      " attention(2).focus-step attention(3).focus-step"
      " attention(4).focus-step' "
      "  embedding_dim: 64 "
      "  source_component: 'ff' "
      "  source_translator: 'identity' "
      "  source_layer: 'layer_0' "
      "} "
      "linked_feature { "
      "  name: 'frame-creation-steps' "
      "  fml: 'attention(0).creation-step attention(1).creation-step"
      " attention(2).creation-step attention(3).creation-step"
      " attention(4).creation-step' "
      "  embedding_dim: 64 "
      "  source_component: 'ff' "
      "  source_translator: 'identity' "
      "  source_layer: 'layer_0' "
      "} "
      "network_unit { "
      "  registered_name: 'FeedForwardNetwork' "
      "  parameters { key: 'hidden_layer_sizes' value: '128' } "
      "} ";
  ComponentSpec ff;
  CHECK(TextFormat::ParseFromString(ff_spec_str, &ff));

  MasterSpec spec;
  *spec.add_component() = lr_lstm;
  *spec.add_component() = rl_lstm;
  *spec.add_component() = ff;
  for (auto &c : *spec.mutable_component()) {
    c.mutable_backend()->set_registered_name("SyntaxNetComponent");
    c.mutable_component_builder()->set_registered_name(
        "DynamicComponentBuilder");
    SpecifyResource(&c, "word-map", artifacts->word_map);
    SpecifyResource(&c, "label-map", artifacts->label_map);
    SpecifyResource(&c, "prefix-table", artifacts->prefix_table);
    SpecifyResource(&c, "suffix-table", artifacts->suffix_table);
    SpecifyResource(&c, "commons", artifacts->commons_filename);
    SpecifyResource(&c, "action-table", artifacts->action_table_filename);

    for (auto &link : *c.mutable_linked_feature()) {
      std::vector<string> parts = Split(link.fml(), ' ');
      link.set_size(parts.size());
    }

    // Fill in the domain and feature sizes for the component.
    syntaxnet::TaskContext context;
    for (const auto &resource : c.resource()) {
      auto *input = context.GetInput(resource.name());
      for (const auto &part : resource.part()) {
        auto *input_part = input->add_part();
        input_part->set_file_pattern(part.file_pattern());
        input_part->set_file_format(part.file_format());
        input_part->set_record_format(part.record_format());
      }
    }
    for (const auto &param : c.transition_system().parameters()) {
      context.SetParameter(param.first, param.second);
    }

    std::vector<string> names;
    std::vector<string> dims;
    std::vector<string> fml;
    std::vector<string> predicates;  // unused
    for (const auto &channel : c.fixed_feature()) {
      names.push_back(channel.name());
      fml.push_back(channel.fml());
      predicates.push_back(channel.predicate_map());
      dims.push_back(StrCat(channel.embedding_dim()));
    }

    context.SetParameter("sempar_embedding_dims", Join(dims, ";"));
    context.SetParameter("sempar_predicate_maps", Join(predicates, ";"));
    context.SetParameter("sempar_features", Join(fml, ";"));
    context.SetParameter("sempar_embedding_names", Join(names, ";"));

    syntaxnet::ParserEmbeddingFeatureExtractor extractor("sempar");
    extractor.Setup(&context);
    extractor.Init(&context);
    for (int i = 0; i < extractor.NumEmbeddings(); ++i) {
      auto *f = c.mutable_fixed_feature(i);
      f->set_size(extractor.FeatureSize(i));
      f->set_vocabulary_size(extractor.EmbeddingSize(i));
      if ((f->name() == "words") && (f->has_pretrained_embedding_matrix())) {
        string vocab_file = FileName("vocab-" + c.name() + "-" + f->name());
        std::vector<string> mapped_words =
            extractor.GetMappingsForEmbedding(f->name());
        string vocab_contents = Join(mapped_words, "\n");
        CHECK(File::WriteContents(vocab_file, vocab_contents));
        LOG(INFO) << "Wrote vocab of size " << mapped_words.size()
                  << " for '" << f->name() << "' pretrained embedding to "
                  << vocab_file;
        f->clear_vocab();
        f->mutable_vocab()->add_part()->set_file_pattern(vocab_file);
      }
    }

    auto *system = syntaxnet::ParserTransitionSystem::Create(
        c.transition_system().registered_name());
    system->Setup(&context);
    system->Init(&context);
    c.set_num_actions(system->NumActions(0 /* |label_map|; irrelevant here */));
    delete system;
  }

  // Dump the master spec.
  artifacts->spec_file = FileName("master_spec");
  artifacts->spec = spec;
  CHECK(File::WriteContents(artifacts->spec_file, spec.DebugString()));
  LOG(INFO) << "Wrote master spec to " << artifacts->spec_file;
}

int main(int argc, char **argv) {
  sling::InitProgram(&argc, &argv);

  CHECK(!FLAGS_documents.empty()) << "No documents specified.";
  CHECK(!FLAGS_commons.empty()) << "No commons specified.";
  CHECK(!FLAGS_output_dir.empty()) << "No output_dir specified.";
  CHECK(FLAGS_documents.find(".tfrecordio") != string::npos);

  if (!File::Exists(FLAGS_output_dir)) {
    CHECK(File::Mkdir(FLAGS_output_dir));
  }

  Artifacts artifacts;
  artifacts.commons_filename = FLAGS_commons;
  artifacts.resources.LoadGlobalStore(FLAGS_commons);
  artifacts.train_corpus = DocumentSource::Create(FLAGS_documents);

  // Dump action table.
  OutputActionTable(&artifacts);

  // Generate lexical resources.
  OutputResources(&artifacts);

  // Output master spec.
  OutputMasterSpec(&artifacts);

  return 0;
}
