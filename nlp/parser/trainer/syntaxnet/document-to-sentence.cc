#include <string>

#include "base/flags.h"
#include "base/init.h"
#include "base/logging.h"
#include "base/macros.h"
#include "frame/serialization.h"
#include "frame/store.h"
#include "nlp/document/document.h"
#include "nlp/document/document-source.h"
#include "nlp/document/token-breaks.h"
#include "nlp/parser/trainer/framed-sentence.pb.h"
#include "nlp/parser/trainer/shared-resources.h"
#include "third_party/syntaxnet/syntaxnet/proto_io.h"
#include "third_party/syntaxnet/syntaxnet/sentence.pb.h"

DEFINE_string(documents, "", "File pattern of documents.");
DEFINE_string(commons, "", "Path to common store.");
DEFINE_string(output, "", "Output filename.");

using sling::Store;
using sling::nlp::SharedResources;
using sling::nlp::Document;
using sling::nlp::DocumentSource;

int main(int argc, char **argv) {
  sling::InitProgram(&argc, &argv);

  CHECK(!FLAGS_commons.empty());
  CHECK(!FLAGS_documents.empty());
  CHECK(!FLAGS_output.empty());

  SharedResources resources;
  resources.LoadGlobalStore(FLAGS_commons);

  DocumentSource *corpus = DocumentSource::Create(FLAGS_documents);
  int count = 0;
  syntaxnet::ProtoRecordWriter writer(FLAGS_output);
  while (true) {
    string name, contents;
    if (!corpus->NextSerialized(&name, &contents)) break;

    Store store(resources.global);
    sling::StringDecoder decoder(&store, contents);
    Document *document = new Document(decoder.Decode().AsFrame());
    if (document == nullptr) break;

    string text = document->GetText();
    if (text.empty()) text = document->PhraseText(0, document->num_tokens());

    syntaxnet::Sentence sentence;
    sentence.set_docid(name);
    sentence.set_text(text);

    for (const sling::nlp::Token &token : document->tokens()) {
      auto *t = sentence.add_token();
      t->set_word(token.text());
      t->set_start(token.begin());
      t->set_end(token.end() - 1);  // DRAGNN token end bytes are inclusive

      int level = static_cast<int>(token.brk());
      int max_level = static_cast<int>(sling::nlp::SENTENCE_BREAK);
      if (level > max_level) level = max_level;
      t->set_break_level(static_cast<syntaxnet::Token::BreakLevel>(level));
    }
    sentence.SetExtension(sling::nlp::FramedSentence::framing, contents);
    writer.Write(sentence);

    count++;
    delete document;
  }
  delete corpus;
  LOG(INFO) << "Converted " << count << " documents to Sentence protos";

  return 0;
}


