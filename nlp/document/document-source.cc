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

#include "nlp/document/document-source.h"

#include <vector>

#include "base/logging.h"
#include "base/macros.h"
#include "file/file.h"
#include "frame/object.h"
#include "frame/serialization.h"
#include "nlp/parser/trainer/syntaxnet/framed-sentence.pb.h"
#include "syntaxnet/proto_io.h"
#include "syntaxnet/sentence.pb.h"
#include "util/zip-iterator.h"

namespace sling {
namespace nlp {

// Iterator implementation which assumes one encoded document per input file.
class EncodedDocumentSource : public DocumentSource {
 public:
  EncodedDocumentSource(const std::vector<string> &files) {
    files_ = files;
    index_ = 0;
  }

  bool NextSerialized(string *name, string *contents) override {
    if (index_ >= files_.size()) return false;
    *name = files_[index_];
    CHECK(File::ReadContents(files_[index_], contents));
    index_++;

    return true;
  }

  void Rewind() override {
    index_ = 0;
  }

 private:
  std::vector<string> files_;
  int index_;
};

// Iterator implementation for zip archives..
// Assumes that each encoded document is a separate file in the zip archive.
class ZipDocumentSource : public DocumentSource {
 public:
  ZipDocumentSource(const string &file) {
    file_ = file;
    iterator_ = new ZipIterator(file);
  }

  ~ZipDocumentSource() override {
    delete iterator_;
  }

  bool NextSerialized(string *name, string *contents) override {
    return iterator_->Next(name, contents);
  }

  void Rewind() override {
    delete iterator_;
    iterator_ = new ZipIterator(file_);
  }

 private:
  ZipIterator *iterator_ = nullptr;
  string file_;
};

// Iterator for TFSentenceRecord files.
class TFSentenceRecordSource : public DocumentSource {
 public:
  TFSentenceRecordSource(const string &file) {
    reader_ = new syntaxnet::ProtoRecordReader(file);
    file_ = file;
  }

  ~TFSentenceRecordSource() override { delete reader_; }

  bool NextSerialized(string *name, string *serialized) override {
    LOG(FATAL) << "Not implemented";
    return false;
  }

  Document *Next(Store *store) override {
    syntaxnet::Sentence sentence;
    auto status = reader_->Read(&sentence);
    if (!status.ok()) return nullptr;

    Frame frame =
        Decode(store, sentence.GetExtension(FramedSentence::framing)).AsFrame();
    CHECK(frame.valid());
    Document *doc = new Document(frame);
    doc->Update();

    return doc;
  }

  void Rewind() override {
    delete reader_;
    reader_ = new syntaxnet::ProtoRecordReader(file_);
  }

 public:
  string file_;
  syntaxnet::ProtoRecordReader *reader_ = nullptr;
};


Document *DocumentSource::Next(Store *store) {
  string name, contents;
  if (!NextSerialized(&name, &contents)) return nullptr;

  StringDecoder decoder(store, contents);
  return new Document(decoder.Decode().AsFrame());
}

namespace {

bool HasSuffix(const string &s, const string &suffix) {
  int len = suffix.size();
  return (s.size() >= len) && (s.substr(s.size() - len) == suffix);
}

}  // namespace

DocumentSource *DocumentSource::Create(const string &file_pattern) {
  // TODO: Add more formats as needed.
  if (HasSuffix(file_pattern, ".zip")) {
    return new ZipDocumentSource(file_pattern);
  } else if (HasSuffix(file_pattern, ".tfrecordio")) {
    return new TFSentenceRecordSource(file_pattern);
  } else {
    std::vector<string> files;
    CHECK(File::Match(file_pattern, &files));
    return new EncodedDocumentSource(files);
  }

  return nullptr;
}

}  // namespace nlp
}  // namespace sling
