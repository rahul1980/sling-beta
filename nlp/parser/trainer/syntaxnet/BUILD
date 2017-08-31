package(default_visibility = ["//visibility:public"])

proto_library(
  name = "framed-sentence-proto",
  srcs = ["framed-sentence.proto"],
  deps = ["//third_party/syntaxnet:sentence-proto"],
)

cc_proto_library(
  name = "framed-sentence",
  deps = ["framed-sentence-proto"],
)

cc_binary(
  name = "generate-master-spec",
  srcs = ["generate-master-spec.cc"],
  deps = [
    ":sempar-features",
    ":sempar-transition-system",
    "//base",
    "//file:posix",
    "//frame:object",
    "//frame:serialization",
    "//frame:store",
    "//nlp/document:document",
    "//nlp/document:document-source",
    "//nlp/parser/trainer:action-table-generator",
    "//nlp/parser/trainer:shared-resources",
    "//string:strcat",
    "//third_party/syntaxnet:dragnn-ops",
    "//third_party/syntaxnet:syntaxnet",
    "//third_party/zlib:miniunzip",
    "//third_party/zlib:zlib",
  ],
)

cc_library(
  name = "sempar-transition-state",
  srcs = ["sempar-transition-state.cc"],
  hdrs = ["sempar-transition-state.h"],
  deps = [
    ":framed-sentence",
    "//base",
    "//nlp/parser:action-table",
    "//nlp/parser:parser-action",
    "//nlp/parser:parser-state",
    "//nlp/parser/trainer:gold-transition-generator",
    "//nlp/parser/trainer:shared-resources",
    "//frame:serialization",
    "//frame:store",
    "//string:strcat",
    "//third_party/syntaxnet",
  ],
)

cc_library(
  name = "sempar-transition-system",
  srcs = ["sempar-transition-system.cc"],
  hdrs = ["sempar-transition-system.h"],
  deps = [
    ":sempar-transition-state",
    "//base",
    "//nlp/parser:action-table",
    "//nlp/parser:parser-action",
    "//nlp/parser:parser-state",
    "//nlp/parser/trainer:gold-transition-generator",
    "//nlp/parser/trainer:shared-resources",
    "//frame:store",
    "//third_party/syntaxnet",
  ],
  alwayslink = 1,
)

cc_library(
  name = "sempar-features",
  srcs = ["sempar-features.cc"],
  hdrs = [],
  deps = [
    ":sempar-transition-state",
    "//base",
    "//nlp/parser:action-table",
    "//nlp/parser:parser-action",
    "//nlp/parser:parser-state",
    "//nlp/parser/trainer:shared-resources",
    "//third_party/syntaxnet",
  ],
  alwayslink = 1,
)

cc_binary(
  name = "sempar-syntaxnet.so",
  deps = [
      ":sempar-features",
      ":sempar-transition-system",
      "//base:libinit",
      "//file:posix",
      "//third_party/syntaxnet:dragnn-ops",
      "//third_party/syntaxnet:syntaxnet",
  ],
  linkshared = 1,
  linkstatic = 1,
)
