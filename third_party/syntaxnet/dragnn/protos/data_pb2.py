# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: dragnn/protos/data.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='dragnn/protos/data.proto',
  package='syntaxnet.dragnn',
  syntax='proto2',
  serialized_pb=_b('\n\x18\x64ragnn/protos/data.proto\x12\x10syntaxnet.dragnn\"U\n\rFixedFeatures\x12\n\n\x02id\x18\x01 \x03(\x04\x12\x0e\n\x06weight\x18\x02 \x03(\x02\x12\x12\n\nvalue_name\x18\x03 \x03(\t\x12\x14\n\x0c\x66\x65\x61ture_name\x18\x04 \x01(\t\"r\n\x0cLinkFeatures\x12\x11\n\tbatch_idx\x18\x01 \x01(\x03\x12\x10\n\x08\x62\x65\x61m_idx\x18\x02 \x01(\x03\x12\x10\n\x08step_idx\x18\x03 \x01(\x03\x12\x15\n\rfeature_value\x18\x04 \x01(\x03\x12\x14\n\x0c\x66\x65\x61ture_name\x18\x05 \x01(\t')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_FIXEDFEATURES = _descriptor.Descriptor(
  name='FixedFeatures',
  full_name='syntaxnet.dragnn.FixedFeatures',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='syntaxnet.dragnn.FixedFeatures.id', index=0,
      number=1, type=4, cpp_type=4, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='weight', full_name='syntaxnet.dragnn.FixedFeatures.weight', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='value_name', full_name='syntaxnet.dragnn.FixedFeatures.value_name', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='feature_name', full_name='syntaxnet.dragnn.FixedFeatures.feature_name', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=46,
  serialized_end=131,
)


_LINKFEATURES = _descriptor.Descriptor(
  name='LinkFeatures',
  full_name='syntaxnet.dragnn.LinkFeatures',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='batch_idx', full_name='syntaxnet.dragnn.LinkFeatures.batch_idx', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='beam_idx', full_name='syntaxnet.dragnn.LinkFeatures.beam_idx', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='step_idx', full_name='syntaxnet.dragnn.LinkFeatures.step_idx', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='feature_value', full_name='syntaxnet.dragnn.LinkFeatures.feature_value', index=3,
      number=4, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='feature_name', full_name='syntaxnet.dragnn.LinkFeatures.feature_name', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=133,
  serialized_end=247,
)

DESCRIPTOR.message_types_by_name['FixedFeatures'] = _FIXEDFEATURES
DESCRIPTOR.message_types_by_name['LinkFeatures'] = _LINKFEATURES

FixedFeatures = _reflection.GeneratedProtocolMessageType('FixedFeatures', (_message.Message,), dict(
  DESCRIPTOR = _FIXEDFEATURES,
  __module__ = 'dragnn.protos.data_pb2'
  # @@protoc_insertion_point(class_scope:syntaxnet.dragnn.FixedFeatures)
  ))
_sym_db.RegisterMessage(FixedFeatures)

LinkFeatures = _reflection.GeneratedProtocolMessageType('LinkFeatures', (_message.Message,), dict(
  DESCRIPTOR = _LINKFEATURES,
  __module__ = 'dragnn.protos.data_pb2'
  # @@protoc_insertion_point(class_scope:syntaxnet.dragnn.LinkFeatures)
  ))
_sym_db.RegisterMessage(LinkFeatures)


# @@protoc_insertion_point(module_scope)
