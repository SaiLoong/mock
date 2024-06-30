# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: chromadb/proto/chroma.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1b\x63hromadb/proto/chroma.proto\x12\x06\x63hroma\"&\n\x06Status\x12\x0e\n\x06reason\x18\x01 \x01(\t\x12\x0c\n\x04\x63ode\x18\x02 \x01(\x05\"U\n\x06Vector\x12\x11\n\tdimension\x18\x01 \x01(\x05\x12\x0e\n\x06vector\x18\x02 \x01(\x0c\x12(\n\x08\x65ncoding\x18\x03 \x01(\x0e\x32\x16.chroma.ScalarEncoding\"\x1a\n\tFilePaths\x12\r\n\x05paths\x18\x01 \x03(\t\"\xa5\x02\n\x07Segment\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04type\x18\x02 \x01(\t\x12#\n\x05scope\x18\x03 \x01(\x0e\x32\x14.chroma.SegmentScope\x12\x17\n\ncollection\x18\x05 \x01(\tH\x00\x88\x01\x01\x12-\n\x08metadata\x18\x06 \x01(\x0b\x32\x16.chroma.UpdateMetadataH\x01\x88\x01\x01\x12\x32\n\nfile_paths\x18\x07 \x03(\x0b\x32\x1e.chroma.Segment.FilePathsEntry\x1a\x43\n\x0e\x46ilePathsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12 \n\x05value\x18\x02 \x01(\x0b\x32\x11.chroma.FilePaths:\x02\x38\x01\x42\r\n\x0b_collectionB\x0b\n\t_metadata\"\xd1\x01\n\nCollection\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04name\x18\x02 \x01(\t\x12-\n\x08metadata\x18\x04 \x01(\x0b\x32\x16.chroma.UpdateMetadataH\x00\x88\x01\x01\x12\x16\n\tdimension\x18\x05 \x01(\x05H\x01\x88\x01\x01\x12\x0e\n\x06tenant\x18\x06 \x01(\t\x12\x10\n\x08\x64\x61tabase\x18\x07 \x01(\t\x12\x14\n\x0clog_position\x18\x08 \x01(\x03\x12\x0f\n\x07version\x18\t \x01(\x05\x42\x0b\n\t_metadataB\x0c\n\n_dimension\"4\n\x08\x44\x61tabase\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x0e\n\x06tenant\x18\x03 \x01(\t\"\x16\n\x06Tenant\x12\x0c\n\x04name\x18\x01 \x01(\t\"x\n\x13UpdateMetadataValue\x12\x16\n\x0cstring_value\x18\x01 \x01(\tH\x00\x12\x13\n\tint_value\x18\x02 \x01(\x03H\x00\x12\x15\n\x0b\x66loat_value\x18\x03 \x01(\x01H\x00\x12\x14\n\nbool_value\x18\x04 \x01(\x08H\x00\x42\x07\n\x05value\"\x96\x01\n\x0eUpdateMetadata\x12\x36\n\x08metadata\x18\x01 \x03(\x0b\x32$.chroma.UpdateMetadata.MetadataEntry\x1aL\n\rMetadataEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12*\n\x05value\x18\x02 \x01(\x0b\x32\x1b.chroma.UpdateMetadataValue:\x02\x38\x01\"\xaf\x01\n\x0fOperationRecord\x12\n\n\x02id\x18\x01 \x01(\t\x12#\n\x06vector\x18\x02 \x01(\x0b\x32\x0e.chroma.VectorH\x00\x88\x01\x01\x12-\n\x08metadata\x18\x03 \x01(\x0b\x32\x16.chroma.UpdateMetadataH\x01\x88\x01\x01\x12$\n\toperation\x18\x04 \x01(\x0e\x32\x11.chroma.OperationB\t\n\x07_vectorB\x0b\n\t_metadata\")\n\x13\x43ountRecordsRequest\x12\x12\n\nsegment_id\x18\x01 \x01(\t\"%\n\x14\x43ountRecordsResponse\x12\r\n\x05\x63ount\x18\x01 \x01(\r\"\xc2\x01\n\x14QueryMetadataRequest\x12\x12\n\nsegment_id\x18\x01 \x01(\t\x12\x1c\n\x05where\x18\x02 \x01(\x0b\x32\r.chroma.Where\x12-\n\x0ewhere_document\x18\x03 \x01(\x0b\x32\x15.chroma.WhereDocument\x12\x0b\n\x03ids\x18\x04 \x03(\t\x12\x12\n\x05limit\x18\x05 \x01(\x05H\x00\x88\x01\x01\x12\x13\n\x06offset\x18\x06 \x01(\x05H\x01\x88\x01\x01\x42\x08\n\x06_limitB\t\n\x07_offset\"I\n\x15QueryMetadataResponse\x12\x30\n\x07records\x18\x01 \x03(\x0b\x32\x1f.chroma.MetadataEmbeddingRecord\"O\n\x17MetadataEmbeddingRecord\x12\n\n\x02id\x18\x01 \x01(\t\x12(\n\x08metadata\x18\x02 \x01(\x0b\x32\x16.chroma.UpdateMetadata\"\x83\x01\n\rWhereDocument\x12-\n\x06\x64irect\x18\x01 \x01(\x0b\x32\x1b.chroma.DirectWhereDocumentH\x00\x12\x31\n\x08\x63hildren\x18\x02 \x01(\x0b\x32\x1d.chroma.WhereDocumentChildrenH\x00\x42\x10\n\x0ewhere_document\"X\n\x13\x44irectWhereDocument\x12\x10\n\x08\x64ocument\x18\x01 \x01(\t\x12/\n\x08operator\x18\x02 \x01(\x0e\x32\x1d.chroma.WhereDocumentOperator\"k\n\x15WhereDocumentChildren\x12\'\n\x08\x63hildren\x18\x01 \x03(\x0b\x32\x15.chroma.WhereDocument\x12)\n\x08operator\x18\x02 \x01(\x0e\x32\x17.chroma.BooleanOperator\"r\n\x05Where\x12\x35\n\x11\x64irect_comparison\x18\x01 \x01(\x0b\x32\x18.chroma.DirectComparisonH\x00\x12)\n\x08\x63hildren\x18\x02 \x01(\x0b\x32\x15.chroma.WhereChildrenH\x00\x42\x07\n\x05where\"\x91\x04\n\x10\x44irectComparison\x12\x0b\n\x03key\x18\x01 \x01(\t\x12?\n\x15single_string_operand\x18\x02 \x01(\x0b\x32\x1e.chroma.SingleStringComparisonH\x00\x12;\n\x13string_list_operand\x18\x03 \x01(\x0b\x32\x1c.chroma.StringListComparisonH\x00\x12\x39\n\x12single_int_operand\x18\x04 \x01(\x0b\x32\x1b.chroma.SingleIntComparisonH\x00\x12\x35\n\x10int_list_operand\x18\x05 \x01(\x0b\x32\x19.chroma.IntListComparisonH\x00\x12?\n\x15single_double_operand\x18\x06 \x01(\x0b\x32\x1e.chroma.SingleDoubleComparisonH\x00\x12;\n\x13\x64ouble_list_operand\x18\x07 \x01(\x0b\x32\x1c.chroma.DoubleListComparisonH\x00\x12\x37\n\x11\x62ool_list_operand\x18\x08 \x01(\x0b\x32\x1a.chroma.BoolListComparisonH\x00\x12;\n\x13single_bool_operand\x18\t \x01(\x0b\x32\x1c.chroma.SingleBoolComparisonH\x00\x42\x0c\n\ncomparison\"[\n\rWhereChildren\x12\x1f\n\x08\x63hildren\x18\x01 \x03(\x0b\x32\r.chroma.Where\x12)\n\x08operator\x18\x02 \x01(\x0e\x32\x17.chroma.BooleanOperator\"S\n\x14StringListComparison\x12\x0e\n\x06values\x18\x01 \x03(\t\x12+\n\rlist_operator\x18\x02 \x01(\x0e\x32\x14.chroma.ListOperator\"V\n\x16SingleStringComparison\x12\r\n\x05value\x18\x01 \x01(\t\x12-\n\ncomparator\x18\x02 \x01(\x0e\x32\x19.chroma.GenericComparator\"T\n\x14SingleBoolComparison\x12\r\n\x05value\x18\x01 \x01(\x08\x12-\n\ncomparator\x18\x02 \x01(\x0e\x32\x19.chroma.GenericComparator\"P\n\x11IntListComparison\x12\x0e\n\x06values\x18\x01 \x03(\x03\x12+\n\rlist_operator\x18\x02 \x01(\x0e\x32\x14.chroma.ListOperator\"\xa2\x01\n\x13SingleIntComparison\x12\r\n\x05value\x18\x01 \x01(\x03\x12\x37\n\x12generic_comparator\x18\x02 \x01(\x0e\x32\x19.chroma.GenericComparatorH\x00\x12\x35\n\x11number_comparator\x18\x03 \x01(\x0e\x32\x18.chroma.NumberComparatorH\x00\x42\x0c\n\ncomparator\"S\n\x14\x44oubleListComparison\x12\x0e\n\x06values\x18\x01 \x03(\x01\x12+\n\rlist_operator\x18\x02 \x01(\x0e\x32\x14.chroma.ListOperator\"Q\n\x12\x42oolListComparison\x12\x0e\n\x06values\x18\x01 \x03(\x08\x12+\n\rlist_operator\x18\x02 \x01(\x0e\x32\x14.chroma.ListOperator\"\xa5\x01\n\x16SingleDoubleComparison\x12\r\n\x05value\x18\x01 \x01(\x01\x12\x37\n\x12generic_comparator\x18\x02 \x01(\x0e\x32\x19.chroma.GenericComparatorH\x00\x12\x35\n\x11number_comparator\x18\x03 \x01(\x0e\x32\x18.chroma.NumberComparatorH\x00\x42\x0c\n\ncomparator\"4\n\x11GetVectorsRequest\x12\x0b\n\x03ids\x18\x01 \x03(\t\x12\x12\n\nsegment_id\x18\x02 \x01(\t\"D\n\x12GetVectorsResponse\x12.\n\x07records\x18\x01 \x03(\x0b\x32\x1d.chroma.VectorEmbeddingRecord\"C\n\x15VectorEmbeddingRecord\x12\n\n\x02id\x18\x01 \x01(\t\x12\x1e\n\x06vector\x18\x03 \x01(\x0b\x32\x0e.chroma.Vector\"\x86\x01\n\x13QueryVectorsRequest\x12\x1f\n\x07vectors\x18\x01 \x03(\x0b\x32\x0e.chroma.Vector\x12\t\n\x01k\x18\x02 \x01(\x05\x12\x13\n\x0b\x61llowed_ids\x18\x03 \x03(\t\x12\x1a\n\x12include_embeddings\x18\x04 \x01(\x08\x12\x12\n\nsegment_id\x18\x05 \x01(\t\"C\n\x14QueryVectorsResponse\x12+\n\x07results\x18\x01 \x03(\x0b\x32\x1a.chroma.VectorQueryResults\"@\n\x12VectorQueryResults\x12*\n\x07results\x18\x01 \x03(\x0b\x32\x19.chroma.VectorQueryResult\"a\n\x11VectorQueryResult\x12\n\n\x02id\x18\x01 \x01(\t\x12\x10\n\x08\x64istance\x18\x03 \x01(\x02\x12#\n\x06vector\x18\x04 \x01(\x0b\x32\x0e.chroma.VectorH\x00\x88\x01\x01\x42\t\n\x07_vector*8\n\tOperation\x12\x07\n\x03\x41\x44\x44\x10\x00\x12\n\n\x06UPDATE\x10\x01\x12\n\n\x06UPSERT\x10\x02\x12\n\n\x06\x44\x45LETE\x10\x03*(\n\x0eScalarEncoding\x12\x0b\n\x07\x46LOAT32\x10\x00\x12\t\n\x05INT32\x10\x01*@\n\x0cSegmentScope\x12\n\n\x06VECTOR\x10\x00\x12\x0c\n\x08METADATA\x10\x01\x12\n\n\x06RECORD\x10\x02\x12\n\n\x06SQLITE\x10\x03*7\n\x15WhereDocumentOperator\x12\x0c\n\x08\x43ONTAINS\x10\x00\x12\x10\n\x0cNOT_CONTAINS\x10\x01*\"\n\x0f\x42ooleanOperator\x12\x07\n\x03\x41ND\x10\x00\x12\x06\n\x02OR\x10\x01*\x1f\n\x0cListOperator\x12\x06\n\x02IN\x10\x00\x12\x07\n\x03NIN\x10\x01*#\n\x11GenericComparator\x12\x06\n\x02\x45Q\x10\x00\x12\x06\n\x02NE\x10\x01*4\n\x10NumberComparator\x12\x06\n\x02GT\x10\x00\x12\x07\n\x03GTE\x10\x01\x12\x06\n\x02LT\x10\x02\x12\x07\n\x03LTE\x10\x03\x32\xad\x01\n\x0eMetadataReader\x12N\n\rQueryMetadata\x12\x1c.chroma.QueryMetadataRequest\x1a\x1d.chroma.QueryMetadataResponse\"\x00\x12K\n\x0c\x43ountRecords\x12\x1b.chroma.CountRecordsRequest\x1a\x1c.chroma.CountRecordsResponse\"\x00\x32\xa2\x01\n\x0cVectorReader\x12\x45\n\nGetVectors\x12\x19.chroma.GetVectorsRequest\x1a\x1a.chroma.GetVectorsResponse\"\x00\x12K\n\x0cQueryVectors\x12\x1b.chroma.QueryVectorsRequest\x1a\x1c.chroma.QueryVectorsResponse\"\x00\x42:Z8github.com/chroma-core/chroma/go/pkg/proto/coordinatorpbb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'chromadb.proto.chroma_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  _globals['DESCRIPTOR']._loaded_options = None
  _globals['DESCRIPTOR']._serialized_options = b'Z8github.com/chroma-core/chroma/go/pkg/proto/coordinatorpb'
  _globals['_SEGMENT_FILEPATHSENTRY']._loaded_options = None
  _globals['_SEGMENT_FILEPATHSENTRY']._serialized_options = b'8\001'
  _globals['_UPDATEMETADATA_METADATAENTRY']._loaded_options = None
  _globals['_UPDATEMETADATA_METADATAENTRY']._serialized_options = b'8\001'
  _globals['_OPERATION']._serialized_start=4148
  _globals['_OPERATION']._serialized_end=4204
  _globals['_SCALARENCODING']._serialized_start=4206
  _globals['_SCALARENCODING']._serialized_end=4246
  _globals['_SEGMENTSCOPE']._serialized_start=4248
  _globals['_SEGMENTSCOPE']._serialized_end=4312
  _globals['_WHEREDOCUMENTOPERATOR']._serialized_start=4314
  _globals['_WHEREDOCUMENTOPERATOR']._serialized_end=4369
  _globals['_BOOLEANOPERATOR']._serialized_start=4371
  _globals['_BOOLEANOPERATOR']._serialized_end=4405
  _globals['_LISTOPERATOR']._serialized_start=4407
  _globals['_LISTOPERATOR']._serialized_end=4438
  _globals['_GENERICCOMPARATOR']._serialized_start=4440
  _globals['_GENERICCOMPARATOR']._serialized_end=4475
  _globals['_NUMBERCOMPARATOR']._serialized_start=4477
  _globals['_NUMBERCOMPARATOR']._serialized_end=4529
  _globals['_STATUS']._serialized_start=39
  _globals['_STATUS']._serialized_end=77
  _globals['_VECTOR']._serialized_start=79
  _globals['_VECTOR']._serialized_end=164
  _globals['_FILEPATHS']._serialized_start=166
  _globals['_FILEPATHS']._serialized_end=192
  _globals['_SEGMENT']._serialized_start=195
  _globals['_SEGMENT']._serialized_end=488
  _globals['_SEGMENT_FILEPATHSENTRY']._serialized_start=393
  _globals['_SEGMENT_FILEPATHSENTRY']._serialized_end=460
  _globals['_COLLECTION']._serialized_start=491
  _globals['_COLLECTION']._serialized_end=700
  _globals['_DATABASE']._serialized_start=702
  _globals['_DATABASE']._serialized_end=754
  _globals['_TENANT']._serialized_start=756
  _globals['_TENANT']._serialized_end=778
  _globals['_UPDATEMETADATAVALUE']._serialized_start=780
  _globals['_UPDATEMETADATAVALUE']._serialized_end=900
  _globals['_UPDATEMETADATA']._serialized_start=903
  _globals['_UPDATEMETADATA']._serialized_end=1053
  _globals['_UPDATEMETADATA_METADATAENTRY']._serialized_start=977
  _globals['_UPDATEMETADATA_METADATAENTRY']._serialized_end=1053
  _globals['_OPERATIONRECORD']._serialized_start=1056
  _globals['_OPERATIONRECORD']._serialized_end=1231
  _globals['_COUNTRECORDSREQUEST']._serialized_start=1233
  _globals['_COUNTRECORDSREQUEST']._serialized_end=1274
  _globals['_COUNTRECORDSRESPONSE']._serialized_start=1276
  _globals['_COUNTRECORDSRESPONSE']._serialized_end=1313
  _globals['_QUERYMETADATAREQUEST']._serialized_start=1316
  _globals['_QUERYMETADATAREQUEST']._serialized_end=1510
  _globals['_QUERYMETADATARESPONSE']._serialized_start=1512
  _globals['_QUERYMETADATARESPONSE']._serialized_end=1585
  _globals['_METADATAEMBEDDINGRECORD']._serialized_start=1587
  _globals['_METADATAEMBEDDINGRECORD']._serialized_end=1666
  _globals['_WHEREDOCUMENT']._serialized_start=1669
  _globals['_WHEREDOCUMENT']._serialized_end=1800
  _globals['_DIRECTWHEREDOCUMENT']._serialized_start=1802
  _globals['_DIRECTWHEREDOCUMENT']._serialized_end=1890
  _globals['_WHEREDOCUMENTCHILDREN']._serialized_start=1892
  _globals['_WHEREDOCUMENTCHILDREN']._serialized_end=1999
  _globals['_WHERE']._serialized_start=2001
  _globals['_WHERE']._serialized_end=2115
  _globals['_DIRECTCOMPARISON']._serialized_start=2118
  _globals['_DIRECTCOMPARISON']._serialized_end=2647
  _globals['_WHERECHILDREN']._serialized_start=2649
  _globals['_WHERECHILDREN']._serialized_end=2740
  _globals['_STRINGLISTCOMPARISON']._serialized_start=2742
  _globals['_STRINGLISTCOMPARISON']._serialized_end=2825
  _globals['_SINGLESTRINGCOMPARISON']._serialized_start=2827
  _globals['_SINGLESTRINGCOMPARISON']._serialized_end=2913
  _globals['_SINGLEBOOLCOMPARISON']._serialized_start=2915
  _globals['_SINGLEBOOLCOMPARISON']._serialized_end=2999
  _globals['_INTLISTCOMPARISON']._serialized_start=3001
  _globals['_INTLISTCOMPARISON']._serialized_end=3081
  _globals['_SINGLEINTCOMPARISON']._serialized_start=3084
  _globals['_SINGLEINTCOMPARISON']._serialized_end=3246
  _globals['_DOUBLELISTCOMPARISON']._serialized_start=3248
  _globals['_DOUBLELISTCOMPARISON']._serialized_end=3331
  _globals['_BOOLLISTCOMPARISON']._serialized_start=3333
  _globals['_BOOLLISTCOMPARISON']._serialized_end=3414
  _globals['_SINGLEDOUBLECOMPARISON']._serialized_start=3417
  _globals['_SINGLEDOUBLECOMPARISON']._serialized_end=3582
  _globals['_GETVECTORSREQUEST']._serialized_start=3584
  _globals['_GETVECTORSREQUEST']._serialized_end=3636
  _globals['_GETVECTORSRESPONSE']._serialized_start=3638
  _globals['_GETVECTORSRESPONSE']._serialized_end=3706
  _globals['_VECTOREMBEDDINGRECORD']._serialized_start=3708
  _globals['_VECTOREMBEDDINGRECORD']._serialized_end=3775
  _globals['_QUERYVECTORSREQUEST']._serialized_start=3778
  _globals['_QUERYVECTORSREQUEST']._serialized_end=3912
  _globals['_QUERYVECTORSRESPONSE']._serialized_start=3914
  _globals['_QUERYVECTORSRESPONSE']._serialized_end=3981
  _globals['_VECTORQUERYRESULTS']._serialized_start=3983
  _globals['_VECTORQUERYRESULTS']._serialized_end=4047
  _globals['_VECTORQUERYRESULT']._serialized_start=4049
  _globals['_VECTORQUERYRESULT']._serialized_end=4146
  _globals['_METADATAREADER']._serialized_start=4532
  _globals['_METADATAREADER']._serialized_end=4705
  _globals['_VECTORREADER']._serialized_start=4708
  _globals['_VECTORREADER']._serialized_end=4870
# @@protoc_insertion_point(module_scope)