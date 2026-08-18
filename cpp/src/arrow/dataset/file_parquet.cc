// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "arrow/dataset/file_parquet.h"

#include <any>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "arrow/array/array_primitive.h"
#include "arrow/array/builder_primitive.h"
#include "arrow/compute/api_vector.h"
#include "arrow/compute/cast.h"
#include "arrow/compute/exec.h"
#include "arrow/dataset/dataset_internal.h"
#include "arrow/dataset/parquet_encryption_config.h"
#include "arrow/dataset/scanner.h"
#include "arrow/filesystem/path_util.h"
#include "arrow/table.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/future.h"
#include "arrow/util/iterator.h"
#include "arrow/util/logging_internal.h"
#include "arrow/util/range.h"
#include "arrow/util/thread_pool.h"
#include "arrow/util/tracing_internal.h"
#include "parquet/arrow/reader.h"
#include "parquet/arrow/schema.h"
#include "parquet/arrow/writer.h"
#include "parquet/encryption/crypto_factory.h"
#include "parquet/encryption/encryption.h"
#include "parquet/encryption/kms_client.h"
#include "parquet/file_reader.h"
#include "parquet/properties.h"
#include "parquet/row_selection.h"
#include "parquet/statistics.h"

namespace arrow {

using internal::checked_cast;
using internal::checked_pointer_cast;
using internal::Iota;

namespace dataset {

using parquet::arrow::SchemaField;
using parquet::arrow::SchemaManifest;
using parquet::arrow::StatisticsAsScalars;

using compute::Cast;

namespace {

parquet::ReaderProperties MakeReaderProperties(
    const ParquetFileFormat& format, ParquetFragmentScanOptions* parquet_scan_options,
    const std::string& path = "", std::shared_ptr<fs::FileSystem> filesystem = nullptr,
    MemoryPool* pool = default_memory_pool()) {
  // Can't mutate pool after construction
  parquet::ReaderProperties properties(pool);
  if (parquet_scan_options->reader_properties->is_buffered_stream_enabled()) {
    properties.enable_buffered_stream();
  } else {
    properties.disable_buffered_stream();
  }
  properties.set_buffer_size(parquet_scan_options->reader_properties->buffer_size());

  auto file_decryption_prop =
      parquet_scan_options->reader_properties->file_decryption_properties();

#ifdef PARQUET_REQUIRE_ENCRYPTION
  auto parquet_decrypt_config = parquet_scan_options->parquet_decryption_config;

  if (parquet_decrypt_config != nullptr) {
    file_decryption_prop =
        parquet_decrypt_config->crypto_factory->GetFileDecryptionProperties(
            *parquet_decrypt_config->kms_connection_config,
            *parquet_decrypt_config->decryption_config, path, filesystem);
  }
#else
  if (parquet_scan_options->parquet_decryption_config != nullptr) {
    parquet::ParquetException::NYI("Encryption is not supported in this build.");
  }
#endif

  properties.file_decryption_properties(file_decryption_prop);

  properties.set_thrift_string_size_limit(
      parquet_scan_options->reader_properties->thrift_string_size_limit());
  properties.set_thrift_container_size_limit(
      parquet_scan_options->reader_properties->thrift_container_size_limit());

  properties.set_page_checksum_verification(
      parquet_scan_options->reader_properties->page_checksum_verification());

  return properties;
}

parquet::ArrowReaderProperties MakeArrowReaderProperties(
    const ParquetFileFormat& format, const parquet::FileMetaData& metadata) {
  parquet::ArrowReaderProperties properties(/* use_threads = */ false);
  for (const std::string& name : format.reader_options.dict_columns) {
    auto column_index = metadata.schema()->ColumnIndex(name);
    properties.set_read_dictionary(column_index, true);
  }
  properties.set_coerce_int96_timestamp_unit(
      format.reader_options.coerce_int96_timestamp_unit);
  properties.set_binary_type(format.reader_options.binary_type);
  properties.set_list_type(format.reader_options.list_type);
  return properties;
}

parquet::ArrowReaderProperties MakeArrowReaderProperties(
    const ParquetFileFormat& format, const parquet::FileMetaData& metadata,
    const ScanOptions& options, const ParquetFragmentScanOptions& parquet_scan_options) {
  auto arrow_properties = MakeArrowReaderProperties(format, metadata);
  arrow_properties.set_batch_size(options.batch_size);
  // Must be set here since the sync ScanTask handles pre-buffering itself
  arrow_properties.set_pre_buffer(
      parquet_scan_options.arrow_reader_properties->pre_buffer());
  arrow_properties.set_cache_options(
      parquet_scan_options.arrow_reader_properties->cache_options());
  arrow_properties.set_io_context(
      parquet_scan_options.arrow_reader_properties->io_context());
  arrow_properties.set_use_threads(options.use_threads);
  arrow_properties.set_arrow_extensions_enabled(
      parquet_scan_options.arrow_reader_properties->get_arrow_extensions_enabled());
  return arrow_properties;
}

Result<std::shared_ptr<SchemaManifest>> GetSchemaManifest(
    const parquet::FileMetaData& metadata,
    const parquet::ArrowReaderProperties& properties) {
  auto manifest = std::make_shared<SchemaManifest>();
  const std::shared_ptr<const ::arrow::KeyValueMetadata>& key_value_metadata =
      metadata.key_value_metadata();
  RETURN_NOT_OK(SchemaManifest::Make(metadata.schema(), key_value_metadata, properties,
                                     manifest.get()));
  return manifest;
}

bool IsNan(const Scalar& value) {
  if (value.is_valid) {
    if (value.type->id() == Type::FLOAT) {
      const FloatScalar& float_scalar = checked_cast<const FloatScalar&>(value);
      return std::isnan(float_scalar.value);
    } else if (value.type->id() == Type::DOUBLE) {
      const DoubleScalar& double_scalar = checked_cast<const DoubleScalar&>(value);
      return std::isnan(double_scalar.value);
    }
  }
  return false;
}

std::optional<compute::Expression> ColumnChunkStatisticsAsExpression(
    const FieldRef& field_ref, const SchemaField& schema_field,
    const parquet::RowGroupMetaData& metadata) {
  // For the remaining of this function, failure to extract/parse statistics
  // are ignored by returning nullptr. The goal is two fold. First
  // avoid an optimization which breaks the computation. Second, allow the
  // following columns to maybe succeed in extracting column statistics.

  // For now, only leaf (primitive) types are supported.
  if (!schema_field.is_leaf()) {
    return std::nullopt;
  }

  auto column_metadata = metadata.ColumnChunk(schema_field.column_index);
  auto statistics = column_metadata->statistics();
  const auto& field = schema_field.field;

  if (statistics == nullptr) {
    return std::nullopt;
  }

  return ParquetFileFragment::EvaluateStatisticsAsExpression(*field, field_ref,
                                                             *statistics);
}

void AddColumnIndices(const SchemaField& schema_field,
                      std::vector<int>* column_projection) {
  if (schema_field.is_leaf()) {
    column_projection->push_back(schema_field.column_index);
  } else {
    // The following ensure that complex types, e.g. struct,  are materialized.
    for (const auto& child : schema_field.children) {
      AddColumnIndices(child, column_projection);
    }
  }
}

Status ResolveOneFieldRef(
    const SchemaManifest& manifest, const FieldRef& field_ref,
    const std::unordered_map<std::string, const SchemaField*>& field_lookup,
    const std::unordered_set<std::string>& duplicate_fields,
    std::vector<int>* columns_selection) {
  if (const std::string* name = field_ref.name()) {
    auto it = field_lookup.find(*name);
    if (it != field_lookup.end()) {
      AddColumnIndices(*it->second, columns_selection);
    } else if (duplicate_fields.find(*name) != duplicate_fields.end()) {
      // We shouldn't generally get here because SetProjection will reject such references
      return Status::Invalid("Ambiguous reference to column '", *name,
                             "' which occurs more than once");
    }
    // "Virtual" column: field is not in file but is in the ScanOptions.
    // Ignore it here, as projection will pad the batch with a null column.
    return Status::OK();
  }

  const SchemaField* toplevel = nullptr;
  const SchemaField* field = nullptr;
  if (const std::vector<FieldRef>* refs = field_ref.nested_refs()) {
    // Only supports a sequence of names
    for (const auto& ref : *refs) {
      if (const std::string* name = ref.name()) {
        if (!field) {
          // First lookup, top-level field
          auto it = field_lookup.find(*name);
          if (it != field_lookup.end()) {
            field = it->second;
            toplevel = field;
          } else if (duplicate_fields.find(*name) != duplicate_fields.end()) {
            return Status::Invalid("Ambiguous reference to column '", *name,
                                   "' which occurs more than once");
          } else {
            // Virtual column
            return Status::OK();
          }
        } else {
          const SchemaField* result = nullptr;
          for (const auto& child : field->children) {
            if (child.field->name() == *name) {
              if (!result) {
                result = &child;
              } else {
                return Status::Invalid("Ambiguous nested reference to column '", *name,
                                       "' which occurs more than once in field ",
                                       field->field->ToString());
              }
            }
          }
          if (!result) {
            // Virtual column
            return Status::OK();
          }
          field = result;
        }
        continue;
      }
      return Status::NotImplemented("Inferring column projection from FieldRef ",
                                    field_ref.ToString());
    }
  } else {
    return Status::NotImplemented("Inferring column projection from FieldRef ",
                                  field_ref.ToString());
  }

  if (field) {
    // TODO(ARROW-1888): support fine-grained column projection. We should be
    // able to materialize only the child fields requested, and not the entire
    // top-level field.
    // Right now, if enabled, projection/filtering will fail when they cast the
    // physical schema to the dataset schema.
    AddColumnIndices(*toplevel, columns_selection);
  }
  return Status::OK();
}

// Converts a field ref into a position-independent ref (containing only a sequence of
// names) based on the dataset schema. Returns `false` if no conversion was needed.
Result<FieldRef> MaybeConvertFieldRef(FieldRef ref, const Schema& dataset_schema) {
  if (ARROW_PREDICT_TRUE(ref.IsNameSequence())) {
    return ref;
  }

  ARROW_ASSIGN_OR_RAISE(auto path, ref.FindOne(dataset_schema));
  std::vector<FieldRef> named_refs;
  named_refs.reserve(path.indices().size());

  const FieldVector* child_fields = &dataset_schema.fields();
  for (auto index : path) {
    const auto& child_field = *(*child_fields)[index];
    named_refs.emplace_back(child_field.name());
    child_fields = &child_field.type()->fields();
  }

  return named_refs.size() == 1 ? std::move(named_refs[0])
                                : FieldRef(std::move(named_refs));
}

// Compute the column projection based on the scan options
Result<std::vector<int>> InferColumnProjection(const parquet::arrow::FileReader& reader,
                                               const ScanOptions& options) {
  auto manifest = reader.manifest();
  // Checks if the field is needed in either the projection or the filter.
  auto field_refs = options.MaterializedFields();

  // Build a lookup table from top level field name to field metadata.
  // This is to avoid quadratic-time mapping of projected fields to
  // column indices, in the common case of selecting top level
  // columns. For nested fields, we will pay the cost of a linear scan
  // assuming for now that this is relatively rare, but this can be
  // optimized. (Also, we don't want to pay the cost of building all
  // the lookup tables up front if they're rarely used.)
  std::unordered_map<std::string, const SchemaField*> field_lookup;
  std::unordered_set<std::string> duplicate_fields;
  for (const auto& schema_field : manifest.schema_fields) {
    const auto it = field_lookup.emplace(schema_field.field->name(), &schema_field);
    if (!it.second) {
      duplicate_fields.emplace(schema_field.field->name());
    }
  }

  std::vector<int> columns_selection;
  for (auto& ref : field_refs) {
    // In the (unlikely) absence of a known dataset schema, we require that all
    // materialized refs are named.
    if (options.dataset_schema) {
      ARROW_ASSIGN_OR_RAISE(
          ref, MaybeConvertFieldRef(std::move(ref), *options.dataset_schema));
    }
    RETURN_NOT_OK(ResolveOneFieldRef(manifest, ref, field_lookup, duplicate_fields,
                                     &columns_selection));
  }
  return columns_selection;
}

Status WrapSourceError(const Status& status, const std::string& path) {
  return status.WithMessage("Could not open Parquet input source '", path,
                            "': ", status.message());
}

Result<bool> IsSupportedParquetFile(const ParquetFileFormat& format,
                                    const FileSource& source) {
  BEGIN_PARQUET_CATCH_EXCEPTIONS
  try {
    ARROW_ASSIGN_OR_RAISE(auto input, source.Open());
    ARROW_ASSIGN_OR_RAISE(
        auto parquet_scan_options,
        GetFragmentScanOptions<ParquetFragmentScanOptions>(
            kParquetTypeName, nullptr, format.default_fragment_scan_options));
    auto reader = parquet::ParquetFileReader::Open(
        std::move(input), MakeReaderProperties(format, parquet_scan_options.get()));
    std::shared_ptr<parquet::FileMetaData> metadata = reader->metadata();
    return metadata != nullptr && metadata->can_decompress();
  } catch (const ::parquet::ParquetInvalidOrCorruptedFileException& e) {
    ARROW_UNUSED(e);
    return false;
  }
  END_PARQUET_CATCH_EXCEPTIONS
}

}  // namespace

std::optional<compute::Expression> ParquetFileFragment::EvaluateStatisticsAsExpression(
    const Field& field, const FieldRef& field_ref,
    const parquet::Statistics& statistics) {
  auto field_expr = compute::field_ref(field_ref);

  bool may_have_null = !statistics.HasNullCount() || statistics.null_count() > 0;
  // Optimize for corner case where all values are nulls
  if (statistics.num_values() == 0) {
    // If there are no non-null values, column `field_ref` in the fragment
    // might be empty or all values are nulls. In this case, we also return
    // a null expression.
    return is_null(std::move(field_expr));
  }

  std::shared_ptr<Scalar> min, max;
  if (!StatisticsAsScalars(statistics, &min, &max).ok()) {
    return std::nullopt;
  }

  auto maybe_min = Cast(min, field.type());
  auto maybe_max = Cast(max, field.type());
  if (maybe_min.ok() && maybe_max.ok()) {
    min = maybe_min.MoveValueUnsafe().scalar();
    max = maybe_max.MoveValueUnsafe().scalar();

    if (min->Equals(*max)) {
      auto single_value = compute::equal(field_expr, compute::literal(std::move(min)));

      if (!may_have_null) {
        return single_value;
      }
      return compute::or_(std::move(single_value), is_null(std::move(field_expr)));
    }

    auto lower_bound = compute::greater_equal(field_expr, compute::literal(min));
    auto upper_bound = compute::less_equal(field_expr, compute::literal(max));
    compute::Expression in_range;

    // Since the minimum & maximum values are NaN, useful statistics
    // cannot be extracted for checking the presence of a value within
    // range
    if (IsNan(*min) && IsNan(*max)) {
      return std::nullopt;
    }

    // If either minimum or maximum is NaN, it should be ignored for the
    // range computation
    if (IsNan(*min)) {
      in_range = std::move(upper_bound);
    } else if (IsNan(*max)) {
      in_range = std::move(lower_bound);
    } else {
      in_range = compute::and_(std::move(lower_bound), std::move(upper_bound));
    }
    if (may_have_null) {
      return compute::or_(std::move(in_range), compute::is_null(std::move(field_expr)));
    }
    return in_range;
  }
  return std::nullopt;
}

std::optional<compute::Expression> ParquetFileFragment::EvaluateStatisticsAsExpression(
    const Field& field, const parquet::Statistics& statistics) {
  auto field_name = field.name();
  return EvaluateStatisticsAsExpression(field, FieldRef(std::move(field_name)),
                                        statistics);
}

ParquetFileFormat::ParquetFileFormat()
    : FileFormat(std::make_shared<ParquetFragmentScanOptions>()) {}

bool ParquetFileFormat::Equals(const FileFormat& other) const {
  if (other.type_name() != type_name()) return false;

  const auto& other_reader_options =
      checked_cast<const ParquetFileFormat&>(other).reader_options;

  // FIXME implement comparison for decryption options
  return (reader_options.dict_columns == other_reader_options.dict_columns &&
          reader_options.coerce_int96_timestamp_unit ==
              other_reader_options.coerce_int96_timestamp_unit &&
          reader_options.binary_type == other_reader_options.binary_type &&
          reader_options.list_type == other_reader_options.list_type);
}

ParquetFileFormat::ParquetFileFormat(const parquet::ReaderProperties& reader_properties)
    : FileFormat(std::make_shared<ParquetFragmentScanOptions>()) {
  auto* default_scan_opts =
      static_cast<ParquetFragmentScanOptions*>(default_fragment_scan_options.get());
  *default_scan_opts->reader_properties = reader_properties;
}

Result<bool> ParquetFileFormat::IsSupported(const FileSource& source) const {
  auto maybe_is_supported = IsSupportedParquetFile(*this, source);
  if (!maybe_is_supported.ok()) {
    return WrapSourceError(maybe_is_supported.status(), source.path());
  }
  return maybe_is_supported;
}

Result<std::shared_ptr<Schema>> ParquetFileFormat::Inspect(
    const FileSource& source) const {
  auto scan_options = std::make_shared<ScanOptions>();
  ARROW_ASSIGN_OR_RAISE(auto reader, GetReader(source, scan_options));
  std::shared_ptr<Schema> schema;
  RETURN_NOT_OK(reader->GetSchema(&schema));
  return schema;
}

Result<std::shared_ptr<parquet::arrow::FileReader>> ParquetFileFormat::GetReader(
    const FileSource& source, const std::shared_ptr<ScanOptions>& options) const {
  return GetReader(source, options, /*metadata=*/nullptr);
}

Result<std::shared_ptr<parquet::arrow::FileReader>> ParquetFileFormat::GetReader(
    const FileSource& source, const std::shared_ptr<ScanOptions>& options,
    const std::shared_ptr<parquet::FileMetaData>& metadata) const {
  ARROW_ASSIGN_OR_RAISE(
      auto parquet_scan_options,
      GetFragmentScanOptions<ParquetFragmentScanOptions>(kParquetTypeName, options.get(),
                                                         default_fragment_scan_options));
  auto properties =
      MakeReaderProperties(*this, parquet_scan_options.get(), "", nullptr, options->pool);
  ARROW_ASSIGN_OR_RAISE(auto input, source.Open());
  // `parquet::ParquetFileReader::Open` will not wrap the exception as status,
  // so using `open_parquet_file` to wrap it.
  auto open_parquet_file = [&]() -> Result<std::unique_ptr<parquet::ParquetFileReader>> {
    BEGIN_PARQUET_CATCH_EXCEPTIONS
    auto reader = parquet::ParquetFileReader::Open(std::move(input),
                                                   std::move(properties), metadata);
    return reader;
    END_PARQUET_CATCH_EXCEPTIONS
  };

  auto reader_opt = open_parquet_file();
  if (!reader_opt.ok()) {
    return WrapSourceError(reader_opt.status(), source.path());
  }
  auto reader = std::move(reader_opt).ValueOrDie();

  std::shared_ptr<parquet::FileMetaData> reader_metadata = reader->metadata();
  auto arrow_properties =
      MakeArrowReaderProperties(*this, *reader_metadata, *options, *parquet_scan_options);
  ARROW_ASSIGN_OR_RAISE(auto arrow_reader,
                        parquet::arrow::FileReader::Make(options->pool, std::move(reader),
                                                         std::move(arrow_properties)));
  return arrow_reader;
}

Future<std::shared_ptr<parquet::arrow::FileReader>> ParquetFileFormat::GetReaderAsync(
    const FileSource& source, const std::shared_ptr<ScanOptions>& options) const {
  return GetReaderAsync(source, options, nullptr);
}

Future<std::shared_ptr<parquet::arrow::FileReader>> ParquetFileFormat::GetReaderAsync(
    const FileSource& source, const std::shared_ptr<ScanOptions>& options,
    const std::shared_ptr<parquet::FileMetaData>& metadata) const {
  ARROW_ASSIGN_OR_RAISE(
      auto parquet_scan_options,
      GetFragmentScanOptions<ParquetFragmentScanOptions>(kParquetTypeName, options.get(),
                                                         default_fragment_scan_options));
  auto properties = MakeReaderProperties(*this, parquet_scan_options.get(), source.path(),
                                         source.filesystem(), options->pool);
  auto self = checked_pointer_cast<const ParquetFileFormat>(shared_from_this());

  return source.OpenAsync().Then(
      [self = self, properties = std::move(properties), source = source,
       options = options, metadata = metadata,
       parquet_scan_options = parquet_scan_options](
          const std::shared_ptr<io::RandomAccessFile>& input) mutable {
        return parquet::ParquetFileReader::OpenAsync(input, properties, metadata)
            .Then(
                [=](const std::unique_ptr<parquet::ParquetFileReader>& reader) mutable
                -> Result<std::shared_ptr<parquet::arrow::FileReader>> {
                  auto arrow_properties = MakeArrowReaderProperties(
                      *self, *reader->metadata(), *options, *parquet_scan_options);

                  ARROW_ASSIGN_OR_RAISE(
                      auto arrow_reader,
                      parquet::arrow::FileReader::Make(
                          options->pool,
                          // TODO(ARROW-12259): workaround since we have Future<(move-only
                          // type)> It *wouldn't* be safe to const_cast reader except that
                          // here we know there are no other waiters on the reader.
                          std::move(
                              const_cast<std::unique_ptr<parquet::ParquetFileReader>&>(
                                  reader)),
                          arrow_properties));

                  return arrow_reader;
                },
                [path = source.path()](const Status& status)
                    -> Result<std::shared_ptr<parquet::arrow::FileReader>> {
                  return WrapSourceError(status, path);
                });
      });
}

struct SlicingGenerator {
  SlicingGenerator(RecordBatchGenerator source, int64_t batch_size)
      : state(std::make_shared<State>(source, batch_size)) {}

  Future<std::shared_ptr<RecordBatch>> operator()() {
    if (state->current) {
      return state->SliceOffABatch();
    } else {
      auto state_capture = state;
      return state->source().Then(
          [state_capture](const std::shared_ptr<RecordBatch>& next) {
            if (IsIterationEnd(next)) {
              return next;
            }
            state_capture->current = next;
            return state_capture->SliceOffABatch();
          });
    }
  }

  struct State {
    State(RecordBatchGenerator source, int64_t batch_size)
        : source(std::move(source)), current(), batch_size(batch_size) {}

    std::shared_ptr<RecordBatch> SliceOffABatch() {
      if (current->num_rows() <= batch_size) {
        auto sliced = current;
        current = nullptr;
        return sliced;
      }
      auto slice = current->Slice(0, batch_size);
      current = current->Slice(batch_size);
      return slice;
    }

    RecordBatchGenerator source;
    std::shared_ptr<RecordBatch> current;
    int64_t batch_size;
  };
  std::shared_ptr<State> state;
};

/// \brief Build a record-batch generator that reads each row group individually
/// with page-level I/O pruning via RowSelection.
///
/// For row groups that have no entry in \p page_selections, the row group is
/// read normally.  For row groups that do have an entry, only the pages
/// matching the RowSelection are read from disk via SparseInputStream,
/// enabling true I/O-level page skipping.
///
/// This replaces the default GetRecordBatchGenerator() call when page-level
/// pruning has produced a non-empty page_selections_ map.
static Result<RecordBatchGenerator> MakePagePrunedGenerator(
    const std::shared_ptr<parquet::arrow::FileReader>& reader,
    const std::vector<int>& row_groups, const std::vector<int>& column_projection,
    const std::map<int, std::shared_ptr<parquet::RowSelection>>& page_selections,
    int64_t batch_size, MemoryPool* pool) {
  // Collect batches for all row groups synchronously (blocking).
  // This is acceptable here because ScanBatchesAsync already runs inside a
  // thread-pool task (via MakeFromFuture → GetReaderAsync).
  std::vector<std::shared_ptr<RecordBatch>> all_batches;
  for (int rg : row_groups) {
    std::shared_ptr<Table> table;

    // Check if this row group has a RowSelection for page-level pruning
    auto it = page_selections.find(rg);
    if (it != page_selections.end() && it->second) {
      // Use sparse I/O path: only selected pages are fetched from disk
      ARROW_ASSIGN_OR_RAISE(
          table, reader->ReadRowGroupWithRowSelection(rg, column_projection, it->second));
    } else {
      // No pruning: read entire row group normally
      ARROW_ASSIGN_OR_RAISE(table, reader->ReadRowGroup(rg, column_projection));
    }

    if (table->num_rows() == 0) continue;

    // Split into batches of at most batch_size rows.
    TableBatchReader tbr(*table);
    tbr.set_chunksize(batch_size);
    ARROW_ASSIGN_OR_RAISE(auto batches, tbr.ToRecordBatches());
    for (auto& batch : batches) {
      all_batches.push_back(std::move(batch));
    }
  }
  return MakeVectorGenerator(std::move(all_batches));
}

Result<RecordBatchGenerator> ParquetFileFormat::ScanBatchesAsync(
    const std::shared_ptr<ScanOptions>& options,
    const std::shared_ptr<FileFragment>& file) const {
  auto parquet_fragment = checked_pointer_cast<ParquetFileFragment>(file);
  std::vector<int> row_groups;
  bool pre_filtered = false;
  // If RowGroup metadata is cached completely we can pre-filter RowGroups before opening
  // a FileReader, potentially avoiding IO altogether if all RowGroups are excluded due to
  // prior statistics knowledge. In the case where a RowGroup doesn't have statistics
  // metadata, it will not be excluded.
  if (parquet_fragment->metadata() != nullptr) {
    ARROW_ASSIGN_OR_RAISE(row_groups, parquet_fragment->FilterRowGroups(options->filter));
    pre_filtered = true;
    if (row_groups.empty()) return MakeEmptyGenerator<std::shared_ptr<RecordBatch>>();
  }
  // Open the reader and pay the real IO cost.
  auto make_generator =
      [this, options, parquet_fragment, pre_filtered,
       row_groups](const std::shared_ptr<parquet::arrow::FileReader>& reader) mutable
      -> Result<RecordBatchGenerator> {
    // Ensure that parquet_fragment has FileMetaData
    RETURN_NOT_OK(parquet_fragment->EnsureCompleteMetadata(reader.get()));
    if (!pre_filtered) {
      // row groups were not already filtered; do this now
      ARROW_ASSIGN_OR_RAISE(row_groups,
                            parquet_fragment->FilterRowGroups(options->filter));
      if (row_groups.empty()) return MakeEmptyGenerator<std::shared_ptr<RecordBatch>>();
    }

    // After row-group pruning, attempt page-level pruning on surviving row groups.
    // ComputePageSelections populates parquet_fragment->page_selections_ with the
    // intersected RowSelection for each row group.  This is a best-effort
    // optimization: failures are logged and fall back to row-group-only pruning.
    ARROW_ASSIGN_OR_RAISE(
        auto parquet_scan_options_for_policy,
        GetFragmentScanOptions<ParquetFragmentScanOptions>(
            kParquetTypeName, options.get(), default_fragment_scan_options));
    const parquet::PageIndexPolicy policy =
        parquet_scan_options_for_policy->arrow_reader_properties->page_index_policy();
    if (policy != parquet::PageIndexPolicy::NEVER &&
        options->filter != compute::literal(true)) {
      // ComputePageSelections sets arrow_reader_properties on the underlying
      // ParquetFileReader and fills parquet_fragment->page_selections_.
      Status page_status =
          parquet_fragment->ComputePageSelections(options->filter, reader.get());
      if (policy == parquet::PageIndexPolicy::ALWAYS) {
        RETURN_NOT_OK(page_status);
      } else if (!page_status.ok()) {
        ARROW_LOG(WARNING)
            << "Page-level pruning failed, falling back to row-group pruning: "
            << page_status.message();
      }
    }

    ARROW_ASSIGN_OR_RAISE(auto column_projection,
                          InferColumnProjection(*reader, *options));
    int batch_readahead = options->batch_readahead;

    // If page-level pruning produced row selections, use the per-row-group
    // read+filter path so that page_selections_ is consumed by the I/O layer.
    if (!parquet_fragment->page_selections_.empty()) {
      ARROW_ASSIGN_OR_RAISE(auto generator,
                            MakePagePrunedGenerator(reader, row_groups, column_projection,
                                                    parquet_fragment->page_selections_,
                                                    options->batch_size, options->pool));
      RecordBatchGenerator sliced =
          SlicingGenerator(std::move(generator), options->batch_size);
      if (batch_readahead == 0) {
        return sliced;
      }
      RecordBatchGenerator sliced_readahead =
          MakeSerialReadaheadGenerator(std::move(sliced), batch_readahead);
      return sliced_readahead;
    }

    int64_t rows_to_readahead = batch_readahead * options->batch_size;
    // Use the executor from scan options if provided.
    auto cpu_executor = options->cpu_executor ? options->cpu_executor
                                              : ::arrow::internal::GetCpuThreadPool();
    ARROW_ASSIGN_OR_RAISE(auto generator, reader->GetRecordBatchGenerator(
                                              reader, row_groups, column_projection,
                                              cpu_executor, rows_to_readahead));
    RecordBatchGenerator sliced =
        SlicingGenerator(std::move(generator), options->batch_size);
    if (batch_readahead == 0) {
      return sliced;
    }
    RecordBatchGenerator sliced_readahead =
        MakeSerialReadaheadGenerator(std::move(sliced), batch_readahead);
    return sliced_readahead;
  };
  auto generator = MakeFromFuture(
      GetReaderAsync(parquet_fragment->source(), options, parquet_fragment->metadata())
          .Then(std::move(make_generator)));
  WRAP_ASYNC_GENERATOR_WITH_CHILD_SPAN(
      generator, "arrow::dataset::ParquetFileFormat::ScanBatchesAsync::Next");
  return generator;
}

Future<std::optional<int64_t>> ParquetFileFormat::CountRows(
    const std::shared_ptr<FileFragment>& file, compute::Expression predicate,
    const std::shared_ptr<ScanOptions>& options) {
  auto parquet_file = checked_pointer_cast<ParquetFileFragment>(file);
  if (parquet_file->metadata()) {
    ARROW_ASSIGN_OR_RAISE(auto maybe_count,
                          parquet_file->TryCountRows(std::move(predicate)));
    return Future<std::optional<int64_t>>::MakeFinished(maybe_count);
  } else {
    return DeferNotOk(options->io_context.executor()->Submit(
        [parquet_file, predicate]() -> Result<std::optional<int64_t>> {
          RETURN_NOT_OK(parquet_file->EnsureCompleteMetadata());
          return parquet_file->TryCountRows(predicate);
        }));
  }
}

Result<std::shared_ptr<ParquetFileFragment>> ParquetFileFormat::MakeFragment(
    FileSource source, compute::Expression partition_expression,
    std::shared_ptr<Schema> physical_schema, std::vector<int> row_groups) {
  return std::shared_ptr<ParquetFileFragment>(new ParquetFileFragment(
      std::move(source), shared_from_this(), std::move(partition_expression),
      std::move(physical_schema), std::move(row_groups)));
}

Result<std::shared_ptr<FileFragment>> ParquetFileFormat::MakeFragment(
    FileSource source, compute::Expression partition_expression,
    std::shared_ptr<Schema> physical_schema) {
  return std::shared_ptr<FileFragment>(new ParquetFileFragment(
      std::move(source), shared_from_this(), std::move(partition_expression),
      std::move(physical_schema), std::nullopt));
}

//
// ParquetFileWriter, ParquetFileWriteOptions
//

std::shared_ptr<FileWriteOptions> ParquetFileFormat::DefaultWriteOptions() {
  std::shared_ptr<ParquetFileWriteOptions> options(
      new ParquetFileWriteOptions(shared_from_this()));
  options->writer_properties = parquet::default_writer_properties();
  options->arrow_writer_properties = parquet::default_arrow_writer_properties();
  return options;
}

Result<std::shared_ptr<FileWriter>> ParquetFileFormat::MakeWriter(
    std::shared_ptr<io::OutputStream> destination, std::shared_ptr<Schema> schema,
    std::shared_ptr<FileWriteOptions> options,
    fs::FileLocator destination_locator) const {
  if (!Equals(*options->format())) {
    return Status::TypeError("Mismatching format/write options");
  }

  auto parquet_options = checked_pointer_cast<ParquetFileWriteOptions>(options);

  std::unique_ptr<parquet::arrow::FileWriter> parquet_writer;

#ifdef PARQUET_REQUIRE_ENCRYPTION
  auto parquet_encrypt_config = parquet_options->parquet_encryption_config;

  if (parquet_encrypt_config != nullptr) {
    auto file_encryption_prop =
        parquet_encrypt_config->crypto_factory->GetFileEncryptionProperties(
            *parquet_encrypt_config->kms_connection_config,
            *parquet_encrypt_config->encryption_config, destination_locator.path,
            destination_locator.filesystem);

    auto writer_properties =
        parquet::WriterProperties::Builder(*parquet_options->writer_properties)
            .encryption(std::move(file_encryption_prop))
            ->build();

    ARROW_ASSIGN_OR_RAISE(
        parquet_writer, parquet::arrow::FileWriter::Open(
                            *schema, writer_properties->memory_pool(), destination,
                            writer_properties, parquet_options->arrow_writer_properties));
  }
#else
  if (parquet_options->parquet_encryption_config != nullptr) {
    return Status::NotImplemented("Encryption is not supported in this build.");
  }
#endif

  if (parquet_writer == nullptr) {
    ARROW_ASSIGN_OR_RAISE(parquet_writer,
                          parquet::arrow::FileWriter::Open(
                              *schema, parquet_options->writer_properties->memory_pool(),
                              destination, parquet_options->writer_properties,
                              parquet_options->arrow_writer_properties));
  }

  return std::shared_ptr<FileWriter>(
      new ParquetFileWriter(std::move(destination), std::move(parquet_writer),
                            std::move(parquet_options), std::move(destination_locator)));
}

ParquetFileWriter::ParquetFileWriter(std::shared_ptr<io::OutputStream> destination,
                                     std::shared_ptr<parquet::arrow::FileWriter> writer,
                                     std::shared_ptr<ParquetFileWriteOptions> options,
                                     fs::FileLocator destination_locator)
    : FileWriter(writer->schema(), std::move(options), std::move(destination),
                 std::move(destination_locator)),
      parquet_writer_(std::move(writer)) {}

Status ParquetFileWriter::Write(const std::shared_ptr<RecordBatch>& batch) {
  ARROW_ASSIGN_OR_RAISE(auto table, Table::FromRecordBatches(batch->schema(), {batch}));
  return parquet_writer_->WriteTable(*table, batch->num_rows());
}

Future<> ParquetFileWriter::FinishInternal() {
  return DeferNotOk(destination_locator_.filesystem->io_context().executor()->Submit(
      [this]() { return parquet_writer_->Close(); }));
}

//
// ParquetFileFragment
//

ParquetFileFragment::ParquetFileFragment(FileSource source,
                                         std::shared_ptr<FileFormat> format,
                                         compute::Expression partition_expression,
                                         std::shared_ptr<Schema> physical_schema,
                                         std::optional<std::vector<int>> row_groups)
    : FileFragment(std::move(source), std::move(format), std::move(partition_expression),
                   std::move(physical_schema)),
      parquet_format_(checked_cast<ParquetFileFormat&>(*format_)),
      row_groups_(std::move(row_groups)) {}

std::shared_ptr<parquet::FileMetaData> ParquetFileFragment::metadata() {
  auto lock = physical_schema_mutex_.Lock();
  return metadata_;
}

Status ParquetFileFragment::EnsureCompleteMetadata(parquet::arrow::FileReader* reader) {
  auto lock = physical_schema_mutex_.Lock();
  if (metadata_ != nullptr) {
    return Status::OK();
  }

  if (reader == nullptr) {
    lock.Unlock();
    auto scan_options = std::make_shared<ScanOptions>();
    ARROW_ASSIGN_OR_RAISE(auto reader, parquet_format_.GetReader(source_, scan_options));
    return EnsureCompleteMetadata(reader.get());
  }

  RETURN_NOT_OK(reader->GetSchema(&physical_schema_));
  if (given_physical_schema_ && !given_physical_schema_->Equals(*physical_schema_)) {
    return Status::Invalid("Fragment initialized with physical schema ",
                           *given_physical_schema_, " but ", source_.path(),
                           " has schema ", *physical_schema_);
  }

  if (!row_groups_) {
    row_groups_ = Iota(reader->num_row_groups());
  }

  ARROW_ASSIGN_OR_RAISE(
      auto manifest,
      GetSchemaManifest(*reader->parquet_reader()->metadata(), reader->properties()));
  return SetMetadata(reader->parquet_reader()->metadata(), std::move(manifest));
}

Status ParquetFileFragment::SetMetadata(
    std::shared_ptr<parquet::FileMetaData> metadata,
    std::shared_ptr<parquet::arrow::SchemaManifest> manifest,
    std::shared_ptr<parquet::FileMetaData> original_metadata) {
  DCHECK(row_groups_.has_value());

  metadata_ = std::move(metadata);
  manifest_ = std::move(manifest);
  original_metadata_ = original_metadata ? std::move(original_metadata) : metadata_;
  // The SchemaDescriptor needs to be owned by a FileMetaData instance,
  // because SchemaManifest only stores a raw pointer (GH-39562).
  DCHECK_EQ(manifest_->descr, original_metadata_->schema())
      << "SchemaDescriptor should be owned by the original FileMetaData";

  if (!physical_schema_) {
    physical_schema_ = given_physical_schema_;
  }

  statistics_expressions_.resize(row_groups_->size(), compute::literal(true));
  statistics_expressions_complete_.resize(manifest_->descr->num_columns(), false);

  for (int row_group : *row_groups_) {
    // Ensure RowGroups are indexing valid RowGroups before augmenting.
    if (row_group < metadata_->num_row_groups()) continue;

    return Status::IndexError("ParquetFileFragment references row group ", row_group,
                              " but ", source_.path(), " only has ",
                              metadata_->num_row_groups(), " row groups");
  }

  return Status::OK();
}

Status ParquetFileFragment::ClearCachedMetadata() {
  metadata_.reset();
  manifest_.reset();
  original_metadata_.reset();
  return FileFragment::ClearCachedMetadata();
}

Result<FragmentVector> ParquetFileFragment::SplitByRowGroup(
    compute::Expression predicate) {
  RETURN_NOT_OK(EnsureCompleteMetadata());
  ARROW_ASSIGN_OR_RAISE(auto row_groups, FilterRowGroups(predicate));

  FragmentVector fragments(row_groups.size());
  int i = 0;
  for (int row_group : row_groups) {
    ARROW_ASSIGN_OR_RAISE(auto fragment,
                          parquet_format_.MakeFragment(source_, partition_expression(),
                                                       physical_schema_, {row_group}));

    RETURN_NOT_OK(fragment->SetMetadata(metadata_, manifest_,
                                        /*original_metadata=*/original_metadata_));
    fragments[i++] = std::move(fragment);
  }

  return fragments;
}

Result<std::shared_ptr<Fragment>> ParquetFileFragment::Subset(
    compute::Expression predicate) {
  RETURN_NOT_OK(EnsureCompleteMetadata());
  ARROW_ASSIGN_OR_RAISE(auto row_groups, FilterRowGroups(predicate));
  return Subset(std::move(row_groups));
}

Result<std::shared_ptr<Fragment>> ParquetFileFragment::Subset(
    std::vector<int> row_groups) {
  RETURN_NOT_OK(EnsureCompleteMetadata());
  ARROW_ASSIGN_OR_RAISE(auto new_fragment, parquet_format_.MakeFragment(
                                               source_, partition_expression(),
                                               physical_schema_, std::move(row_groups)));

  RETURN_NOT_OK(new_fragment->SetMetadata(metadata_, manifest_));
  return new_fragment;
}

inline void FoldingAnd(compute::Expression* l, compute::Expression r) {
  if (*l == compute::literal(true)) {
    *l = std::move(r);
  } else {
    *l = and_(std::move(*l), std::move(r));
  }
}

Result<std::vector<int>> ParquetFileFragment::FilterRowGroups(
    compute::Expression predicate) {
  std::vector<int> row_groups;
  ARROW_ASSIGN_OR_RAISE(auto expressions, TestRowGroups(std::move(predicate)));

  auto lock = physical_schema_mutex_.Lock();
  DCHECK(expressions.empty() || (expressions.size() == row_groups_->size()));
  for (size_t i = 0; i < expressions.size(); i++) {
    if (expressions[i].IsSatisfiable()) {
      row_groups.push_back(row_groups_->at(i));
    }
  }
  return row_groups;
}

Result<std::vector<compute::Expression>> ParquetFileFragment::TestRowGroups(
    compute::Expression predicate) {
  auto lock = physical_schema_mutex_.Lock();

  DCHECK_NE(metadata_, nullptr);
  ARROW_ASSIGN_OR_RAISE(
      predicate, SimplifyWithGuarantee(std::move(predicate), partition_expression_));

  if (!predicate.IsSatisfiable()) {
    return std::vector<compute::Expression>{};
  }

  for (const FieldRef& ref : FieldsInExpression(predicate)) {
    ARROW_ASSIGN_OR_RAISE(auto match, ref.FindOneOrNone(*physical_schema_));

    if (match.empty()) continue;
    const SchemaField* schema_field = &manifest_->schema_fields[match[0]];

    for (size_t i = 1; i < match.indices().size(); ++i) {
      if (schema_field->field->type()->id() != Type::STRUCT) {
        return Status::Invalid("nested paths only supported for structs");
      }
      schema_field = &schema_field->children[match[i]];
    }

    if (!schema_field->is_leaf()) continue;
    if (statistics_expressions_complete_[schema_field->column_index]) continue;
    statistics_expressions_complete_[schema_field->column_index] = true;

    int i = 0;
    for (int row_group : *row_groups_) {
      auto row_group_metadata = metadata_->RowGroup(row_group);

      if (auto minmax = ColumnChunkStatisticsAsExpression(ref, *schema_field,
                                                          *row_group_metadata)) {
        FoldingAnd(&statistics_expressions_[i], std::move(*minmax));
        ARROW_ASSIGN_OR_RAISE(statistics_expressions_[i],
                              statistics_expressions_[i].Bind(*physical_schema_));
      }

      ++i;
    }
  }

  std::vector<compute::Expression> row_groups(row_groups_->size());
  for (size_t i = 0; i < row_groups_->size(); ++i) {
    ARROW_ASSIGN_OR_RAISE(auto row_group_predicate,
                          SimplifyWithGuarantee(predicate, statistics_expressions_[i]));
    row_groups[i] = std::move(row_group_predicate);
  }
  return row_groups;
}

Result<std::optional<int64_t>> ParquetFileFragment::TryCountRows(
    compute::Expression predicate) {
  DCHECK_NE(metadata_, nullptr);
  if (ExpressionHasFieldRefs(predicate)) {
    ARROW_ASSIGN_OR_RAISE(auto expressions, TestRowGroups(std::move(predicate)));
    int64_t rows = 0;
    for (size_t i = 0; i < row_groups_->size(); i++) {
      // If the row group is entirely excluded, exclude it from the row count
      if (!expressions[i].IsSatisfiable()) continue;
      // Unless the row group is entirely included, bail out of fast path
      if (expressions[i] != compute::literal(true)) return std::nullopt;
      BEGIN_PARQUET_CATCH_EXCEPTIONS
      rows += metadata()->RowGroup((*row_groups_)[i])->num_rows();
      END_PARQUET_CATCH_EXCEPTIONS
    }
    return rows;
  }
  return metadata()->num_rows();
}

namespace {

/// \brief A single extracted leaf predicate: column index + operator + optional value.
struct LeafPredicate {
  int column_index;
  parquet::PredicateOp op;
  std::any value;  // empty for is_null / is_not_null
};

/// Map Arrow compute function names to the PredicateOp accepted by
/// ParquetFileReader::ComputePageSelection().
std::optional<parquet::PredicateOp> ArrowFunctionToPageOp(const std::string& fn) {
  if (fn == "equal") return parquet::PredicateOp::EQ;
  if (fn == "greater") return parquet::PredicateOp::GT;
  if (fn == "greater_equal") return parquet::PredicateOp::GTE;
  if (fn == "less") return parquet::PredicateOp::LT;
  if (fn == "less_equal") return parquet::PredicateOp::LTE;
  if (fn == "is_null") return parquet::PredicateOp::IS_NULL;
  if (fn == "is_valid") return parquet::PredicateOp::IS_NOT_NULL;
  return std::nullopt;
}

/// Attempt to extract a scalar value from an Arrow Scalar as a std::any whose
/// C++ type matches the *Parquet physical type* of @p target_col.  This is
/// required because ComputePageSelection performs a typeid() check on the
/// std::any value, so storing INT8/INT16/INT32 values as int64_t would cause a
/// type mismatch for any column whose Parquet physical type is INT32.
///
/// Returns an error Status for combinations that cannot be represented, or
/// Status::NotImplemented for unsupported Arrow types.
arrow::Result<std::any> ScalarToAny(const Scalar& scalar,
                                    const parquet::ColumnDescriptor* target_col) {
  if (!scalar.is_valid) return std::any{};

  const parquet::Type::type parquet_phys = target_col->physical_type();

  switch (scalar.type->id()) {
    case Type::BOOL:
      if (parquet_phys != parquet::Type::BOOLEAN) {
        return Status::TypeError("Cannot compare BOOL scalar to Parquet column of type ",
                                 parquet::TypeToString(parquet_phys));
      }
      return std::any{checked_cast<const BooleanScalar&>(scalar).value};

    case Type::INT8:
    case Type::INT16:
    case Type::INT32: {
      // Arrow INT8/INT16/INT32 scalars all carry an int32_t-range value; select
      // the C++ type to match the Parquet physical type so typeid() succeeds.
      int32_t v32;
      if (scalar.type->id() == Type::INT8)
        v32 = static_cast<int32_t>(checked_cast<const Int8Scalar&>(scalar).value);
      else if (scalar.type->id() == Type::INT16)
        v32 = static_cast<int32_t>(checked_cast<const Int16Scalar&>(scalar).value);
      else
        v32 = checked_cast<const Int32Scalar&>(scalar).value;

      if (parquet_phys == parquet::Type::INT32) return std::any{v32};
      if (parquet_phys == parquet::Type::INT64)
        return std::any{static_cast<int64_t>(v32)};
      return Status::TypeError(
          "Cannot compare INT8/INT16/INT32 scalar to Parquet column of type ",
          parquet::TypeToString(parquet_phys));
    }

    case Type::INT64:
      if (parquet_phys != parquet::Type::INT64) {
        return Status::TypeError("Cannot compare INT64 scalar to Parquet column of type ",
                                 parquet::TypeToString(parquet_phys));
      }
      return std::any{checked_cast<const Int64Scalar&>(scalar).value};

    case Type::UINT8:
    case Type::UINT16: {
      // Small unsigned integers fit in INT32 physical columns.
      uint32_t v32;
      if (scalar.type->id() == Type::UINT8)
        v32 = static_cast<uint32_t>(checked_cast<const UInt8Scalar&>(scalar).value);
      else
        v32 = static_cast<uint32_t>(checked_cast<const UInt16Scalar&>(scalar).value);

      if (parquet_phys == parquet::Type::INT32)
        return std::any{static_cast<int32_t>(v32)};
      if (parquet_phys == parquet::Type::INT64)
        return std::any{static_cast<int64_t>(v32)};
      return Status::TypeError(
          "Cannot compare UINT8/UINT16 scalar to Parquet column of type ",
          parquet::TypeToString(parquet_phys));
    }

    case Type::UINT32: {
      uint32_t v = checked_cast<const UInt32Scalar&>(scalar).value;
      // INT32 physical columns store signed values; promote to INT64 when the
      // value exceeds INT32_MAX to avoid silent truncation.
      if (parquet_phys == parquet::Type::INT32) {
        if (v > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
          return Status::TypeError(
              "UINT32 value ", v,
              " exceeds INT32_MAX; cannot fit in INT32 physical column");
        }
        return std::any{static_cast<int32_t>(v)};
      }
      if (parquet_phys == parquet::Type::INT64) return std::any{static_cast<int64_t>(v)};
      return Status::TypeError("Cannot compare UINT32 scalar to Parquet column of type ",
                               parquet::TypeToString(parquet_phys));
    }

    case Type::UINT64: {
      uint64_t v = checked_cast<const UInt64Scalar&>(scalar).value;
      if (parquet_phys == parquet::Type::INT64) return std::any{static_cast<int64_t>(v)};
      return Status::TypeError("Cannot compare UINT64 scalar to Parquet column of type ",
                               parquet::TypeToString(parquet_phys));
    }

    case Type::FLOAT:
      if (parquet_phys != parquet::Type::FLOAT) {
        return Status::TypeError("Cannot compare FLOAT scalar to Parquet column of type ",
                                 parquet::TypeToString(parquet_phys));
      }
      return std::any{checked_cast<const FloatScalar&>(scalar).value};

    case Type::DOUBLE:
      if (parquet_phys != parquet::Type::DOUBLE) {
        return Status::TypeError(
            "Cannot compare DOUBLE scalar to Parquet column of type ",
            parquet::TypeToString(parquet_phys));
      }
      return std::any{checked_cast<const DoubleScalar&>(scalar).value};

    case Type::STRING:
    case Type::LARGE_STRING:
      if (parquet_phys != parquet::Type::BYTE_ARRAY) {
        return Status::TypeError(
            "Cannot compare STRING scalar to Parquet column of type ",
            parquet::TypeToString(parquet_phys));
      }
      if (scalar.type->id() == Type::STRING)
        return std::any{
            std::string(checked_cast<const StringScalar&>(scalar).value->ToString())};
      return std::any{
          std::string(checked_cast<const LargeStringScalar&>(scalar).value->ToString())};

    default:
      return Status::NotImplemented("ScalarToAny: unsupported Arrow type ",
                                    scalar.type->ToString());
  }
}

/// Walk a compute::Expression tree (depth-first, AND-conjunction aware) and
/// collect all leaf predicates that can be pushed down to the page-index API.
///
/// Supported forms:
///   field_ref op literal  (both orderings; op in ==, <, <=, >, >=)
///   is_null(field_ref)
///   is_valid(field_ref)
///   and_(left, right)      (recursed into)
///
/// Composition model: the extracted leaves form an implicit conjunction (AND).
/// Each leaf is evaluated independently by ComputePageSelection(), and the
/// per-column RowSelections are combined with RowSelection::Intersect() by the
/// caller (ComputePageSelections). This covers the common
/// "col_a > x AND col_b == y AND ..." filter shape.
///
/// Deliberately NOT handled (safe over-selection — never wrong, just less
/// pruning):
///   - OR / or_kleene: an OR branch is ignored rather than combined via
///     RowSelection::Union(). Ignoring it can only *retain* pages that a full
///     evaluation might have dropped, so results stay correct; the post-scan
///     filter removes any surplus rows. Wiring Union() through a tree-shaped
///     combiner is left as a follow-up.
///   - NOT / negation, and any non-comparison function.
/// In all unsupported cases the offending sub-expression is silently skipped,
/// which is safe: page pruning is an optimization, and the exact predicate is
/// always re-applied by the scanner after reading.
///
/// @p parquet_schema is used to look up each column's Parquet physical type so
/// that ScalarToAny() can produce a std::any value whose C++ type matches what
/// ComputePageSelection() expects.
void ExtractLeafPredicates(const compute::Expression& expr,
                           const SchemaManifest& manifest, const Schema& physical_schema,
                           const parquet::SchemaDescriptor& parquet_schema,
                           std::vector<LeafPredicate>* out) {
  const auto* call = expr.call();
  if (!call) return;

  // Recurse into AND conjunctions.
  if (call->function_name == "and" || call->function_name == "and_kleene") {
    for (const auto& arg : call->arguments) {
      ExtractLeafPredicates(arg, manifest, physical_schema, parquet_schema, out);
    }
    return;
  }

  // is_null / is_valid take a single field_ref argument.
  auto maybe_op = ArrowFunctionToPageOp(call->function_name);
  if (!maybe_op) return;

  if (call->function_name == "is_null" || call->function_name == "is_valid") {
    if (call->arguments.size() != 1) return;
    const auto* ref = call->arguments[0].field_ref();
    if (!ref) return;
    auto match = ref->FindOneOrNone(physical_schema);
    if (!match.ok() || match.ValueUnsafe().empty()) return;
    const SchemaField* schema_field = &manifest.schema_fields[match.ValueUnsafe()[0]];
    if (!schema_field->is_leaf()) return;
    out->push_back({schema_field->column_index, *maybe_op, {}});
    return;
  }

  // Binary comparison: expects exactly two arguments.
  if (call->arguments.size() != 2) return;

  // Try both orderings: (field_ref, literal) and (literal, field_ref).
  const compute::Expression* field_expr = nullptr;
  const compute::Expression* literal_expr = nullptr;
  parquet::PredicateOp op = *maybe_op;

  if (call->arguments[0].field_ref() && call->arguments[1].literal()) {
    field_expr = &call->arguments[0];
    literal_expr = &call->arguments[1];
  } else if (call->arguments[1].field_ref() && call->arguments[0].literal()) {
    // Flip the field/literal order: invert the comparison direction.
    field_expr = &call->arguments[1];
    literal_expr = &call->arguments[0];
    // Invert: e.g. 5 > col  =>  col < 5
    if (op == parquet::PredicateOp::GT)
      op = parquet::PredicateOp::LT;
    else if (op == parquet::PredicateOp::LT)
      op = parquet::PredicateOp::GT;
    else if (op == parquet::PredicateOp::GTE)
      op = parquet::PredicateOp::LTE;
    else if (op == parquet::PredicateOp::LTE)
      op = parquet::PredicateOp::GTE;
    // EQ is symmetric, no inversion needed.
  } else {
    return;
  }

  const auto* ref = field_expr->field_ref();
  if (!ref) return;
  auto match = ref->FindOneOrNone(physical_schema);
  if (!match.ok() || match.ValueUnsafe().empty()) return;
  const SchemaField* schema_field = &manifest.schema_fields[match.ValueUnsafe()[0]];
  if (!schema_field->is_leaf()) return;

  const Datum* datum = literal_expr->literal();
  if (!datum || !datum->is_scalar()) return;

  // Look up the Parquet physical type for this column so that ScalarToAny can
  // produce a std::any value whose C++ type matches ComputePageSelection's
  // typeid() check.
  int col_idx = schema_field->column_index;
  if (col_idx < 0 || col_idx >= parquet_schema.num_columns()) return;
  const parquet::ColumnDescriptor* col_desc = parquet_schema.Column(col_idx);
  if (!col_desc) return;

  auto any_value_result = ScalarToAny(*datum->scalar(), col_desc);
  if (!any_value_result.ok()) {
    // The scalar type is incompatible with the Parquet physical type (e.g.
    // string vs INT32).  Skip this predicate rather than silently corrupting
    // page pruning.
    ARROW_LOG(WARNING) << "ScalarToAny failed for column " << col_idx << ": "
                       << any_value_result.status().message();
    return;
  }
  std::any val = std::move(any_value_result).ValueUnsafe();
  if (!val.has_value()) return;

  out->push_back({col_idx, op, std::move(val)});
}

}  // namespace

Status ParquetFileFragment::ComputePageSelections(const compute::Expression& predicate,
                                                  parquet::arrow::FileReader* reader) {
  DCHECK_NE(reader, nullptr);
  DCHECK_NE(metadata_, nullptr);

  // Re-use the lock to safely read manifest_ / row_groups_.
  auto lock = physical_schema_mutex_.Lock();

  // Configure the page-index policy on the underlying ParquetFileReader so that
  // ComputePageSelection() honours the caller's settings.
  parquet::ParquetFileReader* parquet_reader = reader->parquet_reader();
  parquet_reader->set_arrow_reader_properties(reader->properties());

  const parquet::PageIndexPolicy policy = reader->properties().page_index_policy();

  if (policy == parquet::PageIndexPolicy::NEVER) {
    // Page pruning disabled; leave page_selections_ empty.
    return Status::OK();
  }

  // Simplify the predicate against the partition expression before extracting leaves.
  auto simplified = predicate;
  auto simplify_result =
      SimplifyWithGuarantee(std::move(simplified), partition_expression_);
  if (!simplify_result.ok()) {
    // Non-fatal: fall back to no page pruning.
    return Status::OK();
  }
  simplified = std::move(simplify_result).ValueUnsafe();
  if (!simplified.IsSatisfiable()) {
    // Predicate is already unsatisfiable; no rows to select.
    return Status::OK();
  }

  // Extract individual leaf predicates from the (possibly compound) filter.
  // Pass the Parquet file schema so ScalarToAny() can match the physical type
  // of each column and produce a std::any with the correct C++ type.
  const parquet::SchemaDescriptor* parquet_schema = parquet_reader->metadata()->schema();
  DCHECK_NE(parquet_schema, nullptr);
  std::vector<LeafPredicate> leaf_predicates;
  ExtractLeafPredicates(simplified, *manifest_, *physical_schema_, *parquet_schema,
                        &leaf_predicates);

  if (leaf_predicates.empty()) {
    // No pushdown-able predicates found; leave page_selections_ empty.
    return Status::OK();
  }

  // For each leaf predicate, call ComputePageSelection across the surviving
  // row groups, then intersect the results per row group.
  //
  // page_selections_[rg] holds the intersection of all per-column selections.
  // A missing entry means "select all" for that row group.
  std::map<int, std::shared_ptr<parquet::RowSelection>> merged;

  for (const auto& leaf : leaf_predicates) {
    std::map<int, std::shared_ptr<parquet::RowSelection>> per_col;

    if (policy == parquet::PageIndexPolicy::ALWAYS) {
      // ALWAYS: propagate errors back to the caller.
      ARROW_ASSIGN_OR_RAISE(per_col, parquet_reader->ComputePageSelection(
                                         leaf.column_index, leaf.op, leaf.value,
                                         row_groups_ ? &(*row_groups_) : nullptr));
    } else {
      // AUTO: swallow errors (e.g. absent page index) and fall back.
      auto result =
          parquet_reader->ComputePageSelection(leaf.column_index, leaf.op, leaf.value,
                                               row_groups_ ? &(*row_groups_) : nullptr);
      if (!result.ok()) {
        ARROW_LOG(WARNING) << "ComputePageSelection failed for column "
                           << leaf.column_index << " (op="
                           << static_cast<int>(leaf.op)
                           << "); falling back to row-group pruning: "
                           << result.status().message();
        continue;
      }
      per_col = std::move(result).ValueUnsafe();
    }

    // Intersect per_col into merged (AND semantics).
    for (auto& [rg_idx, sel] : per_col) {
      if (!sel) continue;
      auto it = merged.find(rg_idx);
      if (it == merged.end()) {
        merged[rg_idx] = sel;
      } else {
        auto intersected = it->second->Intersect(*sel);
        if (!intersected.ok()) {
          // Incompatible row counts — skip this intersection.
          ARROW_LOG(WARNING) << "RowSelection::Intersect failed for row group " << rg_idx
                             << ": " << intersected.status().message();
          continue;
        }
        it->second =
            std::make_shared<parquet::RowSelection>(std::move(intersected).ValueUnsafe());
      }
    }
  }

  page_selections_ = std::move(merged);
  return Status::OK();
}

//
// ParquetFragmentScanOptions
//

ParquetFragmentScanOptions::ParquetFragmentScanOptions() {
  reader_properties = std::make_shared<parquet::ReaderProperties>();
  arrow_reader_properties =
      std::make_shared<parquet::ArrowReaderProperties>(/*use_threads=*/false);
}

//
// ParquetDatasetFactory
//

static inline Result<std::string> FileFromRowGroup(
    fs::FileSystem* filesystem, const std::string& base_path,
    const parquet::RowGroupMetaData& row_group, bool validate_column_chunk_paths) {
  constexpr auto prefix = "Extracting file path from RowGroup failed. ";

  if (row_group.num_columns() == 0) {
    return Status::Invalid(prefix,
                           "RowGroup must have a least one column to extract path.");
  }

  auto path = row_group.ColumnChunk(0)->file_path();
  if (path == "") {
    return Status::Invalid(
        prefix,
        "The column chunks' file paths should be set, but got an empty file path.");
  }

  if (validate_column_chunk_paths) {
    for (int i = 1; i < row_group.num_columns(); ++i) {
      const auto& column_path = row_group.ColumnChunk(i)->file_path();
      if (column_path != path) {
        return Status::Invalid(prefix, "Path '", column_path, "' not equal to path '",
                               path, ", for ColumnChunk at index ", i,
                               "; ColumnChunks in a RowGroup must have the same path.");
      }
    }
  }

  path = fs::internal::JoinAbstractPath(
      std::vector<std::string>{base_path, std::move(path)});
  // Normalizing path is required for Windows.
  return filesystem->NormalizePath(std::move(path));
}

namespace {

Result<std::shared_ptr<Schema>> GetSchema(
    const parquet::FileMetaData& metadata,
    const parquet::ArrowReaderProperties& properties) {
  std::shared_ptr<Schema> schema;
  RETURN_NOT_OK(parquet::arrow::FromParquetSchema(
      metadata.schema(), properties, metadata.key_value_metadata(), &schema));
  return schema;
}

}  // namespace

Result<std::shared_ptr<DatasetFactory>> ParquetDatasetFactory::Make(
    const std::string& metadata_path, std::shared_ptr<fs::FileSystem> filesystem,
    std::shared_ptr<ParquetFileFormat> format, ParquetFactoryOptions options) {
  // Paths in ColumnChunk are relative to the `_metadata` file. Thus, the base
  // directory of all parquet files is `dirname(metadata_path)`.
  auto dirname = arrow::fs::internal::GetAbstractPathParent(metadata_path).first;
  return Make({metadata_path, filesystem}, dirname, filesystem, std::move(format),
              std::move(options));
}

Result<std::shared_ptr<DatasetFactory>> ParquetDatasetFactory::Make(
    const FileSource& metadata_source, const std::string& base_path,
    std::shared_ptr<fs::FileSystem> filesystem, std::shared_ptr<ParquetFileFormat> format,
    ParquetFactoryOptions options) {
  DCHECK_NE(filesystem, nullptr);
  DCHECK_NE(format, nullptr);

  // By automatically setting the options base_dir to the metadata's base_path,
  // we provide a better experience for user providing Partitioning that are
  // relative to the base_dir instead of the full path.
  if (options.partition_base_dir.empty()) {
    options.partition_base_dir = base_path;
  }

  auto scan_options = std::make_shared<ScanOptions>();
  ARROW_ASSIGN_OR_RAISE(auto reader, format->GetReader(metadata_source, scan_options));
  std::shared_ptr<parquet::FileMetaData> metadata = reader->parquet_reader()->metadata();

  if (metadata->num_columns() == 0) {
    return Status::Invalid(
        "ParquetDatasetFactory must contain a schema with at least one column");
  }

  auto properties = MakeArrowReaderProperties(*format, *metadata);
  ARROW_ASSIGN_OR_RAISE(auto physical_schema, GetSchema(*metadata, properties));
  ARROW_ASSIGN_OR_RAISE(auto manifest, GetSchemaManifest(*metadata, properties));

  std::vector<std::pair<std::string, std::vector<int>>> paths_with_row_group_ids;
  std::unordered_map<std::string, int> paths_to_index;

  for (int i = 0; i < metadata->num_row_groups(); i++) {
    auto row_group = metadata->RowGroup(i);
    ARROW_ASSIGN_OR_RAISE(auto path,
                          FileFromRowGroup(filesystem.get(), base_path, *row_group,
                                           options.validate_column_chunk_paths));

    // Insert the path, or increase the count of row groups. It will be assumed that the
    // RowGroup of a file are ordered exactly as in the metadata file.
    auto inserted_index = paths_to_index.emplace(
        std::move(path), static_cast<int>(paths_with_row_group_ids.size()));
    if (inserted_index.second) {
      paths_with_row_group_ids.push_back({inserted_index.first->first, {}});
    }
    paths_with_row_group_ids[inserted_index.first->second].second.push_back(i);
  }

  return std::shared_ptr<DatasetFactory>(new ParquetDatasetFactory(
      std::move(filesystem), std::move(format), std::move(metadata), std::move(manifest),
      std::move(physical_schema), base_path, std::move(options),
      std::move(paths_with_row_group_ids)));
}

Result<std::vector<std::shared_ptr<FileFragment>>>
ParquetDatasetFactory::CollectParquetFragments(const Partitioning& partitioning) {
  std::vector<std::shared_ptr<FileFragment>> fragments(paths_with_row_group_ids_.size());

  size_t i = 0;
  for (const auto& e : paths_with_row_group_ids_) {
    const auto& path = e.first;
    auto metadata_subset = metadata_->Subset(e.second);

    auto row_groups = Iota(metadata_subset->num_row_groups());

    auto partition_expression =
        partitioning.Parse(StripPrefix(path, options_.partition_base_dir))
            .ValueOr(compute::literal(true));

    ARROW_ASSIGN_OR_RAISE(
        auto fragment,
        format_->MakeFragment({path, filesystem_}, std::move(partition_expression),
                              physical_schema_, std::move(row_groups)));

    RETURN_NOT_OK(fragment->SetMetadata(metadata_subset, manifest_,
                                        /*original_metadata=*/metadata_));
    fragments[i++] = std::move(fragment);
  }

  return fragments;
}

Result<std::vector<std::shared_ptr<Schema>>> ParquetDatasetFactory::InspectSchemas(
    InspectOptions options) {
  // The physical_schema from the _metadata file is always yielded
  std::vector<std::shared_ptr<Schema>> schemas = {physical_schema_};

  if (auto factory = options_.partitioning.factory()) {
    // Gather paths found in RowGroups' ColumnChunks.
    std::vector<std::string> stripped(paths_with_row_group_ids_.size());

    size_t i = 0;
    for (const auto& e : paths_with_row_group_ids_) {
      stripped[i++] = StripPrefixAndFilename(e.first, options_.partition_base_dir);
    }
    ARROW_ASSIGN_OR_RAISE(auto partition_schema, factory->Inspect(stripped));

    schemas.push_back(std::move(partition_schema));
  } else {
    schemas.push_back(options_.partitioning.partitioning()->schema());
  }

  return schemas;
}

Result<std::shared_ptr<Dataset>> ParquetDatasetFactory::Finish(FinishOptions options) {
  std::shared_ptr<Schema> schema = options.schema;
  bool schema_missing = schema == nullptr;
  if (schema_missing) {
    ARROW_ASSIGN_OR_RAISE(schema, Inspect(options.inspect_options));
  }

  std::shared_ptr<Partitioning> partitioning = options_.partitioning.partitioning();
  if (partitioning == nullptr) {
    auto factory = options_.partitioning.factory();
    ARROW_ASSIGN_OR_RAISE(partitioning, factory->Finish(schema));
  }

  ARROW_ASSIGN_OR_RAISE(auto fragments, CollectParquetFragments(*partitioning));
  return FileSystemDataset::Make(std::move(schema), compute::literal(true), format_,
                                 filesystem_, std::move(fragments),
                                 std::move(partitioning));
}

}  // namespace dataset
}  // namespace arrow
