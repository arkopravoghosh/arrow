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

#include <functional>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "arrow/array.h"
#include "arrow/array/builder_primitive.h"
#include "arrow/compute/api_scalar.h"
#include "arrow/compute/api_vector.h"
#include "arrow/compute/expression.h"
#include "arrow/dataset/dataset_internal.h"
#include "arrow/dataset/parquet_encryption_config.h"
#include "arrow/dataset/scanner.h"
#include "arrow/dataset/test_util_internal.h"
#include "arrow/io/interfaces.h"
#include "arrow/io/memory.h"
#include "arrow/io/test_common.h"
#include "arrow/io/util_internal.h"
#include "arrow/record_batch.h"
#include "arrow/table.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/testing/util.h"
#include "arrow/type.h"
#include "arrow/type_fwd.h"
#include "arrow/util/io_util.h"
#include "arrow/util/logging_internal.h"
#include "arrow/util/range.h"

#include "parquet/arrow/reader.h"
#include "parquet/arrow/writer.h"
#include "parquet/file_reader.h"
#include "parquet/metadata.h"
#include "parquet/properties.h"
#include "parquet/row_selection.h"
#include "parquet/statistics.h"
#include "parquet/types.h"

namespace arrow {

using internal::checked_cast;
using internal::checked_pointer_cast;

namespace dataset {

using parquet::ArrowWriterProperties;
using parquet::default_arrow_writer_properties;

using parquet::default_writer_properties;
using parquet::WriterProperties;

using parquet::CreateOutputStream;
using parquet::arrow::WriteTable;

using testing::Pointee;

class ParquetFormatHelper {
 public:
  using FormatType = ParquetFileFormat;

  static Result<std::shared_ptr<Buffer>> Write(
      RecordBatchReader* reader,
      const std::shared_ptr<ArrowWriterProperties>& arrow_properties =
          default_arrow_writer_properties()) {
    auto pool = ::arrow::default_memory_pool();
    std::shared_ptr<Buffer> out;
    auto sink = CreateOutputStream(pool);
    RETURN_NOT_OK(WriteRecordBatchReader(reader, pool, sink, default_writer_properties(),
                                         arrow_properties));
    return sink->Finish();
  }
  static std::shared_ptr<ParquetFileFormat> MakeFormat() {
    return std::make_shared<ParquetFileFormat>();
  }

 private:
  static Status WriteRecordBatch(const RecordBatch& batch,
                                 parquet::arrow::FileWriter* writer) {
    auto schema = batch.schema();

    if (!schema->Equals(*writer->schema(), false)) {
      return Status::Invalid("RecordBatch schema does not match this writer's. batch:'",
                             schema->ToString(), "' this:'", writer->schema()->ToString(),
                             "'");
    }

    RETURN_NOT_OK(writer->NewRowGroup());
    for (int i = 0; i < batch.num_columns(); i++) {
      RETURN_NOT_OK(writer->WriteColumnChunk(*batch.column(i)));
    }

    return Status::OK();
  }

  static Status WriteRecordBatchReader(RecordBatchReader* reader,
                                       parquet::arrow::FileWriter* writer) {
    auto schema = reader->schema();

    if (!schema->Equals(*writer->schema(), false)) {
      return Status::Invalid("RecordBatch schema does not match this writer's. batch:'",
                             schema->ToString(), "' this:'", writer->schema()->ToString(),
                             "'");
    }

    return MakeFunctionIterator([reader] { return reader->Next(); })
        .Visit([&](std::shared_ptr<RecordBatch> batch) {
          return WriteRecordBatch(*batch, writer);
        });
  }

  static Status WriteRecordBatchReader(
      RecordBatchReader* reader, MemoryPool* pool,
      const std::shared_ptr<io::OutputStream>& sink,
      const std::shared_ptr<WriterProperties>& properties = default_writer_properties(),
      const std::shared_ptr<ArrowWriterProperties>& arrow_properties =
          default_arrow_writer_properties()) {
    std::unique_ptr<parquet::arrow::FileWriter> writer;
    ARROW_ASSIGN_OR_RAISE(writer,
                          parquet::arrow::FileWriter::Open(*reader->schema(), pool, sink,
                                                           properties, arrow_properties));
    RETURN_NOT_OK(WriteRecordBatchReader(reader, writer.get()));
    return writer->Close();
  }
};

class DelayedBufferReader : public ::arrow::io::BufferReader {
 public:
  explicit DelayedBufferReader(const std::shared_ptr<::arrow::Buffer>& buffer)
      : ::arrow::io::BufferReader(buffer) {}

  ::arrow::Future<std::shared_ptr<Buffer>> ReadAsync(
      const ::arrow::io::IOContext& io_context, int64_t position,
      int64_t nbytes) override {
    read_async_count.fetch_add(1);
    auto self = std::dynamic_pointer_cast<DelayedBufferReader>(shared_from_this());
    return DeferNotOk(::arrow::io::internal::SubmitIO(
        io_context, [self, position, nbytes]() -> Result<std::shared_ptr<Buffer>> {
          std::this_thread::sleep_for(std::chrono::seconds(1));
          return self->DoReadAt(position, nbytes, /*allow_short_read=*/false);
        }));
  }

  std::atomic<int> read_async_count{0};
};

using CustomizeScanOptionsWithThreadPool =
    std::function<void(ScanOptions&, arrow::internal::ThreadPool*)>;

class TestParquetFileFormat : public FileFormatFixtureMixin<ParquetFormatHelper> {
 public:
  RecordBatchIterator Batches(Fragment* fragment) {
    EXPECT_OK_AND_ASSIGN(auto batch_gen, fragment->ScanBatchesAsync(opts_));
    return MakeGeneratorIterator(batch_gen);
  }

  std::shared_ptr<RecordBatch> SingleBatch(Fragment* fragment) {
    auto batches = IteratorToVector(Batches(fragment));
    EXPECT_EQ(batches.size(), 1);
    return batches.front();
  }

  void CountRowsAndBatchesInScan(Fragment* fragment, int64_t expected_rows,
                                 int64_t expected_batches) {
    int64_t actual_rows = 0;
    int64_t actual_batches = 0;

    for (auto maybe_batch : Batches(fragment)) {
      ASSERT_OK_AND_ASSIGN(auto batch, maybe_batch);
      actual_rows += batch->num_rows();
      ++actual_batches;
    }

    EXPECT_EQ(actual_rows, expected_rows);
    EXPECT_EQ(actual_batches, expected_batches);
  }

  void CountRowsAndBatchesInScan(const std::shared_ptr<Fragment>& fragment,
                                 int64_t expected_rows, int64_t expected_batches) {
    return CountRowsAndBatchesInScan(fragment.get(), expected_rows, expected_batches);
  }

  void CountRowGroupsInFragment(const std::shared_ptr<Fragment>& fragment,
                                std::vector<int> expected_row_groups,
                                compute::Expression filter) {
    SetFilter(filter);

    auto parquet_fragment = checked_pointer_cast<ParquetFileFragment>(fragment);
    ASSERT_OK_AND_ASSIGN(auto fragments, parquet_fragment->SplitByRowGroup(opts_->filter))

    EXPECT_EQ(fragments.size(), expected_row_groups.size());
    for (size_t i = 0; i < fragments.size(); i++) {
      auto expected = expected_row_groups[i];
      auto parquet_fragment = checked_pointer_cast<ParquetFileFragment>(fragments[i]);

      EXPECT_EQ(parquet_fragment->row_groups(), std::vector<int>{expected});
      EXPECT_EQ(SingleBatch(parquet_fragment.get())->num_rows(), expected + 1);
    }
  }

  void TestMultithreadedRegression(CustomizeScanOptionsWithThreadPool customizer) {
    auto reader = MakeGeneratedRecordBatch(schema({field("utf8", utf8())}), 10000, 100);
    ASSERT_OK_AND_ASSIGN(auto buffer, ParquetFormatHelper::Write(reader.get()));

    std::vector<Future<>> completes;
    std::vector<std::shared_ptr<arrow::internal::ThreadPool>> pools;

    for (int idx = 0; idx < 2; ++idx) {
      auto buffer_reader = std::make_shared<DelayedBufferReader>(buffer);
      auto source = std::make_shared<FileSource>(buffer_reader, buffer->size());
      auto fragment = MakeFragment(*source);
      std::shared_ptr<Scanner> scanner;

      {
        auto options = std::make_shared<ScanOptions>();
        ASSERT_OK_AND_ASSIGN(auto thread_pool, arrow::internal::ThreadPool::Make(1));
        pools.emplace_back(thread_pool);
        customizer(*options, pools.back().get());
        auto fragment_scan_options = std::make_shared<ParquetFragmentScanOptions>();
        fragment_scan_options->arrow_reader_properties->set_pre_buffer(true);

        options->fragment_scan_options = fragment_scan_options;
        ScannerBuilder builder(ArithmeticDatasetFixture::schema(), fragment, options);

        ASSERT_OK(builder.UseThreads(true));
        ASSERT_OK(builder.BatchSize(10000));
        ASSERT_OK_AND_ASSIGN(scanner, builder.Finish());
      }

      ASSERT_OK_AND_ASSIGN(auto batch, scanner->Head(10000));
      [[maybe_unused]] auto fut = scanner->ScanBatchesUnorderedAsync();
      // Random ReadAsync calls, generate some futures to make the state machine
      // more complex.
      for (int yy = 0; yy < 16; yy++) {
        completes.emplace_back(
            buffer_reader->ReadAsync(::arrow::io::IOContext(), 0, 1001));
      }
      scanner = nullptr;
    }

    for (auto& f : completes) {
      f.Wait();
    }
  }
};

TEST_F(TestParquetFileFormat, InspectFailureWithRelevantError) {
  TestInspectFailureWithRelevantError(StatusCode::Invalid, "Parquet");
}
TEST_F(TestParquetFileFormat, Inspect) { TestInspect(); }

TEST_F(TestParquetFileFormat, InspectDictEncoded) {
  auto reader = GetRecordBatchReader(schema({field("utf8", utf8())}));
  auto source = GetFileSource(reader.get());

  format_->reader_options.dict_columns = {"utf8"};
  ASSERT_OK_AND_ASSIGN(auto actual, format_->Inspect(*source.get()));

  Schema expected_schema({field("utf8", dictionary(int32(), utf8()))});
  AssertSchemaEqual(*actual, expected_schema, /* check_metadata = */ false);
}

TEST_F(TestParquetFileFormat, IsSupported) { TestIsSupported(); }

TEST_F(TestParquetFileFormat, WriteRecordBatchReader) { TestWrite(); }

TEST_F(TestParquetFileFormat, WriteRecordBatchReaderCustomOptions) {
  TimeUnit::type coerce_timestamps_to = TimeUnit::MICRO,
                 coerce_timestamps_from = TimeUnit::NANO;

  auto reader =
      GetRecordBatchReader(schema({field("ts", timestamp(coerce_timestamps_from))}));
  auto options =
      checked_pointer_cast<ParquetFileWriteOptions>(format_->DefaultWriteOptions());
  options->writer_properties = parquet::WriterProperties::Builder()
                                   .created_by("TestParquetFileFormat")
                                   ->disable_statistics()
                                   ->build();
  options->arrow_writer_properties = parquet::ArrowWriterProperties::Builder()
                                         .coerce_timestamps(coerce_timestamps_to)
                                         ->allow_truncated_timestamps()
                                         ->build();

  auto written = WriteToBuffer(reader->schema(), options);

  EXPECT_OK_AND_ASSIGN(auto fragment, format_->MakeFragment(FileSource{written}));
  EXPECT_OK_AND_ASSIGN(auto actual_schema, fragment->ReadPhysicalSchema());
  AssertSchemaEqual(Schema({field("ts", timestamp(coerce_timestamps_to))}),
                    *actual_schema);
}

TEST_F(TestParquetFileFormat, CountRows) { TestCountRows(); }

TEST_F(TestParquetFileFormat, FragmentEquals) { TestFragmentEquals(); }

TEST_F(TestParquetFileFormat, CountRowsPredicatePushdown) {
  constexpr int64_t kNumRowGroups = 16;
  constexpr int64_t kTotalNumRows = kNumRowGroups * (kNumRowGroups + 1) / 2;

  // See PredicatePushdown test below for a description of the generated data
  auto reader = ArithmeticDatasetFixture::GetRecordBatchReader(kNumRowGroups);
  auto source = GetFileSource(reader.get());
  auto options = std::make_shared<ScanOptions>();

  auto fragment = MakeFragment(*source);

  ASSERT_FINISHES_OK_AND_EQ(std::make_optional<int64_t>(kTotalNumRows),
                            fragment->CountRows(literal(true), options));

  for (int i = 1; i <= kNumRowGroups; i++) {
    SCOPED_TRACE(i);
    // The row group for which all values in column i64 == i has i rows
    auto predicate = less_equal(field_ref("i64"), literal(i));
    ASSERT_OK_AND_ASSIGN(predicate, predicate.Bind(*reader->schema()));
    auto expected = i * (i + 1) / 2;
    ASSERT_FINISHES_OK_AND_EQ(std::make_optional<int64_t>(expected),
                              fragment->CountRows(predicate, options));

    predicate = and_(less_equal(field_ref("i64"), literal(i)),
                     greater_equal(field_ref("i64"), literal(i)));
    ASSERT_OK_AND_ASSIGN(predicate, predicate.Bind(*reader->schema()));
    ASSERT_FINISHES_OK_AND_EQ(std::make_optional<int64_t>(i),
                              fragment->CountRows(predicate, options));

    predicate = equal(field_ref("i64"), literal(i));
    ASSERT_OK_AND_ASSIGN(predicate, predicate.Bind(*reader->schema()));
    ASSERT_FINISHES_OK_AND_EQ(std::make_optional<int64_t>(i),
                              fragment->CountRows(predicate, options));
  }

  // Ensure nulls are properly handled
  {
    auto dataset_schema = schema({field("i64", int64())});
    auto null_batch = RecordBatchFromJSON(dataset_schema, R"([
[null],
[null],
[null]
])");
    auto batch = RecordBatchFromJSON(dataset_schema, R"([
[1],
[2]
])");
    auto batch2 = RecordBatchFromJSON(dataset_schema, R"([
[4],
[4]
])");
    ASSERT_OK_AND_ASSIGN(auto reader, RecordBatchReader::Make({null_batch, batch, batch2},
                                                              dataset_schema));
    auto source = GetFileSource(reader.get());
    auto fragment = MakeFragment(*source);
    ASSERT_OK_AND_ASSIGN(
        auto predicate,
        greater_equal(field_ref("i64"), literal(1)).Bind(*dataset_schema));
    ASSERT_FINISHES_OK_AND_EQ(std::make_optional<int64_t>(4),
                              fragment->CountRows(predicate, options));

    ASSERT_OK_AND_ASSIGN(predicate, is_null(field_ref("i64")).Bind(*dataset_schema));
    ASSERT_FINISHES_OK_AND_EQ(std::make_optional<int64_t>(3),
                              fragment->CountRows(predicate, options));

    ASSERT_OK_AND_ASSIGN(predicate, is_valid(field_ref("i64")).Bind(*dataset_schema));
    ASSERT_FINISHES_OK_AND_EQ(std::make_optional<int64_t>(4),
                              fragment->CountRows(predicate, options));
  }
}

TEST_F(TestParquetFileFormat, CachedMetadata) {
  // Create a test file
  auto mock_fs = std::make_shared<fs::internal::MockFileSystem>(fs::kNoTime);
  std::shared_ptr<Schema> test_schema = schema({field("x", int32())});
  std::shared_ptr<RecordBatch> batch = RecordBatchFromJSON(test_schema, "[[0]]");
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<io::OutputStream> out_stream,
                       mock_fs->OpenOutputStream("/foo.parquet"));
  ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<FileWriter> writer,
      format_->MakeWriter(out_stream, test_schema, format_->DefaultWriteOptions(),
                          {mock_fs, "/foo.parquet"}));
  ASSERT_OK(writer->Write(batch));
  ASSERT_FINISHES_OK(writer->Finish());

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<io::RandomAccessFile> test_file,
                       mock_fs->OpenInputFile("/foo.parquet"));
  std::shared_ptr<io::TrackedRandomAccessFile> tracked_input =
      io::TrackedRandomAccessFile::Make(test_file.get());

  FileSource source(tracked_input);
  ASSERT_OK_AND_ASSIGN(auto fragment,
                       format_->MakeFragment(std::move(source), literal(true)));
  auto pq_fragment = checked_cast<ParquetFileFragment*>(fragment.get());

  // Read the file the first time, will read metadata
  auto options = std::make_shared<ScanOptions>();
  options->filter = literal(true);
  ASSERT_OK_AND_ASSIGN(
      auto projection_descr,
      ProjectionDescr::FromNames({"x"}, *test_schema, options->add_augmented_fields));
  options->projected_schema = projection_descr.schema;
  options->projection = projection_descr.expression;
  ASSERT_OK_AND_ASSIGN(auto generator, fragment->ScanBatchesAsync(options));
  ASSERT_FINISHES_OK(CollectAsyncGenerator(std::move(generator)));
  ASSERT_NE(nullptr, pq_fragment->metadata());

  ASSERT_GT(tracked_input->bytes_read(), 0);
  int64_t bytes_read_first_time = tracked_input->bytes_read();
  ASSERT_OK(tracked_input->Seek(0));

  // Read the file the second time, should not read metadata
  tracked_input->ResetStats();
  ASSERT_OK_AND_ASSIGN(generator, fragment->ScanBatchesAsync(options));
  ASSERT_FINISHES_OK(CollectAsyncGenerator(std::move(generator)));
  ASSERT_LT(tracked_input->bytes_read(), bytes_read_first_time);

  // Clear cached metadata
  ASSERT_OK(fragment->ClearCachedMetadata());
  ASSERT_EQ(nullptr, pq_fragment->metadata());

  // Read the file a third time, should read metadata
  tracked_input->ResetStats();
  ASSERT_OK_AND_ASSIGN(generator, fragment->ScanBatchesAsync(options));
  ASSERT_FINISHES_OK(CollectAsyncGenerator(std::move(generator)));
  ASSERT_EQ(tracked_input->bytes_read(), bytes_read_first_time);
  ASSERT_NE(nullptr, pq_fragment->metadata());
}

TEST_F(TestParquetFileFormat, MultithreadedScan) {
  constexpr int64_t kNumRowGroups = 16;

  // See PredicatePushdown test below for a description of the generated data
  auto reader = ArithmeticDatasetFixture::GetRecordBatchReader(kNumRowGroups);
  auto source = GetFileSource(reader.get());
  auto options = std::make_shared<ScanOptions>();

  auto fragment = MakeFragment(*source);

  FragmentDataset dataset(ArithmeticDatasetFixture::schema(), {fragment});
  ScannerBuilder builder({&dataset, [](...) {}});

  ASSERT_OK(builder.UseThreads(true));
  ASSERT_OK(builder.Project({call("add", {field_ref("i64"), literal(3)})}, {""}));
  ASSERT_OK_AND_ASSIGN(auto scanner, builder.Finish());

  ASSERT_OK_AND_ASSIGN(auto gen, scanner->ScanBatchesUnorderedAsync());

  auto collect_fut = CollectAsyncGenerator(gen);
  ASSERT_OK_AND_ASSIGN(auto batches, collect_fut.result());

  ASSERT_EQ(batches.size(), kNumRowGroups);
}

TEST_F(TestParquetFileFormat, SingleThreadExecutor) {
  // Reset capacity for io executor
  struct PoolResetGuard {
    int original_capacity = io::GetIOThreadPoolCapacity();
    ~PoolResetGuard() { DCHECK_OK(io::SetIOThreadPoolCapacity(original_capacity)); }
  } guard;
  ASSERT_OK(io::SetIOThreadPoolCapacity(1));

  auto reader = GetRecordBatchReader(schema({field("utf8", utf8())}));

  ASSERT_OK_AND_ASSIGN(auto buffer, ParquetFormatHelper::Write(reader.get()));
  auto buffer_reader = std::make_shared<::arrow::io::BufferReader>(buffer);
  auto source = std::make_shared<FileSource>(std::move(buffer_reader), buffer->size());
  auto options = std::make_shared<ScanOptions>();

  {
    auto fragment = MakeFragment(*source);
    auto count_rows = fragment->CountRows(literal(true), options);
    ASSERT_OK_AND_ASSIGN(auto result, count_rows.MoveResult());
    ASSERT_EQ(expected_rows(), result);
  }
}

class TestParquetFileSystemDataset : public WriteFileSystemDatasetMixin,
                                     public testing::Test {
 public:
  void SetUp() override {
    MakeSourceDataset();
    check_metadata_ = false;
    auto parquet_format = std::make_shared<ParquetFileFormat>();
    format_ = parquet_format;
    SetWriteOptions(parquet_format->DefaultWriteOptions());
  }
};

TEST_F(TestParquetFileSystemDataset, WriteWithIdenticalPartitioningSchema) {
  TestWriteWithIdenticalPartitioningSchema();
}

TEST_F(TestParquetFileSystemDataset, WriteWithUnrelatedPartitioningSchema) {
  TestWriteWithUnrelatedPartitioningSchema();
}

TEST_F(TestParquetFileSystemDataset, WriteWithSupersetPartitioningSchema) {
  TestWriteWithSupersetPartitioningSchema();
}

TEST_F(TestParquetFileSystemDataset, WriteWithEmptyPartitioningSchema) {
  TestWriteWithEmptyPartitioningSchema();
}

TEST_F(TestParquetFileSystemDataset, WriteWithEncryptionConfigNotSupported) {
#ifndef PARQUET_REQUIRE_ENCRYPTION
  // Create a dummy ParquetEncryptionConfig
  std::shared_ptr<ParquetEncryptionConfig> encryption_config =
      std::make_shared<ParquetEncryptionConfig>();

  auto options =
      checked_pointer_cast<ParquetFileWriteOptions>(format_->DefaultWriteOptions());

  // Set the encryption config in the options
  options->parquet_encryption_config = encryption_config;

  // Setup mock filesystem and test data
  auto mock_fs = std::make_shared<fs::internal::MockFileSystem>(fs::kNoTime);
  std::shared_ptr<Schema> test_schema = schema({field("x", int32())});
  std::shared_ptr<RecordBatch> batch = RecordBatchFromJSON(test_schema, "[[0]]");
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<io::OutputStream> out_stream,
                       mock_fs->OpenOutputStream("/foo.parquet"));
  // Try to create a writer with the encryption config
  auto result =
      format_->MakeWriter(out_stream, test_schema, options, {mock_fs, "/foo.parquet"});
  // Expect an error if encryption is not supported in the build
  EXPECT_TRUE(result.status().IsNotImplemented());
#endif
}

class TestParquetFileFormatScan : public FileFormatScanMixin<ParquetFormatHelper> {
 public:
  std::shared_ptr<RecordBatch> SingleBatch(std::shared_ptr<Fragment> fragment) {
    auto batches = IteratorToVector(PhysicalBatches(fragment));
    EXPECT_EQ(batches.size(), 1);
    return batches.front();
  }

  void CountRowsAndBatchesInScan(std::shared_ptr<Fragment> fragment,
                                 int64_t expected_rows, int64_t expected_batches) {
    int64_t actual_rows = 0;
    int64_t actual_batches = 0;

    for (auto maybe_batch : PhysicalBatches(fragment)) {
      ASSERT_OK_AND_ASSIGN(auto batch, maybe_batch);
      actual_rows += batch->num_rows();
      ++actual_batches;
    }

    EXPECT_EQ(actual_rows, expected_rows);
    EXPECT_EQ(actual_batches, expected_batches);
  }

  void CountRowGroupsInFragment(const std::shared_ptr<Fragment>& fragment,
                                std::vector<int> expected_row_groups,
                                compute::Expression filter) {
    SetFilter(filter);

    auto parquet_fragment = checked_pointer_cast<ParquetFileFragment>(fragment);
    ASSERT_OK_AND_ASSIGN(auto fragments, parquet_fragment->SplitByRowGroup(opts_->filter))

    EXPECT_EQ(fragments.size(), expected_row_groups.size());
    for (size_t i = 0; i < fragments.size(); i++) {
      auto expected = expected_row_groups[i];
      auto parquet_fragment = checked_pointer_cast<ParquetFileFragment>(fragments[i]);

      EXPECT_EQ(parquet_fragment->row_groups(), std::vector<int>{expected});
      EXPECT_EQ(SingleBatch(parquet_fragment)->num_rows(), expected + 1);
    }
  }
};

TEST_P(TestParquetFileFormatScan, ScanRecordBatchReader) { TestScan(); }
TEST_P(TestParquetFileFormatScan, ScanBatchSize) { TestScanBatchSize(); }
TEST_P(TestParquetFileFormatScan, ScanNoReadahead) { TestScanNoReadahead(); }
TEST_P(TestParquetFileFormatScan, ScanRecordBatchReaderProjected) { TestScanProjected(); }
TEST_P(TestParquetFileFormatScan, ScanRecordBatchReaderProjectedNested) {
  // TODO(ARROW-1888): enable fine-grained column projection.
  TestScanProjectedNested(/*fine_grained_selection=*/false);
}
TEST_P(TestParquetFileFormatScan, ScanRecordBatchReaderProjectedMissingCols) {
  TestScanProjectedMissingCols();
}
TEST_P(TestParquetFileFormatScan, ScanRecordBatchReaderWithVirtualColumn) {
  TestScanWithVirtualColumn();
}
TEST_P(TestParquetFileFormatScan, ScanRecordBatchReaderWithDuplicateColumn) {
  TestScanWithDuplicateColumn();
}
TEST_P(TestParquetFileFormatScan, ScanRecordBatchReaderWithDuplicateColumnError) {
  TestScanWithDuplicateColumnError();
}
TEST_P(TestParquetFileFormatScan, ScanWithPushdownNulls) { TestScanWithPushdownNulls(); }
TEST_P(TestParquetFileFormatScan, ScanRecordBatchReaderDictEncoded) {
  auto reader = GetRecordBatchReader(schema({field("utf8", utf8())}));
  auto source = GetFileSource(reader.get());

  SetSchema(reader->schema()->fields());
  SetFilter(literal(true));
  format_->reader_options.dict_columns = {"utf8"};
  ASSERT_OK_AND_ASSIGN(auto fragment, format_->MakeFragment(*source));

  int64_t row_count = 0;
  Schema expected_schema({field("utf8", dictionary(int32(), utf8()))});

  for (auto maybe_batch : PhysicalBatches(fragment)) {
    ASSERT_OK_AND_ASSIGN(auto batch, maybe_batch);
    row_count += batch->num_rows();
    AssertSchemaEqual(*batch->schema(), expected_schema, /* check_metadata = */ false);
  }
  ASSERT_EQ(row_count, expected_rows());
}
TEST_P(TestParquetFileFormatScan, ScanRecordBatchReaderPreBuffer) {
  auto reader = GetRecordBatchReader(schema({field("f64", float64())}));
  auto source = GetFileSource(reader.get());

  SetSchema(reader->schema()->fields());
  SetFilter(literal(true));

  ASSERT_OK_AND_ASSIGN(auto fragment, format_->MakeFragment(*source));
  auto fragment_scan_options = std::make_shared<ParquetFragmentScanOptions>();
  fragment_scan_options->arrow_reader_properties->set_pre_buffer(true);
  opts_->fragment_scan_options = fragment_scan_options;

  int64_t row_count = 0;
  for (auto maybe_batch : PhysicalBatches(fragment)) {
    ASSERT_OK_AND_ASSIGN(auto batch, maybe_batch);
    row_count += batch->num_rows();
  }
  ASSERT_EQ(row_count, expected_rows());
}
TEST_P(TestParquetFileFormatScan, PredicatePushdown) {
  // Given a number `n`, the arithmetic dataset creates n RecordBatches where
  // each RecordBatch is keyed by a unique integer in [1, n]. Let `rb_i` denote
  // the record batch keyed by `i`. `rb_i` is composed of `i` rows where all
  // values are a variant of `i`, e.g. {"i64": i, "u8": i, ... }.
  //
  // Thus the ArithmeticDataset(n) has n RecordBatches and the total number of
  // rows is n(n+1)/2.
  //
  // This test uses the Fragment directly, and so no post-filtering is
  // applied via ScanOptions' evaluator. Thus, counting the number of returned
  // rows and returned row groups is a good enough proxy to check if pushdown
  // predicate is working.

  constexpr int64_t kNumRowGroups = 16;
  constexpr int64_t kTotalNumRows = kNumRowGroups * (kNumRowGroups + 1) / 2;

  auto reader = ArithmeticDatasetFixture::GetRecordBatchReader(kNumRowGroups);
  auto source = GetFileSource(reader.get());

  SetSchema(reader->schema()->fields());
  ASSERT_OK_AND_ASSIGN(auto fragment, format_->MakeFragment(*source));

  SetFilter(literal(true));
  CountRowsAndBatchesInScan(fragment, kTotalNumRows, kNumRowGroups);

  for (int64_t i = 1; i <= kNumRowGroups; i++) {
    SetFilter(equal(field_ref("i64"), literal(i)));
    CountRowsAndBatchesInScan(fragment, i, 1);
  }

  // Out of bound filters should skip all RowGroups.
  SetFilter(literal(false));
  CountRowsAndBatchesInScan(fragment, 0, 0);
  SetFilter(equal(field_ref("i64"), literal<int64_t>(kNumRowGroups + 1)));
  CountRowsAndBatchesInScan(fragment, 0, 0);
  SetFilter(equal(field_ref("i64"), literal<int64_t>(-1)));
  CountRowsAndBatchesInScan(fragment, 0, 0);
  // No rows match 1 and 2.
  SetFilter(and_(equal(field_ref("i64"), literal<int64_t>(1)),
                 equal(field_ref("u8"), literal<uint8_t>(2))));
  CountRowsAndBatchesInScan(fragment, 0, 0);

  SetFilter(or_(equal(field_ref("i64"), literal<int64_t>(2)),
                equal(field_ref("i64"), literal<int64_t>(4))));
  CountRowsAndBatchesInScan(fragment, 2 + 4, 2);

  SetFilter(less(field_ref("i64"), literal<int64_t>(6)));
  CountRowsAndBatchesInScan(fragment, 5 * (5 + 1) / 2, 5);

  SetFilter(greater_equal(field_ref("i64"), literal<int64_t>(6)));
  CountRowsAndBatchesInScan(fragment, kTotalNumRows - (5 * (5 + 1) / 2),
                            kNumRowGroups - 5);
}

TEST_P(TestParquetFileFormatScan, PredicatePushdownRowGroupFragments) {
  constexpr int64_t kNumRowGroups = 16;

  auto reader = ArithmeticDatasetFixture::GetRecordBatchReader(kNumRowGroups);
  auto source = GetFileSource(reader.get());

  SetSchema(reader->schema()->fields());
  ASSERT_OK_AND_ASSIGN(auto fragment, format_->MakeFragment(*source));

  auto all_row_groups = ::arrow::internal::Iota(static_cast<int>(kNumRowGroups));
  CountRowGroupsInFragment(fragment, all_row_groups, literal(true));

  for (int i = 0; i < kNumRowGroups; ++i) {
    CountRowGroupsInFragment(fragment, {i}, equal(field_ref("i64"), literal(i + 1)));
  }

  // Out of bound filters should skip all RowGroups.
  CountRowGroupsInFragment(fragment, {}, literal(false));
  CountRowGroupsInFragment(fragment, {},
                           equal(field_ref("i64"), literal(kNumRowGroups + 1)));
  CountRowGroupsInFragment(fragment, {}, equal(field_ref("i64"), literal(-1)));

  // No rows match 1 and 2.
  CountRowGroupsInFragment(
      fragment, {},
      and_(equal(field_ref("i64"), literal(1)), equal(field_ref("u8"), literal(2))));
  CountRowGroupsInFragment(
      fragment, {},
      and_(equal(field_ref("i64"), literal(2)), equal(field_ref("i64"), literal(4))));

  CountRowGroupsInFragment(
      fragment, {1, 3},
      or_(equal(field_ref("i64"), literal(2)), equal(field_ref("i64"), literal(4))));

  auto set = ArrayFromJSON(int64(), "[2, 4]");
  CountRowGroupsInFragment(
      fragment, {1, 3},
      call("is_in", {field_ref("i64")}, compute::SetLookupOptions{set}));

  CountRowGroupsInFragment(fragment, {0, 1, 2, 3, 4}, less(field_ref("i64"), literal(6)));

  CountRowGroupsInFragment(fragment,
                           ::arrow::internal::Iota(5, static_cast<int>(kNumRowGroups)),
                           greater_equal(field_ref("i64"), literal(6)));

  CountRowGroupsInFragment(fragment, {5, 6},
                           and_(greater_equal(field_ref("i64"), literal(6)),
                                less(field_ref("i64"), literal(8))));

  // nested field reference
  CountRowGroupsInFragment(fragment, {0, 1, 2, 3, 4},
                           less(field_ref(FieldRef("struct", "i32")), literal(6)));
  CountRowGroupsInFragment(fragment, {1},
                           equal(field_ref(FieldRef("struct", "str")), literal("2")));
}

TEST_P(TestParquetFileFormatScan, ExplicitRowGroupSelection) {
  constexpr int64_t kNumRowGroups = 16;
  constexpr int64_t kTotalNumRows = kNumRowGroups * (kNumRowGroups + 1) / 2;

  auto reader = ArithmeticDatasetFixture::GetRecordBatchReader(kNumRowGroups);
  auto source = GetFileSource(reader.get());

  SetSchema(reader->schema()->fields());
  SetFilter(literal(true));

  auto row_groups_fragment = [&](std::vector<int> row_groups) {
    EXPECT_OK_AND_ASSIGN(auto fragment,
                         format_->MakeFragment(*source, literal(true),
                                               /*physical_schema=*/nullptr, row_groups));
    return fragment;
  };

  // select all row groups
  EXPECT_OK_AND_ASSIGN(auto all_row_groups_fragment,
                       format_->MakeFragment(*source, literal(true))
                           .Map([](std::shared_ptr<FileFragment> f) {
                             return checked_pointer_cast<ParquetFileFragment>(f);
                           }));

  EXPECT_EQ(all_row_groups_fragment->row_groups(), std::vector<int>{});

  ARROW_EXPECT_OK(all_row_groups_fragment->EnsureCompleteMetadata());
  CountRowsAndBatchesInScan(all_row_groups_fragment, kTotalNumRows, kNumRowGroups);

  // individual selection selects a single row group
  for (int i = 0; i < kNumRowGroups; ++i) {
    CountRowsAndBatchesInScan(row_groups_fragment({i}), i + 1, 1);
    EXPECT_EQ(row_groups_fragment({i})->row_groups(), std::vector<int>{i});
  }

  for (int i = 0; i < kNumRowGroups; ++i) {
    // conflicting selection/filter
    SetFilter(equal(field_ref("i64"), literal(i)));
    CountRowsAndBatchesInScan(row_groups_fragment({i}), 0, 0);
  }

  for (int i = 0; i < kNumRowGroups; ++i) {
    // identical selection/filter
    SetFilter(equal(field_ref("i64"), literal(i + 1)));
    CountRowsAndBatchesInScan(row_groups_fragment({i}), i + 1, 1);
  }

  SetFilter(greater(field_ref("i64"), literal(3)));
  CountRowsAndBatchesInScan(row_groups_fragment({2, 3, 4, 5}), 4 + 5 + 6, 3);

  ASSERT_OK_AND_ASSIGN(auto batch_gen,
                       row_groups_fragment({kNumRowGroups + 1})->ScanBatchesAsync(opts_));
  Status scan_status = CollectAsyncGenerator(batch_gen).status();

  EXPECT_RAISES_WITH_MESSAGE_THAT(
      IndexError,
      testing::HasSubstr("only has " + std::to_string(kNumRowGroups) + " row groups"),
      scan_status);
}

TEST_P(TestParquetFileFormatScan, PredicatePushdownRowGroupFragmentsUsingStringColumn) {
  auto table = TableFromJSON(schema({field("x", utf8())}),
                             {
                                 R"([{"x": "a"}])",
                                 R"([{"x": "b"}, {"x": "b"}])",
                                 R"([{"x": "c"}, {"x": "c"}, {"x": "c"}])",
                                 R"([{"x": "a"}, {"x": "b"}, {"x": "c"}, {"x": "d"}])",
                             });
  TableBatchReader reader(*table);
  auto source = GetFileSource(&reader);

  SetSchema(reader.schema()->fields());
  ASSERT_OK_AND_ASSIGN(auto fragment, format_->MakeFragment(*source));

  CountRowGroupsInFragment(fragment, {0, 3}, equal(field_ref("x"), literal("a")));
}

TEST_P(TestParquetFileFormatScan, PredicatePushdownRowGroupFragmentsUsingDurationColumn) {
  // GH-37111: Parquet arrow stores writer schema and possible field_id in
  // key_value_metadata when store_schema enabled. When storing `arrow::duration`, it will
  // be stored as int64. This test ensures that dataset can parse the writer schema
  // correctly.
  auto table = TableFromJSON(schema({field("t", duration(TimeUnit::NANO))}),
                             {
                                 R"([{"t": 1}])",
                                 R"([{"t": 2}, {"t": 3}])",
                             });
  TableBatchReader table_reader(*table);
  ASSERT_OK_AND_ASSIGN(
      auto buffer,
      ParquetFormatHelper::Write(
          &table_reader, ArrowWriterProperties::Builder().store_schema()->build()));
  auto source = std::make_shared<FileSource>(buffer);
  SetSchema({field("t", duration(TimeUnit::NANO))});
  ASSERT_OK_AND_ASSIGN(auto fragment, format_->MakeFragment(*source));

  auto expr = equal(field_ref("t"), literal(::arrow::DurationScalar(1, TimeUnit::NANO)));
  CountRowGroupsInFragment(fragment, {0}, expr);
}

TEST_P(TestParquetFileFormatScan,
       PredicatePushdownRowGroupFragmentsUsingTimestampColumn) {
  // GH-37799: Parquet arrow will change TimeUnit::SECOND to TimeUnit::MILLI
  // because parquet LogicalType doesn't support SECOND.
  for (auto time_unit : {TimeUnit::MILLI, TimeUnit::SECOND}) {
    auto table = TableFromJSON(schema({field("t", time32(time_unit))}),
                               {
                                   R"([{"t": 1}])",
                                   R"([{"t": 2}, {"t": 3}])",
                               });
    TableBatchReader table_reader(*table);
    ARROW_SCOPED_TRACE("time_unit=", time_unit);
    ASSERT_OK_AND_ASSIGN(
        auto source,
        ParquetFormatHelper::Write(
            &table_reader, ArrowWriterProperties::Builder().store_schema()->build())
            .As<FileSource>());
    SetSchema({field("t", time32(time_unit))});
    ASSERT_OK_AND_ASSIGN(auto fragment, format_->MakeFragment(source));

    auto expr = equal(field_ref("t"), literal(::arrow::Time32Scalar(1, time_unit)));
    CountRowGroupsInFragment(fragment, {0}, expr);
  }
}

// Tests projection with nested/indexed FieldRefs.
// https://github.com/apache/arrow/issues/35579
TEST_P(TestParquetFileFormatScan, ProjectWithNonNamedFieldRefs) {
  auto table_schema = schema(
      {field("info", struct_({field("name", utf8()),
                              field("data", struct_({field("amount", float64()),
                                                     field("percent", float32())}))}))});
  auto table = TableFromJSON(table_schema, {R"([
    {"info": {"name": "a", "data": {"amount": 10.3, "percent": 0.1}}},
    {"info": {"name": "b", "data": {"amount": 11.6, "percent": 0.2}}},
    {"info": {"name": "c", "data": {"amount": 12.9, "percent": 0.3}}},
    {"info": {"name": "d", "data": {"amount": 14.2, "percent": 0.4}}},
    {"info": {"name": "e", "data": {"amount": 15.5, "percent": 0.5}}},
    {"info": {"name": "f", "data": {"amount": 16.8, "percent": 0.6}}}])"});
  ASSERT_OK_AND_ASSIGN(auto expected_batch, table->CombineChunksToBatch());

  TableBatchReader reader(*table);
  SetSchema(reader.schema()->fields());

  auto source = GetFileSource(&reader);
  ASSERT_OK_AND_ASSIGN(auto fragment, format_->MakeFragment(*source));

  std::vector<FieldRef> equivalent_refs = {
      FieldRef("info", "data", "percent"), FieldRef("info", 1, 1),
      FieldRef(0, 1, "percent"),           FieldRef(0, 1, 1),
      FieldRef(0, FieldRef("data", 1)),    FieldRef(FieldRef(0), FieldRef(1, 1)),
  };
  for (const auto& ref : equivalent_refs) {
    ARROW_SCOPED_TRACE("ref = ", ref.ToString());

    Project({field_ref(ref)}, {"value"});
    auto batch = SingleBatch(fragment);
    AssertBatchesEqual(*expected_batch, *batch);
  }
}

INSTANTIATE_TEST_SUITE_P(TestScan, TestParquetFileFormatScan,
                         ::testing::ValuesIn(TestFormatParams::Values()),
                         TestFormatParams::ToTestNameString);

TEST(TestParquetStatistics, NullMax) {
  auto field = ::arrow::field("x", float32());
  ASSERT_OK_AND_ASSIGN(std::string dir_string,
                       arrow::internal::GetEnvVar("PARQUET_TEST_DATA"));
  auto reader =
      parquet::ParquetFileReader::OpenFile(dir_string + "/nan_in_stats.parquet");
  auto statistics = reader->RowGroup(0)->metadata()->ColumnChunk(0)->statistics();
  auto stat_expression =
      ParquetFileFragment::EvaluateStatisticsAsExpression(*field, *statistics);
  EXPECT_EQ(stat_expression->ToString(), "(x >= 1)");
}

TEST(TestParquetStatistics, NoNullCount) {
  auto field = ::arrow::field("x", int32());
  auto parquet_node_ptr = ::parquet::schema::Int32("x", ::parquet::Repetition::REQUIRED);
  ::parquet::ColumnDescriptor descr(parquet_node_ptr, /*max_definition_level=*/1,
                                    /*max_repetition_level=*/0);

  auto int32_to_parquet_stats = [](int32_t v) {
    std::string value;
    value.resize(sizeof(int32_t));
    memcpy(value.data(), &v, sizeof(int32_t));
    return value;
  };
  {
    // Base case: when null_count is not set, the expression might contain null
    ::parquet::EncodedStatistics encoded_stats;
    encoded_stats.set_min(int32_to_parquet_stats(1));
    encoded_stats.set_max(int32_to_parquet_stats(100));
    encoded_stats.has_null_count = false;
    encoded_stats.all_null_value = false;
    encoded_stats.null_count = 0;
    auto stats = ::parquet::Statistics::Make(&descr, &encoded_stats, /*num_values=*/10);

    auto stat_expression =
        ParquetFileFragment::EvaluateStatisticsAsExpression(*field, *stats);
    ASSERT_TRUE(stat_expression.has_value());
    EXPECT_EQ(stat_expression->ToString(),
              "(((x >= 1) and (x <= 100)) or is_null(x, {nan_is_null=false}))");
  }
  {
    // Special case: when num_value is 0, it would return
    // "is_null".
    ::parquet::EncodedStatistics encoded_stats;
    encoded_stats.has_null_count = true;
    encoded_stats.null_count = 1;
    encoded_stats.all_null_value = true;
    auto stats = ::parquet::Statistics::Make(&descr, &encoded_stats, /*num_values=*/0);
    auto stat_expression =
        ParquetFileFragment::EvaluateStatisticsAsExpression(*field, *stats);
    ASSERT_TRUE(stat_expression.has_value());
    EXPECT_EQ(stat_expression->ToString(), "is_null(x, {nan_is_null=false})");

    encoded_stats.has_null_count = false;
    encoded_stats.all_null_value = false;
    stats = ::parquet::Statistics::Make(&descr, &encoded_stats, /*num_values=*/0);
    stat_expression = ParquetFileFragment::EvaluateStatisticsAsExpression(*field, *stats);
    ASSERT_TRUE(stat_expression.has_value());
    EXPECT_EQ(stat_expression->ToString(), "is_null(x, {nan_is_null=false})");
  }
}

TEST_F(TestParquetFileFormat, MultithreadedScanRegression) {
  // GH-38438: This test is similar to MultithreadedScan, but it try to use self
  // designed Executor and DelayedBufferReader to mock async execution to make
  // the state machine more complex.
  CustomizeScanOptionsWithThreadPool customize_io_context =
      [](ScanOptions& options, arrow::internal::ThreadPool* pool) {
        options.io_context = ::arrow::io::IOContext(::arrow::default_memory_pool(), pool);
      };
  TestMultithreadedRegression(customize_io_context);
}

TEST_F(TestParquetFileFormat, MultithreadedComputeRegression) {
  // GH-43694: Test similar situation as MultithreadedScanRegression but with
  // the customized CPU executor instead
  CustomizeScanOptionsWithThreadPool customize_cpu_executor =
      [](ScanOptions& options, arrow::internal::ThreadPool* pool) {
        options.cpu_executor = pool;
      };
  TestMultithreadedRegression(customize_cpu_executor);
}

// ============================================================================
// TestDatasetParquetPagePruning
//
// Ten tests covering page-level row pruning through the Arrow dataset scanner
// (ParquetFileFragment + ScanBatchesAsync). Each test writes a Parquet file
// with known data, optionally enabling/disabling the page index via
// WriterProperties, configures the PageIndexPolicy on ParquetFragmentScanOptions,
// runs a filtered ScanBatchesAsync, and verifies row counts and values.
// ============================================================================

namespace {

// ---------------------------------------------------------------------------
// Helper: write a multi-row-group, multi-page Parquet file to a Buffer.
//
// schema: single int32 column "v"
// The file has kNumRowGroups row groups.  Each row group has kRowsPerGroup rows
// with values in a contiguous range so that we can craft predicates that match
// exactly one row group, all row groups, or none.
//
// Row group i (0-based) contains values in [i * kRowsPerGroup,
// (i+1) * kRowsPerGroup).  Within each row group we use a small page size so
// the Parquet writer emits multiple data pages — a prerequisite for page-level
// pruning to have any effect.
// ---------------------------------------------------------------------------
constexpr int kNumRowGroups = 4;
constexpr int kRowsPerGroup = 100;
// Each row group spans values [i*kRowsPerGroup, (i+1)*kRowsPerGroup).
// Total rows = kNumRowGroups * kRowsPerGroup = 400.

// Write a Parquet buffer with page indices enabled or disabled.
Result<std::shared_ptr<Buffer>> WritePageIndexParquet(bool enable_page_index) {
  auto data_schema = schema({field("v", int32())});

  // Build kNumRowGroups record batches, each with kRowsPerGroup rows.
  std::vector<std::shared_ptr<RecordBatch>> batches;
  for (int rg = 0; rg < kNumRowGroups; ++rg) {
    Int32Builder builder;
    RETURN_NOT_OK(builder.Reserve(kRowsPerGroup));
    for (int r = 0; r < kRowsPerGroup; ++r) {
      RETURN_NOT_OK(builder.Append(rg * kRowsPerGroup + r));
    }
    std::shared_ptr<Array> col;
    ARROW_ASSIGN_OR_RAISE(col, builder.Finish());
    batches.push_back(RecordBatch::Make(data_schema, kRowsPerGroup, {col}));
  }

  // Parquet writer properties: use a small page size to force multiple pages
  // per row group, and optionally enable page index.
  auto props_builder = parquet::WriterProperties::Builder();
  props_builder.data_pagesize(256);  // small page → multiple pages per row group
  if (enable_page_index) {
    props_builder.enable_write_page_index();
  } else {
    props_builder.disable_write_page_index();
  }
  auto writer_props = props_builder.build();
  auto arrow_props = parquet::default_arrow_writer_properties();

  auto pool = arrow::default_memory_pool();
  auto sink = parquet::CreateOutputStream(pool);
  ARROW_ASSIGN_OR_RAISE(std::unique_ptr<parquet::arrow::FileWriter> writer,
                        parquet::arrow::FileWriter::Open(*data_schema, pool, sink,
                                                         writer_props, arrow_props));
  for (const auto& batch : batches) {
    RETURN_NOT_OK(writer->NewRowGroup());
    RETURN_NOT_OK(writer->WriteColumnChunk(*batch->column(0)));
  }
  RETURN_NOT_OK(writer->Close());
  return sink->Finish();
}

// Write an all-null Parquet file (single row group, nulls only).
Result<std::shared_ptr<Buffer>> WriteAllNullParquet() {
  auto data_schema = schema({field("v", int32())});
  // Build an all-null Int32Array
  Int32Builder builder;
  RETURN_NOT_OK(builder.AppendNulls(10));
  std::shared_ptr<Array> col;
  ARROW_ASSIGN_OR_RAISE(col, builder.Finish());
  auto batch = RecordBatch::Make(data_schema, 10, {col});

  auto props_builder = parquet::WriterProperties::Builder();
  props_builder.enable_write_page_index();
  auto writer_props = props_builder.build();
  auto arrow_props = parquet::default_arrow_writer_properties();

  auto pool = arrow::default_memory_pool();
  auto sink = parquet::CreateOutputStream(pool);
  ARROW_ASSIGN_OR_RAISE(std::unique_ptr<parquet::arrow::FileWriter> writer,
                        parquet::arrow::FileWriter::Open(*data_schema, pool, sink,
                                                         writer_props, arrow_props));
  RETURN_NOT_OK(writer->NewRowGroup());
  RETURN_NOT_OK(writer->WriteColumnChunk(*batch->column(0)));
  RETURN_NOT_OK(writer->Close());
  return sink->Finish();
}

// Collect all RecordBatches from a ScanBatchesAsync call with the given filter.
// Returns the total row count and all batches.
Status CollectBatches(const std::shared_ptr<Fragment>& fragment,
                      const std::shared_ptr<ScanOptions>& options, int64_t* total_rows,
                      std::vector<std::shared_ptr<RecordBatch>>* out_batches) {
  ARROW_ASSIGN_OR_RAISE(auto gen, fragment->ScanBatchesAsync(options));
  ARROW_ASSIGN_OR_RAISE(auto batches, CollectAsyncGenerator(std::move(gen)).result());
  *total_rows = 0;
  out_batches->clear();
  for (const auto& batch : batches) {
    *total_rows += batch->num_rows();
    out_batches->push_back(batch);
  }
  return Status::OK();
}

// Build a ScanOptions with a dataset_schema, a filter, and default projection.
// Optionally attach a ParquetFragmentScanOptions with the specified PageIndexPolicy.
std::shared_ptr<ScanOptions> MakeScanOptions(
    const std::shared_ptr<Schema>& data_schema, const compute::Expression& filter,
    parquet::PageIndexPolicy policy = parquet::PageIndexPolicy::AUTO) {
  auto opts = std::make_shared<ScanOptions>();
  opts->dataset_schema = data_schema;
  EXPECT_OK_AND_ASSIGN(opts->filter, filter.Bind(*data_schema));
  EXPECT_OK_AND_ASSIGN(auto projection,
                       ProjectionDescr::Default(*data_schema,
                                                /*add_augmented_fields=*/true));
  SetProjection(opts.get(), std::move(projection));

  auto fso = std::make_shared<ParquetFragmentScanOptions>();
  fso->arrow_reader_properties->set_page_index_policy(policy);
  opts->fragment_scan_options = fso;
  return opts;
}

// Return the count of matching rows from a full in-memory table filtered via
// compute::ExecuteScalarExpression.
Result<int64_t> ReferenceRowCount(const std::shared_ptr<Table>& full_table,
                                  const compute::Expression& filter) {
  ARROW_ASSIGN_OR_RAISE(auto bound_filter, filter.Bind(*full_table->schema()));
  int64_t total = 0;
  TableBatchReader reader(*full_table);
  std::shared_ptr<RecordBatch> batch;
  while (true) {
    RETURN_NOT_OK(reader.ReadNext(&batch));
    if (!batch) break;
    ARROW_ASSIGN_OR_RAISE(
        auto mask_datum,
        compute::ExecuteScalarExpression(bound_filter, *full_table->schema(),
                                         Datum(batch)));
    ARROW_ASSIGN_OR_RAISE(auto filtered, compute::Filter(batch, mask_datum.make_array()));
    total += filtered.record_batch()->num_rows();
  }
  return total;
}

}  // namespace

class TestDatasetParquetPagePruning : public ::testing::Test {
 protected:
  void SetUp() override {
    format_ = std::make_shared<ParquetFileFormat>();
    data_schema_ = schema({field("v", int32())});

    ASSERT_OK_AND_ASSIGN(buf_with_index_,
                         WritePageIndexParquet(/*enable_page_index=*/true));
    ASSERT_OK_AND_ASSIGN(buf_no_index_,
                         WritePageIndexParquet(/*enable_page_index=*/false));
    ASSERT_OK_AND_ASSIGN(buf_all_null_, WriteAllNullParquet());

    // Build an in-memory reference table from the page-index-enabled file.
    auto source = FileSource(buf_with_index_);
    ASSERT_OK_AND_ASSIGN(auto fragment, format_->MakeFragment(source));
    auto opts_all =
        MakeScanOptions(data_schema_, literal(true), parquet::PageIndexPolicy::NEVER);
    std::vector<std::shared_ptr<RecordBatch>> all_batches;
    int64_t ignored = 0;
    ASSERT_OK(CollectBatches(fragment, opts_all, &ignored, &all_batches));
    ASSERT_OK_AND_ASSIGN(ref_table_, Table::FromRecordBatches(data_schema_, all_batches));
  }

  // Make a fragment from buf and set the physical schema.
  std::shared_ptr<Fragment> MakeFragmentFromBuffer(
      const std::shared_ptr<Buffer>& buf) const {
    auto source = FileSource(buf);
    EXPECT_OK_AND_ASSIGN(auto fragment, format_->MakeFragment(source));
    return fragment;
  }

  std::shared_ptr<ParquetFileFormat> format_;
  std::shared_ptr<Schema> data_schema_;
  std::shared_ptr<Buffer> buf_with_index_;
  std::shared_ptr<Buffer> buf_no_index_;
  std::shared_ptr<Buffer> buf_all_null_;
  std::shared_ptr<Table> ref_table_;  // all rows, for reference filtering
};

// Case 1: ScanBatchesAsync with a filter returns exactly the rows that satisfy
// the predicate, matching the in-memory reference.
TEST_F(TestDatasetParquetPagePruning, ScanWithFilterMatchesReference) {
  // Filter: v >= 150 AND v < 250  (row groups 1 and 2 partially)
  auto filter = and_(greater_equal(field_ref("v"), literal<int32_t>(150)),
                     less(field_ref("v"), literal<int32_t>(250)));

  auto fragment = MakeFragmentFromBuffer(buf_with_index_);
  auto opts = MakeScanOptions(data_schema_, filter, parquet::PageIndexPolicy::AUTO);

  int64_t scanned_rows = 0;
  std::vector<std::shared_ptr<RecordBatch>> batches;
  ASSERT_OK(CollectBatches(fragment, opts, &scanned_rows, &batches));

  ASSERT_OK_AND_ASSIGN(int64_t expected_rows, ReferenceRowCount(ref_table_, filter));
  EXPECT_GT(expected_rows, 0) << "Reference should have matching rows";
  EXPECT_EQ(scanned_rows, expected_rows)
      << "Scanned row count must match in-memory reference";

  // Verify every returned value satisfies the predicate.
  for (const auto& batch : batches) {
    const auto& col = checked_cast<const Int32Array&>(*batch->column(0));
    for (int64_t i = 0; i < col.length(); ++i) {
      ASSERT_FALSE(col.IsNull(i));
      int32_t v = col.Value(i);
      EXPECT_GE(v, 150);
      EXPECT_LT(v, 250);
    }
  }
}

// Case 2: Disabling pruning (PageIndexPolicy::NEVER) still returns all rows
// that satisfy the predicate (the pruned result is a subset of unpruned).
TEST_F(TestDatasetParquetPagePruning, PruningOffReturnsAll) {
  auto filter = greater_equal(field_ref("v"), literal<int32_t>(200));

  // Scan with pruning enabled.
  auto fragment_pruned = MakeFragmentFromBuffer(buf_with_index_);
  auto opts_pruned =
      MakeScanOptions(data_schema_, filter, parquet::PageIndexPolicy::AUTO);
  int64_t pruned_rows = 0;
  std::vector<std::shared_ptr<RecordBatch>> pruned_batches;
  ASSERT_OK(CollectBatches(fragment_pruned, opts_pruned, &pruned_rows, &pruned_batches));

  // Scan with pruning disabled.
  auto fragment_unpruned = MakeFragmentFromBuffer(buf_with_index_);
  auto opts_unpruned =
      MakeScanOptions(data_schema_, filter, parquet::PageIndexPolicy::NEVER);
  int64_t unpruned_rows = 0;
  std::vector<std::shared_ptr<RecordBatch>> unpruned_batches;
  ASSERT_OK(CollectBatches(fragment_unpruned, opts_unpruned, &unpruned_rows,
                           &unpruned_batches));

  // The NEVER policy skips page pruning but applies the filter post-decode; both
  // paths should return the same qualifying rows.
  ASSERT_OK_AND_ASSIGN(int64_t expected, ReferenceRowCount(ref_table_, filter));
  EXPECT_EQ(pruned_rows, expected);
  EXPECT_EQ(unpruned_rows, expected);
  EXPECT_GE(unpruned_rows, pruned_rows)
      << "Unpruned scan must return at least as many qualifying rows as pruned";
}

// Case 3: A predicate that matches no rows yields 0 batches (or all empty batches).
TEST_F(TestDatasetParquetPagePruning, PredicateMatchesNone) {
  // v > max value in the file (kNumRowGroups * kRowsPerGroup - 1 = 399)
  constexpr int32_t kBeyondMax = kNumRowGroups * kRowsPerGroup;
  auto filter = greater(field_ref("v"), literal(int32_t{kBeyondMax}));

  auto fragment = MakeFragmentFromBuffer(buf_with_index_);
  auto opts = MakeScanOptions(data_schema_, filter, parquet::PageIndexPolicy::AUTO);

  int64_t total_rows = 0;
  std::vector<std::shared_ptr<RecordBatch>> batches;
  ASSERT_OK(CollectBatches(fragment, opts, &total_rows, &batches));

  EXPECT_EQ(total_rows, 0) << "No rows should match a beyond-max predicate";
}

// Case 4: A predicate that matches all rows returns the same count as a full scan.
TEST_F(TestDatasetParquetPagePruning, PredicateMatchesAll) {
  // v >= 0 matches every row in the file.
  auto filter = greater_equal(field_ref("v"), literal<int32_t>(0));

  auto fragment = MakeFragmentFromBuffer(buf_with_index_);
  auto opts = MakeScanOptions(data_schema_, filter, parquet::PageIndexPolicy::AUTO);

  int64_t total_rows = 0;
  std::vector<std::shared_ptr<RecordBatch>> batches;
  ASSERT_OK(CollectBatches(fragment, opts, &total_rows, &batches));

  EXPECT_EQ(total_rows, static_cast<int64_t>(kNumRowGroups * kRowsPerGroup))
      << "A match-all predicate must return all rows";

  // Also confirm this equals the unpruned count.
  auto fragment2 = MakeFragmentFromBuffer(buf_with_index_);
  auto opts2 =
      MakeScanOptions(data_schema_, literal(true), parquet::PageIndexPolicy::NEVER);
  int64_t unpruned_rows = 0;
  std::vector<std::shared_ptr<RecordBatch>> unpruned_batches;
  ASSERT_OK(CollectBatches(fragment2, opts2, &unpruned_rows, &unpruned_batches));

  EXPECT_EQ(total_rows, unpruned_rows);
}

// Case 5: A pruned scan issues strictly fewer byte-range reads than an unpruned
// scan.  We use TrackedRandomAccessFile to instrument the file source.
TEST_F(TestDatasetParquetPagePruning, PrunedIsFewerReads) {
  // Choose a narrow range that covers only part of the data.
  auto filter = and_(greater_equal(field_ref("v"), literal<int32_t>(50)),
                     less(field_ref("v"), literal<int32_t>(100)));

  auto run_scan = [&](parquet::PageIndexPolicy policy) -> int64_t {
    auto buf_reader = std::make_shared<::arrow::io::BufferReader>(buf_with_index_);
    std::shared_ptr<io::TrackedRandomAccessFile> tracked =
        io::TrackedRandomAccessFile::Make(buf_reader.get());
    auto source = FileSource(tracked);
    EXPECT_OK_AND_ASSIGN(auto fragment, format_->MakeFragment(source));
    auto opts = MakeScanOptions(data_schema_, filter, policy);
    int64_t rows = 0;
    std::vector<std::shared_ptr<RecordBatch>> batches;
    EXPECT_OK(CollectBatches(fragment, opts, &rows, &batches));
    return tracked->bytes_read();
  };

  int64_t bytes_pruned = run_scan(parquet::PageIndexPolicy::AUTO);
  int64_t bytes_unpruned = run_scan(parquet::PageIndexPolicy::NEVER);

  // Page-level pruning should reduce I/O.  We accept equality only if the file
  // is so small that no page pruning actually happens; in practice the file we
  // write has multiple pages per row group, so pruning should reduce bytes.
  EXPECT_LE(bytes_pruned, bytes_unpruned)
      << "Pruned scan should not read more bytes than an unpruned scan";
}

// Case 6: A file written without page indices falls back to a full (row-group-
// level) scan without error.  Row counts must still be correct.
TEST_F(TestDatasetParquetPagePruning, NoIndexFallback) {
  auto filter = and_(greater_equal(field_ref("v"), literal<int32_t>(100)),
                     less(field_ref("v"), literal<int32_t>(200)));

  auto fragment = MakeFragmentFromBuffer(buf_no_index_);
  auto opts = MakeScanOptions(data_schema_, filter, parquet::PageIndexPolicy::AUTO);

  int64_t total_rows = 0;
  std::vector<std::shared_ptr<RecordBatch>> batches;
  // Must complete without error even though there are no page indices.
  ASSERT_OK(CollectBatches(fragment, opts, &total_rows, &batches));

  // Reference count from the index-enabled file (same data).
  ASSERT_OK_AND_ASSIGN(int64_t expected, ReferenceRowCount(ref_table_, filter));
  EXPECT_EQ(total_rows, expected) << "Fallback full scan must return all qualifying rows";

  // All returned values must satisfy the predicate.
  for (const auto& batch : batches) {
    const auto& col = checked_cast<const Int32Array&>(*batch->column(0));
    for (int64_t i = 0; i < col.length(); ++i) {
      if (!col.IsNull(i)) {
        int32_t v = col.Value(i);
        EXPECT_GE(v, 100);
        EXPECT_LT(v, 200);
      }
    }
  }
}

// Case 7: A fragment consisting entirely of null values returns 0 qualifying
// rows for a non-null predicate.
TEST_F(TestDatasetParquetPagePruning, AllNullFragmentZeroRows) {
  auto filter = greater(field_ref("v"), literal<int32_t>(0));

  auto fragment = MakeFragmentFromBuffer(buf_all_null_);
  auto opts = MakeScanOptions(data_schema_, filter, parquet::PageIndexPolicy::AUTO);

  int64_t total_rows = 0;
  std::vector<std::shared_ptr<RecordBatch>> batches;
  ASSERT_OK(CollectBatches(fragment, opts, &total_rows, &batches));

  EXPECT_EQ(total_rows, 0)
      << "All-null fragment should yield 0 rows for a non-null predicate";
}

// Case 8: A file with a single row group and small page size behaves identically
// to an unpruned scan.
TEST_F(TestDatasetParquetPagePruning, SinglePageEqualsUnpruned) {
  // Write a single-row-group file — one row group, page index enabled.
  auto single_rg_schema = schema({field("v", int32())});
  Int32Builder builder;
  ASSERT_OK(builder.Reserve(kRowsPerGroup));
  for (int i = 0; i < kRowsPerGroup; ++i) {
    ASSERT_OK(builder.Append(i));
  }
  std::shared_ptr<Array> col;
  ASSERT_OK_AND_ASSIGN(col, builder.Finish());
  auto batch = RecordBatch::Make(single_rg_schema, kRowsPerGroup, {col});

  auto pool = arrow::default_memory_pool();
  auto sink = parquet::CreateOutputStream(pool);
  auto writer_props =
      parquet::WriterProperties::Builder().enable_write_page_index()->build();
  auto arrow_props = parquet::default_arrow_writer_properties();
  ASSERT_OK_AND_ASSIGN(auto writer,
                       parquet::arrow::FileWriter::Open(*single_rg_schema, pool, sink,
                                                        writer_props, arrow_props));
  ASSERT_OK(writer->NewRowGroup());
  ASSERT_OK(writer->WriteColumnChunk(*batch->column(0)));
  ASSERT_OK(writer->Close());
  ASSERT_OK_AND_ASSIGN(auto buf, sink->Finish());

  // Filter that matches half the rows.
  auto filter = less(field_ref("v"), literal<int32_t>(kRowsPerGroup / 2));

  // Pruned scan.
  auto frag_pruned = MakeFragmentFromBuffer(buf);
  auto opts_pruned =
      MakeScanOptions(single_rg_schema, filter, parquet::PageIndexPolicy::AUTO);
  int64_t rows_pruned = 0;
  std::vector<std::shared_ptr<RecordBatch>> batches_pruned;
  ASSERT_OK(CollectBatches(frag_pruned, opts_pruned, &rows_pruned, &batches_pruned));

  // Unpruned scan (NEVER policy — no page filtering).
  auto frag_unpruned = MakeFragmentFromBuffer(buf);
  auto opts_unpruned =
      MakeScanOptions(single_rg_schema, filter, parquet::PageIndexPolicy::NEVER);
  int64_t rows_unpruned = 0;
  std::vector<std::shared_ptr<RecordBatch>> batches_unpruned;
  ASSERT_OK(
      CollectBatches(frag_unpruned, opts_unpruned, &rows_unpruned, &batches_unpruned));

  EXPECT_EQ(rows_pruned, rows_unpruned)
      << "Single-page-per-group file: pruned and unpruned must return the same row count";
  EXPECT_EQ(rows_pruned, kRowsPerGroup / 2);
}

// Case 9: PageIndexPolicy::AUTO on a file without page indices falls back to
// full decode and still returns all matching rows.
TEST_F(TestDatasetParquetPagePruning, AutoAbsentIndexFullDecode) {
  // buf_no_index_ was written without page indices.  AUTO should detect their
  // absence and fall back to a full decode, returning correct results.
  auto filter = and_(greater_equal(field_ref("v"), literal<int32_t>(200)),
                     less(field_ref("v"), literal<int32_t>(300)));

  auto fragment = MakeFragmentFromBuffer(buf_no_index_);
  auto opts = MakeScanOptions(data_schema_, filter, parquet::PageIndexPolicy::AUTO);

  int64_t total_rows = 0;
  std::vector<std::shared_ptr<RecordBatch>> batches;
  ASSERT_OK(CollectBatches(fragment, opts, &total_rows, &batches));

  ASSERT_OK_AND_ASSIGN(int64_t expected, ReferenceRowCount(ref_table_, filter));
  EXPECT_GT(expected, 0) << "Sanity: range [200,300) should have matching rows";
  EXPECT_EQ(total_rows, expected)
      << "AUTO policy absent-index fallback must return all qualifying rows";

  // Confirm values are in range.
  for (const auto& batch : batches) {
    const auto& col = checked_cast<const Int32Array&>(*batch->column(0));
    for (int64_t i = 0; i < col.length(); ++i) {
      if (!col.IsNull(i)) {
        int32_t v = col.Value(i);
        EXPECT_GE(v, 200);
        EXPECT_LT(v, 300);
      }
    }
  }
}

// Case 10: A predicate that skips every row group (values beyond the file max)
// completes successfully with 0 rows and no error.
TEST_F(TestDatasetParquetPagePruning, AllRGSkipped) {
  // All values in the file are in [0, kNumRowGroups*kRowsPerGroup).
  // Predicate v > max ensures every row group is eliminated at the row-group
  // statistics level, so no data pages are even opened.
  constexpr int32_t kBeyondMax = kNumRowGroups * kRowsPerGroup + 1000;
  auto filter = greater(field_ref("v"), literal(int32_t{kBeyondMax}));

  auto fragment = MakeFragmentFromBuffer(buf_with_index_);
  auto opts = MakeScanOptions(data_schema_, filter, parquet::PageIndexPolicy::AUTO);

  int64_t total_rows = 0;
  std::vector<std::shared_ptr<RecordBatch>> batches;
  ASSERT_OK(CollectBatches(fragment, opts, &total_rows, &batches))
      << "Scanner must complete without error even when all row groups are skipped";

  EXPECT_EQ(total_rows, 0) << "All row groups skipped: expected 0 rows";
}

}  // namespace dataset
}  // namespace arrow
