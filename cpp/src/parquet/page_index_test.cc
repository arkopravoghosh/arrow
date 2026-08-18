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

#include "parquet/page_index.h"

#include <gtest/gtest.h>
#include <memory>

#include "arrow/io/file.h"
#include "arrow/util/float16.h"
#include "parquet/file_reader.h"
#include "parquet/metadata.h"
#include "parquet/schema.h"
#include "parquet/test_util.h"
#include "parquet/thrift_internal.h"

namespace parquet {

TEST(PageIndex, ReadOffsetIndex) {
  std::string dir_string(parquet::test::get_data_dir());
  std::string path = dir_string + "/alltypes_tiny_pages.parquet";
  auto reader = ParquetFileReader::OpenFile(path, false);
  auto file_metadata = reader->metadata();

  // Get offset index location to column 0 of row group 0.
  const int row_group_id = 0;
  const int column_id = 0;
  ASSERT_LT(row_group_id, file_metadata->num_row_groups());
  ASSERT_LT(column_id, file_metadata->num_columns());
  auto index_location = file_metadata->RowGroup(row_group_id)
                            ->ColumnChunk(column_id)
                            ->GetOffsetIndexLocation();
  ASSERT_TRUE(index_location.has_value());

  // Read serialized offset index from the file.
  std::shared_ptr<::arrow::io::RandomAccessFile> source;
  PARQUET_ASSIGN_OR_THROW(source, ::arrow::io::ReadableFile::Open(path));
  PARQUET_ASSIGN_OR_THROW(auto buffer,
                          source->ReadAt(index_location->offset, index_location->length));
  PARQUET_THROW_NOT_OK(source->Close());

  // Deserialize offset index.
  auto properties = default_reader_properties();
  std::unique_ptr<OffsetIndex> offset_index = OffsetIndex::Make(
      buffer->data(), static_cast<uint32_t>(buffer->size()), properties);

  // Verify only partial data as it contains 325 pages in total.
  const size_t num_pages = 325;
  const std::vector<size_t> page_indices = {0, 100, 200, 300};
  const std::vector<PageLocation> page_locations = {
      PageLocation{4, 109, 0}, PageLocation{11480, 133, 2244},
      PageLocation{22980, 133, 4494}, PageLocation{34480, 133, 6744}};

  ASSERT_EQ(num_pages, offset_index->page_locations().size());
  for (size_t i = 0; i < page_indices.size(); ++i) {
    size_t page_id = page_indices.at(i);
    const auto& read_page_location = offset_index->page_locations().at(page_id);
    const auto& expected_page_location = page_locations.at(i);
    ASSERT_EQ(expected_page_location.offset, read_page_location.offset);
    ASSERT_EQ(expected_page_location.compressed_page_size,
              read_page_location.compressed_page_size);
    ASSERT_EQ(expected_page_location.first_row_index, read_page_location.first_row_index);
  }
}

template <typename DType, typename T = typename DType::c_type>
void TestReadTypedColumnIndex(const std::string& file_name, int column_id,
                              size_t num_pages, BoundaryOrder::type boundary_order,
                              const std::vector<size_t>& page_indices,
                              const std::vector<bool>& null_pages,
                              const std::vector<T>& min_values,
                              const std::vector<T>& max_values,
                              bool has_null_counts = false,
                              const std::vector<int64_t>& null_counts = {}) {
  std::string dir_string(parquet::test::get_data_dir());
  std::string path = dir_string + "/" + file_name;
  auto reader = ParquetFileReader::OpenFile(path, false);
  auto file_metadata = reader->metadata();

  // Get column index location to a specific column chunk.
  const int row_group_id = 0;
  ASSERT_LT(row_group_id, file_metadata->num_row_groups());
  ASSERT_LT(column_id, file_metadata->num_columns());
  auto index_location = file_metadata->RowGroup(row_group_id)
                            ->ColumnChunk(column_id)
                            ->GetColumnIndexLocation();
  ASSERT_TRUE(index_location.has_value());

  // Read serialized column index from the file.
  std::shared_ptr<::arrow::io::RandomAccessFile> source;
  PARQUET_ASSIGN_OR_THROW(source, ::arrow::io::ReadableFile::Open(path));
  PARQUET_ASSIGN_OR_THROW(auto buffer,
                          source->ReadAt(index_location->offset, index_location->length));
  PARQUET_THROW_NOT_OK(source->Close());

  // Deserialize column index.
  auto properties = default_reader_properties();
  auto descr = file_metadata->schema()->Column(column_id);
  std::unique_ptr<ColumnIndex> column_index = ColumnIndex::Make(
      *descr, buffer->data(), static_cast<uint32_t>(buffer->size()), properties);
  auto typed_column_index = dynamic_cast<TypedColumnIndex<DType>*>(column_index.get());
  ASSERT_TRUE(typed_column_index != nullptr);

  // Verify only partial data as there are too many pages.
  ASSERT_EQ(num_pages, column_index->null_pages().size());
  ASSERT_EQ(has_null_counts, column_index->has_null_counts());
  ASSERT_EQ(boundary_order, column_index->boundary_order());
  for (size_t i = 0; i < page_indices.size(); ++i) {
    size_t page_id = page_indices.at(i);
    ASSERT_EQ(null_pages.at(i), column_index->null_pages().at(page_id));
    if (has_null_counts) {
      ASSERT_EQ(null_counts.at(i), column_index->null_counts().at(page_id));
    }
    // min/max values are only meaningful for non-null pages.
    if (!null_pages.at(i)) {
      if constexpr (std::is_same_v<T, double>) {
        ASSERT_DOUBLE_EQ(min_values.at(i), typed_column_index->min_values().at(page_id));
        ASSERT_DOUBLE_EQ(max_values.at(i), typed_column_index->max_values().at(page_id));
      } else if constexpr (std::is_same_v<T, float>) {
        ASSERT_FLOAT_EQ(min_values.at(i), typed_column_index->min_values().at(page_id));
        ASSERT_FLOAT_EQ(max_values.at(i), typed_column_index->max_values().at(page_id));
      } else if constexpr (std::is_same_v<T, FLBA>) {
        auto len = descr->type_length();
        ASSERT_EQ(0, ::memcmp(min_values.at(i).ptr,
                              typed_column_index->min_values().at(page_id).ptr, len));
        ASSERT_EQ(0, ::memcmp(max_values.at(i).ptr,
                              typed_column_index->max_values().at(page_id).ptr, len));
      } else {
        ASSERT_EQ(min_values.at(i), typed_column_index->min_values().at(page_id));
        ASSERT_EQ(max_values.at(i), typed_column_index->max_values().at(page_id));
      }
    }
  }
}

TEST(PageIndex, ReadInt64ColumnIndex) {
  const int column_id = 5;
  const size_t num_pages = 528;
  const BoundaryOrder::type boundary_order = BoundaryOrder::Unordered;
  const std::vector<size_t> page_indices = {0, 99, 426, 520};
  const std::vector<bool> null_pages = {false, false, false, false};
  const bool has_null_counts = true;
  const std::vector<int64_t> null_counts = {0, 0, 0, 0};
  const std::vector<int64_t> min_values = {0, 10, 0, 0};
  const std::vector<int64_t> max_values = {90, 90, 80, 70};

  TestReadTypedColumnIndex<Int64Type>(
      "alltypes_tiny_pages.parquet", column_id, num_pages, boundary_order, page_indices,
      null_pages, min_values, max_values, has_null_counts, null_counts);
}

TEST(PageIndex, ReadDoubleColumnIndex) {
  const int column_id = 7;
  const size_t num_pages = 528;
  const BoundaryOrder::type boundary_order = BoundaryOrder::Unordered;
  const std::vector<size_t> page_indices = {0, 51, 212, 527};
  const std::vector<bool> null_pages = {false, false, false, false};
  const bool has_null_counts = true;
  const std::vector<int64_t> null_counts = {0, 0, 0, 0};
  const std::vector<double> min_values = {-0, 30.3, 10.1, 40.4};
  const std::vector<double> max_values = {90.9, 90.9, 90.9, 60.6};

  TestReadTypedColumnIndex<DoubleType>(
      "alltypes_tiny_pages.parquet", column_id, num_pages, boundary_order, page_indices,
      null_pages, min_values, max_values, has_null_counts, null_counts);
}

TEST(PageIndex, ReadByteArrayColumnIndex) {
  const int column_id = 9;
  const size_t num_pages = 352;
  const BoundaryOrder::type boundary_order = BoundaryOrder::Ascending;
  const std::vector<size_t> page_indices = {0, 128, 256};
  const std::vector<bool> null_pages = {false, false, false};
  const bool has_null_counts = true;
  const std::vector<int64_t> null_counts = {0, 0, 0};

  // All min values are "0" and max values are "9".
  const std::string_view min_value = "0";
  const std::string_view max_value = "9";
  const std::vector<ByteArray> min_values = {ByteArray{min_value}, ByteArray{min_value},
                                             ByteArray{min_value}};
  const std::vector<ByteArray> max_values = {ByteArray{max_value}, ByteArray{max_value},
                                             ByteArray{max_value}};

  TestReadTypedColumnIndex<ByteArrayType>(
      "alltypes_tiny_pages.parquet", column_id, num_pages, boundary_order, page_indices,
      null_pages, min_values, max_values, has_null_counts, null_counts);
}

TEST(PageIndex, ReadBoolColumnIndex) {
  const int column_id = 1;
  const size_t num_pages = 82;
  const BoundaryOrder::type boundary_order = BoundaryOrder::Ascending;
  const std::vector<size_t> page_indices = {0, 16, 64};
  const std::vector<bool> null_pages = {false, false, false};
  const bool has_null_counts = true;
  const std::vector<int64_t> null_counts = {0, 0, 0};
  const std::vector<bool> min_values = {false, false, false};
  const std::vector<bool> max_values = {true, true, true};

  TestReadTypedColumnIndex<BooleanType>(
      "alltypes_tiny_pages.parquet", column_id, num_pages, boundary_order, page_indices,
      null_pages, min_values, max_values, has_null_counts, null_counts);
}

TEST(PageIndex, ReadFixedLengthByteArrayColumnIndex) {
  auto to_flba = [](const char* ptr) {
    return FLBA{reinterpret_cast<const uint8_t*>(ptr)};
  };

  const int column_id = 0;
  const size_t num_pages = 10;
  const BoundaryOrder::type boundary_order = BoundaryOrder::Descending;
  const std::vector<size_t> page_indices = {0, 4, 8};
  const std::vector<bool> null_pages = {false, false, false};
  const bool has_null_counts = true;
  const std::vector<int64_t> null_counts = {9, 13, 9};
  const std::vector<const char*> min_literals = {"\x00\x00\x03\x85", "\x00\x00\x01\xF5",
                                                 "\x00\x00\x00\x65"};
  const std::vector<const char*> max_literals = {"\x00\x00\x03\xE8", "\x00\x00\x02\x58",
                                                 "\x00\x00\x00\xC8"};
  const std::vector<FLBA> min_values = {
      to_flba(min_literals[0]), to_flba(min_literals[1]), to_flba(min_literals[2])};
  const std::vector<FLBA> max_values = {
      to_flba(max_literals[0]), to_flba(max_literals[1]), to_flba(max_literals[2])};

  TestReadTypedColumnIndex<FLBAType>(
      "fixed_length_byte_array.parquet", column_id, num_pages, boundary_order,
      page_indices, null_pages, min_values, max_values, has_null_counts, null_counts);
}

TEST(PageIndex, ReadColumnIndexWithNullPage) {
  const int column_id = 0;
  const size_t num_pages = 10;
  const BoundaryOrder::type boundary_order = BoundaryOrder::Unordered;
  const std::vector<size_t> page_indices = {2, 4, 8};
  const std::vector<bool> null_pages = {true, false, false};
  const bool has_null_counts = true;
  const std::vector<int64_t> null_counts = {100, 16, 8};
  const std::vector<int32_t> min_values = {0, -2048691758, -2046900272};
  const std::vector<int32_t> max_values = {0, 2143189382, 2087168549};

  TestReadTypedColumnIndex<Int32Type>(
      "int32_with_null_pages.parquet", column_id, num_pages, boundary_order, page_indices,
      null_pages, min_values, max_values, has_null_counts, null_counts);
}

struct PageIndexRanges {
  int64_t column_index_offset;
  int64_t column_index_length;
  int64_t offset_index_offset;
  int64_t offset_index_length;
};

using RowGroupRanges = std::vector<PageIndexRanges>;

/// Creates an FileMetaData object w/ single row group based on data in
/// 'row_group_ranges'. It sets the offsets and sizes of the column index and offset index
/// members of the row group. It doesn't set the member if the input value is -1.
std::shared_ptr<FileMetaData> ConstructFakeMetaData(
    const RowGroupRanges& row_group_ranges) {
  format::RowGroup row_group;
  for (auto& page_index_ranges : row_group_ranges) {
    format::ColumnChunk col_chunk;
    if (page_index_ranges.column_index_offset != -1) {
      col_chunk.__set_column_index_offset(page_index_ranges.column_index_offset);
    }
    if (page_index_ranges.column_index_length != -1) {
      col_chunk.__set_column_index_length(
          static_cast<int32_t>(page_index_ranges.column_index_length));
    }
    if (page_index_ranges.offset_index_offset != -1) {
      col_chunk.__set_offset_index_offset(page_index_ranges.offset_index_offset);
    }
    if (page_index_ranges.offset_index_length != -1) {
      col_chunk.__set_offset_index_length(
          static_cast<int32_t>(page_index_ranges.offset_index_length));
    }
    row_group.columns.push_back(col_chunk);
  }

  format::FileMetaData metadata;
  metadata.row_groups.push_back(row_group);

  metadata.schema.emplace_back();
  schema::NodeVector fields;
  for (size_t i = 0; i < row_group_ranges.size(); ++i) {
    fields.push_back(schema::Int64(std::to_string(i)));
    metadata.schema.emplace_back();
    fields.back()->ToParquet(&metadata.schema.back());
  }
  schema::GroupNode::Make("schema", Repetition::REPEATED, fields)
      ->ToParquet(&metadata.schema.front());

  auto sink = CreateOutputStream();
  ThriftSerializer{}.Serialize(&metadata, sink.get());
  auto buffer = sink->Finish().MoveValueUnsafe();
  uint32_t len = static_cast<uint32_t>(buffer->size());
  return FileMetaData::Make(buffer->data(), &len);
}

/// Validates that 'DeterminePageIndexRangesInRowGroup()' selects the expected file
/// offsets and sizes or returns false when the row group doesn't have a page index.
void ValidatePageIndexRange(const RowGroupRanges& row_group_ranges,
                            const std::vector<int32_t>& column_indices,
                            bool expected_has_column_index,
                            bool expected_has_offset_index, int expected_ci_start,
                            int expected_ci_size, int expected_oi_start,
                            int expected_oi_size) {
  auto file_metadata = ConstructFakeMetaData(row_group_ranges);
  auto read_range = PageIndexReader::DeterminePageIndexRangesInRowGroup(
      *file_metadata->RowGroup(0), column_indices);
  ASSERT_EQ(expected_has_column_index, read_range.column_index.has_value());
  ASSERT_EQ(expected_has_offset_index, read_range.offset_index.has_value());
  if (expected_has_column_index) {
    EXPECT_EQ(expected_ci_start, read_range.column_index->offset);
    EXPECT_EQ(expected_ci_size, read_range.column_index->length);
  }
  if (expected_has_offset_index) {
    EXPECT_EQ(expected_oi_start, read_range.offset_index->offset);
    EXPECT_EQ(expected_oi_size, read_range.offset_index->length);
  }
}

/// This test constructs a couple of artificial row groups with page index offsets in
/// them. Then it validates if PageIndexReader::DeterminePageIndexRangesInRowGroup()
/// properly computes the file range that contains the whole page index.
TEST(PageIndex, DeterminePageIndexRangesInRowGroup) {
  // No Column chunks
  ValidatePageIndexRange({}, {}, false, false, -1, -1, -1, -1);
  // No page index at all.
  ValidatePageIndexRange({{-1, -1, -1, -1}}, {}, false, false, -1, -1, -1, -1);
  // Page index for single column chunk.
  ValidatePageIndexRange({{10, 5, 15, 5}}, {}, true, true, 10, 5, 15, 5);
  // Page index for two column chunks.
  ValidatePageIndexRange({{10, 5, 30, 25}, {15, 15, 50, 20}}, {}, true, true, 10, 20, 30,
                         40);
  // Page index for second column chunk.
  ValidatePageIndexRange({{-1, -1, -1, -1}, {20, 10, 30, 25}}, {}, true, true, 20, 10, 30,
                         25);
  // Page index for first column chunk.
  ValidatePageIndexRange({{10, 5, 15, 5}, {-1, -1, -1, -1}}, {}, true, true, 10, 5, 15,
                         5);
  // Missing offset index for first column chunk. Gap in column index.
  ValidatePageIndexRange({{10, 5, -1, -1}, {20, 10, 30, 25}}, {}, true, true, 10, 20, 30,
                         25);
  // Missing offset index for second column chunk.
  ValidatePageIndexRange({{10, 5, 25, 5}, {20, 10, -1, -1}}, {}, true, true, 10, 20, 25,
                         5);
  // Four column chunks.
  ValidatePageIndexRange(
      {{100, 10, 220, 30}, {110, 25, 250, 10}, {140, 30, 260, 40}, {200, 10, 300, 100}},
      {}, true, true, 100, 110, 220, 180);
}

/// This test constructs a couple of artificial row groups with page index offsets in
/// them. Then it validates if PageIndexReader::DeterminePageIndexRangesInRowGroup()
/// properly computes the file range that contains the page index of selected columns.
TEST(PageIndex, DeterminePageIndexRangesInRowGroupWithPartialColumnsSelected) {
  // No page index at all.
  ValidatePageIndexRange({{-1, -1, -1, -1}}, {0}, false, false, -1, -1, -1, -1);
  // Page index for single column chunk.
  ValidatePageIndexRange({{10, 5, 15, 5}}, {0}, true, true, 10, 5, 15, 5);
  // Page index for the 1st column chunk.
  ValidatePageIndexRange({{10, 5, 30, 25}, {15, 15, 50, 20}}, {0}, true, true, 10, 5, 30,
                         25);
  // Page index for the 2nd column chunk.
  ValidatePageIndexRange({{10, 5, 30, 25}, {15, 15, 50, 20}}, {1}, true, true, 15, 15, 50,
                         20);
  // Only 2nd column is selected among four column chunks.
  ValidatePageIndexRange(
      {{100, 10, 220, 30}, {110, 25, 250, 10}, {140, 30, 260, 40}, {200, 10, 300, 100}},
      {1}, true, true, 110, 25, 250, 10);
  // Only 2nd and 3rd columns are selected among four column chunks.
  ValidatePageIndexRange(
      {{100, 10, 220, 30}, {110, 25, 250, 10}, {140, 30, 260, 40}, {200, 10, 300, 100}},
      {1, 2}, true, true, 110, 60, 250, 50);
  // Only 2nd and 4th columns are selected among four column chunks.
  ValidatePageIndexRange(
      {{100, 10, 220, 30}, {110, 25, 250, 10}, {140, 30, 260, 40}, {200, 10, 300, 100}},
      {1, 3}, true, true, 110, 100, 250, 150);
  // Only 1st, 2nd and 4th columns are selected among four column chunks.
  ValidatePageIndexRange(
      {{100, 10, 220, 30}, {110, 25, 250, 10}, {140, 30, 260, 40}, {200, 10, 300, 100}},
      {0, 1, 3}, true, true, 100, 110, 220, 180);
  // 3rd column is selected but not present in the row group.
  EXPECT_THROW(ValidatePageIndexRange({{10, 5, 30, 25}, {15, 15, 50, 20}}, {2}, false,
                                      false, -1, -1, -1, -1),
               ParquetException);
}

/// This test constructs a couple of artificial row groups with page index offsets in
/// them. Then it validates if PageIndexReader::DeterminePageIndexRangesInRowGroup()
/// properly detects if column index or offset index is missing.
TEST(PageIndex, DeterminePageIndexRangesInRowGroupWithMissingPageIndex) {
  // No column index at all.
  ValidatePageIndexRange({{-1, -1, 15, 5}}, {}, false, true, -1, -1, 15, 5);
  // No offset index at all.
  ValidatePageIndexRange({{10, 5, -1, -1}}, {}, true, false, 10, 5, -1, -1);
  // No column index at all among two column chunks.
  ValidatePageIndexRange({{-1, -1, 30, 25}, {-1, -1, 50, 20}}, {}, false, true, -1, -1,
                         30, 40);
  // No offset index at all among two column chunks.
  ValidatePageIndexRange({{10, 5, -1, -1}, {15, 15, -1, -1}}, {}, true, false, 10, 20, -1,
                         -1);
}

void TestWriteOffsetIndex(bool write_size_stats) {
  /// Create offset index via the OffsetIndexBuilder interface.
  auto builder = OffsetIndexBuilder::Make();
  const size_t num_pages = 5;
  const std::vector<int64_t> offsets = {100, 200, 300, 400, 500};
  const std::vector<int32_t> page_sizes = {1024, 2048, 3072, 4096, 8192};
  const std::vector<int64_t> first_row_indices = {0, 10000, 20000, 30000, 40000};
  const std::vector<int64_t> unencoded_byte_array_lengths = {1111, 2222, 0, 3333, 4444};
  for (size_t i = 0; i < num_pages; ++i) {
    auto unencoded_byte_array_length =
        write_size_stats ? std::make_optional(unencoded_byte_array_lengths[i])
                         : std::nullopt;
    builder->AddPage(offsets[i], page_sizes[i], first_row_indices[i],
                     unencoded_byte_array_length);
  }
  const int64_t final_position = 4096;
  builder->Finish(final_position);

  std::vector<std::unique_ptr<OffsetIndex>> offset_indexes;
  /// 1st element is the offset index just built.
  offset_indexes.emplace_back(builder->Build());
  /// 2nd element is the offset index restored by serialize-then-deserialize round trip.
  auto sink = CreateOutputStream();
  builder->WriteTo(sink.get());
  PARQUET_ASSIGN_OR_THROW(auto buffer, sink->Finish());
  offset_indexes.emplace_back(OffsetIndex::Make(buffer->data(),
                                                static_cast<uint32_t>(buffer->size()),
                                                default_reader_properties()));

  /// Verify the data of the offset index.
  for (const auto& offset_index : offset_indexes) {
    ASSERT_EQ(num_pages, offset_index->page_locations().size());
    if (write_size_stats) {
      ASSERT_EQ(num_pages, offset_index->unencoded_byte_array_data_bytes().size());
    } else {
      ASSERT_TRUE(offset_index->unencoded_byte_array_data_bytes().empty());
    }
    for (size_t i = 0; i < num_pages; ++i) {
      const auto& page_location = offset_index->page_locations().at(i);
      ASSERT_EQ(offsets[i] + final_position, page_location.offset);
      ASSERT_EQ(page_sizes[i], page_location.compressed_page_size);
      ASSERT_EQ(first_row_indices[i], page_location.first_row_index);
      if (write_size_stats) {
        ASSERT_EQ(unencoded_byte_array_lengths[i],
                  offset_index->unencoded_byte_array_data_bytes()[i]);
      }
    }
  }
}

TEST(PageIndex, WriteOffsetIndexWithoutSizeStats) {
  TestWriteOffsetIndex(/*write_size_stats=*/false);
}

TEST(PageIndex, WriteOffsetIndexWithSizeStats) {
  TestWriteOffsetIndex(/*write_size_stats=*/true);
}

struct PageLevelHistogram {
  std::vector<int64_t> def_levels;
  std::vector<int64_t> rep_levels;
};

std::unique_ptr<SizeStatistics> ConstructFakeSizeStatistics(
    const ColumnDescriptor* descr, const PageLevelHistogram& page_level_histogram) {
  auto stats = SizeStatistics::Make(descr);
  stats->definition_level_histogram = page_level_histogram.def_levels;
  stats->repetition_level_histogram = page_level_histogram.rep_levels;
  return stats;
}

void VerifyPageLevelHistogram(size_t page_id,
                              const std::vector<int64_t>& expected_page_levels,
                              const std::vector<int64_t>& all_page_levels) {
  const size_t max_level = expected_page_levels.size() - 1;
  const size_t offset = page_id * (max_level + 1);
  for (size_t level = 0; level <= max_level; ++level) {
    ASSERT_EQ(expected_page_levels[level], all_page_levels[offset + level]);
  }
}

void TestWriteTypedColumnIndex(schema::NodePtr node,
                               const std::vector<EncodedStatistics>& page_stats,
                               BoundaryOrder::type boundary_order, bool has_null_counts,
                               int16_t max_definition_level = 1,
                               int16_t max_repetition_level = 0,
                               const std::vector<PageLevelHistogram>& page_levels = {}) {
  const bool build_size_stats = !page_levels.empty();
  if (build_size_stats) {
    ASSERT_EQ(page_levels.size(), page_stats.size());
  }
  auto descr = std::make_unique<ColumnDescriptor>(node, max_definition_level,
                                                  max_repetition_level);
  auto builder = ColumnIndexBuilder::Make(descr.get());
  for (size_t i = 0; i < page_stats.size(); ++i) {
    auto size_stats = build_size_stats
                          ? ConstructFakeSizeStatistics(descr.get(), page_levels[i])
                          : std::make_unique<SizeStatistics>();
    builder->AddPage(page_stats[i], *size_stats);
  }
  ASSERT_NO_THROW(builder->Finish());

  std::vector<std::unique_ptr<ColumnIndex>> column_indexes;
  /// 1st element is the column index just built.
  column_indexes.emplace_back(builder->Build());
  /// 2nd element is the column index restored by serialize-then-deserialize round trip.
  auto sink = CreateOutputStream();
  builder->WriteTo(sink.get());
  PARQUET_ASSIGN_OR_THROW(auto buffer, sink->Finish());
  column_indexes.emplace_back(ColumnIndex::Make(*descr, buffer->data(),
                                                static_cast<uint32_t>(buffer->size()),
                                                default_reader_properties()));

  /// Verify the data of the column index.
  for (const auto& column_index : column_indexes) {
    ASSERT_EQ(boundary_order, column_index->boundary_order());
    ASSERT_EQ(has_null_counts, column_index->has_null_counts());
    const size_t num_pages = column_index->null_pages().size();
    if (build_size_stats) {
      ASSERT_EQ(num_pages * (max_repetition_level + 1),
                column_index->repetition_level_histograms().size());
      ASSERT_EQ(num_pages * (max_definition_level + 1),
                column_index->definition_level_histograms().size());
    }

    for (size_t i = 0; i < num_pages; ++i) {
      ASSERT_EQ(page_stats[i].all_null_value, column_index->null_pages()[i]);
      ASSERT_EQ(page_stats[i].min(), column_index->encoded_min_values()[i]);
      ASSERT_EQ(page_stats[i].max(), column_index->encoded_max_values()[i]);
      if (has_null_counts) {
        ASSERT_EQ(page_stats[i].null_count, column_index->null_counts()[i]);
      }
      if (build_size_stats) {
        ASSERT_NO_FATAL_FAILURE(VerifyPageLevelHistogram(
            i, page_levels[i].def_levels, column_index->definition_level_histograms()));
        ASSERT_NO_FATAL_FAILURE(VerifyPageLevelHistogram(
            i, page_levels[i].rep_levels, column_index->repetition_level_histograms()));
      }
    }
  }
}

TEST(PageIndex, WriteInt32ColumnIndex) {
  auto encode = [=](int32_t value) {
    return std::string(reinterpret_cast<const char*>(&value), sizeof(int32_t));
  };

  // Integer values in the ascending order.
  std::vector<EncodedStatistics> page_stats(3);
  page_stats.at(0).set_null_count(1).set_min(encode(1)).set_max(encode(2));
  page_stats.at(1).set_null_count(2).set_min(encode(2)).set_max(encode(3));
  page_stats.at(2).set_null_count(3).set_min(encode(3)).set_max(encode(4));

  TestWriteTypedColumnIndex(schema::Int32("c1"), page_stats, BoundaryOrder::Ascending,
                            /*has_null_counts=*/true);
}

TEST(PageIndex, WriteInt64ColumnIndex) {
  auto encode = [=](int64_t value) {
    return std::string(reinterpret_cast<const char*>(&value), sizeof(int64_t));
  };

  // Integer values in the descending order.
  std::vector<EncodedStatistics> page_stats(3);
  page_stats.at(0).set_null_count(4).set_min(encode(-1)).set_max(encode(-2));
  page_stats.at(1).set_null_count(0).set_min(encode(-2)).set_max(encode(-3));
  page_stats.at(2).set_null_count(4).set_min(encode(-3)).set_max(encode(-4));

  TestWriteTypedColumnIndex(schema::Int64("c1"), page_stats, BoundaryOrder::Descending,
                            /*has_null_counts=*/true);
}

TEST(PageIndex, WriteFloatColumnIndex) {
  auto encode = [=](float value) {
    return std::string(reinterpret_cast<const char*>(&value), sizeof(float));
  };

  // Float values with no specific order.
  std::vector<EncodedStatistics> page_stats(3);
  page_stats.at(0).set_null_count(0).set_min(encode(2.2F)).set_max(encode(4.4F));
  page_stats.at(1).set_null_count(0).set_min(encode(1.1F)).set_max(encode(5.5F));
  page_stats.at(2).set_null_count(0).set_min(encode(3.3F)).set_max(encode(6.6F));

  TestWriteTypedColumnIndex(schema::Float("c1"), page_stats, BoundaryOrder::Unordered,
                            /*has_null_counts=*/true);
}

TEST(PageIndex, WriteDoubleColumnIndex) {
  auto encode = [=](double value) {
    return std::string(reinterpret_cast<const char*>(&value), sizeof(double));
  };

  // Double values with no specific order and without null count.
  std::vector<EncodedStatistics> page_stats(3);
  page_stats.at(0).set_min(encode(1.2)).set_max(encode(4.4));
  page_stats.at(1).set_min(encode(2.2)).set_max(encode(5.5));
  page_stats.at(2).set_min(encode(3.3)).set_max(encode(-6.6));

  TestWriteTypedColumnIndex(schema::Double("c1"), page_stats, BoundaryOrder::Unordered,
                            /*has_null_counts=*/false);
}

TEST(PageIndex, WriteByteArrayColumnIndex) {
  // Byte array values with identical min/max.
  std::vector<EncodedStatistics> page_stats(3);
  page_stats.at(0).set_min("bar").set_max("foo");
  page_stats.at(1).set_min("bar").set_max("foo");
  page_stats.at(2).set_min("bar").set_max("foo");

  TestWriteTypedColumnIndex(schema::ByteArray("c1"), page_stats, BoundaryOrder::Ascending,
                            /*has_null_counts=*/false);
}

TEST(PageIndex, WriteFLBAColumnIndex) {
  // FLBA values in the ascending order with some null pages
  std::vector<EncodedStatistics> page_stats(5);
  page_stats.at(0).set_min("abc").set_max("ABC");
  page_stats.at(1).all_null_value = true;
  page_stats.at(2).set_min("foo").set_max("FOO");
  page_stats.at(3).all_null_value = true;
  page_stats.at(4).set_min("xyz").set_max("XYZ");

  auto node =
      schema::PrimitiveNode::Make("c1", Repetition::OPTIONAL, Type::FIXED_LEN_BYTE_ARRAY,
                                  ConvertedType::NONE, /*length=*/3);
  TestWriteTypedColumnIndex(std::move(node), page_stats, BoundaryOrder::Ascending,
                            /*has_null_counts=*/false);
}

TEST(PageIndex, WriteFloat16ColumnIndex) {
  using ::arrow::util::Float16;
  auto encode = [](auto value) {
    auto bytes = Float16(value).ToLittleEndian();
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  };

  // Float16 (FLBA) values in the ascending order and without null count.
  std::vector<EncodedStatistics> page_stats(4);
  page_stats.at(0).set_min(encode(-1.3)).set_max(encode(+3.6));
  page_stats.at(1).set_min(encode(-0.2)).set_max(encode(+4.5));
  page_stats.at(2).set_min(encode(+1.1)).set_max(encode(+5.4));
  page_stats.at(3).set_min(encode(+2.0)).set_max(encode(+6.3));

  auto node = schema::PrimitiveNode::Make(
      "c1", Repetition::OPTIONAL, LogicalType::Float16(), Type::FIXED_LEN_BYTE_ARRAY,
      /*length=*/2);
  TestWriteTypedColumnIndex(std::move(node), page_stats, BoundaryOrder::Ascending,
                            /*has_null_counts=*/false);
}

TEST(PageIndex, WriteColumnIndexWithAllNullPages) {
  // All values are null.
  std::vector<EncodedStatistics> page_stats(3);
  page_stats.at(0).set_null_count(100).all_null_value = true;
  page_stats.at(1).set_null_count(100).all_null_value = true;
  page_stats.at(2).set_null_count(100).all_null_value = true;

  TestWriteTypedColumnIndex(schema::Int32("c1"), page_stats, BoundaryOrder::Unordered,
                            /*has_null_counts=*/true);
}

TEST(PageIndex, WriteColumnIndexWithInvalidNullCounts) {
  auto encode = [=](int32_t value) {
    return std::string(reinterpret_cast<const char*>(&value), sizeof(int32_t));
  };

  // Some pages do not provide null_count
  std::vector<EncodedStatistics> page_stats(3);
  page_stats.at(0).set_min(encode(1)).set_max(encode(2)).set_null_count(0);
  page_stats.at(1).set_min(encode(1)).set_max(encode(3));
  page_stats.at(2).set_min(encode(2)).set_max(encode(3)).set_null_count(0);

  TestWriteTypedColumnIndex(schema::Int32("c1"), page_stats, BoundaryOrder::Ascending,
                            /*has_null_counts=*/false);
}

TEST(PageIndex, WriteColumnIndexWithCorruptedStats) {
  auto encode = [=](int32_t value) {
    return std::string(reinterpret_cast<const char*>(&value), sizeof(int32_t));
  };

  // 2nd page does not set anything
  std::vector<EncodedStatistics> page_stats(3);
  page_stats.at(0).set_min(encode(1)).set_max(encode(2));
  page_stats.at(2).set_min(encode(3)).set_max(encode(4));

  ColumnDescriptor descr(schema::Int32("c1"), /*max_definition_level=*/1, 0);
  auto builder = ColumnIndexBuilder::Make(&descr);
  for (const auto& stats : page_stats) {
    builder->AddPage(stats, SizeStatistics());
  }
  ASSERT_NO_THROW(builder->Finish());
  ASSERT_EQ(nullptr, builder->Build());

  auto sink = CreateOutputStream();
  builder->WriteTo(sink.get());
  PARQUET_ASSIGN_OR_THROW(auto buffer, sink->Finish());
  EXPECT_EQ(0, buffer->size());
}

TEST(PageIndex, WriteInt64ColumnIndexWithSizeStats) {
  auto encode = [=](int64_t value) {
    return std::string(reinterpret_cast<const char*>(&value), sizeof(int64_t));
  };

  // Integer values in the descending order.
  std::vector<EncodedStatistics> page_stats(3);
  page_stats.at(0).set_null_count(4).set_min(encode(-1)).set_max(encode(-2));
  page_stats.at(1).set_null_count(0).set_min(encode(-2)).set_max(encode(-3));
  page_stats.at(2).set_null_count(4).set_min(encode(-3)).set_max(encode(-4));

  // Page level histograms.
  std::vector<PageLevelHistogram> page_levels;
  page_levels.push_back(
      PageLevelHistogram{/*def_levels=*/{2, 4, 6, 8}, /*rep_levels=*/{10, 5, 5}});
  page_levels.push_back(
      PageLevelHistogram{/*def_levels=*/{1, 3, 5, 7}, /*rep_levels=*/{4, 8, 4}});
  page_levels.push_back(
      PageLevelHistogram{/*def_levels=*/{0, 2, 4, 6}, /*rep_levels=*/{3, 4, 5}});

  TestWriteTypedColumnIndex(schema::Int64("c1"), page_stats, BoundaryOrder::Descending,
                            /*has_null_counts=*/true, /*max_definition_level=*/3,
                            /*max_repetition_level=*/2, page_levels);
}

TEST(PageIndex, TestPageIndexBuilderWithZeroRowGroup) {
  schema::NodeVector fields = {schema::Int32("c1"), schema::ByteArray("c2")};
  schema::NodePtr root = schema::GroupNode::Make("schema", Repetition::REPEATED, fields);
  SchemaDescriptor schema;
  schema.Init(root);

  auto builder = PageIndexBuilder::Make(&schema);

  // AppendRowGroup() is not called and expect throw.
  ASSERT_THROW(builder->GetColumnIndexBuilder(0), ParquetException);
  ASSERT_THROW(builder->GetOffsetIndexBuilder(0), ParquetException);

  // Finish the builder without calling AppendRowGroup().
  ASSERT_NO_THROW(builder->Finish());

  // Verify WriteTo does not write anything.
  auto sink = CreateOutputStream();
  auto write_result = builder->WriteTo(sink.get());
  PARQUET_ASSIGN_OR_THROW(auto buffer, sink->Finish());
  ASSERT_EQ(0, buffer->size());
  ASSERT_TRUE(write_result.column_index_locations.empty());
  ASSERT_TRUE(write_result.offset_index_locations.empty());
}

class PageIndexBuilderTest : public ::testing::Test {
 public:
  void WritePageIndexes(int num_row_groups, int num_columns,
                        const std::vector<std::vector<EncodedStatistics>>& page_stats,
                        const std::vector<std::vector<PageLocation>>& page_locations,
                        int final_position) {
    auto builder = PageIndexBuilder::Make(&schema_);
    for (int row_group = 0; row_group < num_row_groups; ++row_group) {
      ASSERT_NO_THROW(builder->AppendRowGroup());

      for (int column = 0; column < num_columns; ++column) {
        if (static_cast<size_t>(column) < page_stats[row_group].size()) {
          auto column_index_builder = builder->GetColumnIndexBuilder(column);
          ASSERT_NO_THROW(
              column_index_builder->AddPage(page_stats[row_group][column], {}));
          ASSERT_NO_THROW(column_index_builder->Finish());
        }

        if (static_cast<size_t>(column) < page_locations[row_group].size()) {
          auto offset_index_builder = builder->GetOffsetIndexBuilder(column);
          ASSERT_NO_THROW(offset_index_builder->AddPage(page_locations[row_group][column],
                                                        /*size_stats=*/{}));
          ASSERT_NO_THROW(offset_index_builder->Finish(final_position));
        }
      }
    }
    ASSERT_NO_THROW(builder->Finish());

    auto sink = CreateOutputStream();
    write_result_ = builder->WriteTo(sink.get());
    PARQUET_ASSIGN_OR_THROW(buffer_, sink->Finish());

    ASSERT_LE(write_result_.column_index_locations.size(),
              static_cast<size_t>(num_row_groups * num_columns));
    ASSERT_LE(write_result_.offset_index_locations.size(),
              static_cast<size_t>(num_row_groups * num_columns));
    ASSERT_GT(write_result_.column_index_locations.size(), size_t{0});
    ASSERT_GT(write_result_.offset_index_locations.size(), size_t{0});
  }

  void CheckColumnIndex(int row_group, int column, const EncodedStatistics& stats) {
    auto column_index = ReadColumnIndex(row_group, column);
    ASSERT_NE(nullptr, column_index);
    ASSERT_EQ(size_t{1}, column_index->null_pages().size());
    ASSERT_EQ(stats.all_null_value, column_index->null_pages()[0]);
    ASSERT_EQ(stats.min(), column_index->encoded_min_values()[0]);
    ASSERT_EQ(stats.max(), column_index->encoded_max_values()[0]);
    ASSERT_EQ(stats.has_null_count, column_index->has_null_counts());
    if (stats.has_null_count) {
      ASSERT_EQ(stats.null_count, column_index->null_counts()[0]);
    }
  }

  void CheckOffsetIndex(int row_group, int column, const PageLocation& expected_location,
                        int64_t final_location) {
    auto offset_index = ReadOffsetIndex(row_group, column);
    ASSERT_NE(nullptr, offset_index);
    ASSERT_EQ(size_t{1}, offset_index->page_locations().size());
    const auto& location = offset_index->page_locations()[0];
    ASSERT_EQ(expected_location.offset + final_location, location.offset);
    ASSERT_EQ(expected_location.compressed_page_size, location.compressed_page_size);
    ASSERT_EQ(expected_location.first_row_index, location.first_row_index);
  }

 protected:
  std::unique_ptr<ColumnIndex> ReadColumnIndex(int row_group, int column) {
    const auto& column_index_locations = write_result_.column_index_locations;
    auto location = std::find_if(column_index_locations.begin(),
                                 column_index_locations.end(), [&](const auto& location) {
                                   return location.first.row_group_index == row_group &&
                                          location.first.column_index == column;
                                 });
    if (location == column_index_locations.end()) {
      return nullptr;
    }
    auto properties = default_reader_properties();
    return ColumnIndex::Make(*schema_.Column(column),
                             buffer_->data() + location->second.offset,
                             static_cast<uint32_t>(location->second.length), properties);
  }

  std::unique_ptr<OffsetIndex> ReadOffsetIndex(int row_group, int column) {
    const auto& offset_index_locations = write_result_.offset_index_locations;
    auto location = std::find_if(offset_index_locations.begin(),
                                 offset_index_locations.end(), [&](const auto& location) {
                                   return location.first.row_group_index == row_group &&
                                          location.first.column_index == column;
                                 });
    if (location == offset_index_locations.end()) {
      return nullptr;
    }
    auto properties = default_reader_properties();
    return OffsetIndex::Make(buffer_->data() + location->second.offset,
                             static_cast<uint32_t>(location->second.length), properties);
  }

  SchemaDescriptor schema_;
  std::shared_ptr<Buffer> buffer_;
  PageIndexBuilder::WriteResult write_result_;
};

TEST_F(PageIndexBuilderTest, SingleRowGroup) {
  schema::NodePtr root = schema::GroupNode::Make(
      "schema", Repetition::REPEATED,
      {schema::ByteArray("c1"), schema::ByteArray("c2"), schema::ByteArray("c3")});
  schema_.Init(root);

  // Prepare page stats and page locations for single row group.
  // Note that the 3rd column does not have any stats and its page index is disabled.
  const int num_row_groups = 1;
  const int num_columns = 3;
  const std::vector<std::vector<EncodedStatistics>> page_stats = {
      /*row_group_id=0*/
      {/*column_id=0*/ EncodedStatistics().set_null_count(0).set_min("a").set_max("b"),
       /*column_id=1*/ EncodedStatistics().set_null_count(0).set_min("A").set_max("B")}};
  const std::vector<std::vector<PageLocation>> page_locations = {
      /*row_group_id=0*/
      {/*column_id=0*/ {/*offset=*/128, /*compressed_page_size=*/512,
                        /*first_row_index=*/0},
       /*column_id=1*/ {/*offset=*/1024, /*compressed_page_size=*/512,
                        /*first_row_index=*/0}}};
  const int64_t final_position = 200;

  WritePageIndexes(num_row_groups, num_columns, page_stats, page_locations,
                   final_position);

  // Verify that first two columns have good page indexes.
  for (int column = 0; column < 2; ++column) {
    CheckColumnIndex(/*row_group=*/0, column, page_stats[0][column]);
    CheckOffsetIndex(/*row_group=*/0, column, page_locations[0][column], final_position);
  }

  // Verify the 3rd column does not have page indexes.
  ASSERT_EQ(nullptr, ReadColumnIndex(/*row_group=*/0, /*column=*/2));
  ASSERT_EQ(nullptr, ReadOffsetIndex(/*row_group=*/0, /*column=*/2));
}

TEST_F(PageIndexBuilderTest, TwoRowGroups) {
  schema::NodePtr root = schema::GroupNode::Make(
      "schema", Repetition::REPEATED, {schema::ByteArray("c1"), schema::ByteArray("c2")});
  schema_.Init(root);

  // Prepare page stats and page locations for two row groups.
  // Note that the 2nd column in the 2nd row group has corrupted stats.
  const int num_row_groups = 2;
  const int num_columns = 2;
  const std::vector<std::vector<EncodedStatistics>> page_stats = {
      /*row_group_id=0*/
      {/*column_id=0*/ EncodedStatistics().set_min("a").set_max("b"),
       /*column_id=1*/ EncodedStatistics().set_null_count(0).set_min("A").set_max("B")},
      /*row_group_id=1*/
      {/*column_id=0*/ EncodedStatistics() /* corrupted stats */,
       /*column_id=1*/ EncodedStatistics().set_null_count(0).set_min("bar").set_max(
           "foo")}};
  const std::vector<std::vector<PageLocation>> page_locations = {
      /*row_group_id=0*/
      {/*column_id=0*/ {/*offset=*/128, /*compressed_page_size=*/512,
                        /*first_row_index=*/0},
       /*column_id=1*/ {/*offset=*/1024, /*compressed_page_size=*/512,
                        /*first_row_index=*/0}},
      /*row_group_id=0*/
      {/*column_id=0*/ {/*offset=*/128, /*compressed_page_size=*/512,
                        /*first_row_index=*/0},
       /*column_id=1*/ {/*offset=*/1024, /*compressed_page_size=*/512,
                        /*first_row_index=*/0}}};
  const int64_t final_position = 200;

  WritePageIndexes(num_row_groups, num_columns, page_stats, page_locations,
                   final_position);

  // Verify that all columns have good column indexes except the 2nd column in the 2nd row
  // group.
  CheckColumnIndex(/*row_group=*/0, /*column=*/0, page_stats[0][0]);
  CheckColumnIndex(/*row_group=*/0, /*column=*/1, page_stats[0][1]);
  CheckColumnIndex(/*row_group=*/1, /*column=*/1, page_stats[1][1]);
  ASSERT_EQ(nullptr, ReadColumnIndex(/*row_group=*/1, /*column=*/0));

  // Verify that two columns have good offset indexes.
  CheckOffsetIndex(/*row_group=*/0, /*column=*/0, page_locations[0][0], final_position);
  CheckOffsetIndex(/*row_group=*/0, /*column=*/1, page_locations[0][1], final_position);
  CheckOffsetIndex(/*row_group=*/1, /*column=*/0, page_locations[1][0], final_position);
  CheckOffsetIndex(/*row_group=*/1, /*column=*/1, page_locations[1][1], final_position);
}

// ============================================================================
// Tests for ColumnIndex::FilterPages() predicate evaluation
// ============================================================================
//
// These tests exercise all ten specified cases (plus an all-types parametrized
// sweep) for the FilterPages() method introduced on TypedColumnIndexImpl.
//
// Helper infrastructure
// ---------------------
//
// BuildColumnIndex(node, page_stats) – creates a ColumnIndex from a schema
//   node and a vector of EncodedStatistics, using the existing
//   ColumnIndexBuilder path.
//
// BuildOffsetIndex(page_locations) – creates an OffsetIndex from an explicit
//   vector of PageLocations via OffsetIndexBuilder::Make(); Finish(0) is used
//   so that the stored offsets equal the supplied values verbatim.

namespace {

/// Build a ColumnIndex from a schema node and per-page EncodedStatistics.
/// Returns both the ColumnIndex and the ColumnDescriptor that backs it.
/// The ColumnDescriptor MUST outlive the ColumnIndex (TypedColumnIndexImpl stores a
/// raw pointer to it).
std::pair<std::unique_ptr<ColumnIndex>, std::unique_ptr<ColumnDescriptor>>
BuildColumnIndex(schema::NodePtr node, const std::vector<EncodedStatistics>& page_stats,
                 int16_t max_definition_level = 1, int16_t max_repetition_level = 0) {
  auto descr = std::make_unique<ColumnDescriptor>(std::move(node), max_definition_level,
                                                  max_repetition_level);
  auto builder = ColumnIndexBuilder::Make(descr.get());
  for (const auto& stats : page_stats) {
    builder->AddPage(stats, SizeStatistics());
  }
  builder->Finish();
  return {builder->Build(), std::move(descr)};
}

/// Build an OffsetIndex from an explicit vector of PageLocations.
/// Finish(0) is called so offsets are stored as-is.
std::unique_ptr<OffsetIndex> BuildOffsetIndex(
    const std::vector<PageLocation>& page_locations) {
  auto builder = OffsetIndexBuilder::Make();
  for (const auto& loc : page_locations) {
    builder->AddPage(loc.offset, loc.compressed_page_size, loc.first_row_index);
  }
  builder->Finish(/*final_position=*/0);
  return builder->Build();
}

/// Encode a plain (non-FLBA, non-ByteArray) scalar value to a raw byte string
/// suitable for use in EncodedStatistics.  The column descriptor is needed
/// for Bool which uses a bit-packed representation.
template <typename DType>
std::string EncodePlainValue(typename DType::c_type value,
                             const ColumnDescriptor* descr = nullptr) {
  auto encoder =
      MakeTypedEncoder<DType>(Encoding::PLAIN, /*use_dictionary=*/false, descr);
  encoder->Put(&value, 1);
  auto buf = encoder->FlushValues();
  return std::string(reinterpret_cast<const char*>(buf->data()),
                     static_cast<size_t>(buf->size()));
}

/// Specialisation for ByteArray: the plain encoding for statistics is just the
/// raw bytes of the value (length is implicit).
template <>
std::string EncodePlainValue<ByteArrayType>(ByteArray value, const ColumnDescriptor*) {
  return std::string(reinterpret_cast<const char*>(value.ptr),
                     static_cast<size_t>(value.len));
}

/// Specialisation for FLBA: the plain encoding for statistics is just the raw
/// bytes (type_length bytes, no length prefix).
template <>
std::string EncodePlainValue<FLBAType>(FLBA value, const ColumnDescriptor* descr) {
  int32_t len = descr ? descr->type_length() : FLBA_LENGTH;
  return std::string(reinterpret_cast<const char*>(value.ptr), static_cast<size_t>(len));
}

}  // namespace

// ---------------------------------------------------------------------------
// Fixture: TestColumnIndexFilterPages<TestType>
//
// TYPED_TEST_SUITE is instantiated over all 8 ParquetTypes so that case 3
// (AllTypes) runs for each.  Type-specific cases (Int32Predicate15,
// FloatPredicateGT) only execute meaningful logic for the matching physical
// type; all other types in the parametrized run exercise the "AllTypes" path
// instead.
// ---------------------------------------------------------------------------

template <typename TestType>
class TestColumnIndexFilterPages : public test::PrimitiveTypedTest<TestType> {
 public:
  using T = typename TestType::c_type;

 protected:
  // Build a ColumnDescriptor for the test type.  FLBA uses FLBA_LENGTH bytes.
  std::unique_ptr<ColumnDescriptor> MakeDescriptor() {
    auto node =
        schema::PrimitiveNode::Make("col", Repetition::OPTIONAL, TestType::type_num,
                                    ConvertedType::NONE, FLBA_LENGTH);
    return std::make_unique<ColumnDescriptor>(std::move(node), /*max_def=*/1,
                                              /*max_rep=*/0);
  }

  std::string Encode(T value) {
    auto descr = MakeDescriptor();
    return EncodePlainValue<TestType>(value, descr.get());
  }

  // Build min/max stats for a single page.
  EncodedStatistics MakeStats(T min_val, T max_val) {
    EncodedStatistics stats;
    stats.set_min(Encode(min_val)).set_max(Encode(max_val)).set_null_count(0);
    return stats;
  }

  // Build an all-null page marker.
  EncodedStatistics MakeNullStats() {
    EncodedStatistics stats;
    stats.all_null_value = true;
    stats.set_null_count(100);
    return stats;
  }
};

TYPED_TEST_SUITE(TestColumnIndexFilterPages, test::ParquetTypes);

// ---------------------------------------------------------------------------
// Case 1 – Int32 pages [1,10][11,20][21,30], value=15 → skip pages 0 & 2,
//           select page 1.
//
// This test only runs substantively for Int32Type; all other types in the
// parametrized sweep pass trivially.
// ---------------------------------------------------------------------------
TYPED_TEST(TestColumnIndexFilterPages, Int32Predicate15) {
  if constexpr (!std::is_same_v<TypeParam, Int32Type>) {
    GTEST_SKIP() << "Case 1 is Int32-specific; tested via AllTypes for other types";
  }

  // Three pages: [1,10], [11,20], [21,30].
  auto node = schema::Int32("col");
  std::vector<EncodedStatistics> page_stats(3);
  auto encode = [](int32_t v) {
    return std::string(reinterpret_cast<const char*>(&v), sizeof(int32_t));
  };
  page_stats[0].set_min(encode(1)).set_max(encode(10)).set_null_count(0);
  page_stats[1].set_min(encode(11)).set_max(encode(20)).set_null_count(0);
  page_stats[2].set_min(encode(21)).set_max(encode(30)).set_null_count(0);

  auto [col_index, col_descr] = BuildColumnIndex(node, page_stats);
  ASSERT_NE(nullptr, col_index);

  // OffsetIndex: 3 pages, first_row_index = 0, 100, 200; total rows = 300.
  std::vector<PageLocation> page_locs = {
      {/*offset=*/0, /*size=*/100, /*first_row=*/0},
      {/*offset=*/100, /*size=*/100, /*first_row=*/100},
      {/*offset=*/200, /*size=*/100, /*first_row=*/200},
  };
  auto off_index = BuildOffsetIndex(page_locs);
  ASSERT_NE(nullptr, off_index);

  // EQ predicate for value=15 should skip pages 0 ([1,10]) and 2 ([21,30]),
  // and select page 1 ([11,20]).
  int32_t pred_value = 15;
  PARQUET_ASSIGN_OR_THROW(
      auto selection,
      col_index->FilterPages(std::any(pred_value), PredicateOp::EQ, *off_index,
                             /*row_group_row_count=*/300));

  ASSERT_EQ(3, selection.page_count());
  EXPECT_TRUE(selection.selector(0).skip);   // page 0 [1,10]  – skip
  EXPECT_FALSE(selection.selector(1).skip);  // page 1 [11,20] – select
  EXPECT_TRUE(selection.selector(2).skip);   // page 2 [21,30] – skip

  // Verify row counts derived from OffsetIndex.
  EXPECT_EQ(100, selection.selector(0).row_count);
  EXPECT_EQ(100, selection.selector(1).row_count);
  EXPECT_EQ(100, selection.selector(2).row_count);
}

// ---------------------------------------------------------------------------
// Case 2 – Float pages value > 7.0.
//   Page 0: [0.0, 5.0]  → max(5.0) < 7.0  → skip  (GT: skip if max < value)
//   Page 1: [6.0, 10.0] → max(10.0) >= 7.0 → select
//
// Only runs substantively for FloatType.
// ---------------------------------------------------------------------------
TYPED_TEST(TestColumnIndexFilterPages, FloatPredicateGT) {
  if constexpr (!std::is_same_v<TypeParam, FloatType>) {
    GTEST_SKIP() << "Case 2 is Float-specific; tested via AllTypes for other types";
  }

  auto encode_f = [](float v) {
    return std::string(reinterpret_cast<const char*>(&v), sizeof(float));
  };

  auto node = schema::Float("col");
  std::vector<EncodedStatistics> page_stats(2);
  page_stats[0].set_min(encode_f(0.0f)).set_max(encode_f(5.0f)).set_null_count(0);
  page_stats[1].set_min(encode_f(6.0f)).set_max(encode_f(10.0f)).set_null_count(0);

  auto [col_index, col_descr] = BuildColumnIndex(node, page_stats);
  ASSERT_NE(nullptr, col_index);

  std::vector<PageLocation> page_locs = {
      {0, 100, 0},
      {100, 100, 50},
  };
  auto off_index = BuildOffsetIndex(page_locs);
  ASSERT_NE(nullptr, off_index);

  float pred_value = 7.0f;
  PARQUET_ASSIGN_OR_THROW(
      auto selection,
      col_index->FilterPages(std::any(pred_value), PredicateOp::GT, *off_index,
                             /*row_group_row_count=*/100));

  ASSERT_EQ(2, selection.page_count());
  EXPECT_TRUE(selection.selector(0).skip);   // page 0: max=5.0 < 7.0 → skip
  EXPECT_FALSE(selection.selector(1).skip);  // page 1: max=10.0 >= 7.0 → select
}

// ---------------------------------------------------------------------------
// Case 3 – AllTypes: one typed ColumnIndex with three pages; predicate selects
//           only the middle page.  Runs for all 8 ParquetTypes.
//
// For each type we construct:
//   page 0: [lo, lo]   – EQ mid skips (lo ≠ mid)
//   page 1: [mid, mid] – EQ mid selects
//   page 2: [hi, hi]   – EQ mid skips (hi ≠ mid)
// ---------------------------------------------------------------------------

TYPED_TEST(TestColumnIndexFilterPages, AllTypes) {
  using T = typename TypeParam::c_type;

  if constexpr (std::is_same_v<TypeParam, Int96Type>) {
    GTEST_SKIP() << "Int96 has no defined sort order; FilterPages is not supported";
  }

  // We need three distinct "ordered" values for the type.  All numeric types
  // have an obvious ordering.  For ByteArray we use "a"/"b"/"c".  For FLBA we
  // use 12-byte strings padded with zeros.
  auto descr = this->MakeDescriptor();

  // lo < mid < hi values, type-specific.
  T lo{}, mid{}, hi{};

  if constexpr (std::is_same_v<TypeParam, BooleanType>) {
    // Bool only has two values; make page 0 = [false,false], page 1 = [true,true],
    // page 2 = all-null (to avoid needing a third distinct value).
    auto encode_bool = [&](bool v) {
      return EncodePlainValue<BooleanType>(v, descr.get());
    };

    std::vector<EncodedStatistics> page_stats(3);
    page_stats[0]
        .set_min(encode_bool(false))
        .set_max(encode_bool(false))
        .set_null_count(0);
    page_stats[1].set_min(encode_bool(true)).set_max(encode_bool(true)).set_null_count(0);
    page_stats[2].all_null_value = true;
    page_stats[2].set_null_count(50);

    auto node = schema::Boolean("col");
    auto [col_index, col_descr] = BuildColumnIndex(node, page_stats);
    ASSERT_NE(nullptr, col_index);

    std::vector<PageLocation> page_locs = {{0, 50, 0}, {50, 50, 50}, {100, 50, 100}};
    auto off_index = BuildOffsetIndex(page_locs);
    ASSERT_NE(nullptr, off_index);

    // EQ true → only page 1 qualifies; page 2 is all-null → skipped by EQ.
    bool pred_val = true;
    PARQUET_ASSIGN_OR_THROW(
        auto sel, col_index->FilterPages(std::any(pred_val), PredicateOp::EQ, *off_index,
                                         /*row_group_row_count=*/150));
    ASSERT_EQ(3, sel.page_count());
    EXPECT_TRUE(sel.selector(0).skip);   // [false,false] does not contain true
    EXPECT_FALSE(sel.selector(1).skip);  // [true,true] contains true
    EXPECT_TRUE(sel.selector(2).skip);   // all-null page → skip for EQ
    return;

  } else if constexpr (std::is_same_v<TypeParam, Int32Type>) {
    lo = 1;
    mid = 5;
    hi = 10;
  } else if constexpr (std::is_same_v<TypeParam, Int64Type>) {
    lo = 100L;
    mid = 500L;
    hi = 1000L;
  } else if constexpr (std::is_same_v<TypeParam, Int96Type>) {
    lo = Int96{{1, 0, 0}};
    mid = Int96{{5, 0, 0}};
    hi = Int96{{10, 0, 0}};
  } else if constexpr (std::is_same_v<TypeParam, FloatType>) {
    lo = 1.0f;
    mid = 5.0f;
    hi = 10.0f;
  } else if constexpr (std::is_same_v<TypeParam, DoubleType>) {
    lo = 1.0;
    mid = 5.0;
    hi = 10.0;
  } else if constexpr (std::is_same_v<TypeParam, ByteArrayType>) {
    // ByteArray comparison is lexicographic on raw bytes.
    lo = ByteArray{std::string_view{"a"}};
    mid = ByteArray{std::string_view{"e"}};
    hi = ByteArray{std::string_view{"z"}};
  } else if constexpr (std::is_same_v<TypeParam, FLBAType>) {
    // FLBA: use FLBA_LENGTH=12 byte buffers.  lo < mid < hi lexicographically.
    static const uint8_t lo_bytes[FLBA_LENGTH] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    static const uint8_t mid_bytes[FLBA_LENGTH] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5};
    static const uint8_t hi_bytes[FLBA_LENGTH] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10};
    lo = FLBA{lo_bytes};
    mid = FLBA{mid_bytes};
    hi = FLBA{hi_bytes};
  }

  // Encode values using the column descriptor (needed for Bool, FLBA).
  auto encode = [&](T v) { return EncodePlainValue<TypeParam>(v, descr.get()); };

  // page 0: [lo, lo]   → does not contain mid
  // page 1: [mid, mid] → contains mid  (selected by EQ mid)
  // page 2: [hi, hi]   → does not contain mid
  std::vector<EncodedStatistics> page_stats(3);
  page_stats[0].set_min(encode(lo)).set_max(encode(lo)).set_null_count(0);
  page_stats[1].set_min(encode(mid)).set_max(encode(mid)).set_null_count(0);
  page_stats[2].set_min(encode(hi)).set_max(encode(hi)).set_null_count(0);

  auto node = schema::PrimitiveNode::Make(
      "col", Repetition::OPTIONAL, TypeParam::type_num, ConvertedType::NONE, FLBA_LENGTH);
  auto [col_index, col_descr] = BuildColumnIndex(node, page_stats);
  ASSERT_NE(nullptr, col_index);

  std::vector<PageLocation> page_locs = {{0, 100, 0}, {100, 100, 100}, {200, 100, 200}};
  auto off_index = BuildOffsetIndex(page_locs);
  ASSERT_NE(nullptr, off_index);

  PARQUET_ASSIGN_OR_THROW(
      auto sel, col_index->FilterPages(std::any(mid), PredicateOp::EQ, *off_index,
                                       /*row_group_row_count=*/300));

  ASSERT_EQ(3, sel.page_count());
  EXPECT_TRUE(sel.selector(0).skip);   // page 0 has only lo → skip
  EXPECT_FALSE(sel.selector(1).skip);  // page 1 has mid → select
  EXPECT_TRUE(sel.selector(2).skip);   // page 2 has only hi → skip
}

// ---------------------------------------------------------------------------
// Case 4 – All-null page always skipped for value predicates.
// ---------------------------------------------------------------------------
TYPED_TEST(TestColumnIndexFilterPages, AllNullPage) {
  using T = typename TypeParam::c_type;

  if constexpr (std::is_same_v<TypeParam, Int96Type>) {
    GTEST_SKIP() << "Int96 has no defined sort order; FilterPages is not supported";
  }

  auto descr = this->MakeDescriptor();

  // For simplicity construct a T value that the column index builder can
  // encode.  Use a per-type constant.
  T some_value{};
  if constexpr (std::is_same_v<TypeParam, Int32Type>) {
    some_value = 42;
  } else if constexpr (std::is_same_v<TypeParam, Int64Type>) {
    some_value = 42L;
  } else if constexpr (std::is_same_v<TypeParam, Int96Type>) {
    some_value = Int96{{42, 0, 0}};
  } else if constexpr (std::is_same_v<TypeParam, FloatType>) {
    some_value = 42.0f;
  } else if constexpr (std::is_same_v<TypeParam, DoubleType>) {
    some_value = 42.0;
  } else if constexpr (std::is_same_v<TypeParam, BooleanType>) {
    some_value = true;
  } else if constexpr (std::is_same_v<TypeParam, ByteArrayType>) {
    some_value = ByteArray{std::string_view{"x"}};
  } else if constexpr (std::is_same_v<TypeParam, FLBAType>) {
    static const uint8_t buf[FLBA_LENGTH] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    some_value = FLBA{buf};
  }

  auto encode = [&](T v) { return EncodePlainValue<TypeParam>(v, descr.get()); };

  // Two pages: page 0 is all-null, page 1 has data.
  std::vector<EncodedStatistics> page_stats(2);
  page_stats[0].all_null_value = true;
  page_stats[0].set_null_count(100);
  page_stats[1].set_min(encode(some_value)).set_max(encode(some_value)).set_null_count(0);

  auto node = schema::PrimitiveNode::Make(
      "col", Repetition::OPTIONAL, TypeParam::type_num, ConvertedType::NONE, FLBA_LENGTH);
  auto [col_index, col_descr] = BuildColumnIndex(node, page_stats);
  ASSERT_NE(nullptr, col_index);

  std::vector<PageLocation> page_locs = {{0, 50, 0}, {50, 50, 100}};
  auto off_index = BuildOffsetIndex(page_locs);
  ASSERT_NE(nullptr, off_index);

  // EQ predicate: all-null page must be skipped, non-null page selected
  // (since min==max==some_value matches EQ some_value).
  PARQUET_ASSIGN_OR_THROW(
      auto sel, col_index->FilterPages(std::any(some_value), PredicateOp::EQ, *off_index,
                                       /*row_group_row_count=*/200));

  ASSERT_EQ(2, sel.page_count());
  EXPECT_TRUE(sel.selector(0).skip);   // all-null page → must skip
  EXPECT_FALSE(sel.selector(1).skip);  // non-null page containing some_value → select

  // Also verify IS_NULL on the all-null page: must NOT skip.
  PARQUET_ASSIGN_OR_THROW(
      auto sel_null, col_index->FilterPages(std::any{}, PredicateOp::IS_NULL, *off_index,
                                            /*row_group_row_count=*/200));

  ASSERT_EQ(2, sel_null.page_count());
  EXPECT_FALSE(sel_null.selector(0).skip);  // all-null page → keep for IS_NULL
  EXPECT_FALSE(sel_null.selector(1).skip);  // non-null page → conservatively keep
}

// ---------------------------------------------------------------------------
// Case 5 – Row counts derived from OffsetIndex first_row_index deltas.
//   first_row_index = [0, 100, 250]; total rows = 400
//   Expected row counts: page 0 → 100, page 1 → 150, page 2 → 150
// ---------------------------------------------------------------------------
TYPED_TEST(TestColumnIndexFilterPages, RowCountsFromOffsetIndex) {
  using T = typename TypeParam::c_type;

  auto descr = this->MakeDescriptor();

  T some_value{};
  if constexpr (std::is_same_v<TypeParam, Int32Type>) {
    some_value = 1;
  } else if constexpr (std::is_same_v<TypeParam, Int64Type>) {
    some_value = 1L;
  } else if constexpr (std::is_same_v<TypeParam, Int96Type>) {
    some_value = Int96{{1, 0, 0}};
  } else if constexpr (std::is_same_v<TypeParam, FloatType>) {
    some_value = 1.0f;
  } else if constexpr (std::is_same_v<TypeParam, DoubleType>) {
    some_value = 1.0;
  } else if constexpr (std::is_same_v<TypeParam, BooleanType>) {
    some_value = false;
  } else if constexpr (std::is_same_v<TypeParam, ByteArrayType>) {
    some_value = ByteArray{std::string_view{"a"}};
  } else if constexpr (std::is_same_v<TypeParam, FLBAType>) {
    static const uint8_t buf[FLBA_LENGTH] = {};
    some_value = FLBA{buf};
  }

  auto encode = [&](T v) { return EncodePlainValue<TypeParam>(v, descr.get()); };
  auto encoded_val = encode(some_value);

  // Three pages, all with the same min/max == some_value (IS_NOT_NULL selects all).
  std::vector<EncodedStatistics> page_stats(3);
  for (int i = 0; i < 3; ++i) {
    page_stats[i].set_min(encoded_val).set_max(encoded_val).set_null_count(0);
  }

  auto node = schema::PrimitiveNode::Make(
      "col", Repetition::OPTIONAL, TypeParam::type_num, ConvertedType::NONE, FLBA_LENGTH);
  auto [col_index, col_descr] = BuildColumnIndex(node, page_stats);
  ASSERT_NE(nullptr, col_index);

  // first_row_index: [0, 100, 250]; total = 400.
  std::vector<PageLocation> page_locs = {
      {0, 100, 0},
      {100, 100, 100},
      {200, 100, 250},
  };
  auto off_index = BuildOffsetIndex(page_locs);
  ASSERT_NE(nullptr, off_index);

  // IS_NOT_NULL selects all pages; use it to get RowSelection with correct row counts.
  PARQUET_ASSIGN_OR_THROW(
      auto sel, col_index->FilterPages(std::any{}, PredicateOp::IS_NOT_NULL, *off_index,
                                       /*row_group_row_count=*/400));

  ASSERT_EQ(3, sel.page_count());
  EXPECT_EQ(100, sel.selector(0).row_count);  // 100 - 0 = 100
  EXPECT_EQ(150, sel.selector(1).row_count);  // 250 - 100 = 150
  EXPECT_EQ(150, sel.selector(2).row_count);  // 400 - 250 = 150
  EXPECT_EQ(400, sel.row_count());
}

// ---------------------------------------------------------------------------
// Case 6 – Predicate overlaps every page → select all (no page skipped).
// ---------------------------------------------------------------------------
TYPED_TEST(TestColumnIndexFilterPages, PredicateOverlapsAll) {
  using T = typename TypeParam::c_type;

  if constexpr (std::is_same_v<TypeParam, Int96Type>) {
    GTEST_SKIP() << "Int96 has no defined sort order; FilterPages is not supported";
  }

  auto descr = this->MakeDescriptor();

  // Construct lo <= predicate <= hi per type.
  T lo{}, pred{}, hi{};
  if constexpr (std::is_same_v<TypeParam, Int32Type>) {
    lo = 1;
    pred = 50;
    hi = 100;
  } else if constexpr (std::is_same_v<TypeParam, Int64Type>) {
    lo = 1L;
    pred = 50L;
    hi = 100L;
  } else if constexpr (std::is_same_v<TypeParam, Int96Type>) {
    lo = Int96{{1, 0, 0}};
    pred = Int96{{50, 0, 0}};
    hi = Int96{{100, 0, 0}};
  } else if constexpr (std::is_same_v<TypeParam, FloatType>) {
    lo = 1.0f;
    pred = 50.0f;
    hi = 100.0f;
  } else if constexpr (std::is_same_v<TypeParam, DoubleType>) {
    lo = 1.0;
    pred = 50.0;
    hi = 100.0;
  } else if constexpr (std::is_same_v<TypeParam, BooleanType>) {
    // For Bool use IS_NOT_NULL which always selects non-null pages.
    lo = false;
    pred = false;
    hi = true;
  } else if constexpr (std::is_same_v<TypeParam, ByteArrayType>) {
    lo = ByteArray{std::string_view{"a"}};
    pred = ByteArray{std::string_view{"m"}};
    hi = ByteArray{std::string_view{"z"}};
  } else if constexpr (std::is_same_v<TypeParam, FLBAType>) {
    static const uint8_t lo_b[FLBA_LENGTH] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    static const uint8_t hi_b[FLBA_LENGTH] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100};
    static const uint8_t pred_b[FLBA_LENGTH] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50};
    lo = FLBA{lo_b};
    hi = FLBA{hi_b};
    pred = FLBA{pred_b};
  }

  auto encode = [&](T v) { return EncodePlainValue<TypeParam>(v, descr.get()); };

  // Two pages, both covering [lo, hi].  pred is in [lo, hi] → EQ selects both.
  std::vector<EncodedStatistics> page_stats(2);
  page_stats[0].set_min(encode(lo)).set_max(encode(hi)).set_null_count(0);
  page_stats[1].set_min(encode(lo)).set_max(encode(hi)).set_null_count(0);

  auto node = schema::PrimitiveNode::Make(
      "col", Repetition::OPTIONAL, TypeParam::type_num, ConvertedType::NONE, FLBA_LENGTH);
  auto [col_index, col_descr] = BuildColumnIndex(node, page_stats);
  ASSERT_NE(nullptr, col_index);

  std::vector<PageLocation> page_locs = {{0, 100, 0}, {100, 100, 100}};
  auto off_index = BuildOffsetIndex(page_locs);
  ASSERT_NE(nullptr, off_index);

  PredicateOp op =
      std::is_same_v<TypeParam, BooleanType> ? PredicateOp::IS_NOT_NULL : PredicateOp::EQ;

  PARQUET_ASSIGN_OR_THROW(auto sel, col_index->FilterPages(std::any(pred), op, *off_index,
                                                           /*row_group_row_count=*/200));

  ASSERT_EQ(2, sel.page_count());
  EXPECT_FALSE(sel.selector(0).skip);  // pred in [lo,hi] → select
  EXPECT_FALSE(sel.selector(1).skip);  // pred in [lo,hi] → select
}

// ---------------------------------------------------------------------------
// Case 7 – Predicate overlaps no page → skip all.
// ---------------------------------------------------------------------------
TYPED_TEST(TestColumnIndexFilterPages, PredicateOverlapsNone) {
  using T = typename TypeParam::c_type;

  if constexpr (std::is_same_v<TypeParam, Int96Type>) {
    GTEST_SKIP() << "Int96 has no defined sort order; FilterPages is not supported";
  }

  if constexpr (std::is_same_v<TypeParam, BooleanType>) {
    // Bool only has two distinct values; EQ on a value not in [false,false]
    // range. Use true as pred against pages with [false,false].
    auto encode_bool = [](bool v) { return EncodePlainValue<BooleanType>(v, nullptr); };
    auto node = schema::Boolean("col");
    std::vector<EncodedStatistics> page_stats(2);
    page_stats[0]
        .set_min(encode_bool(false))
        .set_max(encode_bool(false))
        .set_null_count(0);
    page_stats[1]
        .set_min(encode_bool(false))
        .set_max(encode_bool(false))
        .set_null_count(0);

    auto [col_index, col_descr] = BuildColumnIndex(node, page_stats);
    ASSERT_NE(nullptr, col_index);

    std::vector<PageLocation> page_locs = {{0, 100, 0}, {100, 100, 100}};
    auto off_index = BuildOffsetIndex(page_locs);
    ASSERT_NE(nullptr, off_index);

    bool pred_val = true;
    PARQUET_ASSIGN_OR_THROW(
        auto sel, col_index->FilterPages(std::any(pred_val), PredicateOp::EQ, *off_index,
                                         /*row_group_row_count=*/200));

    ASSERT_EQ(2, sel.page_count());
    EXPECT_TRUE(sel.selector(0).skip);
    EXPECT_TRUE(sel.selector(1).skip);
    return;
  }

  auto descr = this->MakeDescriptor();

  // All pages contain values in [lo, hi]; pred is strictly above hi.
  T lo{}, hi{}, pred_above{};
  if constexpr (std::is_same_v<TypeParam, Int32Type>) {
    lo = 1;
    hi = 10;
    pred_above = 100;
  } else if constexpr (std::is_same_v<TypeParam, Int64Type>) {
    lo = 1L;
    hi = 10L;
    pred_above = 100L;
  } else if constexpr (std::is_same_v<TypeParam, Int96Type>) {
    lo = Int96{{1, 0, 0}};
    hi = Int96{{10, 0, 0}};
    pred_above = Int96{{100, 0, 0}};
  } else if constexpr (std::is_same_v<TypeParam, FloatType>) {
    lo = 1.0f;
    hi = 10.0f;
    pred_above = 100.0f;
  } else if constexpr (std::is_same_v<TypeParam, DoubleType>) {
    lo = 1.0;
    hi = 10.0;
    pred_above = 100.0;
  } else if constexpr (std::is_same_v<TypeParam, ByteArrayType>) {
    lo = ByteArray{std::string_view{"a"}};
    hi = ByteArray{std::string_view{"b"}};
    pred_above = ByteArray{std::string_view{"z"}};
  } else if constexpr (std::is_same_v<TypeParam, FLBAType>) {
    static const uint8_t lo_b[FLBA_LENGTH] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    static const uint8_t hi_b[FLBA_LENGTH] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10};
    static const uint8_t above_b[FLBA_LENGTH] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100};
    lo = FLBA{lo_b};
    hi = FLBA{hi_b};
    pred_above = FLBA{above_b};
  }

  auto encode = [&](T v) { return EncodePlainValue<TypeParam>(v, descr.get()); };

  std::vector<EncodedStatistics> page_stats(2);
  page_stats[0].set_min(encode(lo)).set_max(encode(hi)).set_null_count(0);
  page_stats[1].set_min(encode(lo)).set_max(encode(hi)).set_null_count(0);

  auto node = schema::PrimitiveNode::Make(
      "col", Repetition::OPTIONAL, TypeParam::type_num, ConvertedType::NONE, FLBA_LENGTH);
  auto [col_index, col_descr] = BuildColumnIndex(node, page_stats);
  ASSERT_NE(nullptr, col_index);

  std::vector<PageLocation> page_locs = {{0, 100, 0}, {100, 100, 100}};
  auto off_index = BuildOffsetIndex(page_locs);
  ASSERT_NE(nullptr, off_index);

  // EQ pred_above: all pages have max <= hi < pred_above → all skipped.
  PARQUET_ASSIGN_OR_THROW(
      auto sel, col_index->FilterPages(std::any(pred_above), PredicateOp::EQ, *off_index,
                                       /*row_group_row_count=*/200));

  ASSERT_EQ(2, sel.page_count());
  EXPECT_TRUE(sel.selector(0).skip);
  EXPECT_TRUE(sel.selector(1).skip);
}

// ---------------------------------------------------------------------------
// Case 8 – Single page: match and non-match.
// ---------------------------------------------------------------------------
TYPED_TEST(TestColumnIndexFilterPages, SinglePageMatch) {
  using T = typename TypeParam::c_type;

  if constexpr (std::is_same_v<TypeParam, Int96Type>) {
    GTEST_SKIP() << "Int96 has no defined sort order; FilterPages is not supported";
  }

  auto descr = this->MakeDescriptor();

  T in_range{}, out_of_range{};
  if constexpr (std::is_same_v<TypeParam, Int32Type>) {
    in_range = 5;
    out_of_range = 99;
  } else if constexpr (std::is_same_v<TypeParam, Int64Type>) {
    in_range = 5L;
    out_of_range = 99L;
  } else if constexpr (std::is_same_v<TypeParam, Int96Type>) {
    in_range = Int96{{5, 0, 0}};
    out_of_range = Int96{{99, 0, 0}};
  } else if constexpr (std::is_same_v<TypeParam, FloatType>) {
    in_range = 5.0f;
    out_of_range = 99.0f;
  } else if constexpr (std::is_same_v<TypeParam, DoubleType>) {
    in_range = 5.0;
    out_of_range = 99.0;
  } else if constexpr (std::is_same_v<TypeParam, BooleanType>) {
    in_range = true;
    out_of_range = false;  // [true,true] does not contain false
  } else if constexpr (std::is_same_v<TypeParam, ByteArrayType>) {
    in_range = ByteArray{std::string_view{"e"}};
    out_of_range = ByteArray{std::string_view{"z"}};
  } else if constexpr (std::is_same_v<TypeParam, FLBAType>) {
    static const uint8_t in_b[FLBA_LENGTH] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5};
    static const uint8_t out_b[FLBA_LENGTH] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 99};
    in_range = FLBA{in_b};
    out_of_range = FLBA{out_b};
  }

  auto encode = [&](T v) { return EncodePlainValue<TypeParam>(v, descr.get()); };

  // Single page [in_range, in_range].
  std::vector<EncodedStatistics> page_stats(1);
  page_stats[0].set_min(encode(in_range)).set_max(encode(in_range)).set_null_count(0);

  auto node = schema::PrimitiveNode::Make(
      "col", Repetition::OPTIONAL, TypeParam::type_num, ConvertedType::NONE, FLBA_LENGTH);
  auto [col_index, col_descr] = BuildColumnIndex(node, page_stats);
  ASSERT_NE(nullptr, col_index);

  std::vector<PageLocation> page_locs = {{0, 200, 0}};
  auto off_index = BuildOffsetIndex(page_locs);
  ASSERT_NE(nullptr, off_index);

  // --- Match: EQ in_range against [in_range,in_range] → select.
  PARQUET_ASSIGN_OR_THROW(
      auto sel_match,
      col_index->FilterPages(std::any(in_range), PredicateOp::EQ, *off_index,
                             /*row_group_row_count=*/200));

  ASSERT_EQ(1, sel_match.page_count());
  EXPECT_FALSE(sel_match.selector(0).skip);
  EXPECT_EQ(200, sel_match.selector(0).row_count);

  // --- Non-match: EQ out_of_range against [in_range,in_range] → skip.
  PARQUET_ASSIGN_OR_THROW(
      auto sel_no_match,
      col_index->FilterPages(std::any(out_of_range), PredicateOp::EQ, *off_index,
                             /*row_group_row_count=*/200));

  ASSERT_EQ(1, sel_no_match.page_count());
  EXPECT_TRUE(sel_no_match.selector(0).skip);
}

// ---------------------------------------------------------------------------
// Case 9 – min == max per page (degenerate pages).
// ---------------------------------------------------------------------------
TYPED_TEST(TestColumnIndexFilterPages, MinEqualsMax) {
  using T = typename TypeParam::c_type;

  if constexpr (std::is_same_v<TypeParam, Int96Type>) {
    GTEST_SKIP() << "Int96 has no defined sort order; FilterPages is not supported";
  }

  auto descr = this->MakeDescriptor();

  T val_a{}, val_b{};
  if constexpr (std::is_same_v<TypeParam, Int32Type>) {
    val_a = 7;
    val_b = 42;
  } else if constexpr (std::is_same_v<TypeParam, Int64Type>) {
    val_a = 7L;
    val_b = 42L;
  } else if constexpr (std::is_same_v<TypeParam, Int96Type>) {
    val_a = Int96{{7, 0, 0}};
    val_b = Int96{{42, 0, 0}};
  } else if constexpr (std::is_same_v<TypeParam, FloatType>) {
    val_a = 7.0f;
    val_b = 42.0f;
  } else if constexpr (std::is_same_v<TypeParam, DoubleType>) {
    val_a = 7.0;
    val_b = 42.0;
  } else if constexpr (std::is_same_v<TypeParam, BooleanType>) {
    val_a = false;
    val_b = true;
  } else if constexpr (std::is_same_v<TypeParam, ByteArrayType>) {
    val_a = ByteArray{std::string_view{"g"}};
    val_b = ByteArray{std::string_view{"q"}};
  } else if constexpr (std::is_same_v<TypeParam, FLBAType>) {
    static const uint8_t a_b[FLBA_LENGTH] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7};
    static const uint8_t b_b[FLBA_LENGTH] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42};
    val_a = FLBA{a_b};
    val_b = FLBA{b_b};
  }

  auto encode = [&](T v) { return EncodePlainValue<TypeParam>(v, descr.get()); };

  // Two degenerate pages: page 0 = [val_a, val_a], page 1 = [val_b, val_b].
  std::vector<EncodedStatistics> page_stats(2);
  page_stats[0].set_min(encode(val_a)).set_max(encode(val_a)).set_null_count(0);
  page_stats[1].set_min(encode(val_b)).set_max(encode(val_b)).set_null_count(0);

  auto node = schema::PrimitiveNode::Make(
      "col", Repetition::OPTIONAL, TypeParam::type_num, ConvertedType::NONE, FLBA_LENGTH);
  auto [col_index, col_descr] = BuildColumnIndex(node, page_stats);
  ASSERT_NE(nullptr, col_index);

  std::vector<PageLocation> page_locs = {{0, 100, 0}, {100, 100, 100}};
  auto off_index = BuildOffsetIndex(page_locs);
  ASSERT_NE(nullptr, off_index);

  // EQ val_a: page 0 has exactly val_a → select; page 1 has val_b ≠ val_a → skip.
  PARQUET_ASSIGN_OR_THROW(
      auto sel, col_index->FilterPages(std::any(val_a), PredicateOp::EQ, *off_index,
                                       /*row_group_row_count=*/200));

  ASSERT_EQ(2, sel.page_count());
  EXPECT_FALSE(sel.selector(0).skip);  // [val_a, val_a] contains val_a
  EXPECT_TRUE(sel.selector(1).skip);   // [val_b, val_b] does not contain val_a

  // EQ val_b: page 1 has exactly val_b → select; page 0 has val_a ≠ val_b → skip.
  PARQUET_ASSIGN_OR_THROW(
      auto sel2, col_index->FilterPages(std::any(val_b), PredicateOp::EQ, *off_index,
                                        /*row_group_row_count=*/200));

  ASSERT_EQ(2, sel2.page_count());
  EXPECT_TRUE(sel2.selector(0).skip);   // [val_a, val_a] does not contain val_b
  EXPECT_FALSE(sel2.selector(1).skip);  // [val_b, val_b] contains val_b
}

// ---------------------------------------------------------------------------
// Case 10 – Zero pages → empty RowSelection.
// ---------------------------------------------------------------------------
TYPED_TEST(TestColumnIndexFilterPages, ZeroPages) {
  using T = typename TypeParam::c_type;

  // Build a column index with zero pages (no stats entries).
  auto node = schema::PrimitiveNode::Make(
      "col", Repetition::OPTIONAL, TypeParam::type_num, ConvertedType::NONE, FLBA_LENGTH);
  auto [col_index, col_descr] = BuildColumnIndex(node, /*page_stats=*/{});
  // A column index with zero pages may return nullptr from Build()
  // (no pages means no data to index).  In that case the test cannot proceed
  // against FilterPages on a nullptr — so we build the offset index and verify
  // only when a valid column index is returned.
  if (col_index == nullptr) {
    // Expected: builder returns nullptr for zero-page indexes.
    SUCCEED() << "ColumnIndexBuilder::Build() returns nullptr for zero pages";
    return;
  }

  std::vector<PageLocation> page_locs = {};
  auto off_index = BuildOffsetIndex(page_locs);
  ASSERT_NE(nullptr, off_index);

  T dummy_val{};
  PARQUET_ASSIGN_OR_THROW(
      auto sel, col_index->FilterPages(std::any(dummy_val), PredicateOp::EQ, *off_index,
                                       /*row_group_row_count=*/0));

  EXPECT_EQ(0, sel.page_count());
  EXPECT_EQ(0, sel.row_count());
}

}  // namespace parquet
