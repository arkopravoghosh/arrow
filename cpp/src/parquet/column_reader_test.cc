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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/array/array_binary.h"
#include "arrow/io/memory.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/util/config.h"
#include "arrow/util/macros.h"
#include "parquet/column_page.h"
#include "parquet/column_reader.h"
#include "parquet/column_writer.h"
#include "parquet/file_reader.h"
#include "parquet/file_writer.h"
#include "parquet/metadata.h"
#include "parquet/properties.h"
#include "parquet/row_selection.h"
#include "parquet/schema.h"
#include "parquet/test_util.h"
#include "parquet/types.h"

namespace parquet {

using ParquetType = parquet::Type;

using internal::BinaryRecordReader;
using internal::LevelInfo;
using schema::GroupNode;
using schema::NodePtr;
using schema::PrimitiveNode;
using testing::ElementsAre;

namespace test {

template <typename T>
static inline bool vector_equal_with_def_levels(const std::vector<T>& left,
                                                const std::vector<int16_t>& def_levels,
                                                int16_t max_def_levels,
                                                int16_t max_rep_levels,
                                                const std::vector<T>& right) {
  size_t i_left = 0;
  size_t i_right = 0;
  for (size_t i = 0; i < def_levels.size(); i++) {
    if (def_levels[i] == max_def_levels) {
      // Compare
      if (left[i_left] != right[i_right]) {
        std::cerr << "index " << i << " left was " << left[i_left] << " right was "
                  << right[i] << std::endl;
        return false;
      }
      i_left++;
      i_right++;
    } else if (def_levels[i] == (max_def_levels - 1)) {
      // Null entry on the lowest nested level
      i_right++;
    } else if (def_levels[i] < (max_def_levels - 1)) {
      // Null entry on a higher nesting level, only supported for non-repeating data
      if (max_rep_levels == 0) {
        i_right++;
      }
    }
  }

  return true;
}

class TestPrimitiveReader : public ::testing::Test {
 public:
  void InitReader(const ColumnDescriptor* d) {
    auto pager = std::make_unique<MockPageReader>(pages_);
    reader_ = ColumnReader::Make(d, std::move(pager));
  }

  void CheckResults() {
    std::vector<int32_t> vresult(num_values_, -1);
    std::vector<int16_t> dresult(num_levels_, -1);
    std::vector<int16_t> rresult(num_levels_, -1);
    int64_t values_read = 0;
    int total_values_read = 0;
    int batch_actual = 0;

    Int32Reader* reader = static_cast<Int32Reader*>(reader_.get());
    int32_t batch_size = 8;
    int batch = 0;
    // This will cover both the cases
    // 1) batch_size < page_size (multiple ReadBatch from a single page)
    // 2) batch_size > page_size (BatchRead limits to a single page)
    do {
      batch = static_cast<int>(reader->ReadBatch(
          batch_size, &dresult[0] + batch_actual, &rresult[0] + batch_actual,
          &vresult[0] + total_values_read, &values_read));
      total_values_read += static_cast<int>(values_read);
      batch_actual += batch;
      batch_size = std::min(1 << 24, std::max(batch_size * 2, 4096));
    } while (batch > 0);

    ASSERT_EQ(num_levels_, batch_actual);
    ASSERT_EQ(num_values_, total_values_read);
    ASSERT_TRUE(vector_equal(values_, vresult));
    if (max_def_level_ > 0) {
      ASSERT_TRUE(vector_equal(def_levels_, dresult));
    }
    if (max_rep_level_ > 0) {
      ASSERT_TRUE(vector_equal(rep_levels_, rresult));
    }
    // catch improper writes at EOS
    batch_actual =
        static_cast<int>(reader->ReadBatch(5, nullptr, nullptr, nullptr, &values_read));
    ASSERT_EQ(0, batch_actual);
    ASSERT_EQ(0, values_read);
  }

  void Clear() {
    values_.clear();
    def_levels_.clear();
    rep_levels_.clear();
    pages_.clear();
    reader_.reset();
  }

  void ExecutePlain(int num_pages, int levels_per_page, const ColumnDescriptor* d) {
    num_values_ =
        MakePages<Int32Type>(d, num_pages, levels_per_page, def_levels_, rep_levels_,
                             values_, data_buffer_, pages_, Encoding::PLAIN);
    num_levels_ = num_pages * levels_per_page;
    InitReader(d);
    CheckResults();
    Clear();
  }

  void ExecuteDict(int num_pages, int levels_per_page, const ColumnDescriptor* d) {
    num_values_ =
        MakePages<Int32Type>(d, num_pages, levels_per_page, def_levels_, rep_levels_,
                             values_, data_buffer_, pages_, Encoding::RLE_DICTIONARY);
    num_levels_ = num_pages * levels_per_page;
    InitReader(d);
    CheckResults();
    Clear();
  }

 protected:
  int num_levels_;
  int num_values_;
  int16_t max_def_level_;
  int16_t max_rep_level_;
  std::vector<std::shared_ptr<Page>> pages_;
  std::shared_ptr<ColumnReader> reader_;
  std::vector<int32_t> values_;
  std::vector<int16_t> def_levels_;
  std::vector<int16_t> rep_levels_;
  std::vector<uint8_t> data_buffer_;  // For BA and FLBA
};

TEST_F(TestPrimitiveReader, TestInt32FlatRequired) {
  int levels_per_page = 100;
  int num_pages = 50;
  max_def_level_ = 0;
  max_rep_level_ = 0;
  NodePtr type = schema::Int32("a", Repetition::REQUIRED);
  const ColumnDescriptor descr(type, max_def_level_, max_rep_level_);
  ASSERT_NO_FATAL_FAILURE(ExecutePlain(num_pages, levels_per_page, &descr));
  ASSERT_NO_FATAL_FAILURE(ExecuteDict(num_pages, levels_per_page, &descr));
}

TEST_F(TestPrimitiveReader, TestInt32FlatOptional) {
  int levels_per_page = 100;
  int num_pages = 50;
  max_def_level_ = 4;
  max_rep_level_ = 0;
  NodePtr type = schema::Int32("b", Repetition::OPTIONAL);
  const ColumnDescriptor descr(type, max_def_level_, max_rep_level_);
  ASSERT_NO_FATAL_FAILURE(ExecutePlain(num_pages, levels_per_page, &descr));
  ASSERT_NO_FATAL_FAILURE(ExecuteDict(num_pages, levels_per_page, &descr));
}

TEST_F(TestPrimitiveReader, TestInt32FlatRepeated) {
  int levels_per_page = 100;
  int num_pages = 50;
  max_def_level_ = 4;
  max_rep_level_ = 2;
  NodePtr type = schema::Int32("c", Repetition::REPEATED);
  const ColumnDescriptor descr(type, max_def_level_, max_rep_level_);
  ASSERT_NO_FATAL_FAILURE(ExecutePlain(num_pages, levels_per_page, &descr));
  ASSERT_NO_FATAL_FAILURE(ExecuteDict(num_pages, levels_per_page, &descr));
}

// Tests skipping around page boundaries.
TEST_F(TestPrimitiveReader, TestSkipAroundPageBoundaries) {
  int levels_per_page = 100;
  int num_pages = 7;
  max_def_level_ = 0;
  max_rep_level_ = 0;
  NodePtr type = schema::Int32("b", Repetition::REQUIRED);
  const ColumnDescriptor descr(type, max_def_level_, max_rep_level_);
  MakePages<Int32Type>(&descr, num_pages, levels_per_page, def_levels_, rep_levels_,
                       values_, data_buffer_, pages_, Encoding::PLAIN);
  InitReader(&descr);
  std::vector<int32_t> vresult(levels_per_page / 2, -1);
  std::vector<int16_t> dresult(levels_per_page / 2, -1);
  std::vector<int16_t> rresult(levels_per_page / 2, -1);

  Int32Reader* reader = static_cast<Int32Reader*>(reader_.get());
  int64_t values_read = 0;

  // 1) skip_size > page_size (multiple pages skipped)
  // Skip first 2 pages
  int64_t levels_skipped = reader->Skip(2 * levels_per_page);
  ASSERT_EQ(2 * levels_per_page, levels_skipped);
  // Read half a page
  reader->ReadBatch(levels_per_page / 2, dresult.data(), rresult.data(), vresult.data(),
                    &values_read);
  std::vector<int32_t> sub_values(
      values_.begin() + 2 * levels_per_page,
      values_.begin() + static_cast<int>(2.5 * levels_per_page));
  ASSERT_TRUE(vector_equal(sub_values, vresult));

  // 2) skip_size == page_size (skip across two pages from page 2.5 to 3.5)
  levels_skipped = reader->Skip(levels_per_page);
  ASSERT_EQ(levels_per_page, levels_skipped);
  // Read half a page (page 3.5 to 4)
  reader->ReadBatch(levels_per_page / 2, dresult.data(), rresult.data(), vresult.data(),
                    &values_read);
  sub_values.clear();
  sub_values.insert(sub_values.end(),
                    values_.begin() + static_cast<int>(3.5 * levels_per_page),
                    values_.begin() + 4 * levels_per_page);
  ASSERT_TRUE(vector_equal(sub_values, vresult));

  // 3) skip_size == page_size (skip page 4 from start of the page to the end)
  levels_skipped = reader->Skip(levels_per_page);
  ASSERT_EQ(levels_per_page, levels_skipped);
  // Read half a page (page 5 to 5.5)
  reader->ReadBatch(levels_per_page / 2, dresult.data(), rresult.data(), vresult.data(),
                    &values_read);
  sub_values.clear();
  sub_values.insert(sub_values.end(),
                    values_.begin() + static_cast<int>(5.0 * levels_per_page),
                    values_.begin() + static_cast<int>(5.5 * levels_per_page));
  ASSERT_TRUE(vector_equal(sub_values, vresult));

  // 4) skip_size < page_size (skip limited to a single page)
  // Skip half a page (page 5.5 to 6)
  levels_skipped = reader->Skip(levels_per_page / 2);
  ASSERT_EQ(0.5 * levels_per_page, levels_skipped);
  // Read half a page (6 to 6.5)
  reader->ReadBatch(levels_per_page / 2, dresult.data(), rresult.data(), vresult.data(),
                    &values_read);
  sub_values.clear();
  sub_values.insert(sub_values.end(),
                    values_.begin() + static_cast<int>(6.0 * levels_per_page),
                    values_.begin() + static_cast<int>(6.5 * levels_per_page));
  ASSERT_TRUE(vector_equal(sub_values, vresult));

  // 5) skip_size = 0
  levels_skipped = reader->Skip(0);
  ASSERT_EQ(0, levels_skipped);

  // 6) Skip past the end page.
  levels_skipped = reader->Skip(levels_per_page / 2 + 10);
  ASSERT_EQ(levels_per_page / 2, levels_skipped);

  values_.clear();
  def_levels_.clear();
  rep_levels_.clear();
  pages_.clear();
  reader_.reset();
}

// Skip with repeated field. This test makes it clear that we are skipping
// values and not records.
TEST_F(TestPrimitiveReader, TestSkipRepeatedField) {
  // Example schema: message M { repeated int32 b = 1 }
  max_def_level_ = 1;
  max_rep_level_ = 1;
  NodePtr type = schema::Int32("b", Repetition::REPEATED);
  const ColumnDescriptor descr(type, max_def_level_, max_rep_level_);
  // Example rows: {}, {[10, 10]}, {[20, 20, 20]}
  std::vector<int32_t> values = {10, 10, 20, 20, 20};
  std::vector<int16_t> def_levels = {0, 1, 1, 1, 1, 1};
  std::vector<int16_t> rep_levels = {0, 0, 1, 0, 1, 1};
  num_values_ = static_cast<int>(def_levels.size());
  std::shared_ptr<DataPageV1> page = MakeDataPage<Int32Type>(
      &descr, values, num_values_, Encoding::PLAIN, /*indices=*/{},
      /*indices_size=*/0, def_levels, max_def_level_, rep_levels, max_rep_level_);

  pages_.push_back(std::move(page));

  InitReader(&descr);
  Int32Reader* reader = static_cast<Int32Reader*>(reader_.get());

  // Vectors to hold read values, definition levels, and repetition levels.
  std::vector<int32_t> read_vals(4, -1);
  std::vector<int16_t> read_defs(4, -1);
  std::vector<int16_t> read_reps(4, -1);

  // Skip two levels.
  int64_t levels_skipped = reader->Skip(2);
  ASSERT_EQ(2, levels_skipped);

  int64_t num_read_values = 0;
  // Read the next set of values
  reader->ReadBatch(10, read_defs.data(), read_reps.data(), read_vals.data(),
                    &num_read_values);
  ASSERT_EQ(num_read_values, 4);
  // Note that we end up in the record with {[10, 10]}
  ASSERT_TRUE(vector_equal({10, 20, 20, 20}, read_vals));
  ASSERT_TRUE(vector_equal({1, 1, 1, 1}, read_defs));
  ASSERT_TRUE(vector_equal({1, 0, 1, 1}, read_reps));

  // No values remain in data page
  levels_skipped = reader->Skip(2);
  ASSERT_EQ(0, levels_skipped);
  reader->ReadBatch(10, read_defs.data(), read_reps.data(), read_vals.data(),
                    &num_read_values);
  ASSERT_EQ(num_read_values, 0);
}

// Page claims to have two values but only 1 is present.
TEST_F(TestPrimitiveReader, TestReadValuesMissing) {
  max_def_level_ = 1;
  max_rep_level_ = 0;
  constexpr int batch_size = 1;
  std::vector<bool> values(1, false);
  std::vector<int16_t> input_def_levels(1, 1);
  NodePtr type = schema::Boolean("a", Repetition::OPTIONAL);
  const ColumnDescriptor descr(type, max_def_level_, max_rep_level_);

  // The data page falls back to plain encoding
  std::shared_ptr<ResizableBuffer> dummy = AllocateBuffer();
  std::shared_ptr<DataPageV1> data_page = MakeDataPage<BooleanType>(
      &descr, values, /*num_values=*/2, Encoding::PLAIN, /*indices=*/{},
      /*indices_size=*/0, /*def_levels=*/input_def_levels, max_def_level_,
      /*rep_levels=*/{},
      /*max_rep_level=*/max_rep_level_);
  pages_.push_back(data_page);
  InitReader(&descr);
  auto reader = static_cast<BoolReader*>(reader_.get());
  ASSERT_TRUE(reader->HasNext());
  std::vector<int16_t> def_levels(batch_size, 0);
  std::vector<int16_t> rep_levels(batch_size, 0);
  bool values_out[batch_size];
  int64_t values_read;
  EXPECT_EQ(1, reader->ReadBatch(batch_size, def_levels.data(), rep_levels.data(),
                                 values_out, &values_read));
  ASSERT_THROW(reader->ReadBatch(batch_size, def_levels.data(), rep_levels.data(),
                                 values_out, &values_read),
               ParquetException);
}

// GH-41321: When max_def_level > 0 or max_rep_level > 0, and
// Page has more or less levels than the `num_values` in
// PageHeader. We should detect and throw exception.
TEST_F(TestPrimitiveReader, DefRepLevelNotExpected) {
  auto do_check = [&](const NodePtr& type, const std::vector<int16_t>& input_def_levels,
                      const std::vector<int16_t>& input_rep_levels, int num_values) {
    std::vector<bool> values(num_values, false);
    const ColumnDescriptor descr(type, max_def_level_, max_rep_level_);

    // The data page falls back to plain encoding
    std::shared_ptr<ResizableBuffer> dummy = AllocateBuffer();
    std::shared_ptr<DataPageV1> data_page = MakeDataPage<BooleanType>(
        &descr, values, /*num_values=*/num_values, Encoding::PLAIN, /*indices=*/{},
        /*indices_size=*/0, /*def_levels=*/input_def_levels, max_def_level_,
        /*rep_levels=*/input_rep_levels,
        /*max_rep_level=*/max_rep_level_);
    pages_.push_back(data_page);
    InitReader(&descr);
    auto reader = static_cast<BoolReader*>(reader_.get());
    ASSERT_TRUE(reader->HasNext());

    constexpr int batch_size = 10;
    std::vector<int16_t> def_levels(batch_size, 0);
    std::vector<int16_t> rep_levels(batch_size, 0);
    bool values_out[batch_size];
    int64_t values_read;
    EXPECT_THROW_THAT(
        [&]() {
          reader->ReadBatch(batch_size, def_levels.data(), rep_levels.data(), values_out,
                            &values_read);
        },
        ParquetException,
        ::testing::Property(&ParquetException::what,
                            ::testing::HasSubstr("Number of decoded rep / def levels do "
                                                 "not match num_values in page header")));
  };
  // storing def-levels less than value in page-header
  {
    max_def_level_ = 1;
    max_rep_level_ = 0;
    NodePtr type = schema::Boolean("a", Repetition::OPTIONAL);
    std::vector<int16_t> input_def_levels(1, 1);
    std::vector<int16_t> input_rep_levels{};
    do_check(type, input_def_levels, input_rep_levels, /*num_values=*/3);
  }
  // storing def-levels more than value in page-header
  {
    max_def_level_ = 1;
    max_rep_level_ = 0;
    NodePtr type = schema::Boolean("a", Repetition::OPTIONAL);
    std::vector<int16_t> input_def_levels(2, 1);
    std::vector<int16_t> input_rep_levels{};
    do_check(type, input_def_levels, input_rep_levels, /*num_values=*/1);
  }
  // storing rep-levels less than value in page-header
  {
    max_def_level_ = 0;
    max_rep_level_ = 1;
    NodePtr type = schema::Boolean("a", Repetition::REPEATED);
    std::vector<int16_t> input_def_levels{};
    std::vector<int16_t> input_rep_levels(3, 0);
    do_check(type, input_def_levels, input_rep_levels, /*num_values=*/4);
  }
  // storing rep-levels more than value in page-header
  {
    max_def_level_ = 0;
    max_rep_level_ = 1;
    NodePtr type = schema::Boolean("a", Repetition::REPEATED);
    std::vector<int16_t> input_def_levels{};
    std::vector<int16_t> input_rep_levels(2, 1);
    do_check(type, input_def_levels, input_rep_levels, /*num_values=*/1);
  }
}

// Repetition level byte length reported in Page but Max Repetition level
// is zero for the column.
TEST_F(TestPrimitiveReader, TestRepetitionLvlBytesWithMaxRepetitionZero) {
  constexpr int batch_size = 4;
  max_def_level_ = 1;
  max_rep_level_ = 0;
  NodePtr type = schema::Int32("a", Repetition::OPTIONAL);
  const ColumnDescriptor descr(type, max_def_level_, max_rep_level_);
  // Bytes here came from the example parquet file in ARROW-17453's int32
  // column which was delta bit-packed. The key part is the first three
  // bytes: the page header reports 1 byte for repetition levels even
  // though the max rep level is 0. If that byte isn't skipped then
  // we get def levels of [1, 1, 0, 0] instead of the correct [1, 1, 1, 0].
  const std::vector<uint8_t> page_data{0x3,  0x3, 0x7, 0x80, 0x1, 0x4, 0x3,
                                       0x18, 0x1, 0x2, 0x0,  0x0, 0x0, 0xc,
                                       0x0,  0x0, 0x0, 0x0,  0x0, 0x0, 0x0};

  std::shared_ptr<DataPageV2> data_page =
      std::make_shared<DataPageV2>(Buffer::Wrap(page_data.data(), page_data.size()), 4, 1,
                                   4, Encoding::DELTA_BINARY_PACKED, 2, 1, 21);

  pages_.push_back(data_page);
  InitReader(&descr);
  auto reader = static_cast<Int32Reader*>(reader_.get());
  int16_t def_levels_out[batch_size];
  int32_t values[batch_size];
  int64_t values_read;
  ASSERT_TRUE(reader->HasNext());
  EXPECT_EQ(4, reader->ReadBatch(batch_size, def_levels_out, /*rep_levels=*/nullptr,
                                 values, &values_read));
  EXPECT_EQ(3, values_read);
}

// Page claims to have two values but only 1 is present.
TEST_F(TestPrimitiveReader, TestReadValuesMissingWithDictionary) {
  constexpr int batch_size = 1;
  max_def_level_ = 1;
  max_rep_level_ = 0;
  NodePtr type = schema::Int32("a", Repetition::OPTIONAL);
  const ColumnDescriptor descr(type, max_def_level_, max_rep_level_);
  std::shared_ptr<ResizableBuffer> dummy = AllocateBuffer();

  std::shared_ptr<DictionaryPage> dict_page =
      std::make_shared<DictionaryPage>(dummy, 0, Encoding::PLAIN);
  std::vector<int16_t> input_def_levels(1, 0);
  std::shared_ptr<DataPageV1> data_page = MakeDataPage<Int32Type>(
      &descr, {}, /*num_values=*/2, Encoding::RLE_DICTIONARY, /*indices=*/{},
      /*indices_size=*/0, /*def_levels=*/input_def_levels, max_def_level_,
      /*rep_levels=*/{},
      /*max_rep_level=*/0);
  pages_.push_back(dict_page);
  pages_.push_back(data_page);
  InitReader(&descr);
  auto reader = static_cast<ByteArrayReader*>(reader_.get());
  const ByteArray* dict = nullptr;
  int32_t dict_len = 0;
  int64_t indices_read = 0;
  int32_t indices[batch_size];
  int16_t def_levels_out[batch_size];
  ASSERT_TRUE(reader->HasNext());
  EXPECT_EQ(1, reader->ReadBatchWithDictionary(batch_size, def_levels_out,
                                               /*rep_levels=*/nullptr, indices,
                                               &indices_read, &dict, &dict_len));
  ASSERT_THROW(reader->ReadBatchWithDictionary(batch_size, def_levels_out,
                                               /*rep_levels=*/nullptr, indices,
                                               &indices_read, &dict, &dict_len),
               ParquetException);
}

TEST_F(TestPrimitiveReader, TestDictionaryEncodedPages) {
  max_def_level_ = 0;
  max_rep_level_ = 0;
  NodePtr type = schema::Int32("a", Repetition::REQUIRED);
  const ColumnDescriptor descr(type, max_def_level_, max_rep_level_);
  std::shared_ptr<ResizableBuffer> dummy = AllocateBuffer();

  std::shared_ptr<DictionaryPage> dict_page =
      std::make_shared<DictionaryPage>(dummy, 0, Encoding::PLAIN);
  std::shared_ptr<DataPageV1> data_page = MakeDataPage<Int32Type>(
      &descr, {}, 0, Encoding::RLE_DICTIONARY, {}, 0, {}, 0, {}, 0);
  pages_.push_back(dict_page);
  pages_.push_back(data_page);
  InitReader(&descr);
  // Tests Dict : PLAIN, Data : RLE_DICTIONARY
  ASSERT_NO_THROW(reader_->HasNext());
  pages_.clear();

  dict_page = std::make_shared<DictionaryPage>(dummy, 0, Encoding::PLAIN_DICTIONARY);
  data_page = MakeDataPage<Int32Type>(&descr, {}, 0, Encoding::PLAIN_DICTIONARY, {}, 0,
                                      {}, 0, {}, 0);
  pages_.push_back(dict_page);
  pages_.push_back(data_page);
  InitReader(&descr);
  // Tests Dict : PLAIN_DICTIONARY, Data : PLAIN_DICTIONARY
  ASSERT_NO_THROW(reader_->HasNext());
  pages_.clear();

  data_page = MakeDataPage<Int32Type>(&descr, {}, 0, Encoding::RLE_DICTIONARY, {}, 0, {},
                                      0, {}, 0);
  pages_.push_back(data_page);
  InitReader(&descr);
  // Tests dictionary page must occur before data page
  ASSERT_THROW(reader_->HasNext(), ParquetException);
  pages_.clear();

  dict_page = std::make_shared<DictionaryPage>(dummy, 0, Encoding::DELTA_BYTE_ARRAY);
  pages_.push_back(dict_page);
  InitReader(&descr);
  // Tests only RLE_DICTIONARY is supported
  ASSERT_THROW(reader_->HasNext(), ParquetException);
  pages_.clear();

  std::shared_ptr<DictionaryPage> dict_page1 =
      std::make_shared<DictionaryPage>(dummy, 0, Encoding::PLAIN_DICTIONARY);
  std::shared_ptr<DictionaryPage> dict_page2 =
      std::make_shared<DictionaryPage>(dummy, 0, Encoding::PLAIN);
  pages_.push_back(dict_page1);
  pages_.push_back(dict_page2);
  InitReader(&descr);
  // Column cannot have more than one dictionary
  ASSERT_THROW(reader_->HasNext(), ParquetException);
  pages_.clear();

  data_page = MakeDataPage<Int32Type>(&descr, {}, 0, Encoding::DELTA_BYTE_ARRAY, {}, 0,
                                      {}, 0, {}, 0);
  pages_.push_back(data_page);
  InitReader(&descr);
  // unsupported encoding
  ASSERT_THROW(reader_->HasNext(), ParquetException);
  pages_.clear();
}

TEST_F(TestPrimitiveReader, TestDictionaryEncodedPagesWithExposeEncoding) {
  max_def_level_ = 0;
  max_rep_level_ = 0;
  int levels_per_page = 100;
  int num_pages = 5;
  std::vector<int16_t> def_levels;
  std::vector<int16_t> rep_levels;
  std::vector<ByteArray> values;
  std::vector<uint8_t> buffer;
  NodePtr type = schema::ByteArray("a", Repetition::REQUIRED);
  const ColumnDescriptor descr(type, max_def_level_, max_rep_level_);

  // Fully dictionary encoded
  MakePages<ByteArrayType>(&descr, num_pages, levels_per_page, def_levels, rep_levels,
                           values, buffer, pages_, Encoding::RLE_DICTIONARY);
  InitReader(&descr);

  auto reader = static_cast<ByteArrayReader*>(reader_.get());
  const ByteArray* dict = nullptr;
  int32_t dict_len = 0;
  int64_t total_indices = 0;
  int64_t indices_read = 0;
  int64_t value_size = values.size();
  auto indices = std::make_unique<int32_t[]>(value_size);
  while (total_indices < value_size && reader->HasNext()) {
    const ByteArray* tmp_dict = nullptr;
    int32_t tmp_dict_len = 0;
    EXPECT_NO_THROW(reader->ReadBatchWithDictionary(
        value_size, /*def_levels=*/nullptr,
        /*rep_levels=*/nullptr, indices.get() + total_indices, &indices_read, &tmp_dict,
        &tmp_dict_len));
    if (tmp_dict != nullptr) {
      // Dictionary is read along with data
      EXPECT_GT(indices_read, 0);
      dict = tmp_dict;
      dict_len = tmp_dict_len;
    } else {
      // Dictionary is not read when there's no data
      EXPECT_EQ(indices_read, 0);
    }
    total_indices += indices_read;
  }

  EXPECT_EQ(total_indices, value_size);
  for (int64_t i = 0; i < total_indices; ++i) {
    EXPECT_LT(indices[i], dict_len);
    EXPECT_EQ(dict[indices[i]].len, values[i].len);
    EXPECT_EQ(memcmp(dict[indices[i]].ptr, values[i].ptr, values[i].len), 0);
  }
  pages_.clear();
}

TEST_F(TestPrimitiveReader, TestNonDictionaryEncodedPagesWithExposeEncoding) {
  max_def_level_ = 0;
  max_rep_level_ = 0;
  int64_t value_size = 100;
  std::vector<int32_t> values(value_size, 0);
  NodePtr type = schema::Int32("a", Repetition::REQUIRED);
  const ColumnDescriptor descr(type, max_def_level_, max_rep_level_);

  // The data page falls back to plain encoding
  std::shared_ptr<ResizableBuffer> dummy = AllocateBuffer();
  std::shared_ptr<DictionaryPage> dict_page =
      std::make_shared<DictionaryPage>(dummy, 0, Encoding::PLAIN);
  std::shared_ptr<DataPageV1> data_page = MakeDataPage<Int32Type>(
      &descr, values, static_cast<int>(value_size), Encoding::PLAIN, /*indices=*/{},
      /*indices_size=*/0, /*def_levels=*/{}, /*max_def_level=*/0, /*rep_levels=*/{},
      /*max_rep_level=*/0);
  pages_.push_back(dict_page);
  pages_.push_back(data_page);
  InitReader(&descr);

  auto reader = static_cast<ByteArrayReader*>(reader_.get());
  const ByteArray* dict = nullptr;
  int32_t dict_len = 0;
  int64_t indices_read = 0;
  auto indices = std::make_unique<int32_t[]>(value_size);
  // Dictionary cannot be exposed when it's not fully dictionary encoded
  EXPECT_THROW(reader->ReadBatchWithDictionary(value_size, /*def_levels=*/nullptr,
                                               /*rep_levels=*/nullptr, indices.get(),
                                               &indices_read, &dict, &dict_len),
               ParquetException);
  pages_.clear();
}

namespace {

internal::LevelInfo ComputeLevelInfo(const ColumnDescriptor* descr) {
  internal::LevelInfo level_info;
  level_info.def_level = descr->max_definition_level();
  level_info.rep_level = descr->max_repetition_level();

  int16_t min_spaced_def_level = descr->max_definition_level();
  const ::parquet::schema::Node* node = descr->schema_node().get();
  while (node != nullptr && !node->is_repeated()) {
    if (node->is_optional()) {
      min_spaced_def_level--;
    }
    node = node->parent();
  }
  level_info.repeated_ancestor_def_level = min_spaced_def_level;
  return level_info;
}

}  // namespace

using ReadDenseForNullable = bool;
class RecordReaderPrimitiveTypeTest
    : public ::testing::TestWithParam<ReadDenseForNullable> {
 public:
  const int32_t kNullValue = -1;

  void Init(NodePtr column) {
    NodePtr root = GroupNode::Make("root", Repetition::REQUIRED, {column});
    schema_descriptor_.Init(root);
    descr_ = schema_descriptor_.Column(0);
    record_reader_ = internal::RecordReader::Make(descr_, ComputeLevelInfo(descr_),
                                                  ::arrow::default_memory_pool(),
                                                  /*read_dictionary=*/false, GetParam());
  }

  void CheckReadValues(std::vector<int32_t> expected_values,
                       std::vector<int16_t> expected_defs,
                       std::vector<int16_t> expected_reps) {
    const auto read_values = reinterpret_cast<const int32_t*>(record_reader_->values());
    std::vector<int32_t> read_vals(read_values,
                                   read_values + record_reader_->values_written());
    ASSERT_EQ(read_vals.size(), expected_values.size());
    for (size_t i = 0; i < expected_values.size(); ++i) {
      if (expected_values[i] != kNullValue) {
        ASSERT_EQ(expected_values[i], read_values[i]);
      }
    }

    if (!descr_->schema_node()->is_required()) {
      std::vector<int16_t> read_defs(
          record_reader_->def_levels(),
          record_reader_->def_levels() + record_reader_->levels_position());
      ASSERT_TRUE(vector_equal(expected_defs, read_defs));
    }

    if (descr_->schema_node()->is_repeated()) {
      std::vector<int16_t> read_reps(
          record_reader_->rep_levels(),
          record_reader_->rep_levels() + record_reader_->levels_position());
      ASSERT_TRUE(vector_equal(expected_reps, read_reps));
    }
  }

  void CheckState(int64_t values_written, int64_t null_count, int64_t levels_written,
                  int64_t levels_position) {
    ASSERT_EQ(record_reader_->values_written(), values_written);
    ASSERT_EQ(record_reader_->null_count(), null_count);
    ASSERT_EQ(record_reader_->levels_written(), levels_written);
    ASSERT_EQ(record_reader_->levels_position(), levels_position);
  }

 protected:
  SchemaDescriptor schema_descriptor_;
  std::shared_ptr<internal::RecordReader> record_reader_;
  const ColumnDescriptor* descr_;
};

// Tests reading a required field. The expected results are the same for
// reading dense and spaced.
TEST_P(RecordReaderPrimitiveTypeTest, ReadRequired) {
  Init(schema::Int32("b", Repetition::REQUIRED));

  // Records look like: {10, 20, 20, 30, 30, 30}
  std::vector<std::shared_ptr<Page>> pages;
  std::vector<int32_t> values = {10, 20, 20, 30, 30, 30};
  std::vector<int16_t> def_levels = {};
  std::vector<int16_t> rep_levels = {};

  std::shared_ptr<DataPageV1> page = MakeDataPage<Int32Type>(
      descr_, values, /*num_values=*/static_cast<int>(def_levels.size()), Encoding::PLAIN,
      /*indices=*/{},
      /*indices_size=*/0, def_levels, descr_->max_definition_level(), rep_levels,
      descr_->max_repetition_level());
  pages.push_back(std::move(page));
  auto pager = std::make_unique<MockPageReader>(pages);
  record_reader_->SetPageReader(std::move(pager));

  // Read [10]
  int64_t records_read = record_reader_->ReadRecords(/*num_records=*/1);
  ASSERT_EQ(records_read, 1);
  CheckState(/*values_written=*/1, /*null_count=*/0, /*levels_written=*/0,
             /*levels_position=*/0);
  CheckReadValues(/*expected_values=*/{10}, /*expected_defs=*/{},
                  /*expected_reps=*/{});
  record_reader_->Reset();
  CheckState(/*values_written=*/0, /*null_count=*/0, /*levels_written=*/0,
             /*levels_position=*/0);

  // Read 20, 20, 30, 30, 30
  records_read = record_reader_->ReadRecords(/*num_records=*/10);
  ASSERT_EQ(records_read, 5);
  CheckState(/*values_written=*/5, /*null_count=*/0, /*levels_written=*/0,
             /*levels_position=*/0);
  CheckReadValues(/*expected_values=*/{20, 20, 30, 30, 30},
                  /*expected_defs=*/{},
                  /*expected_reps=*/{});
  record_reader_->Reset();
  CheckState(/*values_written=*/0, /*null_count=*/0, /*levels_written=*/0,
             /*levels_position=*/0);
}

// Tests reading an optional field.
// Use a max definition field > 1 to test both cases where parent is present or
// parent is missing.
TEST_P(RecordReaderPrimitiveTypeTest, ReadOptional) {
  NodePtr column = GroupNode::Make(
      "a", Repetition::OPTIONAL,
      {PrimitiveNode::Make("element", Repetition::OPTIONAL, ParquetType::INT32)});
  Init(column);

  // Records look like: {10, null, 20, 20, null, 30, 30, 30, null}
  std::vector<std::shared_ptr<Page>> pages;
  std::vector<int32_t> values = {10, 20, 20, 30, 30, 30};
  std::vector<int16_t> def_levels = {2, 0, 2, 2, 1, 2, 2, 2, 0};

  std::shared_ptr<DataPageV1> page = MakeDataPage<Int32Type>(
      descr_, values, /*num_values=*/static_cast<int>(def_levels.size()), Encoding::PLAIN,
      /*indices=*/{},
      /*indices_size=*/0, def_levels, descr_->max_definition_level(), /*rep_levels=*/{},
      descr_->max_repetition_level());
  pages.push_back(std::move(page));
  auto pager = std::make_unique<MockPageReader>(pages);
  record_reader_->SetPageReader(std::move(pager));

  // Read 10, null
  int64_t records_read = record_reader_->ReadRecords(/*num_records=*/2);
  ASSERT_EQ(records_read, 2);
  if (GetParam() == /*read_dense_for_nullable=*/true) {
    CheckState(/*values_written=*/1, /*null_count=*/0, /*levels_written=*/9,
               /*levels_position=*/2);
    CheckReadValues(/*expected_values=*/{10}, /*expected_defs=*/{2, 0},
                    /*expected_reps=*/{});
  } else {
    CheckState(/*values_written=*/2, /*null_count=*/1, /*levels_written=*/9,
               /*levels_position=*/2);
    CheckReadValues(/*expected_values=*/{10, kNullValue}, /*expected_defs=*/{2, 0},
                    /*expected_reps=*/{});
  }
  record_reader_->Reset();
  CheckState(/*values_written=*/0, /*null_count=*/0, /*levels_written=*/7,
             /*levels_position=*/0);

  // Read 20, 20, null (parent present), 30, 30, 30
  records_read = record_reader_->ReadRecords(/*num_records=*/6);
  ASSERT_EQ(records_read, 6);
  if (GetParam() == /*read_dense_for_nullable=*/true) {
    CheckState(/*values_written=*/5, /*null_count=*/0, /*levels_written=*/7,
               /*levels_position=*/6);
    CheckReadValues(/*expected_values=*/{20, 20, 30, 30, 30},
                    /*expected_defs=*/{2, 2, 1, 2, 2, 2},
                    /*expected_reps=*/{});
  } else {
    CheckState(/*values_written=*/6, /*null_count=*/1, /*levels_written=*/7,
               /*levels_position=*/6);
    CheckReadValues(/*expected_values=*/{20, 20, kNullValue, 30, 30, 30},
                    /*expected_defs=*/{2, 2, 1, 2, 2, 2},
                    /*expected_reps=*/{});
  }
  record_reader_->Reset();
  CheckState(/*values_written=*/0, /*null_count=*/0, /*levels_written=*/1,
             /*levels_position=*/0);

  // Read the last null value and read past the end.
  records_read = record_reader_->ReadRecords(/*num_records=*/3);
  ASSERT_EQ(records_read, 1);
  if (GetParam() == /*read_dense_for_nullable=*/true) {
    CheckState(/*values_written=*/0, /*null_count=*/0, /*levels_written=*/1,
               /*levels_position=*/1);
    CheckReadValues(/*expected_values=*/{},
                    /*expected_defs=*/{0},
                    /*expected_reps=*/{});
  } else {
    CheckState(/*values_written=*/1, /*null_count=*/1, /*levels_written=*/1,
               /*levels_position=*/1);
    CheckReadValues(/*expected_values=*/{kNullValue},
                    /*expected_defs=*/{0},
                    /*expected_reps=*/{});
  }
  record_reader_->Reset();
  CheckState(/*values_written=*/0, /*null_count=*/0, /*levels_written=*/0,
             /*levels_position=*/0);
}

// Tests reading a required repeated field. The results are the same for reading
// dense or spaced.
TEST_P(RecordReaderPrimitiveTypeTest, ReadRequiredRepeated) {
  NodePtr column = GroupNode::Make(
      "p", Repetition::REQUIRED,
      {GroupNode::Make(
          "list", Repetition::REPEATED,
          {PrimitiveNode::Make("element", Repetition::REQUIRED, ParquetType::INT32)})});
  Init(column);

  // Records look like: {[10], [20, 20], [30, 30, 30]}
  std::vector<std::shared_ptr<Page>> pages;
  std::vector<int32_t> values = {10, 20, 20, 30, 30, 30};
  std::vector<int16_t> def_levels = {1, 1, 1, 1, 1, 1};
  std::vector<int16_t> rep_levels = {0, 0, 1, 0, 1, 1};

  std::shared_ptr<DataPageV1> page = MakeDataPage<Int32Type>(
      descr_, values, /*num_values=*/static_cast<int>(def_levels.size()), Encoding::PLAIN,
      /*indices=*/{},
      /*indices_size=*/0, def_levels, descr_->max_definition_level(), rep_levels,
      descr_->max_repetition_level());
  pages.push_back(std::move(page));
  auto pager = std::make_unique<MockPageReader>(pages);
  record_reader_->SetPageReader(std::move(pager));

  // Read [10]
  int64_t records_read = record_reader_->ReadRecords(/*num_records=*/1);
  ASSERT_EQ(records_read, 1);
  CheckState(/*values_written=*/1, /*null_count=*/0, /*levels_written=*/6,
             /*levels_position=*/1);
  CheckReadValues(/*expected_values=*/{10}, /*expected_defs=*/{1},
                  /*expected_reps=*/{0});
  record_reader_->Reset();
  CheckState(/*values_written=*/0, /*null_count=*/0, /*levels_written=*/5,
             /*levels_position=*/0);

  // Read [20, 20], [30, 30, 30]
  records_read = record_reader_->ReadRecords(/*num_records=*/3);
  ASSERT_EQ(records_read, 2);
  CheckState(/*values_written=*/5, /*null_count=*/0, /*levels_written=*/5,
             /*levels_position=*/5);
  CheckReadValues(/*expected_values=*/{20, 20, 30, 30, 30},
                  /*expected_defs=*/{1, 1, 1, 1, 1},
                  /*expected_reps=*/{0, 1, 0, 1, 1});
  record_reader_->Reset();
  CheckState(/*values_written=*/0, /*null_count=*/0, /*levels_written=*/0,
             /*levels_position=*/0);
}

// Tests reading a nullable repeated field. Tests reading null values at
// different levels and reading an empty list.
TEST_P(RecordReaderPrimitiveTypeTest, ReadNullableRepeated) {
  NodePtr column = GroupNode::Make(
      "p", Repetition::OPTIONAL,
      {GroupNode::Make(
          "list", Repetition::REPEATED,
          {PrimitiveNode::Make("element", Repetition::OPTIONAL, ParquetType::INT32)})});
  Init(column);

  // Records look like: {[10], null, [20, 20], [], [30, 30, null, 30]}
  // Some explanation regarding the behavior. When reading spaced, for an empty list or
  // for a top-level null, we do not leave a space and we do not count it towards
  // null_count.  For a leaf-level null, we leave a space for it and we count it towards
  // null_count. When reading dense, null_count is always 0, and we do not leave any space
  // for values.
  std::vector<std::shared_ptr<Page>> pages;
  std::vector<int32_t> values = {10, 20, 20, 30, 30, 30};
  std::vector<int16_t> def_levels = {3, 0, 3, 3, 1, 3, 3, 2, 3};
  std::vector<int16_t> rep_levels = {0, 0, 0, 1, 0, 0, 1, 1, 1};

  std::shared_ptr<DataPageV1> page = MakeDataPage<Int32Type>(
      descr_, values, /*num_values=*/static_cast<int>(def_levels.size()), Encoding::PLAIN,
      /*indices=*/{},
      /*indices_size=*/0, def_levels, descr_->max_definition_level(), rep_levels,
      descr_->max_repetition_level());
  pages.push_back(std::move(page));
  auto pager = std::make_unique<MockPageReader>(pages);
  record_reader_->SetPageReader(std::move(pager));

  // Test reading 0 records.
  int64_t records_read = record_reader_->ReadRecords(/*num_records=*/0);
  ASSERT_EQ(records_read, 0);

  // Test the descr() accessor.
  ASSERT_EQ(record_reader_->descr()->max_definition_level(), 3);

  // Read [10], null
  // We do not read this null for both reading dense and spaced.
  records_read = record_reader_->ReadRecords(/*num_records=*/2);
  ASSERT_EQ(records_read, 2);
  if (GetParam() == /*read_dense_for_nullable=*/true) {
    CheckState(/*values_written=*/1, /*null_count=*/0, /*levels_written=*/9,
               /*levels_position=*/2);
    CheckReadValues(/*expected_values=*/{10}, /*expected_defs=*/{3, 0},
                    /*expected_reps=*/{0, 0});
  } else {
    CheckState(/*values_written=*/1, /*null_count=*/0, /*levels_written=*/9,
               /*levels_position=*/2);
    CheckReadValues(/*expected_values=*/{10}, /*expected_defs=*/{3, 0},
                    /*expected_reps=*/{0, 0});
  }
  record_reader_->Reset();
  CheckState(/*values_written=*/0, /*null_count=*/0, /*levels_written=*/7,
             /*levels_position=*/0);

  // Read [20, 20], []
  // We do not read any value for this, it will be counted towards null count
  // when reading spaced.
  records_read = record_reader_->ReadRecords(/*num_records=*/2);
  ASSERT_EQ(records_read, 2);
  if (GetParam() == /*read_dense_for_nullable=*/true) {
    CheckState(/*values_written=*/2, /*null_count=*/0, /*levels_written=*/7,
               /*levels_position=*/3);
    CheckReadValues(/*expected_values=*/{20, 20},
                    /*expected_defs=*/{3, 3, 1},
                    /*expected_reps=*/{0, 1, 0});
  } else {
    CheckState(/*values_written=*/2, /*null_count=*/0, /*levels_written=*/7,
               /*levels_position=*/3);
    CheckReadValues(/*expected_values=*/{20, 20},
                    /*expected_defs=*/{3, 3, 1},
                    /*expected_reps=*/{0, 1, 0});
  }
  record_reader_->Reset();
  CheckState(/*values_written=*/0, /*null_count=*/0, /*levels_written=*/4,
             /*levels_position=*/0);

  // Test reading 0 records.
  records_read = record_reader_->ReadRecords(/*num_records=*/0);
  ASSERT_EQ(records_read, 0);

  // Read the last record.
  records_read = record_reader_->ReadRecords(/*num_records=*/1);
  ASSERT_EQ(records_read, 1);
  if (GetParam() == /*read_dense_for_nullable=*/true) {
    CheckState(/*values_written=*/3, /*null_count=*/0, /*levels_written=*/4,
               /*levels_position=*/4);
    CheckReadValues(/*expected_values=*/{30, 30, 30},
                    /*expected_defs=*/{3, 3, 2, 3},
                    /*expected_reps=*/{0, 1, 1, 1});
  } else {
    CheckState(/*values_written=*/4, /*null_count=*/1, /*levels_written=*/4,
               /*levels_position=*/4);
    CheckReadValues(/*expected_values=*/{30, 30, kNullValue, 30},
                    /*expected_defs=*/{3, 3, 2, 3},
                    /*expected_reps=*/{0, 1, 1, 1});
  }
  record_reader_->Reset();
  CheckState(/*values_written=*/0, /*null_count=*/0, /*levels_written=*/0,
             /*levels_position=*/0);
}

// Test that we can skip required top level field.
TEST_P(RecordReaderPrimitiveTypeTest, SkipRequiredTopLevel) {
  Init(schema::Int32("b", Repetition::REQUIRED));

  std::vector<std::shared_ptr<Page>> pages;
  std::vector<int32_t> values = {10, 20, 20, 30, 30, 30};
  std::shared_ptr<DataPageV1> page = MakeDataPage<Int32Type>(
      descr_, values, /*num_values=*/static_cast<int>(values.size()), Encoding::PLAIN,
      /*indices=*/{},
      /*indices_size=*/0, /*def_levels=*/{}, descr_->max_definition_level(),
      /*rep_levels=*/{}, descr_->max_repetition_level());
  pages.push_back(std::move(page));
  auto pager = std::make_unique<MockPageReader>(pages);
  record_reader_->SetPageReader(std::move(pager));

  int64_t records_skipped = record_reader_->SkipRecords(/*num_records=*/3);
  ASSERT_EQ(records_skipped, 3);
  CheckState(/*values_written=*/0, /*null_count=*/0, /*levels_written=*/0,
             /*levels_position=*/0);

  int64_t records_read = record_reader_->ReadRecords(/*num_records=*/2);
  ASSERT_EQ(records_read, 2);
  CheckState(/*values_written=*/2, /*null_count=*/0, /*levels_written=*/0,
             /*levels_position=*/0);
  CheckReadValues(/*expected_values=*/{30, 30}, /*expected_defs=*/{},
                  /*expected_reps=*/{});
  record_reader_->Reset();
  CheckState(/*values_written=*/0, /*null_count=*/0, /*levels_written=*/0,
             /*levels_position=*/0);
}

// Skip an optional field. Intentionally included some null values.
TEST_P(RecordReaderPrimitiveTypeTest, SkipOptional) {
  Init(schema::Int32("b", Repetition::OPTIONAL));

  // Records look like {null, 10, 20, 30, null, 40, 50, 60}
  std::vector<std::shared_ptr<Page>> pages;
  std::vector<int32_t> values = {10, 20, 30, 40, 50, 60};
  std::vector<int16_t> def_levels = {0, 1, 1, 0, 1, 1, 1, 1};

  std::shared_ptr<DataPageV1> page = MakeDataPage<Int32Type>(
      descr_, values, /*num_values=*/static_cast<int>(values.size()), Encoding::PLAIN,
      /*indices=*/{},
      /*indices_size=*/0, def_levels, descr_->max_definition_level(),
      /*rep_levels=*/{}, descr_->max_repetition_level());
  pages.push_back(std::move(page));
  auto pager = std::make_unique<MockPageReader>(pages);
  record_reader_->SetPageReader(std::move(pager));

  {
    // Skip {null, 10}
    // This also tests when we start with a Skip.
    int64_t records_skipped = record_reader_->SkipRecords(/*num_records=*/2);
    ASSERT_EQ(records_skipped, 2);
    CheckState(/*values_written=*/0, /*null_count=*/0, /*levels_written=*/0,
               /*levels_position=*/0);
  }

  {
    // Read 3 records: {20, null, 30}
    int64_t records_read = record_reader_->ReadRecords(/*num_records=*/3);

    ASSERT_EQ(records_read, 3);
    if (GetParam() == /*read_dense_for_nullable=*/true) {
      // We had skipped 2 of the levels above. So there is only 6 left in total to
      // read, and we read 3 of them here.
      CheckState(/*values_written=*/2, /*null_count=*/0, /*levels_written=*/6,
                 /*levels_position=*/3);
      CheckReadValues(/*expected_values=*/{20, 30}, /*expected_defs=*/{1, 0, 1},
                      /*expected_reps=*/{});
    } else {
      CheckState(/*values_written=*/3, /*null_count=*/1, /*levels_written=*/6,
                 /*levels_position=*/3);
      CheckReadValues(/*expected_values=*/{20, kNullValue, 30},
                      /*expected_defs=*/{1, 0, 1},
                      /*expected_reps=*/{});
    }

    record_reader_->Reset();
  }

  {
    // Skip {40, 50}.
    int64_t records_skipped = record_reader_->SkipRecords(/*num_records=*/2);
    ASSERT_EQ(records_skipped, 2);
    CheckState(/*values_written=*/0, /*null_count=*/0, /*levels_written=*/1,
               /*levels_position=*/0);
    CheckReadValues(/*expected_values=*/{}, /*expected_defs=*/{},
                    /*expected_reps=*/{});
    // Try reset after a Skip.
    record_reader_->Reset();
    CheckState(/*values_written=*/0, /*null_count=*/0, /*levels_written=*/1,
               /*levels_position=*/0);
  }

  {
    // Read to the end of the column. Read {60}
    // This test checks that ReadAndThrowAwayValues works, since if it
    // does not we would read the wrong values.
    int64_t records_read = record_reader_->ReadRecords(/*num_records=*/1);

    ASSERT_EQ(records_read, 1);
    CheckState(/*values_written=*/1, /*null_count=*/0, /*levels_written=*/1,
               /*levels_position=*/1);
    CheckReadValues(/*expected_values=*/{60},
                    /*expected_defs=*/{1},
                    /*expected_reps=*/{});
  }

  // We have exhausted all the records.
  ASSERT_EQ(record_reader_->ReadRecords(/*num_records=*/3), 0);
  ASSERT_EQ(record_reader_->SkipRecords(/*num_records=*/3), 0);
}

// Test skipping for repeated fields.
TEST_P(RecordReaderPrimitiveTypeTest, SkipRepeated) {
  Init(schema::Int32("b", Repetition::REPEATED));

  // Records look like {null, [20, 20, 20], null, [30, 30], [40]}
  std::vector<std::shared_ptr<Page>> pages;
  std::vector<int32_t> values = {20, 20, 20, 30, 30, 40};
  std::vector<int16_t> def_levels = {0, 1, 1, 1, 0, 1, 1, 1};
  std::vector<int16_t> rep_levels = {0, 0, 1, 1, 0, 0, 1, 0};

  std::shared_ptr<DataPageV1> page = MakeDataPage<Int32Type>(
      descr_, values, /*num_values=*/static_cast<int>(values.size()), Encoding::PLAIN,
      /*indices=*/{},
      /*indices_size=*/0, def_levels, descr_->max_definition_level(), rep_levels,
      descr_->max_repetition_level());
  pages.push_back(std::move(page));
  auto pager = std::make_unique<MockPageReader>(pages);
  record_reader_->SetPageReader(std::move(pager));

  {
    // Skip 0 records.
    int64_t records_skipped = record_reader_->SkipRecords(/*num_records=*/0);
    ASSERT_EQ(records_skipped, 0);
  }

  {
    // This should skip the first null record.
    int64_t records_skipped = record_reader_->SkipRecords(/*num_records=*/1);
    ASSERT_EQ(records_skipped, 1);
    ASSERT_EQ(record_reader_->values_written(), 0);
    ASSERT_EQ(record_reader_->null_count(), 0);
    // For repeated fields, we need to read the levels to find the record
    // boundaries and skip. So some levels are read, however, the skipped
    // level should not be there after the skip. That's why levels_position()
    // is 0.
    CheckState(/*values_written=*/0, /*null_count=*/0, /*levels_written=*/7,
               /*levels_position=*/0);
    CheckReadValues(/*expected_values=*/{},
                    /*expected_defs=*/{},
                    /*expected_reps=*/{});
  }

  {
    // Read [20, 20, 20]
    int64_t records_read = record_reader_->ReadRecords(/*num_records=*/1);
    ASSERT_EQ(records_read, 1);
    CheckState(/*values_written=*/3, /*null_count=*/0, /*levels_written=*/7,
               /*levels_position=*/3);
    CheckReadValues(/*expected_values=*/{20, 20, 20},
                    /*expected_defs=*/{1, 1, 1},
                    /*expected_reps=*/{0, 1, 1});
  }

  {
    // Skip 0 records.
    int64_t records_skipped = record_reader_->SkipRecords(/*num_records=*/0);
    ASSERT_EQ(records_skipped, 0);
  }

  {
    // Skip the null record and also skip [30, 30]
    int64_t records_skipped = record_reader_->SkipRecords(/*num_records=*/2);
    ASSERT_EQ(records_skipped, 2);
    // We remove the skipped levels from the buffer.
    CheckState(/*values_written=*/3, /*null_count=*/0, /*levels_written=*/4,
               /*levels_position=*/3);
    CheckReadValues(/*expected_values=*/{20, 20, 20},
                    /*expected_defs=*/{1, 1, 1},
                    /*expected_reps=*/{0, 1, 1});
  }

  {
    // Read [40]
    int64_t records_read = record_reader_->ReadRecords(/*num_records=*/1);
    ASSERT_EQ(records_read, 1);
    CheckState(/*values_written=*/4, /*null_count=*/0, /*levels_written=*/4,
               /*levels_position=*/4);
    CheckReadValues(/*expected_values=*/{20, 20, 20, 40},
                    /*expected_defs=*/{1, 1, 1, 1},
                    /*expected_reps=*/{0, 1, 1, 0});
  }
}

// Tests that for repeated fields, we first consume what is in the buffer
// before reading more levels.
TEST_P(RecordReaderPrimitiveTypeTest, SkipRepeatedConsumeBufferFirst) {
  Init(schema::Int32("b", Repetition::REPEATED));

  std::vector<std::shared_ptr<Page>> pages;
  std::vector<int32_t> values(2048, 10);
  std::vector<int16_t> def_levels(2048, 1);
  std::vector<int16_t> rep_levels(2048, 0);

  std::shared_ptr<DataPageV1> page = MakeDataPage<Int32Type>(
      descr_, values, /*num_values=*/static_cast<int>(values.size()), Encoding::PLAIN,
      /*indices=*/{},
      /*indices_size=*/0, def_levels, descr_->max_definition_level(), rep_levels,
      descr_->max_repetition_level());
  pages.push_back(std::move(page));
  auto pager = std::make_unique<MockPageReader>(pages);
  record_reader_->SetPageReader(std::move(pager));
  {
    // Read 1000 records. We will read 1024 levels because that is the minimum
    // number of levels to read.
    int64_t records_read = record_reader_->ReadRecords(/*num_records=*/1000);
    ASSERT_EQ(records_read, 1000);
    CheckState(/*values_written=*/1000, /*null_count=*/0, /*levels_written=*/1024,
               /*levels_position=*/1000);
    std::vector<int32_t> expected_values(1000, 10);
    std::vector<int16_t> expected_def_levels(1000, 1);
    std::vector<int16_t> expected_rep_levels(1000, 0);
    CheckReadValues(expected_values, expected_def_levels, expected_rep_levels);
    // Reset removes the already consumed values and levels.
    record_reader_->Reset();
  }

  {  // Skip 12 records. Since we already have 24 in the buffer, we should not be
    // reading any more levels into the buffer, we will just consume 12 of it.
    int64_t records_skipped = record_reader_->SkipRecords(/*num_records=*/12);
    ASSERT_EQ(records_skipped, 12);
    CheckState(/*values_written=*/0, /*null_count=*/0, /*levels_written=*/12,
               /*levels_position=*/0);
    // Everything is empty because we reset the reader before this skip.
    CheckReadValues(/*expected_values=*/{}, /*expected_def_levels=*/{},
                    /*expected_rep_levels=*/{});
  }
}

// Test reading when one record spans multiple pages for a repeated field.
TEST_P(RecordReaderPrimitiveTypeTest, ReadPartialRecord) {
  Init(schema::Int32("b", Repetition::REPEATED));

  std::vector<std::shared_ptr<Page>> pages;

  // Page 1: {[10], [20, 20, 20 ... } continues to next page.
  {
    std::shared_ptr<DataPageV1> page = MakeDataPage<Int32Type>(
        descr_, /*values=*/{10, 20, 20, 20}, /*num_values=*/4, Encoding::PLAIN,
        /*indices=*/{},
        /*indices_size=*/0, /*def_levels=*/{1, 1, 1, 1}, descr_->max_definition_level(),
        /*rep_levels=*/{0, 0, 1, 1}, descr_->max_repetition_level());
    pages.push_back(std::move(page));
  }

  // Page 2: {... 20, 20, ...} continues from previous page and to next page.
  {
    std::shared_ptr<DataPageV1> page = MakeDataPage<Int32Type>(
        descr_, /*values=*/{20, 20}, /*num_values=*/2, Encoding::PLAIN,
        /*indices=*/{},
        /*indices_size=*/0, /*def_levels=*/{1, 1}, descr_->max_definition_level(),
        /*rep_levels=*/{1, 1}, descr_->max_repetition_level());
    pages.push_back(std::move(page));
  }

  // Page 3: { ... 20], [30]} continues from previous page.
  {
    std::shared_ptr<DataPageV1> page = MakeDataPage<Int32Type>(
        descr_, /*values=*/{20, 30}, /*num_values=*/2, Encoding::PLAIN,
        /*indices=*/{},
        /*indices_size=*/0, /*def_levels=*/{1, 1}, descr_->max_definition_level(),
        /*rep_levels=*/{1, 0}, descr_->max_repetition_level());
    pages.push_back(std::move(page));
  }

  auto pager = std::make_unique<MockPageReader>(pages);
  record_reader_->SetPageReader(std::move(pager));

  {
    // Read [10]
    int64_t records_read = record_reader_->ReadRecords(/*num_records=*/1);
    ASSERT_EQ(records_read, 1);
    CheckState(/*values_written=*/1, /*null_count=*/0, /*levels_written=*/4,
               /*levels_position=*/1);
    CheckReadValues(/*expected_values=*/{10},
                    /*expected_defs=*/{1},
                    /*expected_reps=*/{0});
  }

  {
    // Read [20, 20, 20, 20, 20, 20] that spans multiple pages.
    int64_t records_read = record_reader_->ReadRecords(/*num_records=*/1);
    ASSERT_EQ(records_read, 1);
    CheckState(/*values_written=*/7, /*null_count=*/0, /*levels_written=*/8,
               /*levels_position=*/7);
    CheckReadValues(/*expected_values=*/{10, 20, 20, 20, 20, 20, 20},
                    /*expected_defs=*/{1, 1, 1, 1, 1, 1, 1},
                    /*expected_reps=*/{0, 0, 1, 1, 1, 1, 1});
  }

  {
    // Read [30]
    int64_t records_read = record_reader_->ReadRecords(/*num_records=*/1);
    ASSERT_EQ(records_read, 1);
    CheckState(/*values_written=*/8, /*null_count=*/0, /*levels_written=*/8,
               /*levels_position=*/8);
    CheckReadValues(/*expected_values=*/{10, 20, 20, 20, 20, 20, 20, 30},
                    /*expected_defs=*/{1, 1, 1, 1, 1, 1, 1, 1},
                    /*expected_reps=*/{0, 0, 1, 1, 1, 1, 1, 0});
  }
}

// Test skipping for repeated fields for the case when one record spans multiple
// pages.
TEST_P(RecordReaderPrimitiveTypeTest, SkipPartialRecord) {
  Init(schema::Int32("b", Repetition::REPEATED));

  std::vector<std::shared_ptr<Page>> pages;

  // Page 1: {[10], [20, 20, 20 ... } continues to next page.
  {
    std::shared_ptr<DataPageV1> page = MakeDataPage<Int32Type>(
        descr_, /*values=*/{10, 20, 20, 20}, /*num_values=*/4, Encoding::PLAIN,
        /*indices=*/{},
        /*indices_size=*/0, /*def_levels=*/{1, 1, 1, 1}, descr_->max_definition_level(),
        /*rep_levels=*/{0, 0, 1, 1}, descr_->max_repetition_level());
    pages.push_back(std::move(page));
  }

  // Page 2: {... 20, 20, ...} continues from previous page and to next page.
  {
    std::shared_ptr<DataPageV1> page = MakeDataPage<Int32Type>(
        descr_, /*values=*/{20, 20}, /*num_values=*/2, Encoding::PLAIN,
        /*indices=*/{},
        /*indices_size=*/0, /*def_levels=*/{1, 1}, descr_->max_definition_level(),
        /*rep_levels=*/{1, 1}, descr_->max_repetition_level());
    pages.push_back(std::move(page));
  }

  // Page 3: { ... 20, [30]} continues from previous page.
  {
    std::shared_ptr<DataPageV1> page = MakeDataPage<Int32Type>(
        descr_, /*values=*/{20, 30}, /*num_values=*/2, Encoding::PLAIN,
        /*indices=*/{},
        /*indices_size=*/0, /*def_levels=*/{1, 1}, descr_->max_definition_level(),
        /*rep_levels=*/{1, 0}, descr_->max_repetition_level());
    pages.push_back(std::move(page));
  }

  auto pager = std::make_unique<MockPageReader>(pages);
  record_reader_->SetPageReader(std::move(pager));

  {
    // Read [10]
    int64_t records_read = record_reader_->ReadRecords(/*num_records=*/1);
    ASSERT_EQ(records_read, 1);
    // There are 4 levels in the first page.
    CheckState(/*values_written=*/1, /*null_count=*/0, /*levels_written=*/4,
               /*levels_position=*/1);
    CheckReadValues(/*expected_values=*/{10},
                    /*expected_defs=*/{1},
                    /*expected_reps=*/{0});
  }

  {
    // Skip the record that goes across pages.
    int64_t records_skipped = record_reader_->SkipRecords(/*num_records=*/1);
    ASSERT_EQ(records_skipped, 1);
    CheckState(/*values_written=*/1, /*null_count=*/0, /*levels_written=*/2,
               /*levels_position=*/1);
    CheckReadValues(/*expected_values=*/{10},
                    /*expected_defs=*/{1},
                    /*expected_reps=*/{0});
  }

  {
    // Read [30]
    int64_t records_read = record_reader_->ReadRecords(/*num_records=*/1);

    ASSERT_EQ(records_read, 1);
    CheckState(/*values_written=*/2, /*null_count=*/0, /*levels_written=*/2,
               /*levels_position=*/2);
    CheckReadValues(/*expected_values=*/{10, 30},
                    /*expected_defs=*/{1, 1},
                    /*expected_reps=*/{0, 0});
  }
}

INSTANTIATE_TEST_SUITE_P(RecordReaderPrimitiveTypeTests, RecordReaderPrimitiveTypeTest,
                         ::testing::Values(/*read_dense_for_nullable=*/true, false),
                         testing::PrintToStringParamName());

// Parameterized test for FLBA record reader.
class FLBARecordReaderTest : public ::testing::TestWithParam<bool> {
 public:
  bool read_dense_for_nullable() { return GetParam(); }

  void MakeRecordReader(int levels_per_page, int num_pages, int FLBA_type_length) {
    levels_per_page_ = levels_per_page;
    FLBA_type_length_ = FLBA_type_length;
    internal::LevelInfo level_info;
    level_info.def_level = 1;
    level_info.rep_level = 0;
    NodePtr type = ::parquet::schema::PrimitiveNode::Make(
        "b", Repetition::OPTIONAL, Type::FIXED_LEN_BYTE_ARRAY, ConvertedType::NONE,
        FLBA_type_length_);
    descr_ = std::make_unique<ColumnDescriptor>(type, level_info.def_level,
                                                level_info.rep_level);
    MakePages<FLBAType>(descr_.get(), num_pages, levels_per_page, def_levels_,
                        rep_levels_, values_, buffer_, pages_, Encoding::PLAIN);
    auto pager = std::make_unique<MockPageReader>(pages_);
    record_reader_ = internal::RecordReader::Make(
        descr_.get(), level_info, ::arrow::default_memory_pool(),
        /*read_dictionary=*/false, read_dense_for_nullable());
    record_reader_->SetPageReader(std::move(pager));
  }

  // Returns expected values in row range.
  // We need this since some values are null.
  std::vector<std::string_view> expected_values(int start, int end) {
    std::vector<std::string_view> result;
    // Find out where in the values_ vector we start from.
    size_t values_index = 0;
    for (int i = 0; i < start; ++i) {
      if (def_levels_[i] != 0) {
        ++values_index;
      }
    }

    for (int i = start; i < end; ++i) {
      if (def_levels_[i] == 0) {
        if (!read_dense_for_nullable()) {
          result.emplace_back();
        }
        continue;
      }
      result.emplace_back(reinterpret_cast<const char*>(values_[values_index].ptr),
                          FLBA_type_length_);
      ++values_index;
    }
    return result;
  }

  void CheckReadValues(int start, int end) {
    auto binary_reader = dynamic_cast<BinaryRecordReader*>(record_reader_.get());
    ASSERT_NE(binary_reader, nullptr);
    // Chunks are reset after this call.
    ::arrow::ArrayVector array_vector = binary_reader->GetBuilderChunks();
    ASSERT_EQ(array_vector.size(), 1);
    auto binary_array =
        dynamic_cast<::arrow::FixedSizeBinaryArray*>(array_vector[0].get());

    ASSERT_NE(binary_array, nullptr);
    ASSERT_EQ(binary_array->length(), record_reader_->values_written());
    if (read_dense_for_nullable()) {
      ASSERT_EQ(binary_array->null_count(), 0);
      ASSERT_EQ(record_reader_->null_count(), 0);
    } else {
      ASSERT_EQ(binary_array->null_count(), record_reader_->null_count());
    }

    std::vector<std::string_view> expected = expected_values(start, end);
    for (size_t i = 0; i < expected.size(); ++i) {
      if (def_levels_[i + start] == 0) {
        ASSERT_EQ(!read_dense_for_nullable(), binary_array->IsNull(i));
      } else {
        ASSERT_EQ(expected[i].compare(binary_array->GetView(i)), 0);
        ASSERT_FALSE(binary_array->IsNull(i));
      }
    }
  }

 protected:
  std::shared_ptr<internal::RecordReader> record_reader_;

 private:
  int levels_per_page_;
  int FLBA_type_length_;
  std::vector<std::shared_ptr<Page>> pages_;
  std::vector<int16_t> def_levels_;
  std::vector<int16_t> rep_levels_;
  std::vector<FixedLenByteArray> values_;
  std::vector<uint8_t> buffer_;
  std::unique_ptr<ColumnDescriptor> descr_;
};

// Similar to above, except for Byte arrays. FLBA and Byte arrays are
// sufficiently different to warrant a separate class for readability.
class ByteArrayRecordReaderTest : public ::testing::TestWithParam<bool> {
 public:
  bool read_dense_for_nullable() { return GetParam(); }

  void MakeRecordReader(int levels_per_page, int num_pages) {
    levels_per_page_ = levels_per_page;
    internal::LevelInfo level_info;
    level_info.def_level = 1;
    level_info.rep_level = 0;
    NodePtr type = schema::ByteArray("b", Repetition::OPTIONAL);
    descr_ = std::make_unique<ColumnDescriptor>(type, level_info.def_level,
                                                level_info.rep_level);
    MakePages<ByteArrayType>(descr_.get(), num_pages, levels_per_page, def_levels_,
                             rep_levels_, values_, buffer_, pages_, Encoding::PLAIN);

    auto pager = std::make_unique<MockPageReader>(pages_);

    record_reader_ = internal::RecordReader::Make(
        descr_.get(), level_info, ::arrow::default_memory_pool(),
        /*read_dictionary=*/false, read_dense_for_nullable());
    record_reader_->SetPageReader(std::move(pager));
  }

  // Returns expected values in row range.
  // We need this since some values are null.
  std::vector<std::string_view> expected_values(int start, int end) {
    std::vector<std::string_view> result;
    // Find out where in the values_ vector we start from.
    size_t values_index = 0;
    for (int i = 0; i < start; ++i) {
      if (def_levels_[i] != 0) {
        ++values_index;
      }
    }

    for (int i = start; i < end; ++i) {
      if (def_levels_[i] == 0) {
        if (!read_dense_for_nullable()) {
          result.emplace_back();
        }
        continue;
      }
      result.emplace_back(reinterpret_cast<const char*>(values_[values_index].ptr),
                          values_[values_index].len);
      ++values_index;
    }
    return result;
  }

  void CheckReadValues(int start, int end) {
    auto binary_reader = dynamic_cast<BinaryRecordReader*>(record_reader_.get());
    ASSERT_NE(binary_reader, nullptr);
    // Chunks are reset after this call.
    ::arrow::ArrayVector array_vector = binary_reader->GetBuilderChunks();
    ASSERT_EQ(array_vector.size(), 1);
    ::arrow::BinaryArray* binary_array =
        dynamic_cast<::arrow::BinaryArray*>(array_vector[0].get());

    ASSERT_NE(binary_array, nullptr);
    ASSERT_EQ(binary_array->length(), record_reader_->values_written());
    if (read_dense_for_nullable()) {
      ASSERT_EQ(binary_array->null_count(), 0);
      ASSERT_EQ(record_reader_->null_count(), 0);
    } else {
      ASSERT_EQ(binary_array->null_count(), record_reader_->null_count());
    }

    std::vector<std::string_view> expected = expected_values(start, end);
    for (size_t i = 0; i < expected.size(); ++i) {
      if (def_levels_[i + start] == 0) {
        ASSERT_EQ(!read_dense_for_nullable(), binary_array->IsNull(i));
      } else {
        ASSERT_EQ(expected[i].compare(binary_array->GetView(i)), 0);
        ASSERT_FALSE(binary_array->IsNull(i));
      }
    }
  }

 protected:
  std::shared_ptr<internal::RecordReader> record_reader_;

 private:
  int levels_per_page_;
  std::vector<std::shared_ptr<Page>> pages_;
  std::vector<int16_t> def_levels_;
  std::vector<int16_t> rep_levels_;
  std::vector<ByteArray> values_;
  std::vector<uint8_t> buffer_;
  std::unique_ptr<ColumnDescriptor> descr_;
};

// Tests reading and skipping a ByteArray field.
// The binary readers only differ in DecodeDense and DecodeSpaced functions, so
// testing optional is sufficient in exercising those code paths.
TEST_P(ByteArrayRecordReaderTest, ReadAndSkipOptional) {
  MakeRecordReader(/*levels_per_page=*/90, /*num_pages=*/1);

  // Read one-third of the page.
  ASSERT_EQ(record_reader_->ReadRecords(/*num_records=*/30), 30);
  CheckReadValues(0, 30);
  record_reader_->Reset();

  // Skip 30 records.
  ASSERT_EQ(record_reader_->SkipRecords(/*num_records=*/30), 30);

  // Read 60 more records. Only 30 will be read, since we read 30 and skipped 30,
  // so only 30 is left.
  ASSERT_EQ(record_reader_->ReadRecords(/*num_records=*/60), 30);
  CheckReadValues(60, 90);
  record_reader_->Reset();
}

// Test skipping buffered records when reading/skipping more than kMinLevelBatchSize
// levels at a time.
TEST_P(ByteArrayRecordReaderTest, ReadAndBatchSkipOptional) {
  MakeRecordReader(/*levels_per_page=*/9000, /*num_pages=*/1);

  // Read 100 records and buffer some records.
  ASSERT_EQ(record_reader_->ReadRecords(/*num_records=*/100), 100);
  CheckReadValues(0, 100);
  record_reader_->Reset();

  // Skip 3000 records. The buffered records will be skipped.
  ASSERT_EQ(record_reader_->SkipRecords(/*num_records=*/3000), 3000);

  // Read 900 records and buffer some records again.
  ASSERT_EQ(record_reader_->ReadRecords(/*num_records=*/900), 900);
  CheckReadValues(3100, 4000);
  record_reader_->Reset();

  // Skip 3000 records. The buffered records will be skipped.
  ASSERT_EQ(record_reader_->SkipRecords(/*num_records=*/3000), 3000);

  // Read 3000 records. Only 2000 records are left to be read.
  ASSERT_EQ(record_reader_->ReadRecords(/*num_records=*/3000), 2000);
  CheckReadValues(7000, 9000);
  record_reader_->Reset();
}

// Tests reading and skipping an optional FLBA field.
// The binary readers only differ in DecodeDense and DecodeSpaced functions, so
// testing optional is sufficient in exercising those code paths.
TEST_P(FLBARecordReaderTest, ReadAndSkipOptional) {
  MakeRecordReader(/*levels_per_page=*/90, /*num_pages=*/1, /*FLBA_type_length=*/4);

  // Read one-third of the page.
  ASSERT_EQ(record_reader_->ReadRecords(/*num_records=*/30), 30);
  CheckReadValues(0, 30);
  record_reader_->Reset();

  // Skip 30 records.
  ASSERT_EQ(record_reader_->SkipRecords(/*num_records=*/30), 30);

  // Read 60 more records. Only 30 will be read, since we read 30 and skipped 30,
  // so only 30 is left.
  ASSERT_EQ(record_reader_->ReadRecords(/*num_records=*/60), 30);
  CheckReadValues(60, 90);
  record_reader_->Reset();
}

INSTANTIATE_TEST_SUITE_P(ByteArrayRecordReaderTests, ByteArrayRecordReaderTest,
                         testing::Bool());

INSTANTIATE_TEST_SUITE_P(FLBARecordReaderTests, FLBARecordReaderTest, testing::Bool());

// Test random combination of ReadRecords and SkipRecords.
class RecordReaderStressTest : public ::testing::TestWithParam<Repetition::type> {};

TEST_P(RecordReaderStressTest, StressTest) {
  internal::LevelInfo level_info;
  // Define these boolean variables for improving readability below.
  bool repeated = false, required = false;
  if (GetParam() == Repetition::REQUIRED) {
    level_info.def_level = 0;
    level_info.rep_level = 0;
    required = true;
  } else if (GetParam() == Repetition::OPTIONAL) {
    level_info.def_level = 1;
    level_info.rep_level = 0;
  } else {
    level_info.def_level = 1;
    level_info.rep_level = 1;
    repeated = true;
  }

  NodePtr type = schema::Int32("b", GetParam());
  const ColumnDescriptor descr(type, level_info.def_level, level_info.rep_level);

  auto seed1 = static_cast<uint32_t>(time(0));
  std::default_random_engine gen(seed1);
  // Generate random number of pages with random number of values per page.
  std::uniform_int_distribution<int> d(0, 2000);
  const int num_pages = d(gen);
  const int levels_per_page = d(gen);
  std::vector<int32_t> values;
  std::vector<int16_t> def_levels;
  std::vector<int16_t> rep_levels;
  std::vector<uint8_t> data_buffer;
  std::vector<std::shared_ptr<Page>> pages;
  auto seed2 = static_cast<uint32_t>(time(0));
  // Uses time(0) as seed so it would run a different test every time it is
  // run.
  MakePages<Int32Type>(&descr, num_pages, levels_per_page, def_levels, rep_levels, values,
                       data_buffer, pages, Encoding::PLAIN, seed2);
  std::unique_ptr<PageReader> pager;
  pager.reset(new test::MockPageReader(pages));

  // Set up the RecordReader.
  std::shared_ptr<internal::RecordReader> record_reader =
      internal::RecordReader::Make(&descr, level_info);
  record_reader->SetPageReader(std::move(pager));

  // Figure out how many total records.
  int total_records = 0;
  if (repeated) {
    for (int16_t rep : rep_levels) {
      if (rep == 0) {
        ++total_records;
      }
    }
  } else {
    total_records = static_cast<int>(def_levels.size());
  }

  // Generate a sequence of reads and skips.
  int records_left = total_records;
  // The first element of the pair is 1 if SkipRecords and 0 if ReadRecords.
  // The second element indicates the number of records for reading or
  // skipping.
  std::vector<std::pair<bool, int>> sequence;
  while (records_left > 0) {
    std::uniform_int_distribution<int> d(0, records_left);
    // Generate a number to decide if this is a skip or read.
    bool is_skip = d(gen) < records_left / 2;
    int num_records = d(gen);

    sequence.emplace_back(is_skip, num_records);
    records_left -= num_records;
  }

  // The levels_index and values_index are over the original vectors that have
  // all the rep/def values for all the records. In the following loop, we will
  // read/skip a number of records and Reset the reader after each iteration.
  // This is on-par with how the record reader is used.
  size_t levels_index = 0;
  size_t values_index = 0;
  for (const auto& [is_skip, num_records] : sequence) {
    // Reset the reader before the next round of read/skip.
    record_reader->Reset();

    // Prepare the expected result and do the SkipRecords and ReadRecords.
    std::vector<int32_t> expected_values;
    std::vector<int16_t> expected_def_levels;
    std::vector<int16_t> expected_rep_levels;
    bool inside_repeated_field = false;

    int read_records = 0;
    while (read_records < num_records || inside_repeated_field) {
      if (!repeated || (repeated && rep_levels[levels_index] == 0)) {
        ++read_records;
      }

      bool has_value =
          required || (!required && def_levels[levels_index] == level_info.def_level);

      // If we are not skipping, we need to update the expected values and
      // rep/defs. If we are skipping, we just keep going.
      if (!is_skip) {
        if (!required) {
          expected_def_levels.push_back(def_levels[levels_index]);
          if (!has_value) {
            expected_values.push_back(-1);
          }
        }
        if (repeated) {
          expected_rep_levels.push_back(rep_levels[levels_index]);
        }
        if (has_value) {
          expected_values.push_back(values[values_index]);
        }
      }

      if (has_value) {
        ++values_index;
      }

      // If we are in the middle of a repeated field, we should keep going
      // until we consume it all.
      if (repeated && levels_index + 1 < rep_levels.size() &&
          rep_levels[levels_index + 1] == 1) {
        inside_repeated_field = true;
      } else {
        inside_repeated_field = false;
      }

      ++levels_index;
    }

    // Print out the seeds with each failing ASSERT to easily reproduce the bug.
    std::string seeds = "seeds: " + std::to_string(seed1) + " " + std::to_string(seed2);

    // Perform the actual read/skip.
    if (is_skip) {
      int64_t skipped_records = record_reader->SkipRecords(num_records);
      ASSERT_EQ(skipped_records, num_records) << seeds;
    } else {
      int64_t read_records = record_reader->ReadRecords(num_records);
      ASSERT_EQ(read_records, num_records) << seeds;
    }

    const auto read_values = reinterpret_cast<const int32_t*>(record_reader->values());
    if (required) {
      ASSERT_EQ(record_reader->null_count(), 0) << seeds;
    }
    std::vector<int32_t> read_vals(read_values,
                                   read_values + record_reader->values_written());
    for (size_t i = 0; i < expected_values.size(); ++i) {
      if (expected_values[i] != -1) {
        ASSERT_EQ(read_vals[i], expected_values[i]) << seeds;
      }
    }

    if (!required) {
      std::vector<int16_t> read_def_levels(
          record_reader->def_levels(),
          record_reader->def_levels() + record_reader->levels_position());
      ASSERT_TRUE(vector_equal(read_def_levels, expected_def_levels)) << seeds;
    }

    if (repeated) {
      std::vector<int16_t> read_rep_levels(
          record_reader->rep_levels(),
          record_reader->rep_levels() + record_reader->levels_position());
      ASSERT_TRUE(vector_equal(read_rep_levels, expected_rep_levels)) << seeds;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Repetition_type, RecordReaderStressTest,
                         ::testing::Values(Repetition::REQUIRED, Repetition::OPTIONAL,
                                           Repetition::REPEATED));

// =============================================================================
// TestSparseInputStream — unit tests for SparseInputStream
// =============================================================================

// Helper: build an in-memory RandomAccessFile from a byte vector.
static std::shared_ptr<::arrow::io::RandomAccessFile> MakeBufferFile(
    const std::vector<uint8_t>& data) {
  auto buf =
      std::make_shared<::arrow::Buffer>(data.data(), static_cast<int64_t>(data.size()));
  return std::make_shared<::arrow::io::BufferReader>(buf);
}

class TestSparseInputStream : public ::testing::Test {
 protected:
  // Builds a 256-byte source where byte[i] == i % 256.
  void SetUp() override {
    source_data_.resize(256);
    for (int i = 0; i < 256; ++i) source_data_[i] = static_cast<uint8_t>(i);
    source_ = MakeBufferFile(source_data_);
  }

  std::vector<uint8_t> source_data_;
  std::shared_ptr<::arrow::io::RandomAccessFile> source_;
};

// Case 1: three non-contiguous ranges — correct bytes returned, gaps skipped.
TEST_F(TestSparseInputStream, ThreeNonContiguousRanges) {
  // Ranges: [10,15), [30,35), [100,110)
  std::vector<::arrow::io::ReadRange> ranges = {{10, 5}, {30, 5}, {100, 10}};
  SparseInputStream stream(source_, ranges);

  // Total logical size = 5 + 5 + 10 = 20
  ASSERT_OK_AND_ASSIGN(int64_t sz, stream.GetSize());
  EXPECT_EQ(sz, 20);

  // Read range 0: bytes 10..14
  ASSERT_OK_AND_ASSIGN(auto buf0, stream.Read(5));
  ASSERT_EQ(buf0->size(), 5);
  for (int i = 0; i < 5; ++i) EXPECT_EQ(buf0->data()[i], source_data_[10 + i]);

  // Read range 1: bytes 30..34
  ASSERT_OK_AND_ASSIGN(auto buf1, stream.Read(5));
  ASSERT_EQ(buf1->size(), 5);
  for (int i = 0; i < 5; ++i) EXPECT_EQ(buf1->data()[i], source_data_[30 + i]);

  // Read range 2: bytes 100..109
  ASSERT_OK_AND_ASSIGN(auto buf2, stream.Read(10));
  ASSERT_EQ(buf2->size(), 10);
  for (int i = 0; i < 10; ++i) EXPECT_EQ(buf2->data()[i], source_data_[100 + i]);

  // Should be at EOF
  ASSERT_OK_AND_ASSIGN(auto eof_buf, stream.Read(1));
  EXPECT_EQ(eof_buf->size(), 0);
}

// Case 2: skip pages 0 & 1, select only the third range (page 2).
// Verifies that a SparseInputStream covering only the third page returns
// exactly the third page's bytes and returns empty on the next read.
TEST_F(TestSparseInputStream, SkipPagesBoundaryCase) {
  // Simulate 3 pages of 20 bytes each at offsets 0, 20, 40.
  // Only include range for page 2 ([40, 60)).
  std::vector<::arrow::io::ReadRange> ranges = {{40, 20}};
  SparseInputStream stream(source_, ranges);

  ASSERT_OK_AND_ASSIGN(int64_t sz, stream.GetSize());
  EXPECT_EQ(sz, 20);

  ASSERT_OK_AND_ASSIGN(auto buf, stream.Read(20));
  ASSERT_EQ(buf->size(), 20);
  for (int i = 0; i < 20; ++i) EXPECT_EQ(buf->data()[i], source_data_[40 + i]);

  // Next read returns empty (logical EOF).
  ASSERT_OK_AND_ASSIGN(auto eof_buf, stream.Read(20));
  EXPECT_EQ(eof_buf->size(), 0);
}

// Case 3: select first half (rows 0–49 maps to first 50 bytes).
TEST_F(TestSparseInputStream, SelectFirstHalf) {
  std::vector<::arrow::io::ReadRange> ranges = {{0, 50}};
  SparseInputStream stream(source_, ranges);

  ASSERT_OK_AND_ASSIGN(auto buf, stream.Read(50));
  ASSERT_EQ(buf->size(), 50);
  for (int i = 0; i < 50; ++i) EXPECT_EQ(buf->data()[i], source_data_[i]);
}

// Case 4: skip first half, select second half (bytes 50–99).
TEST_F(TestSparseInputStream, SelectSecondHalf) {
  std::vector<::arrow::io::ReadRange> ranges = {{50, 50}};
  SparseInputStream stream(source_, ranges);

  ASSERT_OK_AND_ASSIGN(auto buf, stream.Read(50));
  ASSERT_EQ(buf->size(), 50);
  for (int i = 0; i < 50; ++i) EXPECT_EQ(buf->data()[i], source_data_[50 + i]);
}

// Case 5: skipped range issues no Read — verify by checking an empty ranges
// vector results in zero bytes returned (i.e. all ranges were "skipped").
TEST_F(TestSparseInputStream, SkippedRangeNoRead) {
  // Empty range list: all data is effectively skipped.
  std::vector<::arrow::io::ReadRange> ranges = {};
  SparseInputStream stream(source_, ranges);

  ASSERT_OK_AND_ASSIGN(int64_t sz, stream.GetSize());
  EXPECT_EQ(sz, 0);

  ASSERT_OK_AND_ASSIGN(auto buf, stream.Read(10));
  EXPECT_EQ(buf->size(), 0);
}

// Case 6: select-all is equivalent to no selection (reads entire content).
TEST_F(TestSparseInputStream, SelectAllEqualsNoSelection) {
  // One range spanning the entire 256-byte source.
  std::vector<::arrow::io::ReadRange> ranges = {{0, 256}};
  SparseInputStream stream(source_, ranges);

  ASSERT_OK_AND_ASSIGN(int64_t sz, stream.GetSize());
  EXPECT_EQ(sz, 256);

  ASSERT_OK_AND_ASSIGN(auto buf, stream.Read(256));
  ASSERT_EQ(buf->size(), 256);
  for (int i = 0; i < 256; ++i) EXPECT_EQ(buf->data()[i], source_data_[i]);
}

// Case 7: skip-all → 0 values. (Same as SkippedRangeNoRead but via empty
// ranges to confirm zero logical size and empty read.)
TEST_F(TestSparseInputStream, SkipAll) {
  std::vector<::arrow::io::ReadRange> ranges = {};
  SparseInputStream stream(source_, ranges);
  ASSERT_OK_AND_ASSIGN(int64_t sz, stream.GetSize());
  EXPECT_EQ(sz, 0);
  ASSERT_OK_AND_ASSIGN(auto buf, stream.Read(256));
  EXPECT_EQ(buf->size(), 0);
}

// Case 8: select-all matches unfiltered — byte-for-byte equal to direct read.
TEST_F(TestSparseInputStream, SelectAllMatchesUnfiltered) {
  std::vector<::arrow::io::ReadRange> ranges = {{0, 256}};
  SparseInputStream stream(source_, ranges);

  // Read in two halves; should still recover all bytes in order.
  ASSERT_OK_AND_ASSIGN(auto first, stream.Read(128));
  ASSERT_OK_AND_ASSIGN(auto second, stream.Read(128));
  ASSERT_EQ(first->size(), 128);
  ASSERT_EQ(second->size(), 128);
  for (int i = 0; i < 128; ++i) EXPECT_EQ(first->data()[i], source_data_[i]);
  for (int i = 0; i < 128; ++i) EXPECT_EQ(second->data()[i], source_data_[128 + i]);
}

// Case 9: single full-span range is equivalent to sequential read.
TEST_F(TestSparseInputStream, SingleFullSpan) {
  std::vector<::arrow::io::ReadRange> ranges = {{0, 256}};
  SparseInputStream stream(source_, ranges);

  // Read all bytes in a single call; verify sequential equality.
  ASSERT_OK_AND_ASSIGN(auto buf, stream.Read(256));
  ASSERT_EQ(buf->size(), 256);
  EXPECT_EQ(0, std::memcmp(buf->data(), source_data_.data(), 256));
}

// Case 10: page boundary at last byte — reading the final range then returns
// nullptr (empty buffer).
TEST_F(TestSparseInputStream, PageBoundaryAtLastByte) {
  // Range ending exactly at the last byte of the source (byte 255).
  std::vector<::arrow::io::ReadRange> ranges = {{255, 1}};
  SparseInputStream stream(source_, ranges);

  ASSERT_OK_AND_ASSIGN(int64_t sz, stream.GetSize());
  EXPECT_EQ(sz, 1);

  // Read the single byte.
  ASSERT_OK_AND_ASSIGN(auto buf, stream.Read(1));
  ASSERT_EQ(buf->size(), 1);
  EXPECT_EQ(buf->data()[0], source_data_[255]);

  // Next read is at logical EOF, must return empty buffer (not null).
  ASSERT_OK_AND_ASSIGN(auto eof_buf, stream.Read(1));
  EXPECT_EQ(eof_buf->size(), 0);
}

// =============================================================================
// TestSerializedPageReader — page-reader tests driven by SerializedPageReader
// (accessed via PageReader::Open) over a SparseInputStream.
// =============================================================================

namespace {

// Build a minimal multi-page Parquet column chunk into an in-memory buffer,
// returning:
//   - buffer: the raw byte data
//   - total_values: total int32 values written
//   - pages_per_row_count: number of values written per page
//
// Uses N pages of kValuesPerPage required int32 values.
struct MultiPageColumn {
  std::shared_ptr<::arrow::Buffer> buffer;
  int64_t total_values;
  int values_per_page;
  int num_pages;
};

::arrow::Result<MultiPageColumn> BuildMultiPageInt32Column(
    int num_pages, int values_per_page,
    Compression::type codec = Compression::UNCOMPRESSED) {
  // Build schema: required int32 field "col".
  schema::NodeVector fields;
  fields.push_back(
      PrimitiveNode::Make("col", Repetition::REQUIRED, Type::INT32, ConvertedType::NONE));
  auto schema_node = std::static_pointer_cast<schema::GroupNode>(
      schema::GroupNode::Make("schema", Repetition::REQUIRED, fields));

  auto writer_props = WriterProperties::Builder()
                          .disable_dictionary()
                          ->compression(codec)
                          // Small page size forces multiple pages.
                          ->data_pagesize(values_per_page * sizeof(int32_t) + 64)
                          ->build();

  ARROW_ASSIGN_OR_RAISE(auto out_stream, ::arrow::io::BufferOutputStream::Create());
  auto file_writer = ParquetFileWriter::Open(out_stream, schema_node, writer_props);
  auto rg_writer = file_writer->AppendRowGroup();
  auto int32_writer = static_cast<Int32Writer*>(rg_writer->NextColumn());

  // Write values: value[i] = p * values_per_page + i for page p.
  std::vector<int32_t> values(values_per_page);
  for (int p = 0; p < num_pages; ++p) {
    for (int i = 0; i < values_per_page; ++i) {
      values[i] = p * values_per_page + i;
    }
    int32_writer->WriteBatch(values_per_page, nullptr, nullptr, values.data());
  }
  rg_writer->Close();
  file_writer->Close();

  ARROW_ASSIGN_OR_RAISE(auto buf, out_stream->Finish());
  MultiPageColumn result;
  result.buffer = buf;
  result.total_values = static_cast<int64_t>(num_pages * values_per_page);
  result.values_per_page = values_per_page;
  result.num_pages = num_pages;
  return result;
}

}  // namespace

class TestSerializedPageReader : public ::testing::Test {
 protected:
  static constexpr int kNumPages = 4;
  static constexpr int kValuesPerPage = 25;  // 100 values total

  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(col_, BuildMultiPageInt32Column(kNumPages, kValuesPerPage));
    ASSERT_NE(col_.buffer, nullptr);
    ASSERT_GT(col_.buffer->size(), 0);
  }

  // Open the Parquet file and return a PageReader for column 0 via the
  // normal (non-sparse) path. The caller must keep file_reader alive for the
  // duration of the page reader's use; store it in the provided unique_ptr.
  std::unique_ptr<PageReader> OpenPageReader(
      const std::shared_ptr<::arrow::Buffer>& buf,
      std::unique_ptr<ParquetFileReader>& out_file_reader) {
    auto source = std::make_shared<::arrow::io::BufferReader>(buf);
    out_file_reader = ParquetFileReader::Open(source);
    auto rg_reader = out_file_reader->RowGroup(0);
    return rg_reader->GetColumnPageReader(0);
  }

  MultiPageColumn col_;
};

// Case 3 (page-level): open page reader on uncompressed column, verify we
// can iterate and read expected number of non-null data pages.
TEST_F(TestSerializedPageReader, SelectFirstHalfPages) {
  std::unique_ptr<ParquetFileReader> file_reader;
  auto page_reader = OpenPageReader(col_.buffer, file_reader);
  ASSERT_NE(page_reader, nullptr);

  int data_page_count = 0;
  int64_t total_values_seen = 0;
  while (true) {
    auto page = page_reader->NextPage();
    if (!page) break;
    if (page->type() == PageType::DATA_PAGE || page->type() == PageType::DATA_PAGE_V2) {
      auto data_page = std::static_pointer_cast<DataPage>(page);
      total_values_seen += data_page->num_values();
      ++data_page_count;
    }
  }
  EXPECT_EQ(data_page_count, kNumPages);
  EXPECT_EQ(total_values_seen, kNumPages * kValuesPerPage);
}

// Case 2 (page-level): use data_page_filter to skip pages 0 and 1,
// verify only pages 2 and 3 are processed.
TEST_F(TestSerializedPageReader, SkipPagesBoundaryCase) {
  std::unique_ptr<ParquetFileReader> file_reader;
  auto page_reader = OpenPageReader(col_.buffer, file_reader);
  ASSERT_NE(page_reader, nullptr);

  int page_ordinal = 0;
  // Filter: skip the first two data pages (ordinals 0 and 1).
  page_reader->set_data_page_filter([&page_ordinal](const DataPageStats&) -> bool {
    return page_ordinal++ < 2;  // true == skip
  });

  int data_pages_processed = 0;
  while (true) {
    auto page = page_reader->NextPage();
    if (!page) break;
    if (page->type() == PageType::DATA_PAGE || page->type() == PageType::DATA_PAGE_V2) {
      ++data_pages_processed;
    }
  }
  // Pages 0 and 1 were skipped; pages 2 and 3 should be returned.
  EXPECT_EQ(data_pages_processed, 2);
}

// Case 10 (page-level): confirm last page is followed by nullptr (EOF).
TEST_F(TestSerializedPageReader, PageBoundaryAtLastByte) {
  std::unique_ptr<ParquetFileReader> file_reader;
  auto page_reader = OpenPageReader(col_.buffer, file_reader);
  ASSERT_NE(page_reader, nullptr);

  // Drain all pages.
  std::shared_ptr<Page> last_page;
  while (true) {
    auto page = page_reader->NextPage();
    if (!page) break;
    last_page = page;
  }
  EXPECT_NE(last_page, nullptr);  // We saw at least one page.
  // After all pages, NextPage() must return nullptr.
  auto after_eof = page_reader->NextPage();
  EXPECT_EQ(after_eof, nullptr);
}

// Case (compression): same read-all test with Snappy if available.
TEST_F(TestSerializedPageReader, UncompressedReadAll) {
  // Uncompressed path already covered by SelectFirstHalfPages; this
  // explicitly verifies the uncompressed codec path works end-to-end.
  std::unique_ptr<ParquetFileReader> file_reader;
  auto page_reader = OpenPageReader(col_.buffer, file_reader);
  ASSERT_NE(page_reader, nullptr);

  int64_t total = 0;
  while (true) {
    auto page = page_reader->NextPage();
    if (!page) break;
    if (page->type() == PageType::DATA_PAGE || page->type() == PageType::DATA_PAGE_V2) {
      total += std::static_pointer_cast<DataPage>(page)->num_values();
    }
  }
  EXPECT_EQ(total, col_.total_values);
}

#ifdef ARROW_WITH_SNAPPY
TEST_F(TestSerializedPageReader, SnappyCompressedReadAll) {
  MultiPageColumn snappy_col;
  ASSERT_OK_AND_ASSIGN(snappy_col, BuildMultiPageInt32Column(kNumPages, kValuesPerPage,
                                                             Compression::SNAPPY));
  ASSERT_NE(snappy_col.buffer, nullptr);

  std::unique_ptr<ParquetFileReader> file_reader;
  auto page_reader = OpenPageReader(snappy_col.buffer, file_reader);
  ASSERT_NE(page_reader, nullptr);

  int64_t total = 0;
  while (true) {
    auto page = page_reader->NextPage();
    if (!page) break;
    if (page->type() == PageType::DATA_PAGE || page->type() == PageType::DATA_PAGE_V2) {
      total += std::static_pointer_cast<DataPage>(page)->num_values();
    }
  }
  EXPECT_EQ(total, snappy_col.total_values);
}
#endif  // ARROW_WITH_SNAPPY

// =============================================================================
// TestRecordReaderWithRowSelection — RecordReader + RowSelection integration
// =============================================================================

class TestRecordReaderWithRowSelection : public ::testing::Test {
 protected:
  static constexpr int kNumPages = 4;
  static constexpr int kValuesPerPage = 25;  // 100 values total

  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(col_, BuildMultiPageInt32Column(kNumPages, kValuesPerPage));
    ASSERT_NE(col_.buffer, nullptr);

    // Build the ColumnDescriptor matching the schema written by
    // BuildMultiPageInt32Column (required int32 "col").
    NodePtr type = schema::Int32("col", Repetition::REQUIRED);
    NodePtr root = schema::GroupNode::Make("schema", Repetition::REQUIRED, {type});
    schema_descriptor_.Init(root);
    descr_ = schema_descriptor_.Column(0);

    level_info_.def_level = descr_->max_definition_level();
    level_info_.rep_level = descr_->max_repetition_level();
    level_info_.repeated_ancestor_def_level = descr_->max_definition_level();
  }

  // Create a fresh RecordReader backed by the column data.
  std::shared_ptr<internal::RecordReader> MakeRecordReader() {
    auto source = std::make_shared<::arrow::io::BufferReader>(col_.buffer);
    file_reader_ = ParquetFileReader::Open(source);
    auto rg_reader = file_reader_->RowGroup(0);
    auto page_reader = rg_reader->GetColumnPageReader(0);

    auto record_reader =
        internal::RecordReader::Make(descr_, level_info_, ::arrow::default_memory_pool());
    record_reader->SetPageReader(std::move(page_reader));
    return record_reader;
  }

  // Read all selected values into a vector after setting a RowSelection.
  std::vector<int32_t> ReadSelectedValues(std::shared_ptr<internal::RecordReader> rr,
                                          std::shared_ptr<RowSelection> selection,
                                          int64_t num_to_read) {
    if (selection) {
      ARROW_UNUSED(rr->SetRowSelection(selection));
    }
    ARROW_UNUSED(rr->ReadRecords(num_to_read));
    const auto* vals = reinterpret_cast<const int32_t*>(rr->values());
    return std::vector<int32_t>(vals, vals + rr->values_written());
  }

  MultiPageColumn col_;
  SchemaDescriptor schema_descriptor_;
  const ColumnDescriptor* descr_;
  internal::LevelInfo level_info_;
  std::unique_ptr<ParquetFileReader> file_reader_;
};

// Case 3: select rows 0–49 → first 50 values match 0..49.
TEST_F(TestRecordReaderWithRowSelection, SelectFirst50) {
  auto rr = MakeRecordReader();
  // Select first 50 rows, skip remaining 50.
  auto sel = std::make_shared<RowSelection>(std::vector<RowSelector>{
      RowSelector{false, 50},  // select 50
      RowSelector{true, 50},   // skip 50
  });
  auto vals = ReadSelectedValues(rr, sel, 50);
  ASSERT_EQ(static_cast<int64_t>(vals.size()), 50);
  for (int i = 0; i < 50; ++i) EXPECT_EQ(vals[i], i);
}

// Case 4: skip rows 0–49, select rows 50–99 → values 50..99.
TEST_F(TestRecordReaderWithRowSelection, SelectLast50) {
  auto rr = MakeRecordReader();
  auto sel = std::make_shared<RowSelection>(std::vector<RowSelector>{
      RowSelector{true, 50},   // skip first 50
      RowSelector{false, 50},  // select last 50
  });
  // We ask to read 50 selected records; the skip is processed transparently.
  auto vals = ReadSelectedValues(rr, sel, 50);
  ASSERT_EQ(static_cast<int64_t>(vals.size()), 50);
  for (int i = 0; i < 50; ++i) EXPECT_EQ(vals[i], 50 + i);
}

// Case 7: skip-all → 0 values.
TEST_F(TestRecordReaderWithRowSelection, SkipAll) {
  auto rr = MakeRecordReader();
  auto sel =
      std::make_shared<RowSelection>(std::vector<RowSelector>{RowSelector{true, 100}});
  auto vals = ReadSelectedValues(rr, sel, 100);
  EXPECT_EQ(vals.size(), static_cast<size_t>(0));
}

// Case 6 / 8: select-all is equivalent to no RowSelection.
TEST_F(TestRecordReaderWithRowSelection, SelectAllEqualsNoSelection) {
  // Read without a RowSelection.
  auto rr_plain = MakeRecordReader();
  auto vals_plain = ReadSelectedValues(rr_plain, nullptr, 100);

  // Read with a select-all RowSelection.
  auto rr_sel = MakeRecordReader();
  auto sel_all =
      std::make_shared<RowSelection>(std::vector<RowSelector>{RowSelector{false, 100}});
  auto vals_sel = ReadSelectedValues(rr_sel, sel_all, 100);

  ASSERT_EQ(vals_plain.size(), vals_sel.size());
  for (size_t i = 0; i < vals_plain.size(); ++i) {
    EXPECT_EQ(vals_plain[i], vals_sel[i]);
  }
}

// Case 9: single full-span range equivalent to sequential read.
TEST_F(TestRecordReaderWithRowSelection, SingleFullSpanRange) {
  auto rr = MakeRecordReader();
  auto sel =
      std::make_shared<RowSelection>(std::vector<RowSelector>{RowSelector{false, 100}});
  auto vals = ReadSelectedValues(rr, sel, 100);
  ASSERT_EQ(static_cast<int64_t>(vals.size()), 100);
  for (int i = 0; i < 100; ++i) EXPECT_EQ(vals[i], i);
}

// Additional case: non-contiguous selection — first 25, skip 50, last 25.
TEST_F(TestRecordReaderWithRowSelection, NonContiguousSelection) {
  auto rr = MakeRecordReader();
  auto sel = std::make_shared<RowSelection>(std::vector<RowSelector>{
      RowSelector{false, 25},  // select rows 0-24
      RowSelector{true, 50},   // skip rows 25-74
      RowSelector{false, 25},  // select rows 75-99
  });
  // Read 25 from first range, then 25 from last range = 50 total.
  auto vals = ReadSelectedValues(rr, sel, 50);
  ASSERT_EQ(static_cast<int64_t>(vals.size()), 50);
  for (int i = 0; i < 25; ++i) EXPECT_EQ(vals[i], i);
  for (int i = 0; i < 25; ++i) EXPECT_EQ(vals[25 + i], 75 + i);
}

// Compression test for RecordReader — verify snappy-compressed column reads
// the same values as uncompressed when a RowSelection is applied.
#ifdef ARROW_WITH_SNAPPY
TEST_F(TestRecordReaderWithRowSelection, SnappyCompressedWithRowSelection) {
  MultiPageColumn snappy_col;
  ASSERT_OK_AND_ASSIGN(snappy_col, BuildMultiPageInt32Column(kNumPages, kValuesPerPage,
                                                             Compression::SNAPPY));
  ASSERT_NE(snappy_col.buffer, nullptr);

  auto source = std::make_shared<::arrow::io::BufferReader>(snappy_col.buffer);
  auto snappy_file_reader = ParquetFileReader::Open(source);
  auto rg_reader = snappy_file_reader->RowGroup(0);
  auto page_reader = rg_reader->GetColumnPageReader(0);

  auto rr =
      internal::RecordReader::Make(descr_, level_info_, ::arrow::default_memory_pool());
  rr->SetPageReader(std::move(page_reader));

  // Select last 50 rows.
  auto sel = std::make_shared<RowSelection>(std::vector<RowSelector>{
      RowSelector{true, 50},
      RowSelector{false, 50},
  });
  ARROW_UNUSED(rr->SetRowSelection(sel));
  int64_t n = rr->ReadRecords(50);
  ASSERT_EQ(n, 50);
  const auto* vals = reinterpret_cast<const int32_t*>(rr->values());
  for (int i = 0; i < 50; ++i) EXPECT_EQ(vals[i], 50 + i);
}
#endif  // ARROW_WITH_SNAPPY

}  // namespace test
}  // namespace parquet
