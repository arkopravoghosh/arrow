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

#pragma once

#include <algorithm>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "parquet/arrow/schema.h"
#include "parquet/column_reader.h"
#include "parquet/file_reader.h"
#include "parquet/metadata.h"
#include "parquet/page_index.h"
#include "parquet/platform.h"
#include "parquet/row_selection.h"
#include "parquet/schema.h"

namespace arrow {

class Array;
class ChunkedArray;
class DataType;
class Field;
class KeyValueMetadata;
class Schema;

}  // namespace arrow

using arrow::Status;

namespace parquet {

class ArrowReaderProperties;

namespace arrow {

class ColumnReaderImpl;

// ----------------------------------------------------------------------
// Iteration utilities

// Abstraction to decouple row group iteration details from the ColumnReader,
// so we can read only a single row group if we want
class FileColumnIterator {
 public:
  explicit FileColumnIterator(int column_index, ParquetFileReader* reader,
                              std::vector<int> row_groups)
      : column_index_(column_index),
        reader_(reader),
        schema_(reader->metadata()->schema()),
        row_groups_(row_groups.begin(), row_groups.end()),
        row_group_index_(-1) {}

  virtual ~FileColumnIterator() {}

  virtual std::unique_ptr<::parquet::PageReader> NextChunk() {
    if (row_groups_.empty()) {
      return nullptr;
    }

    row_group_index_ = row_groups_.front();
    auto row_group_reader = reader_->RowGroup(row_group_index_);
    row_groups_.pop_front();
    return row_group_reader->GetColumnPageReader(column_index_);
  }

  const SchemaDescriptor* schema() const { return schema_; }

  const ColumnDescriptor* descr() const { return schema_->Column(column_index_); }

  std::shared_ptr<FileMetaData> metadata() const { return reader_->metadata(); }

  std::unique_ptr<RowGroupMetaData> row_group_metadata() const {
    return metadata()->RowGroup(row_group_index_);
  }

  std::unique_ptr<ColumnChunkMetaData> column_chunk_metadata() const {
    return row_group_metadata()->ColumnChunk(column_index_);
  }

  int column_index() const { return column_index_; }

  int row_group_index() const { return row_group_index_; }

 protected:
  int column_index_;
  ParquetFileReader* reader_;
  const SchemaDescriptor* schema_;
  std::deque<int> row_groups_;
  int row_group_index_;
};

using FileColumnIteratorFactory =
    std::function<FileColumnIterator*(int, ParquetFileReader*)>;

// SparsePagesColumnIterator enables page-level I/O pruning by using RowSelection
// to skip non-matching pages during reads. It overrides NextChunk() to call
// GetColumnPageReaderWithRowSelection instead of GetColumnPageReader.
class SparsePagesColumnIterator : public FileColumnIterator {
 public:
  explicit SparsePagesColumnIterator(
      int column_index, ParquetFileReader* reader, std::vector<int> row_groups,
      std::map<int, std::shared_ptr<RowSelection>> row_selections)
      : FileColumnIterator(column_index, reader, std::move(row_groups)),
        row_selections_(std::move(row_selections)) {}

  std::unique_ptr<::parquet::PageReader> NextChunk() override {
    if (row_groups_.empty()) {
      return nullptr;
    }

    row_group_index_ = row_groups_.front();
    auto row_group_reader = reader_->RowGroup(row_group_index_);
    row_groups_.pop_front();

    // Check if this row group has a RowSelection
    auto sel_it = row_selections_.find(row_group_index_);
    if (sel_it == row_selections_.end() || !sel_it->second) {
      // No selection for this row group, use plain read
      return row_group_reader->GetColumnPageReader(column_index_);
    }

    // Load OffsetIndex for this column+row_group from the page index reader
    std::shared_ptr<OffsetIndex> offset_index;
    auto page_index_reader = reader_->GetPageIndexReader();
    if (page_index_reader) {
      auto rg_index = page_index_reader->RowGroup(row_group_index_);
      if (rg_index) {
        offset_index = rg_index->GetOffsetIndex(column_index_);
      }
    }

    if (!offset_index) {
      // No OffsetIndex available, fall back to plain read
      return row_group_reader->GetColumnPageReader(column_index_);
    }

    int64_t rg_row_count = reader_->metadata()->RowGroup(row_group_index_)->num_rows();

    // Use sparse I/O: only selected page byte ranges are fetched from disk
    return row_group_reader->GetColumnPageReaderWithRowSelection(
        column_index_, sel_it->second.get(), offset_index.get(), rg_row_count);
  }

 private:
  std::map<int, std::shared_ptr<RowSelection>> row_selections_;
};

struct ReaderContext {
  ParquetFileReader* reader;
  ::arrow::MemoryPool* pool;
  FileColumnIteratorFactory iterator_factory;
  bool filter_leaves;
  std::shared_ptr<std::unordered_set<int>> included_leaves;
  ArrowReaderProperties* reader_properties;

  bool IncludesLeaf(int leaf_index) const {
    if (this->filter_leaves) {
      return this->included_leaves->find(leaf_index) != this->included_leaves->end();
    }
    return true;
  }
};

Status TransferColumnData(::parquet::internal::RecordReader* reader,
                          std::unique_ptr<::parquet::ColumnChunkMetaData> metadata,
                          const std::shared_ptr<::arrow::Field>& value_field,
                          const ColumnDescriptor* descr, const ReaderContext* ctx,
                          std::shared_ptr<::arrow::ChunkedArray>* out);

}  // namespace arrow
}  // namespace parquet
