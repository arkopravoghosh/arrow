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

#include <any>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "arrow/io/caching.h"
#include "arrow/result.h"
#include "arrow/util/type_fwd.h"
#include "parquet/metadata.h"  // IWYU pragma: keep
#include "parquet/platform.h"
#include "parquet/properties.h"
#include "parquet/row_selection.h"

namespace parquet {

class ColumnReader;
class FileMetaData;
class OffsetIndex;
class PageIndexReader;
class BloomFilterReader;
class PageReader;
class RowGroupMetaData;

namespace internal {
class RecordReader;
}

class PARQUET_EXPORT RowGroupReader {
 public:
  // Forward declare a virtual class 'Contents' to aid dependency injection and more
  // easily create test fixtures
  // An implementation of the Contents class is defined in the .cc file
  struct Contents {
    virtual ~Contents() {}
    virtual std::unique_ptr<PageReader> GetColumnPageReader(int i) = 0;

    // \brief Variant of GetColumnPageReader that wraps the column stream in a
    // SparseInputStream when a RowSelection and OffsetIndex are provided.
    //
    // If row_selection is nullptr or offset_index is nullptr the method falls
    // back to the plain sequential read path (identical to GetColumnPageReader).
    //
    // \note API EXPERIMENTAL
    virtual std::unique_ptr<PageReader> GetColumnPageReaderWithRowSelection(
        int i, const RowSelection* row_selection, const OffsetIndex* offset_index,
        int64_t row_group_row_count);

    virtual const RowGroupMetaData* metadata() const = 0;
    virtual const ReaderProperties* properties() const = 0;
  };

  explicit RowGroupReader(std::unique_ptr<Contents> contents);

  // Returns the rowgroup metadata
  const RowGroupMetaData* metadata() const;

  // Construct a ColumnReader for the indicated row group-relative
  // column. Ownership is shared with the RowGroupReader.
  std::shared_ptr<ColumnReader> Column(int i);

  // EXPERIMENTAL: Construct a RecordReader for the indicated column of the row group.
  // Ownership is shared with the RowGroupReader.
  std::shared_ptr<internal::RecordReader> RecordReader(int i,
                                                       bool read_dictionary = false);

  // Construct a ColumnReader, trying to enable exposed encoding.
  //
  // For dictionary encoding, currently we only support column chunks that are fully
  // dictionary encoded, i.e., all data pages in the column chunk are dictionary encoded.
  // If a column chunk uses dictionary encoding but then falls back to plain encoding, the
  // encoding will not be exposed.
  //
  // The returned column reader provides an API GetExposedEncoding() for the
  // users to check the exposed encoding and determine how to read the batches.
  //
  // \note API EXPERIMENTAL
  std::shared_ptr<ColumnReader> ColumnWithExposeEncoding(
      int i, ExposedEncoding encoding_to_expose);

  // Construct a RecordReader, trying to enable exposed encoding.
  //
  // For dictionary encoding, currently we only support column chunks that are
  // fully dictionary encoded byte arrays. The caller should verify if the reader can read
  // and expose the dictionary by checking the reader's read_dictionary(). If a column
  // chunk uses dictionary encoding but then falls back to plain encoding, the returned
  // reader will read decoded data without exposing the dictionary.
  //
  // \note API EXPERIMENTAL
  std::shared_ptr<internal::RecordReader> RecordReaderWithExposeEncoding(
      int i, ExposedEncoding encoding_to_expose);

  std::unique_ptr<PageReader> GetColumnPageReader(int i);

  // \brief Obtain a PageReader for column i, restricting I/O to the pages
  // needed for the given RowSelection (when row_selection and offset_index are
  // non-null).
  //
  // When both row_selection and offset_index are provided, the underlying
  // column stream is wrapped in a SparseInputStream so that only the byte
  // ranges required for selected rows are read.  Falls back to the plain
  // sequential read path when either pointer is null.
  //
  // \note API EXPERIMENTAL
  std::unique_ptr<PageReader> GetColumnPageReaderWithRowSelection(
      int i, const RowSelection* row_selection, const OffsetIndex* offset_index,
      int64_t row_group_row_count);

 private:
  // Holds a pointer to an instance of Contents implementation
  std::unique_ptr<Contents> contents_;
};

class PARQUET_EXPORT ParquetFileReader {
 public:
  // Declare a virtual class 'Contents' to aid dependency injection and more
  // easily create test fixtures
  // An implementation of the Contents class is defined in the .cc file
  struct PARQUET_EXPORT Contents {
    static std::unique_ptr<Contents> Open(
        std::shared_ptr<::arrow::io::RandomAccessFile> source,
        const ReaderProperties& props = default_reader_properties(),
        std::shared_ptr<FileMetaData> metadata = NULLPTR);

    static ::arrow::Future<std::unique_ptr<Contents>> OpenAsync(
        std::shared_ptr<::arrow::io::RandomAccessFile> source,
        const ReaderProperties& props = default_reader_properties(),
        std::shared_ptr<FileMetaData> metadata = NULLPTR);

    virtual ~Contents() = default;
    // Perform any cleanup associated with the file contents
    virtual void Close() = 0;
    virtual std::shared_ptr<RowGroupReader> GetRowGroup(int i) = 0;
    virtual std::shared_ptr<FileMetaData> metadata() const = 0;
    virtual std::shared_ptr<PageIndexReader> GetPageIndexReader() = 0;
    virtual BloomFilterReader& GetBloomFilterReader() = 0;
  };

  ParquetFileReader();
  ~ParquetFileReader();

  // Create a file reader instance from an Arrow file object. Thread-safety is
  // the responsibility of the file implementation
  static std::unique_ptr<ParquetFileReader> Open(
      std::shared_ptr<::arrow::io::RandomAccessFile> source,
      const ReaderProperties& props = default_reader_properties(),
      std::shared_ptr<FileMetaData> metadata = NULLPTR);

  // API Convenience to open a serialized Parquet file on disk, using Arrow IO
  // interfaces.
  static std::unique_ptr<ParquetFileReader> OpenFile(
      const std::string& path, bool memory_map = false,
      const ReaderProperties& props = default_reader_properties(),
      std::shared_ptr<FileMetaData> metadata = NULLPTR);

  // Asynchronously open a file reader from an Arrow file object.
  // Does not throw - all errors are reported through the Future.
  static ::arrow::Future<std::unique_ptr<ParquetFileReader>> OpenAsync(
      std::shared_ptr<::arrow::io::RandomAccessFile> source,
      const ReaderProperties& props = default_reader_properties(),
      std::shared_ptr<FileMetaData> metadata = NULLPTR);

  void Open(std::unique_ptr<Contents> contents);
  void Close();

  // The RowGroupReader is owned by the FileReader
  std::shared_ptr<RowGroupReader> RowGroup(int i);

  // Returns the file metadata. Only one instance is ever created
  std::shared_ptr<FileMetaData> metadata() const;

  /// Returns the PageIndexReader. Only one instance is ever created.
  ///
  /// If the file does not have the page index, nullptr may be returned.
  /// Because it pays to check existence of page index in the file, it
  /// is possible to return a non null value even if page index does
  /// not exist. It is the caller's responsibility to check the return
  /// value and follow-up calls to PageIndexReader.
  ///
  /// WARNING: The returned PageIndexReader must not outlive the ParquetFileReader.
  /// Initialize GetPageIndexReader() is not thread-safety.
  std::shared_ptr<PageIndexReader> GetPageIndexReader();

  /// Returns the BloomFilterReader. Only one instance is ever created.
  ///
  /// WARNING: The returned BloomFilterReader must not outlive the ParquetFileReader.
  /// Initialize GetBloomFilterReader() is not thread-safety.
  BloomFilterReader& GetBloomFilterReader();

  /// Pre-buffer the specified column indices in all row groups.
  ///
  /// Readers can optionally call this to cache the necessary slices
  /// of the file in-memory before deserialization. Arrow readers can
  /// automatically do this via an option. This is intended to
  /// increase performance when reading from high-latency filesystems
  /// (e.g. Amazon S3).
  ///
  /// After calling this, creating readers for row groups/column
  /// indices that were not buffered may fail. Creating multiple
  /// readers for the a subset of the buffered regions is
  /// acceptable. This may be called again to buffer a different set
  /// of row groups/columns.
  ///
  /// If memory usage is a concern, note that data will remain
  /// buffered in memory until either \a PreBuffer() is called again,
  /// or the reader itself is destructed. Reading - and buffering -
  /// only one row group at a time may be useful.
  ///
  /// This method may throw.
  void PreBuffer(const std::vector<int>& row_groups,
                 const std::vector<int>& column_indices,
                 const ::arrow::io::IOContext& ctx,
                 const ::arrow::io::CacheOptions& options);

  /// Retrieve the list of byte ranges that would need to be read to retrieve
  /// the data for the specified row groups and column indices.
  ///
  /// A reader can optionally call this if they wish to handle their own
  /// caching and management of file reads (or offload them to other readers).
  /// Unlike PreBuffer, this method will not perform any actual caching or
  /// reads, instead just using the file metadata to determine the byte ranges
  /// that would need to be read if you were to consume the entirety of the column
  /// chunks for the provided columns in the specified row groups.
  ///
  /// If row_groups or column_indices are empty, then the result of this will be empty.
  ///
  /// hole_size_limit represents the maximum distance, in bytes, between two
  /// consecutive ranges; beyond this value, ranges will not be combined. The default
  /// value is 1MB.
  ///
  /// range_size_limit is the maximum size in bytes of a combined range; if combining
  /// two consecutive ranges would produce a range larger than this, they are not
  /// combined. The default values is 64MB. This *must* be larger than hole_size_limit.
  ///
  /// This will not take into account page indexes or any other predicate push down
  /// benefits that may be available.
  ::arrow::Result<std::vector<::arrow::io::ReadRange>> GetReadRanges(
      const std::vector<int>& row_groups, const std::vector<int>& column_indices,
      int64_t hole_size_limit = 1024 * 1024, int64_t range_size_limit = 64 * 1024 * 1024);

  /// Wait for the specified row groups and column indices to be pre-buffered.
  ///
  /// After the returned Future completes, reading the specified row
  /// groups/columns will not block.
  ///
  /// PreBuffer must be called first. This method does not throw.
  ::arrow::Future<> WhenBuffered(const std::vector<int>& row_groups,
                                 const std::vector<int>& column_indices) const;

  /// \brief Set the ArrowReaderProperties used by page-pruning APIs.
  ///
  /// Must be called before ComputePageSelection() to configure the
  /// PageIndexPolicy.  If not called, the default ArrowReaderProperties
  /// (PageIndexPolicy::AUTO) are used.
  ///
  /// \note API EXPERIMENTAL
  void set_arrow_reader_properties(const ArrowReaderProperties& properties);

  /// \brief Return the ArrowReaderProperties currently in effect.
  ///
  /// \note API EXPERIMENTAL
  const ArrowReaderProperties& arrow_reader_properties() const;

  /// \brief Compute a row selection for page-level filtering based on a predicate.
  ///
  /// Evaluates a predicate (e.g., column > 5) against the ColumnIndex statistics for
  /// the specified physical column across the requested row groups.  Returns a map from
  /// row-group index to RowSelection, indicating which rows in each row group might
  /// match the predicate.
  ///
  /// Behavior depends on the PageIndexPolicy stored in the reader's
  /// ArrowReaderProperties:
  ///   - NEVER:  returns a select-all RowSelection for each row group (no pruning).
  ///   - AUTO:   if ColumnIndex/OffsetIndex are present, evaluates the predicate;
  ///             otherwise falls back to a select-all RowSelection.
  ///   - ALWAYS: returns Status::Invalid if the indices are absent.
  ///
  /// The returned RowSelections can be passed to GetRecordReader() or directly to
  /// RowGroupReader::GetColumnPageReaderWithRowSelection() to restrict I/O to
  /// pages that can satisfy the predicate.
  ///
  /// \param column_index  Physical column index (0-based).
  /// \param op            Predicate operator (see PredicateOp).
  /// \param predicate_value  Scalar value to compare against (type must match
  ///                         the column's physical type; ignored for IS_NULL /
  ///                         IS_NOT_NULL).
  /// \param row_group_indices  Optional list of row-group indices to filter.
  ///                           Pass nullptr or an empty vector to use all row
  ///                           groups.
  /// \returns Map from row-group index to the corresponding RowSelection, or an
  ///          error status if column_index is out of range, or the
  ///          PageIndexPolicy is ALWAYS and indices are absent.
  ///
  /// \note API EXPERIMENTAL
  ::arrow::Result<std::map<int, std::shared_ptr<RowSelection>>> ComputePageSelection(
      int column_index, PredicateOp op, const std::any& predicate_value,
      const std::vector<int>* row_group_indices = nullptr) const;

  /// \brief Construct a RecordReader with optional row selection for page-level
  /// filtering.
  ///
  /// Builds a reader for the given row group and column.  If \p row_selection is
  /// provided (typically obtained from ComputePageSelection()), the reader will
  /// skip/read rows according to the selection, restricting I/O to only the pages
  /// that overlap selected row ranges.  If \p row_selection is nullptr, all rows are
  /// read (existing behavior).
  ///
  /// Typical usage:
  /// \code{.cpp}
  ///   auto selections = reader->ComputePageSelection(col, PredicateOp::GT,
  ///                                                  std::any(int32_t(50))).ValueOrDie();
  ///   for (auto& [rg, sel] : selections) {
  ///     auto record_reader = reader->GetRecordReader(rg, col, sel).ValueOrDie();
  ///     // read batches from record_reader ...
  ///   }
  /// \endcode
  ///
  /// \param rg_index    Row-group index (0-based).
  /// \param col_index   Physical column index (0-based) within the row group.
  /// \param row_selection  Optional selection; nullptr means read all rows.
  /// \returns Shared RecordReader with the selection applied (if any), or an error
  ///          status if rg_index or col_index is out of range.
  ///
  /// \note API EXPERIMENTAL
  ::arrow::Result<std::shared_ptr<internal::RecordReader>> GetRecordReader(
      int rg_index, int col_index,
      const std::shared_ptr<RowSelection>& row_selection = nullptr) const;

 private:
  // Holds a pointer to an instance of Contents implementation
  std::unique_ptr<Contents> contents_;

  // ArrowReaderProperties governing page-index behaviour for ComputePageSelection.
  ArrowReaderProperties arrow_reader_properties_;
};

// Read only Parquet file metadata
std::shared_ptr<FileMetaData> PARQUET_EXPORT
ReadMetaData(const std::shared_ptr<::arrow::io::RandomAccessFile>& source);

/// \brief Scan all values in file. Useful for performance testing
/// \param[in] columns the column numbers to scan. If empty scans all
/// \param[in] column_batch_size number of values to read at a time when scanning column
/// \param[in] reader a ParquetFileReader instance
/// \return number of semantic rows in file
PARQUET_EXPORT
int64_t ScanFileContents(std::vector<int> columns, const int32_t column_batch_size,
                         ParquetFileReader* reader);

}  // namespace parquet
