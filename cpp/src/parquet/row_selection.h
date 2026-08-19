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

#include <cstdint>
#include <utility>
#include <vector>

#include "arrow/io/interfaces.h"
#include "arrow/result.h"
#include "parquet/platform.h"
#include "parquet/type_fwd.h"

namespace parquet {

/// \brief Comparison operators for page-level predicate evaluation.
///
/// These operators are used with ColumnIndex::FilterPages() and
/// ParquetFileReader::ComputePageSelection() to express predicates of the form
/// ``column OP value``.  IS_NULL and IS_NOT_NULL do not require a predicate value.
enum class PredicateOp {
  GT,           ///< greater than
  LT,           ///< less than
  EQ,           ///< equal
  GTE,          ///< greater than or equal
  LTE,          ///< less than or equal
  IS_NULL,      ///< value IS NULL
  IS_NOT_NULL,  ///< value IS NOT NULL
};

/// \brief A selector indicating whether to skip or select a contiguous range of rows.
///
/// Used in a run-length encoded sequence (RowSelection) to compactly represent
/// arbitrary row-selection patterns for page pruning.  skip=true means rows should
/// be skipped (not read); skip=false means rows should be selected (read).
struct PARQUET_EXPORT RowSelector {
  /// If true, this run of rows should be skipped; if false, they are selected.
  bool skip;
  /// Number of rows in this run.
  int64_t row_count;

  RowSelector(bool skip, int64_t row_count);
};

/// \brief Run-length encoded representation of row selections for page-level pruning.
///
/// RowSelection encodes which rows in a row group should be read vs skipped.  It is
/// constructed from page statistics and a predicate, then used to direct I/O to only
/// the necessary page byte ranges.
///
/// A RowSelection is composed of a sequence of RowSelector entries, each describing
/// a consecutive run of selected (skip=false) or skipped (skip=true) rows.
///
/// Typical usage:
/// \code{.cpp}
///   // Obtain a RowSelection from ComputePageSelection(), then pass it to
///   // RowGroupReader::GetColumnPageReaderWithRowSelection() or
///   // ParquetFileReader::GetRecordReader() to restrict I/O to matching pages.
///   ARROW_ASSIGN_OR_RAISE(auto selections,
///                         reader->ComputePageSelection(col_idx, PredicateOp::GT,
///                                                      arrow::Int32Scalar(50)));
///   const auto& row_sel = selections[rg_idx];
/// \endcode
class PARQUET_EXPORT RowSelection {
 public:
  /// \brief Construct an empty RowSelection (zero rows covered).
  RowSelection();

  /// \brief Construct a RowSelection from an explicit sequence of selectors.
  explicit RowSelection(std::vector<RowSelector> selectors);

  /// \brief Total number of rows covered by all selectors (selected + skipped).
  int64_t row_count() const;

  /// \brief Number of selected (skip=false) rows.
  int64_t selected_row_count() const;

  /// \brief Number of skipped (skip=true) rows.
  int64_t skipped_row_count() const;

  /// \brief Number of selectors (i.e. RLE page count).
  int64_t page_count() const;

  /// \brief Return the i-th selector.
  ///
  /// \throws std::out_of_range if i >= page_count().
  const RowSelector& selector(size_t i) const;

  /// \brief Return the intersection (AND) of this selection with another.
  ///
  /// Logically AND two RowSelections: a row is selected in the result only if it is
  /// selected in *both* this selection and \p other.  All other rows are skipped.
  ///
  /// Both selections must cover the same total row_count(); returns
  /// Status::Invalid otherwise.
  ::arrow::Result<RowSelection> Intersect(const RowSelection& other) const;

  /// \brief Return the union (OR) of this selection with another.
  ///
  /// Logically OR two RowSelections: a row is selected in the result if it is selected
  /// in *either* this selection or \p other (or both).
  ///
  /// Both selections must cover the same total row_count(); returns
  /// Status::Invalid otherwise.
  ::arrow::Result<RowSelection> Union(const RowSelection& other) const;

  /// \brief Create a RowSelection that selects all rows in [0, row_count).
  static RowSelection All(int64_t row_count);

  /// \brief Create a RowSelection that skips all rows in [0, row_count).
  static RowSelection None(int64_t row_count);

  /// \brief Construct a RowSelection from a sorted list of non-overlapping [start, end)
  /// row ranges.
  ///
  /// Each pair in \p ranges is a half-open interval [start, end).  Rows that fall
  /// within any of these ranges are selected (skip=false); all other rows up to
  /// \p total_rows are skipped (skip=true).
  ///
  /// \param ranges  Sorted, non-overlapping [start, end) half-open pairs.
  /// \param total_rows  Total number of rows in the row group.  Rows beyond
  ///                    the last range are treated as skipped.
  /// \returns RowSelection or Status::Invalid if the ranges are unsorted, overlap, or
  ///          contain out-of-bound indices.
  static ::arrow::Result<RowSelection> FromConsecutiveRanges(
      const std::vector<std::pair<int64_t, int64_t>>& ranges, int64_t total_rows);

  /// \brief Map this RowSelection to page byte offsets via an OffsetIndex.
  ///
  /// For each page whose row interval overlaps a selected (skip=false) RowSelector,
  /// the corresponding ReadRange is included in the result.  The returned ranges are
  /// suitable for sparse I/O via SparseInputStream.  Contiguous ReadRanges are merged.
  ///
  /// \param offset_index  The OffsetIndex for the column chunk.
  /// \param row_group_row_count  Total rows in the row group (needed to
  ///                             derive the row count of the last page).
  /// \returns Sorted, merged vector of ReadRange values covering all pages
  ///          that contain at least one selected row.
  std::vector<::arrow::io::ReadRange> ScanRanges(const OffsetIndex& offset_index,
                                                 int64_t row_group_row_count) const;

 private:
  std::vector<RowSelector> selectors_;
  int64_t row_count_;  // cached sum of all selector row_counts
};

}  // namespace parquet
