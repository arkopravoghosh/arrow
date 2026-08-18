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

#include "parquet/row_selection.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "arrow/status.h"

#include "parquet/page_index.h"

namespace parquet {

using ::arrow::Result;
using ::arrow::Status;
using ::arrow::io::ReadRange;

// ---------------------------------------------------------------------------
// RowSelector

RowSelector::RowSelector(bool skip, int64_t row_count)
    : skip(skip), row_count(row_count) {}

// ---------------------------------------------------------------------------
// RowSelection helpers

namespace {

/// Accumulate a selector into an output list, merging with the last entry if
/// it has the same skip flag.
void PushSelector(std::vector<RowSelector>& out, bool skip, int64_t row_count) {
  if (row_count <= 0) {
    return;
  }
  if (!out.empty() && out.back().skip == skip) {
    out.back().row_count += row_count;
  } else {
    out.emplace_back(skip, row_count);
  }
}

int64_t ComputeRowCount(const std::vector<RowSelector>& selectors) {
  int64_t total = 0;
  for (const auto& sel : selectors) {
    total += sel.row_count;
  }
  return total;
}

}  // namespace

// ---------------------------------------------------------------------------
// RowSelection

RowSelection::RowSelection() : row_count_(0) {}

RowSelection::RowSelection(std::vector<RowSelector> selectors)
    : selectors_(std::move(selectors)), row_count_(ComputeRowCount(selectors_)) {}

int64_t RowSelection::row_count() const { return row_count_; }

int64_t RowSelection::selected_row_count() const {
  int64_t total = 0;
  for (const auto& sel : selectors_) {
    if (!sel.skip) total += sel.row_count;
  }
  return total;
}

int64_t RowSelection::skipped_row_count() const {
  int64_t total = 0;
  for (const auto& sel : selectors_) {
    if (sel.skip) total += sel.row_count;
  }
  return total;
}

int64_t RowSelection::page_count() const {
  return static_cast<int64_t>(selectors_.size());
}

const RowSelector& RowSelection::selector(size_t i) const {
  if (i >= selectors_.size()) {
    throw std::out_of_range("RowSelection::selector index out of range");
  }
  return selectors_[i];
}

// ---------------------------------------------------------------------------
// Intersect

Result<RowSelection> RowSelection::Intersect(const RowSelection& other) const {
  if (row_count_ != other.row_count_) {
    return Status::Invalid(
        "RowSelection::Intersect requires both selections to cover the same row count: ",
        row_count_, " vs ", other.row_count_);
  }

  std::vector<RowSelector> result;
  result.reserve(selectors_.size() + other.selectors_.size());

  size_t ai = 0, bi = 0;
  int64_t a_rem = 0, b_rem = 0;

  // Initialise remaining counts from the first selector of each side.
  if (!selectors_.empty()) {
    a_rem = selectors_[0].row_count;
  }
  if (!other.selectors_.empty()) {
    b_rem = other.selectors_[0].row_count;
  }

  while (ai < selectors_.size() && bi < other.selectors_.size()) {
    const bool a_skip = selectors_[ai].skip;
    const bool b_skip = other.selectors_[bi].skip;

    // The current chunk covers min(a_rem, b_rem) rows.
    int64_t chunk = std::min(a_rem, b_rem);

    // Intersection: select only when both sides select.
    bool out_skip = a_skip || b_skip;
    PushSelector(result, out_skip, chunk);

    a_rem -= chunk;
    b_rem -= chunk;

    if (a_rem == 0) {
      ++ai;
      if (ai < selectors_.size()) {
        a_rem = selectors_[ai].row_count;
      }
    }
    if (b_rem == 0) {
      ++bi;
      if (bi < other.selectors_.size()) {
        b_rem = other.selectors_[bi].row_count;
      }
    }
  }

  return RowSelection(std::move(result));
}

// ---------------------------------------------------------------------------
// Union

Result<RowSelection> RowSelection::Union(const RowSelection& other) const {
  if (row_count_ != other.row_count_) {
    return Status::Invalid(
        "RowSelection::Union requires both selections to cover the same row count: ",
        row_count_, " vs ", other.row_count_);
  }

  std::vector<RowSelector> result;
  result.reserve(selectors_.size() + other.selectors_.size());

  size_t ai = 0, bi = 0;
  int64_t a_rem = 0, b_rem = 0;

  if (!selectors_.empty()) {
    a_rem = selectors_[0].row_count;
  }
  if (!other.selectors_.empty()) {
    b_rem = other.selectors_[0].row_count;
  }

  while (ai < selectors_.size() && bi < other.selectors_.size()) {
    const bool a_skip = selectors_[ai].skip;
    const bool b_skip = other.selectors_[bi].skip;

    int64_t chunk = std::min(a_rem, b_rem);

    // Union: select when either side selects.
    bool out_skip = a_skip && b_skip;
    PushSelector(result, out_skip, chunk);

    a_rem -= chunk;
    b_rem -= chunk;

    if (a_rem == 0) {
      ++ai;
      if (ai < selectors_.size()) {
        a_rem = selectors_[ai].row_count;
      }
    }
    if (b_rem == 0) {
      ++bi;
      if (bi < other.selectors_.size()) {
        b_rem = other.selectors_[bi].row_count;
      }
    }
  }

  return RowSelection(std::move(result));
}

// ---------------------------------------------------------------------------
// Static builders

RowSelection RowSelection::All(int64_t row_count) {
  if (row_count <= 0) {
    return RowSelection();
  }
  std::vector<RowSelector> selectors;
  selectors.emplace_back(/*skip=*/false, row_count);
  return RowSelection(std::move(selectors));
}

RowSelection RowSelection::None(int64_t row_count) {
  if (row_count <= 0) {
    return RowSelection();
  }
  std::vector<RowSelector> selectors;
  selectors.emplace_back(/*skip=*/true, row_count);
  return RowSelection(std::move(selectors));
}

Result<RowSelection> RowSelection::FromConsecutiveRanges(
    const std::vector<std::pair<int64_t, int64_t>>& ranges, int64_t total_rows) {
  std::vector<RowSelector> selectors;

  int64_t cursor = 0;
  for (size_t i = 0; i < ranges.size(); ++i) {
    const int64_t start = ranges[i].first;
    const int64_t end = ranges[i].second;

    if (start < 0 || end < start) {
      return Status::Invalid("Invalid range [", start, ", ", end,
                             "): start must be >= 0 and end >= start");
    }
    if (end > total_rows) {
      return Status::Invalid("Range [", start, ", ", end,
                             ") exceeds total_rows=", total_rows);
    }
    if (start < cursor) {
      return Status::Invalid("Ranges must be sorted and non-overlapping: range [", start,
                             ", ", end, ") overlaps with previous end=", cursor);
    }

    if (start > cursor) {
      // Gap before this range: skip rows [cursor, start).
      PushSelector(selectors, /*skip=*/true, start - cursor);
    }
    if (end > start) {
      PushSelector(selectors, /*skip=*/false, end - start);
    }
    cursor = end;
  }

  // Remaining rows after the last range.
  if (cursor < total_rows) {
    PushSelector(selectors, /*skip=*/true, total_rows - cursor);
  }

  return RowSelection(std::move(selectors));
}

// ---------------------------------------------------------------------------
// ScanRanges

std::vector<ReadRange> RowSelection::ScanRanges(const OffsetIndex& offset_index,
                                                int64_t row_group_row_count) const {
  const auto& page_locs = offset_index.page_locations();
  const size_t num_pages = page_locs.size();

  if (num_pages == 0 || selectors_.empty()) {
    return {};
  }

  std::vector<ReadRange> ranges;

  // Walk the RowSelection and page_locations in parallel using a dual-cursor
  // approach.  `sel_idx` indexes into selectors_; `sel_rem` is the number of
  // rows remaining in the current selector.  `page_idx` indexes into
  // page_locs; `page_rem` is the number of rows remaining in the current page.
  size_t sel_idx = 0;
  size_t page_idx = 0;
  int64_t sel_rem = selectors_[0].row_count;

  // Derive page row count: rows in page i = first_row_index[i+1] - first_row_index[i],
  // with the last page taking rows up to row_group_row_count.
  auto page_row_count = [&](size_t i) -> int64_t {
    if (i + 1 < num_pages) {
      return page_locs[i + 1].first_row_index - page_locs[i].first_row_index;
    }
    return row_group_row_count - page_locs[i].first_row_index;
  };

  int64_t page_rem = page_row_count(0);

  // Track the last page that was emitted to avoid duplicate entries when
  // multiple selectors overlap the same page (e.g. a select+select boundary
  // that falls mid-page).
  size_t last_emitted_page = static_cast<size_t>(-1);

  while (sel_idx < selectors_.size() && page_idx < num_pages) {
    const bool skip = selectors_[sel_idx].skip;
    int64_t chunk = std::min(sel_rem, page_rem);

    if (!skip && page_idx != last_emitted_page) {
      // This selector selects rows — include the current page (once).
      const auto& loc = page_locs[page_idx];
      ReadRange rr{loc.offset, static_cast<int64_t>(loc.compressed_page_size)};

      // Merge with the previous ReadRange if contiguous.
      if (!ranges.empty() && ranges.back().offset + ranges.back().length == rr.offset) {
        ranges.back().length += rr.length;
      } else {
        ranges.push_back(rr);
      }
      last_emitted_page = page_idx;
    }

    sel_rem -= chunk;
    page_rem -= chunk;

    if (sel_rem == 0) {
      ++sel_idx;
      if (sel_idx < selectors_.size()) {
        sel_rem = selectors_[sel_idx].row_count;
      }
    }
    if (page_rem == 0) {
      ++page_idx;
      if (page_idx < num_pages) {
        page_rem = page_row_count(page_idx);
      }
    }
  }

  return ranges;
}

}  // namespace parquet
