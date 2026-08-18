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

#include "gtest/gtest.h"

#include <memory>
#include <vector>

#include "arrow/testing/gtest_util.h"
#include "parquet/page_index.h"
#include "parquet/row_selection.h"

namespace parquet {

// ---------------------------------------------------------------------------
// Helper: build an OffsetIndex from explicit PageLocation values.
//
// Each entry in `page_locs` is {offset, compressed_page_size, first_row_index}.
// The builder's Finish() call takes a `final_position` argument that is added
// to every stored offset; pass 0 to keep the offsets as-is.
static std::unique_ptr<OffsetIndex> MakeOffsetIndex(
    const std::vector<PageLocation>& page_locs) {
  auto builder = OffsetIndexBuilder::Make();
  for (const auto& loc : page_locs) {
    builder->AddPage(loc.offset, loc.compressed_page_size, loc.first_row_index);
  }
  builder->Finish(/*final_position=*/0);
  return builder->Build();
}

// ---------------------------------------------------------------------------
// Case 1: Constructor stores selectors correctly and row_count is summed.
TEST(RowSelectionTest, ConstructorAndAccessors) {
  // A selection of 100 selected rows followed by 50 skipped rows.
  std::vector<RowSelector> selectors = {RowSelector(/*skip=*/false, /*row_count=*/100),
                                        RowSelector(/*skip=*/true, /*row_count=*/50)};
  RowSelection sel(selectors);

  ASSERT_EQ(150, sel.row_count());
  ASSERT_EQ(2, sel.page_count());

  // First selector: select 100 rows.
  ASSERT_FALSE(sel.selector(0).skip);
  ASSERT_EQ(100, sel.selector(0).row_count);

  // Second selector: skip 50 rows.
  ASSERT_TRUE(sel.selector(1).skip);
  ASSERT_EQ(50, sel.selector(1).row_count);
}

// ---------------------------------------------------------------------------
// Case 2: scan_ranges() returns ReadRanges for only the selected pages.
//
// RowSelection: select rows 0–99 and 200–299.
// OffsetIndex:  3 pages at rows 0–99, 100–199, 200–299 with known byte extents.
TEST(RowSelectionTest, ScanRangesSelectiveTwoPages) {
  // Page 0: bytes [1000, 1000+512), rows 0–99.
  // Page 1: bytes [2000, 2000+512), rows 100–199.
  // Page 2: bytes [3000, 3000+512), rows 200–299.
  const std::vector<PageLocation> page_locs = {
      {/*offset=*/1000, /*compressed_page_size=*/512, /*first_row_index=*/0},
      {/*offset=*/2000, /*compressed_page_size=*/512, /*first_row_index=*/100},
      {/*offset=*/3000, /*compressed_page_size=*/512, /*first_row_index=*/200},
  };
  auto offset_index = MakeOffsetIndex(page_locs);

  // Select rows [0,100) and [200,300); skip rows [100,200).
  std::vector<RowSelector> selectors = {
      RowSelector(/*skip=*/false, 100),  // rows 0–99
      RowSelector(/*skip=*/true, 100),   // rows 100–199
      RowSelector(/*skip=*/false, 100),  // rows 200–299
  };
  RowSelection sel(selectors);

  const int64_t row_group_row_count = 300;
  auto ranges = sel.ScanRanges(*offset_index, row_group_row_count);

  ASSERT_EQ(2u, ranges.size());

  // Page 0: offset=1000, length=512.
  ASSERT_EQ(1000, ranges[0].offset);
  ASSERT_EQ(512, ranges[0].length);

  // Page 2: offset=3000, length=512.
  ASSERT_EQ(3000, ranges[1].offset);
  ASSERT_EQ(512, ranges[1].length);
}

// ---------------------------------------------------------------------------
// Case 3: scan_ranges() on a select-all RowSelection returns one ReadRange per page.
//
// With no skip selectors, every page must appear exactly once.
TEST(RowSelectionTest, ScanRangesSelectAll) {
  // 4 pages, each 256 bytes, each covering 50 rows.
  const std::vector<PageLocation> page_locs = {
      {/*offset=*/100, /*compressed_page_size=*/256, /*first_row_index=*/0},
      {/*offset=*/356, /*compressed_page_size=*/256, /*first_row_index=*/50},
      {/*offset=*/612, /*compressed_page_size=*/256, /*first_row_index=*/100},
      {/*offset=*/868, /*compressed_page_size=*/256, /*first_row_index=*/150},
  };
  const int64_t num_pages = static_cast<int64_t>(page_locs.size());
  auto offset_index = MakeOffsetIndex(page_locs);

  const int64_t total_rows = 200;
  RowSelection sel = RowSelection::All(total_rows);

  auto ranges = sel.ScanRanges(*offset_index, total_rows);

  // Because all pages are contiguous (each page starts right where the
  // previous one ends), ScanRanges merges them into a single ReadRange.
  // Verify the union covers the entire byte extent.
  ASSERT_FALSE(ranges.empty());
  const int64_t total_bytes = num_pages * 256;
  int64_t covered = 0;
  for (const auto& rr : ranges) {
    covered += rr.length;
  }
  ASSERT_EQ(total_bytes, covered);

  // Also verify that the number of ReadRanges is exactly num_pages when the
  // pages are non-contiguous (i.e. gaps between them).
  const std::vector<PageLocation> non_contiguous_locs = {
      {/*offset=*/1000, /*compressed_page_size=*/100, /*first_row_index=*/0},
      {/*offset=*/2000, /*compressed_page_size=*/100, /*first_row_index=*/50},
      {/*offset=*/3000, /*compressed_page_size=*/100, /*first_row_index=*/100},
      {/*offset=*/4000, /*compressed_page_size=*/100, /*first_row_index=*/150},
  };
  auto nc_index = MakeOffsetIndex(non_contiguous_locs);
  RowSelection sel2 = RowSelection::All(total_rows);
  auto ranges2 = sel2.ScanRanges(*nc_index, total_rows);

  ASSERT_EQ(static_cast<size_t>(num_pages), ranges2.size());
}

// ---------------------------------------------------------------------------
// Case 4: Intersect() of two partially overlapping selections.
//
// A: select [0,100), skip [100,200) — total 200 rows.
// B: skip [0,50),   select [50,200) — total 200 rows.
// A ∩ B: skip [0,50), select [50,100), skip [100,200).
TEST(RowSelectionTest, IntersectTwoSelections) {
  RowSelection a({
      RowSelector(/*skip=*/false, 100),  // rows 0–99
      RowSelector(/*skip=*/true, 100),   // rows 100–199
  });

  RowSelection b({
      RowSelector(/*skip=*/true, 50),    // rows 0–49
      RowSelector(/*skip=*/false, 150),  // rows 50–199
  });

  ASSERT_OK_AND_ASSIGN(auto result, a.Intersect(b));

  ASSERT_EQ(200, result.row_count());

  // Expected: skip 50, select 50, skip 100.
  ASSERT_EQ(3, result.page_count());

  ASSERT_TRUE(result.selector(0).skip);
  ASSERT_EQ(50, result.selector(0).row_count);

  ASSERT_FALSE(result.selector(1).skip);
  ASSERT_EQ(50, result.selector(1).row_count);

  ASSERT_TRUE(result.selector(2).skip);
  ASSERT_EQ(100, result.selector(2).row_count);
}

// ---------------------------------------------------------------------------
// Case 5: FromConsecutiveRanges() builds the correct RLE selectors.
//
// Ranges {[0,50), [100,150)} with total_rows=150 should produce:
//   select 50, skip 50, select 50   (three selectors, total row_count=150).
TEST(RowSelectionTest, FromConsecutiveRanges) {
  const int64_t total_rows = 150;
  ASSERT_OK_AND_ASSIGN(
      auto sel, RowSelection::FromConsecutiveRanges({{0, 50}, {100, 150}}, total_rows));

  ASSERT_EQ(total_rows, sel.row_count());
  ASSERT_EQ(3, sel.page_count());

  ASSERT_FALSE(sel.selector(0).skip);
  ASSERT_EQ(50, sel.selector(0).row_count);

  ASSERT_TRUE(sel.selector(1).skip);
  ASSERT_EQ(50, sel.selector(1).row_count);

  ASSERT_FALSE(sel.selector(2).skip);
  ASSERT_EQ(50, sel.selector(2).row_count);
}

// ---------------------------------------------------------------------------
// Case 6: Intersect(All, None) produces a skip-all selection.
TEST(RowSelectionTest, IntersectAllVsNone) {
  const int64_t total_rows = 300;
  RowSelection all = RowSelection::All(total_rows);
  RowSelection none = RowSelection::None(total_rows);

  ASSERT_OK_AND_ASSIGN(auto result, all.Intersect(none));

  ASSERT_EQ(total_rows, result.row_count());

  // Every row must be skipped: no selected rows.
  int64_t selected = 0;
  for (int64_t i = 0; i < result.page_count(); ++i) {
    if (!result.selector(static_cast<size_t>(i)).skip) {
      selected += result.selector(static_cast<size_t>(i)).row_count;
    }
  }
  ASSERT_EQ(0, selected);
}

// ---------------------------------------------------------------------------
// Case 7: Empty RowSelection has zero rows; scan_ranges() returns empty vector.
TEST(RowSelectionTest, EmptySelection) {
  RowSelection sel;

  ASSERT_EQ(0, sel.row_count());
  ASSERT_EQ(0, sel.page_count());

  // scan_ranges() against any OffsetIndex must return empty.
  const std::vector<PageLocation> page_locs = {
      {/*offset=*/100, /*compressed_page_size=*/512, /*first_row_index=*/0},
      {/*offset=*/612, /*compressed_page_size=*/512, /*first_row_index=*/100},
  };
  auto offset_index = MakeOffsetIndex(page_locs);

  auto ranges = sel.ScanRanges(*offset_index, /*row_group_row_count=*/200);
  ASSERT_TRUE(ranges.empty());
}

// ---------------------------------------------------------------------------
// Case 8: scan_ranges() on a skip-all RowSelection returns an empty vector.
TEST(RowSelectionTest, ScanRangesSkipAll) {
  const int64_t total_rows = 300;
  RowSelection sel = RowSelection::None(total_rows);

  const std::vector<PageLocation> page_locs = {
      {/*offset=*/1000, /*compressed_page_size=*/256, /*first_row_index=*/0},
      {/*offset=*/1256, /*compressed_page_size=*/256, /*first_row_index=*/100},
      {/*offset=*/1512, /*compressed_page_size=*/256, /*first_row_index=*/200},
  };
  auto offset_index = MakeOffsetIndex(page_locs);

  auto ranges = sel.ScanRanges(*offset_index, total_rows);
  ASSERT_TRUE(ranges.empty());
}

// ---------------------------------------------------------------------------
// Case 9: FromConsecutiveRanges() with a single [0, N) range produces one
//         select-all selector of row_count N.
TEST(RowSelectionTest, FromConsecutiveRangesSingle) {
  const int64_t total_rows = 500;
  ASSERT_OK_AND_ASSIGN(
      auto sel, RowSelection::FromConsecutiveRanges({{0, total_rows}}, total_rows));

  ASSERT_EQ(total_rows, sel.row_count());
  ASSERT_EQ(1, sel.page_count());

  ASSERT_FALSE(sel.selector(0).skip);
  ASSERT_EQ(total_rows, sel.selector(0).row_count);
}

// ---------------------------------------------------------------------------
// Case 10: Intersect() with mismatched row counts returns a non-OK status.
TEST(RowSelectionTest, IntersectMismatchedRowCounts) {
  RowSelection a({RowSelector(/*skip=*/false, 100)});  // 100 rows
  RowSelection b({RowSelector(/*skip=*/false, 200)});  // 200 rows

  auto result = a.Intersect(b);
  ASSERT_RAISES(Invalid, result);
}

}  // namespace parquet
