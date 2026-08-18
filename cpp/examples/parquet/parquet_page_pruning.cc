// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.

// Parquet Page-Level Pruning Example
// ===================================
// This program demonstrates how to use the page-level pruning API to read only
// the rows from a Parquet file that satisfy a predicate, avoiding unnecessary
// page I/O.
//
// The example:
//   1. Writes a multi-row-group Parquet file with a page index.
//   2. Opens the file and configures PageIndexPolicy::AUTO.
//   3. Calls ComputePageSelection() to evaluate "column_0 > 50".
//   4. Iterates the resulting RowSelections and counts selected rows per row group.
//   5. Compares the pruned row count against the full (unpruned) row count.
//
// Usage: parquet-page-pruning-example [parquet_file]
//
// If no file path is given, a temporary file is generated and then read back.

#include <any>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "arrow/api.h"
#include "arrow/io/api.h"
#include "parquet/arrow/reader.h"
#include "parquet/arrow/writer.h"
#include "parquet/file_reader.h"
#include "parquet/page_index.h"
#include "parquet/properties.h"
#include "parquet/row_selection.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static const char* kTempFile = "page_pruning_example.parquet";

// Build an Arrow Table with two columns:
//   column_0 (int32): values 0..num_rows-1
//   column_1 (float): values 0.0..num_rows-1
static std::shared_ptr<arrow::Table> BuildTable(int64_t num_rows) {
  arrow::Int32Builder col0_builder;
  arrow::FloatBuilder col1_builder;

  for (int32_t i = 0; i < static_cast<int32_t>(num_rows); ++i) {
    PARQUET_THROW_NOT_OK(col0_builder.Append(i));
    PARQUET_THROW_NOT_OK(col1_builder.Append(static_cast<float>(i)));
  }

  std::shared_ptr<arrow::Array> col0, col1;
  PARQUET_THROW_NOT_OK(col0_builder.Finish(&col0));
  PARQUET_THROW_NOT_OK(col1_builder.Finish(&col1));

  auto schema = arrow::schema({arrow::field("column_0", arrow::int32()),
                               arrow::field("column_1", arrow::float32())});
  return arrow::Table::Make(schema, {col0, col1});
}

// Write the table to a Parquet file, splitting into multiple row groups.
// Page index writing is enabled so ComputePageSelection() can use it.
static void WriteParquetFile(const std::string& path, const arrow::Table& table,
                             int64_t rows_per_group) {
  std::shared_ptr<arrow::io::FileOutputStream> outfile;
  PARQUET_ASSIGN_OR_THROW(outfile, arrow::io::FileOutputStream::Open(path));

  // Enable page index writing (enabled by default, shown here for clarity).
  auto writer_props = parquet::WriterProperties::Builder()
                          .enable_write_page_index()
                          ->version(parquet::ParquetVersion::PARQUET_2_6)
                          ->build();

  PARQUET_THROW_NOT_OK(parquet::arrow::WriteTable(table, arrow::default_memory_pool(),
                                                  outfile, rows_per_group, writer_props));
}

// ---------------------------------------------------------------------------
// Page-pruning read path
// ---------------------------------------------------------------------------

// Read the file using page-level pruning: only fetch pages whose statistics
// are compatible with column_0 > 50.
static int64_t ReadWithPruning(const std::string& path) {
  std::shared_ptr<arrow::io::ReadableFile> source;
  PARQUET_ASSIGN_OR_THROW(source, arrow::io::ReadableFile::Open(path));

  // Open a low-level ParquetFileReader.
  auto file_reader = parquet::ParquetFileReader::Open(source);

  // Configure PageIndexPolicy::AUTO so the reader uses the page index when
  // it is present and falls back gracefully when it is not.
  parquet::ArrowReaderProperties arrow_props;
  arrow_props.set_page_index_policy(parquet::PageIndexPolicy::AUTO);
  file_reader->set_arrow_reader_properties(arrow_props);

  // ComputePageSelection evaluates "column_0 > 50" against each row group's
  // ColumnIndex and returns a RowSelection per row group.
  auto selection_result = file_reader->ComputePageSelection(
      /*column_index=*/0, /*op=*/parquet::PredicateOp::GT,
      /*predicate_value=*/std::any(int32_t(50)));
  PARQUET_THROW_NOT_OK(selection_result.status());
  auto selections = selection_result.ValueOrDie();

  int64_t total_selected_rows = 0;

  std::cout << "--- Page-pruned read (column_0 > 50) ---\n";
  for (const auto& [rg_idx, row_sel] : selections) {
    if (!row_sel) {
      // Null indicates select-all (no index available for this row group).
      auto rg_meta = file_reader->metadata()->RowGroup(rg_idx);
      int64_t rg_rows = rg_meta->num_rows();
      std::cout << "  Row group " << rg_idx << ": select-all (no page index) — "
                << rg_rows << " rows\n";
      total_selected_rows += rg_rows;
      continue;
    }

    // Count selected rows (skip=false entries).
    int64_t selected = 0;
    for (int64_t p = 0; p < row_sel->page_count(); ++p) {
      const auto& sel = row_sel->selector(static_cast<size_t>(p));
      if (!sel.skip) {
        selected += sel.row_count;
      }
    }

    std::cout << "  Row group " << rg_idx << ": " << selected << " / "
              << row_sel->row_count() << " rows selected by pruning\n";

    // Obtain a RecordReader restricted to the selected rows.
    // This builds a SparseInputStream that reads only the byte ranges for
    // the selected pages.
    auto rr_result = file_reader->GetRecordReader(rg_idx, /*col_index=*/0, row_sel);
    PARQUET_THROW_NOT_OK(rr_result.status());

    // The RecordReader reports how many rows it will yield; the caller would
    // consume batches here in a real scan.
    total_selected_rows += selected;
  }

  return total_selected_rows;
}

// ---------------------------------------------------------------------------
// Full (unpruned) row count for comparison
// ---------------------------------------------------------------------------

static int64_t ReadAllRows(const std::string& path) {
  std::shared_ptr<arrow::io::ReadableFile> source;
  PARQUET_ASSIGN_OR_THROW(source, arrow::io::ReadableFile::Open(path));

  std::unique_ptr<parquet::arrow::FileReader> reader;
  PARQUET_ASSIGN_OR_THROW(reader,
                          parquet::arrow::OpenFile(source, arrow::default_memory_pool()));

  std::shared_ptr<arrow::Table> table;
  PARQUET_ASSIGN_OR_THROW(table, reader->ReadTable());
  return table->num_rows();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  std::string path = (argc >= 2) ? argv[1] : kTempFile;

  bool wrote_temp = false;
  if (argc < 2) {
    // Generate a sample file: 300 rows split into 3 row groups of 100 each.
    // With column_0 = 0..299, the predicate column_0 > 50 matches 249 rows.
    std::cout << "No file specified; writing sample file to " << path << "\n\n";
    auto table = BuildTable(300);
    WriteParquetFile(path, *table, /*rows_per_group=*/100);
    wrote_temp = true;
  }

  // --- Pruned read ---
  int64_t pruned_count = ReadWithPruning(path);
  std::cout << "  Total rows returned by pruned scan: " << pruned_count << "\n\n";

  // --- Full read (baseline) ---
  int64_t full_count = ReadAllRows(path);
  std::cout << "--- Unpruned read ---\n";
  std::cout << "  Total rows in file: " << full_count << "\n\n";

  // Summary
  std::cout << "--- Summary ---\n";
  std::cout << "  Rows in file        : " << full_count << "\n";
  std::cout << "  Rows after pruning  : " << pruned_count << "\n";
  if (full_count > 0) {
    double pct = 100.0 * static_cast<double>(full_count - pruned_count) /
                 static_cast<double>(full_count);
    std::cout << "  Pages skipped (est) : " << pct << "% of total rows\n";
  }

  if (wrote_temp) {
    // Clean up the temporary file.
    std::remove(path.c_str());
  }

  return 0;
}
