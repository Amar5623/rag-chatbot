# ingestion/xlsx_loader.py

import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion.pdf_loader import BaseLoader


class XLSXLoader(BaseLoader):
    """
    Loads Excel (.xlsx) files.
    Handles multiple sheets — each sheet is chunked separately.
    """

    def __init__(self, file_path: str, rows_per_chunk: int = 50):
        super().__init__(file_path)
        self.rows_per_chunk = rows_per_chunk
        self.sheet_names    = []
        self.dataframes     = {}

    def load(self) -> list[dict]:
        """Load all sheets and convert to text chunks."""
        print(f"\n📗 Loading XLSX: {self.file_name}")

        xl           = pd.ExcelFile(self.file_path)
        self.sheet_names = xl.sheet_names
        self.chunks  = []

        print(f"  [SHEETS] Found {len(self.sheet_names)} sheet(s): {self.sheet_names}")

        for sheet_name in self.sheet_names:
            df = xl.parse(sheet_name)
            self.dataframes[sheet_name] = df
            sheet_chunks = self._process_sheet(df, sheet_name)
            self.chunks.extend(sheet_chunks)

        print(f"  ✅ {self.get_summary()}")
        return self.chunks

    def _process_sheet(self, df: pd.DataFrame, sheet_name: str) -> list[dict]:
        """Chunk a single sheet into row batches."""
        results   = []
        schema    = self._get_schema(df, sheet_name)
        total_rows = len(df)

        for start in range(0, total_rows, self.rows_per_chunk):
            end     = min(start + self.rows_per_chunk, total_rows)
            batch   = df.iloc[start:end]
            content = (
                f"{schema}\n\n"
                f"[Rows {start+1} to {end}]\n"
                f"{batch.to_string(index=False)}"
            )
            chunk               = self._make_chunk(content, page=1, chunk_type="xlsx")
            chunk["sheet_name"] = sheet_name
            chunk["row_range"]  = f"{start+1}-{end}"
            results.append(chunk)

        print(f"  [SHEET: {sheet_name}] {len(results)} chunks")
        return results

    def _get_schema(self, df: pd.DataFrame, sheet_name: str) -> str:
        """Human-readable schema for a single sheet."""
        col_info = ", ".join(
            [f"{col} ({str(dtype)})" for col, dtype in zip(df.columns, df.dtypes)]
        )
        return (
            f"[EXCEL FILE: {self.file_name} | SHEET: {sheet_name}]\n"
            f"Total Rows: {len(df)} | Columns: {df.shape[1]}\n"
            f"Schema: {col_info}"
        )

    def get_sheet(self, sheet_name: str) -> pd.DataFrame:
        """Return a specific sheet's dataframe."""
        return self.dataframes.get(sheet_name)