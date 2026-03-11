# ingestion/csv_loader.py

import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion.pdf_loader import BaseLoader


class CSVLoader(BaseLoader):
    """
    Loads CSV files and converts them into text chunks.
    Each chunk = a batch of rows + column schema header.
    """

    def __init__(self, file_path: str, rows_per_chunk: int = 50):
        super().__init__(file_path)
        self.rows_per_chunk = rows_per_chunk
        self.df = None

    def load(self) -> list[dict]:
        """Read CSV and split into row-batched chunks."""
        print(f"\n📊 Loading CSV: {self.file_name}")

        self.df = pd.read_csv(self.file_path)
        self.chunks = []

        schema = self._get_schema()
        total_rows = len(self.df)

        for start in range(0, total_rows, self.rows_per_chunk):
            end        = min(start + self.rows_per_chunk, total_rows)
            batch      = self.df.iloc[start:end]
            content    = f"{schema}\n\n[Rows {start+1} to {end}]\n{batch.to_string(index=False)}"
            chunk      = self._make_chunk(content, page=1, chunk_type="csv")
            chunk["row_range"] = f"{start+1}-{end}"
            self.chunks.append(chunk)

        print(f"  ✅ {self.get_summary()}")
        return self.chunks

    def _get_schema(self) -> str:
        """Return a human-readable schema description of the CSV."""
        col_info = ", ".join(
            [f"{col} ({str(dtype)})" for col, dtype in zip(self.df.columns, self.df.dtypes)]
        )
        return (
            f"[CSV FILE: {self.file_name}]\n"
            f"Total Rows: {len(self.df)} | Columns: {self.df.shape[1]}\n"
            f"Schema: {col_info}"
        )

    def get_dataframe(self) -> pd.DataFrame:
        """Return the raw dataframe if needed elsewhere."""
        if self.df is None:
            self.df = pd.read_csv(self.file_path)
        return self.df