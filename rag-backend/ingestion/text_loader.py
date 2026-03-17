# ingestion/text_loader.py

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion.pdf_loader import BaseLoader


class TextLoader(BaseLoader):
    """
    Loads plain .txt files.
    Splits by paragraphs first, then by character limit if needed.
    """

    def __init__(self, file_path: str, encoding: str = "utf-8"):
        super().__init__(file_path)
        self.encoding = encoding
        self.raw_text = ""

    def load(self) -> list[dict]:
        """Read the text file and split into paragraph chunks."""
        print(f"\n📝 Loading TXT: {self.file_name}")

        with open(self.file_path, "r", encoding=self.encoding, errors="ignore") as f:
            self.raw_text = f.read()

        self.chunks = []
        paragraphs  = self._split_by_paragraphs()

        for idx, para in enumerate(paragraphs):
            if para.strip():
                self.chunks.append(
                    self._make_chunk(para.strip(), page=idx + 1, chunk_type="text")
                )

        print(f"  ✅ {self.get_summary()}")
        return self.chunks

    def _split_by_paragraphs(self) -> list[str]:
        """Split on double newlines — natural paragraph boundaries."""
        return self.raw_text.split("\n\n")

    def get_raw_text(self) -> str:
        """Return the full raw text string."""
        return self.raw_text