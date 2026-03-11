# ingestion/pdf_loader.py

import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
import io
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TESSERACT_PATH


# ─────────────────────────────────────────
# BASE CLASS — all loaders inherit from this
# ─────────────────────────────────────────

class BaseLoader:
    """Abstract base class for all document loaders."""

    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.chunks = []

    def load(self) -> list[dict]:
        """Override this in each subclass."""
        raise NotImplementedError("Subclasses must implement load()")

    def _make_chunk(self, content: str, page: int, chunk_type: str) -> dict:
        """Helper — standard chunk format used across all loaders."""
        return {
            "content": content,
            "page": page,
            "type": chunk_type,
            "source": self.file_name
        }

    def get_summary(self) -> str:
        """Print a summary of what was loaded."""
        types = {}
        for c in self.chunks:
            types[c["type"]] = types.get(c["type"], 0) + 1
        return f"📄 {self.file_name} → {len(self.chunks)} chunks {types}"


# ─────────────────────────────────────────
# PDF LOADER CLASS
# ─────────────────────────────────────────

class PDFLoader(BaseLoader):
    """
    Loads PDF files and extracts:
      - Text  (via PyMuPDF)
      - Tables (via pdfplumber)
      - Images (via PyMuPDF + OCR via pytesseract)
    """

    def __init__(self, file_path: str, extract_images: bool = True,
                 image_output_dir: str = "extracted_images"):
        super().__init__(file_path)
        self.extract_images = extract_images
        self.image_output_dir = image_output_dir

        # Configure tesseract for Windows
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

    # ── PUBLIC ──────────────────────────────

    def load(self) -> list[dict]:
        """Master method — runs all extractors and returns unified chunks."""
        print(f"\n📄 Loading PDF: {self.file_name}")

        self.chunks = []
        self.chunks.extend(self._extract_text())
        self.chunks.extend(self._extract_tables())

        if self.extract_images:
            self.chunks.extend(self._extract_images())

        print(f"  ✅ {self.get_summary()}")
        return self.chunks

    # ── PRIVATE EXTRACTORS ───────────────────

    def _extract_text(self) -> list[dict]:
        """Extract plain text page by page using PyMuPDF."""
        results = []
        doc = fitz.open(self.file_path)

        for page_num, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                results.append(
                    self._make_chunk(text, page_num + 1, "text")
                )

        doc.close()
        print(f"  [TEXT]   {len(results)} pages extracted")
        return results

    def _extract_tables(self) -> list[dict]:
        """Extract tables using pdfplumber → convert to markdown."""
        results = []

        with pdfplumber.open(self.file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                for t_idx, table in enumerate(page.extract_tables()):
                    md = self._table_to_markdown(table)
                    if md:
                        content = f"[TABLE {t_idx+1} — Page {page_num+1}]\n{md}"
                        results.append(
                            self._make_chunk(content, page_num + 1, "table")
                        )

        print(f"  [TABLES] {len(results)} tables extracted")
        return results

    def _extract_images(self) -> list[dict]:
        """Extract images and run OCR to get text content."""
        os.makedirs(self.image_output_dir, exist_ok=True)
        results = []
        doc = fitz.open(self.file_path)

        for page_num, page in enumerate(doc):
            for img_idx, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    img_bytes  = base_image["image"]
                    img_ext    = base_image["ext"]

                    # Save to disk
                    img_filename = f"page{page_num+1}_img{img_idx+1}.{img_ext}"
                    img_path = os.path.join(self.image_output_dir, img_filename)
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)

                    # OCR
                    pil_img  = Image.open(io.BytesIO(img_bytes))
                    ocr_text = pytesseract.image_to_string(pil_img).strip()

                    if ocr_text:
                        content = f"[IMAGE — Page {page_num+1}] OCR: {ocr_text}"
                        chunk   = self._make_chunk(content, page_num + 1, "image")
                        chunk["image_path"] = img_path
                        results.append(chunk)

                except Exception as e:
                    print(f"  [IMAGE]  Skipped img {img_idx+1} p{page_num+1}: {e}")

        doc.close()
        print(f"  [IMAGES] {len(results)} images OCR'd")
        return results

    # ── STATIC HELPER ────────────────────────

    @staticmethod
    def _table_to_markdown(table: list) -> str:
        """Convert pdfplumber raw table (list of lists) → markdown string."""
        if not table:
            return ""

        cleaned = [
            [str(cell) if cell is not None else "" for cell in row]
            for row in table
        ]

        header    = "| " + " | ".join(cleaned[0]) + " |"
        separator = "|" + "|".join(["---"] * len(cleaned[0])) + "|"
        rows      = ["| " + " | ".join(row) + " |" for row in cleaned[1:]]

        return "\n".join([header, separator] + rows)
