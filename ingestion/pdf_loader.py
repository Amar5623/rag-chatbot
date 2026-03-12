# ingestion/pdf_loader.py
# IMPROVED:
#   - Heading detection via PyMuPDF font-size analysis
#   - image_path stored in metadata for later display in UI
#   - heading propagated to every text chunk on that page
#   - _make_chunk now accepts optional extra metadata kwargs

import os
import io
import sys
import fitz                  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TESSERACT_PATH

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ─────────────────────────────────────────
# BASE LOADER
# ─────────────────────────────────────────

class BaseLoader:
    """
    Abstract base for all document loaders.
    Provides _make_chunk() helper used by every subclass.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.chunks    = []

    def load(self) -> list[dict]:
        raise NotImplementedError("Subclasses must implement load()")

    def _make_chunk(self, content: str, page: int, chunk_type: str, **extra) -> dict:
        """
        Standard chunk format used across all loaders.
        Extra keyword args are merged directly — lets subclasses inject
        heading, image_path, row_range, sheet_name, etc. cleanly.
        """
        chunk = {
            "content": content,
            "page"   : page,
            "type"   : chunk_type,
            "source" : self.file_name,
            "heading": "",          # default — overridden by PDF loader
        }
        chunk.update(extra)         # merge any extra metadata
        return chunk

    def get_summary(self) -> str:
        types = {}
        for c in self.chunks:
            types[c["type"]] = types.get(c["type"], 0) + 1
        return f"📄 {self.file_name} → {len(self.chunks)} chunks {types}"


# ─────────────────────────────────────────
# PDF LOADER
# ─────────────────────────────────────────

class PDFLoader(BaseLoader):
    """
    Loads PDF files and extracts:
      - Text   (via PyMuPDF) with heading detection
      - Tables (via pdfplumber)
      - Images (via PyMuPDF + Tesseract OCR), saving image_path for UI display

    HEADING DETECTION
    -----------------
    PyMuPDF's get_text("dict") returns every text span with its font size.
    We compute the median body font size per page, then classify spans whose
    font size is >= body_size * HEADING_SCALE_FACTOR as headings.
    The detected heading is attached to the chunk as metadata["heading"],
    allowing retrieval to filter by section.

    IMAGE METADATA
    --------------
    Every image chunk stores:
        image_path : absolute path to the saved .png / .jpg on disk
    The RAG chain passes this through ChainResponse so app.py can display
    the actual image alongside the LLM answer.
    """

    HEADING_SCALE_FACTOR = 1.15   # font must be ≥ 15% larger than body to count

    def __init__(
        self,
        file_path        : str,
        extract_images   : bool = True,
        image_output_dir : str  = "extracted_images",
    ):
        super().__init__(file_path)
        self.extract_images   = extract_images
        self.image_output_dir = image_output_dir

    # ── PUBLIC ──────────────────────────────

    def load(self) -> list[dict]:
        """Master method — runs all extractors and returns unified chunks."""
        print(f"\n📄 Loading PDF: {self.file_name}")

        self.chunks = []
        self.chunks.extend(self._extract_text_with_headings())
        self.chunks.extend(self._extract_tables())

        if self.extract_images:
            self.chunks.extend(self._extract_images())

        print(f"  ✅ {self.get_summary()}")
        return self.chunks

    # ── TEXT + HEADING EXTRACTION ────────────

    def _extract_text_with_headings(self) -> list[dict]:
        """
        Extract text page by page using PyMuPDF dict mode.
        Detects headings by comparing each span's font size to the
        median body font size of that page.

        Each returned chunk carries:
            heading : the last heading seen before this text block (or "")
        """
        results     = []
        doc         = fitz.open(self.file_path)

        for page_num, page in enumerate(doc):
            page_dict   = page.get_text("dict")
            blocks      = page_dict.get("blocks", [])

            # ── Step 1: collect all font sizes on this page ──
            all_sizes = []
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text", "").strip():
                            all_sizes.append(span["size"])

            if not all_sizes:
                continue

            # median font size = body text size for this page
            all_sizes.sort()
            body_size    = all_sizes[len(all_sizes) // 2]
            heading_threshold = body_size * self.HEADING_SCALE_FACTOR

            # ── Step 2: walk blocks, detect headings, group text ──
            current_heading = ""
            page_text_parts = []

            for block in blocks:
                if block.get("type") != 0:   # 0 = text block
                    continue

                block_lines = []
                is_heading_block = False

                for line in block.get("lines", []):
                    line_text  = ""
                    line_sizes = []
                    for span in line.get("spans", []):
                        t = span.get("text", "").strip()
                        if t:
                            line_text  += t + " "
                            line_sizes.append(span["size"])

                    line_text = line_text.strip()
                    if not line_text:
                        continue

                    # Classify line as heading if avg font size is above threshold
                    avg_size = sum(line_sizes) / len(line_sizes) if line_sizes else 0
                    if avg_size >= heading_threshold and len(line_text) < 120:
                        is_heading_block = True
                        current_heading  = line_text
                    else:
                        block_lines.append(line_text)

                if not is_heading_block and block_lines:
                    page_text_parts.append(
                        (current_heading, "\n".join(block_lines))
                    )

            # ── Step 3: emit one chunk per logical text block ──
            for heading, text in page_text_parts:
                text = text.strip()
                if text:
                    results.append(
                        self._make_chunk(
                            content    = text,
                            page       = page_num + 1,
                            chunk_type = "text",
                            heading    = heading,
                        )
                    )

        doc.close()
        print(f"  [TEXT]   {len(results)} text blocks extracted")
        return results

    # ── TABLE EXTRACTION ─────────────────────

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
                            self._make_chunk(
                                content    = content,
                                page       = page_num + 1,
                                chunk_type = "table",
                            )
                        )

        print(f"  [TABLES] {len(results)} tables extracted")
        return results

    # ── IMAGE EXTRACTION ─────────────────────

    def _extract_images(self) -> list[dict]:
        """
        Extract images, save to disk, and ALWAYS create a searchable chunk.

        ROOT CAUSE FIX
        --------------
        The previous version only stored a chunk when OCR returned text.
        Diagrams, figures, and architecture drawings return EMPTY OCR
        because Tesseract reads characters, not drawings. This meant the
        transformer architecture figure was silently dropped — stored on
        disk but never indexed, never retrievable.

        Now we ALWAYS store an image chunk using this priority:
            1. OCR text        — if Tesseract finds characters (text-heavy figures)
            2. Page context    — surrounding page text as semantic description
                                 (makes "show me transformer architecture" retrieve
                                  the diagram because the page text mentions it)
            3. Generic label   — absolute fallback

        Also skips tiny images < 5KB (icons, decorative elements).
        """
        os.makedirs(self.image_output_dir, exist_ok=True)
        results = []
        doc     = fitz.open(self.file_path)

        for page_num, page in enumerate(doc):
            # Grab surrounding page text — used as semantic fallback description
            page_text    = page.get_text("text").strip()
            page_context = page_text[:300].replace("\n", " ").strip()

            for img_idx, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    img_bytes  = base_image["image"]
                    img_ext    = base_image["ext"]

                    # Skip tiny images — likely icons or decorative elements
                    if len(img_bytes) < 5000:
                        continue

                    # Save image to disk with stable filename
                    img_filename = (
                        f"{os.path.splitext(self.file_name)[0]}"
                        f"_page{page_num+1}_img{img_idx+1}.{img_ext}"
                    )
                    img_path = os.path.abspath(
                        os.path.join(self.image_output_dir, img_filename)
                    )
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)

                    # ── Try OCR ──
                    pil_img  = Image.open(io.BytesIO(img_bytes))
                    ocr_text = ""
                    try:
                        ocr_text = pytesseract.image_to_string(pil_img).strip()
                    except Exception:
                        pass

                    # ── Always build a content string ──
                    if ocr_text:
                        content = (
                            f"[IMAGE — Page {page_num+1}, Figure {img_idx+1}] "
                            f"OCR text: {ocr_text}"
                        )
                    elif page_context:
                        content = (
                            f"[IMAGE — Page {page_num+1}, Figure {img_idx+1}] "
                            f"Figure found on this page. "
                            f"Page context: {page_context}"
                        )
                    else:
                        content = (
                            f"[IMAGE — Page {page_num+1}, Figure {img_idx+1}] "
                            f"Visual figure or diagram from {self.file_name}"
                        )

                    results.append(
                        self._make_chunk(
                            content    = content,
                            page       = page_num + 1,
                            chunk_type = "image",
                            image_path = img_path,
                        )
                    )

                except Exception as e:
                    print(f"  [IMAGE]  Skipped img {img_idx+1} p{page_num+1}: {e}")

        doc.close()
        print(f"  [IMAGES] {len(results)} images extracted")
        return results

    # ── STATIC HELPER ────────────────────────

    @staticmethod
    def _table_to_markdown(table: list) -> str:
        """Convert pdfplumber raw table (list of lists) → markdown string."""
        if not table:
            return ""
        cleaned   = [
            [str(cell) if cell is not None else "" for cell in row]
            for row in table
        ]
        header    = "| " + " | ".join(cleaned[0]) + " |"
        separator = "|" + "|".join(["---"] * len(cleaned[0])) + "|"
        rows      = ["| " + " | ".join(row) + " |" for row in cleaned[1:]]
        return "\n".join([header, separator] + rows)