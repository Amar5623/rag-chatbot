# retrieval/naive_retriever.py
#
# CHANGES vs original:
#   - RetrievalResult.to_context_string() now includes section_path
#     in the header so the LLM knows exactly which section each chunk came from
#   - RetrievalResult.get_citations() now includes heading + section_path
#     so ChainResponse can show richer source attribution in the UI
#   - NaiveRetriever gains optional parent_store parameter
#     When provided, retrieved child chunks are expanded to their parent
#     (same small-to-big pattern as HybridRetriever)
#   - NaiveRetriever.retrieve() now embeds query itself via embedder.embed_text()
#     (previously used store.search(query_string) which re-embedded internally)
#   - All class names and method signatures unchanged

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectorstore.qdrant_store import QdrantVectorStore, BaseVectorStore
from embeddings.embedder      import BaseEmbedder, EmbedderFactory
from config import TOP_K


# ─────────────────────────────────────────────────────────
# RETRIEVAL RESULT
# ─────────────────────────────────────────────────────────

class RetrievalResult:
    """
    Wraps the output of a retrieval call.
    Provides raw chunks, formatted context, and citations
    all from one object.

    CHANGES vs original:
      - to_context_string() includes section_path in header
      - get_citations() returns heading + section_path
      - max_chars bumped 4000 → 6000 (denser PDFs need more context)
    """

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks   # list of {content, score, source, page, type, ...}

    # ── FORMATTED CONTEXT ─────────────────────────────────

    def to_context_string(self, max_chars: int = 6000) -> str:
        """
        Format chunks into a single LLM-ready context block.
        Header now includes section_path for richer attribution.

        Example output:
            [Source: report.pdf | Page: 2 | Section: Chapter 3 > Results]
            Sales increased by 30% in Q2...
        """
        parts: list[str] = []
        total = 0

        for chunk in self.chunks:
            source  = chunk.get("source", "unknown")
            page    = chunk.get("page")
            section = chunk.get("section_path") or chunk.get("heading") or ""
            content = chunk.get("content", "").strip()

            header = f"[Source: {source}"
            if page is not None:
                header += f" | Page: {page}"
            if section:
                header += f" | Section: {section}"
            header += "]"

            block = f"{header}\n{content}"

            if total + len(block) > max_chars:
                break

            parts.append(block)
            total += len(block)

        return "\n\n".join(parts)

    # ── CITATIONS ─────────────────────────────────────────

    def get_citations(self) -> list[dict]:
        """
        Return deduplicated citations with source, page, heading, section_path.
        IMPROVED: includes section_path for richer UI display.
        """
        seen:    set        = set()
        results: list[dict] = []

        for chunk in self.chunks:
            key = (chunk.get("source", "unknown"), chunk.get("page"))
            if key not in seen:
                seen.add(key)
                results.append({
                    "source"      : chunk.get("source", "unknown"),
                    "page"        : chunk.get("page"),
                    "heading"     : chunk.get("heading", ""),
                    "section_path": chunk.get("section_path", ""),
                    "type"        : chunk.get("type", "text"),
                })

        return results

    def format_citations(self) -> str:
        """Human-readable citation string."""
        cites = self.get_citations()
        if not cites:
            return ""
        lines = []
        for i, c in enumerate(cites, 1):
            page_str    = f", p.{c['page']}"     if c["page"] is not None else ""
            section_str = f" [{c['section_path']}]" if c.get("section_path") else (
                          f" [{c['heading']}]"      if c.get("heading")      else "")
            lines.append(f"  [{i}] {c['source']}{page_str}{section_str}")
        return "Sources:\n" + "\n".join(lines)

    # ── RAW ACCESS ────────────────────────────────────────

    def get_chunks(self) -> list[dict]:
        return self.chunks

    def get_top_chunk(self) -> dict | None:
        return self.chunks[0] if self.chunks else None

    def get_images(self) -> list[str]:
        """Return absolute paths of retrieved image chunks that exist on disk.
        Also tries resolving relative to the project root if the stored path
        doesn't exist at face value (handles CWD differences).
        """
        import pathlib
        project_root = pathlib.Path(__file__).parent.parent
        results = []
        for c in self.chunks:
            if c.get("type") != "image" or not c.get("image_path"):
                continue
            p = c["image_path"]
            if os.path.exists(p):
                results.append(p)
                continue
            # Try resolving basename against project_root/extracted_images
            alt = str(project_root / "extracted_images" / os.path.basename(p))
            if os.path.exists(alt):
                results.append(alt)
        return results

    def best_score(self) -> float:
        """Highest rerank/retrieval score in this result set."""
        if not self.chunks:
            return 0.0
        return self.chunks[0].get("rerank_score", self.chunks[0].get("score", 0.0))

    def __len__(self) -> int:
        return len(self.chunks)


# ─────────────────────────────────────────────────────────
# NAIVE RETRIEVER
# ─────────────────────────────────────────────────────────

class NaiveRetriever:
    """
    Dense-only vector retriever with optional parent expansion.

    CHANGES vs original:
      - parent_store parameter added (optional)
        When provided, child chunks are expanded to their parent
        before being returned — same small-to-big pattern as HybridRetriever
      - retrieve() embeds query via embedder.embed_text() directly
        (avoids double-embedding that happened when passing string to store.search())
      - get_info() unchanged
    """

    def __init__(
        self,
        vector_store : BaseVectorStore = None,
        embedder     : BaseEmbedder    = None,
        top_k        : int             = TOP_K,
        parent_store : dict            = None,  # {parent_id: parent_dict}
    ):
        self.embedder     = embedder or EmbedderFactory.get("huggingface")
        self.store        = vector_store or QdrantVectorStore(embedder=self.embedder)
        self.top_k        = top_k
        self.parent_store = parent_store or {}

        print(f"  [NAIVE] NaiveRetriever ready. top_k={top_k}")

    def retrieve(self, query: str, top_k: int = None) -> RetrievalResult:
        """
        Dense vector search with optional parent expansion.

        Args:
            query : search string
            top_k : override instance default

        Returns:
            RetrievalResult — same as before, but with parent-expanded content
        """
        k        = top_k or self.top_k
        q_vec    = self.embedder.embed_text(query)
        results  = self.store.search(query_vector=q_vec, top_k=k)

        if self.parent_store:
            results = self._expand_to_parents(results)

        return RetrievalResult(results)

    # ── PARENT EXPANSION ──────────────────────────────────

    def _expand_to_parents(self, chunks: list[dict]) -> list[dict]:
        """
        Swap child chunks for their parent chunks.
        Falls back to child if parent not found.
        Deduplicates on parent_id.
        """
        expanded:     list[dict] = []
        seen_parents: set        = set()

        for child in chunks:
            parent_id = child.get("parent_id", "")

            if parent_id and parent_id in self.parent_store:
                if parent_id in seen_parents:
                    continue
                seen_parents.add(parent_id)

                parent = self.parent_store[parent_id]
                merged = {
                    "content"     : parent["content"],
                    "score"       : child.get("score", 0.0),
                    "source"      : child.get("source", parent.get("source", "")),
                    "page"        : child.get("page",   parent.get("page")),
                    "type"        : child.get("type",   parent.get("type", "text")),
                    "heading"     : child.get("heading",      parent.get("heading", "")),
                    "section_path": child.get("section_path", parent.get("section_path", "")),
                    "parent_id"   : parent_id,
                    "image_path"  : child.get("image_path", ""),
                }
                expanded.append(merged)
            else:
                expanded.append(child)

        return expanded

    # ── CONVENIENCE ───────────────────────────────────────

    def get_context(self, query: str, **kwargs) -> str:
        """One-liner: retrieve and return formatted context string."""
        return self.retrieve(query, **kwargs).to_context_string()

    def get_info(self) -> dict:
        return {
            "type"        : "NaiveRetriever",
            "top_k"       : self.top_k,
            "parent_store": len(self.parent_store),
            "vector_store": self.store.get_stats(),
        }


__all__ = ["RetrievalResult", "NaiveRetriever"]