# retrieval/naive_retriever.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectorstore.qdrant_store import QdrantVectorStore, BaseVectorStore
from embeddings.embedder import BaseEmbedder, EmbedderFactory
from config import TOP_K


# ─────────────────────────────────────────
# RETRIEVAL RESULT
# ─────────────────────────────────────────

class RetrievalResult:
    """
    Wraps the output of a retrieval call.
    Provides raw chunks, formatted context, and citations
    all from one object.
    """

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks            # list of {content, score, source, page, type}

    # ── FORMATTED CONTEXT ────────────────────

    def to_context_string(self, max_chars: int = 4000) -> str:
        """
        Format chunks into a single LLM-ready context block.
        Each chunk is labelled with its source and page.

        Example output:
            [Source: report.pdf | Page: 2]
            Sales increased by 30% in Q2...

            [Source: notes.txt | Page: 0]
            Python is used for data science...
        """
        parts   = []
        total   = 0

        for chunk in self.chunks:
            source  = chunk.get("source", "unknown")
            page    = chunk.get("page")
            content = chunk.get("content", "").strip()

            header  = f"[Source: {source}"
            if page is not None:
                header += f" | Page: {page}"
            header += "]"

            block = f"{header}\n{content}"

            if total + len(block) > max_chars:
                break                   # stop before exceeding LLM context budget

            parts.append(block)
            total += len(block)

        return "\n\n".join(parts)

    # ── CITATIONS ────────────────────────────

    def get_citations(self) -> list[dict]:
        """
        Return unique source citations (file + page) from retrieved chunks.
        Deduplicates by (source, page) pair.

        Example:
            [
                {"source": "report.pdf",  "page": 2},
                {"source": "notes.txt",   "page": 0},
            ]
        """
        seen    = set()
        results = []
        for chunk in self.chunks:
            key = (chunk.get("source", "unknown"), chunk.get("page"))
            if key not in seen:
                seen.add(key)
                results.append({
                    "source": chunk.get("source", "unknown"),
                    "page"  : chunk.get("page"),
                })
        return results

    def format_citations(self) -> str:
        """Human-readable citation string, e.g. for appending to LLM answers."""
        cites = self.get_citations()
        if not cites:
            return ""
        lines = []
        for i, c in enumerate(cites, 1):
            page_str = f", p.{c['page']}" if c["page"] is not None else ""
            lines.append(f"  [{i}] {c['source']}{page_str}")
        return "Sources:\n" + "\n".join(lines)

    # ── RAW ACCESS ───────────────────────────

    def get_chunks(self) -> list[dict]:
        """Return raw chunk dicts with scores."""
        return self.chunks

    def get_top_chunk(self) -> dict | None:
        """Return the single highest-scoring chunk."""
        return self.chunks[0] if self.chunks else None

    def __len__(self) -> int:
        return len(self.chunks)

    def __repr__(self) -> str:
        return (
            f"RetrievalResult("
            f"{len(self.chunks)} chunks, "
            f"top_score={self.chunks[0]['score'] if self.chunks else 'n/a'})"
        )


# ─────────────────────────────────────────
# NAIVE RETRIEVER
# ─────────────────────────────────────────

class NaiveRetriever:
    """
    Simple dense retriever — embeds query, searches vector store,
    deduplicates, and returns a RetrievalResult.

    'Naive' means: pure cosine similarity, no reranking, no hybrid search.
    That's handled by HybridRetriever and Reranker in separate modules.

    Default vector store : Qdrant (local)
    Accepts any store    : QdrantVectorStore or PineconeVectorStore
    """

    def __init__(
        self,
        vector_store     : BaseVectorStore = None,
        embedder         : BaseEmbedder    = None,
        top_k            : int             = TOP_K,
        score_threshold  : float           = 0.0,     # min score to keep a chunk
        deduplicate      : bool            = True,
    ):
        # Default to local Qdrant if no store provided
        self.store           = vector_store or QdrantVectorStore(
            embedder=embedder or EmbedderFactory.get("huggingface")
        )
        self.top_k           = top_k
        self.score_threshold = score_threshold
        self.deduplicate     = deduplicate

        print(f"  [RETRIEVER] NaiveRetriever ready.")
        print(f"  [RETRIEVER] top_k={top_k} | "
              f"score_threshold={score_threshold} | "
              f"deduplicate={deduplicate}")

    # ── CORE RETRIEVAL ───────────────────────

    def retrieve(
        self,
        query : str,
        top_k : int   = None,
        filter_by  : str = None,
        filter_val : str = None,
    ) -> RetrievalResult:
        """
        Retrieve top-k most relevant chunks for a query.

        Args:
            query      : the search string
            top_k      : override instance default
            filter_by  : optional metadata field to filter on (e.g. "source")
            filter_val : value to match for filter_by (e.g. "report.pdf")

        Returns:
            RetrievalResult with chunks, context string, and citations
        """
        k = top_k or self.top_k

        # ── Search ──
        if filter_by and filter_val:
            raw = self.store.search_with_filter(
                query      = query,
                filter_by  = filter_by,
                filter_val = filter_val,
                top_k      = k,
            )
        else:
            raw = self.store.search(query=query, top_k=k)

        # ── Score threshold ──
        if self.score_threshold > 0:
            raw = [r for r in raw if r["score"] >= self.score_threshold]

        # ── Deduplication ──
        if self.deduplicate:
            raw = self._deduplicate(raw)

        return RetrievalResult(raw)

    def _deduplicate(self, chunks: list[dict]) -> list[dict]:
        """
        Remove chunks with identical content.
        Keeps the first occurrence (highest score since results are sorted).
        """
        seen    = set()
        unique  = []
        for chunk in chunks:
            content = chunk.get("content", "").strip()
            if content not in seen:
                seen.add(content)
                unique.append(chunk)
        return unique

    # ── CONVENIENCE ──────────────────────────

    def get_context(self, query: str, **kwargs) -> str:
        """One-liner: retrieve and immediately return formatted context string."""
        return self.retrieve(query, **kwargs).to_context_string()

    def get_info(self) -> dict:
        return {
            "type"            : "NaiveRetriever",
            "top_k"           : self.top_k,
            "score_threshold" : self.score_threshold,
            "deduplicate"     : self.deduplicate,
            "vector_store"    : self.store.get_stats(),
        }


__all__ = ["RetrievalResult", "NaiveRetriever"]