# retrieval/hybrid_retriever.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.bm25_store      import BM25Store
from retrieval.naive_retriever import RetrievalResult
from vectorstore.qdrant_store  import QdrantVectorStore, BaseVectorStore
from embeddings.embedder       import BaseEmbedder, EmbedderFactory
from utils.parent_store        import ParentStore
from config                    import TOP_K, RRF_K


# ─────────────────────────────────────────────────────────
# RECIPROCAL RANK FUSION
# ─────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_results  : list[dict],
    sparse_results : list[dict],
    k              : int   = RRF_K,
    dense_weight   : float = 1.0,
    sparse_weight  : float = 1.0,
) -> list[dict]:
    """
    Fuse dense and sparse ranked lists using Reciprocal Rank Fusion (RRF).

    RRF formula: score(d) = Σ weight / (k + rank(d))
    where rank is 1-based position in each list.

    k=60 is the standard default (from the original RRF paper).
    Robust to score-scale differences — no normalisation needed.

    Returns merged list sorted by RRF score descending,
    with rrf_score added to each chunk dict.
    """
    rrf_scores: dict[str, float] = {}
    chunk_map : dict[str, dict]  = {}

    def _key(chunk: dict) -> str:
        """Stable identity key — first 200 chars of content."""
        return chunk.get("content", "").strip()[:200]

    for rank, chunk in enumerate(dense_results, start=1):
        key             = _key(chunk)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + dense_weight / (k + rank)
        chunk_map[key]  = chunk

    for rank, chunk in enumerate(sparse_results, start=1):
        key             = _key(chunk)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + sparse_weight / (k + rank)
        if key not in chunk_map:
            chunk_map[key] = chunk

    sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    fused: list[dict] = []
    for key in sorted_keys:
        chunk              = chunk_map[key].copy()
        chunk["rrf_score"] = round(rrf_scores[key], 6)
        chunk["score"]     = chunk["rrf_score"]
        fused.append(chunk)

    return fused


# ─────────────────────────────────────────────────────────
# HYBRID RETRIEVER
# ─────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Hybrid retriever: Dense (cosine) + Sparse (BM25) fused with RRF.

    Pipeline per query:
      1. Embed query once (BGE query prefix applied here)
      2. Dense search on Qdrant (with optional metadata filter)
      3. BM25 keyword search on persisted BM25Store
      4. RRF fusion + content deduplication
      5. Trim to top_k
      6. Parent expansion via batch fetch from ParentStore

    parent_store is now a ParentStore instance (SQLite-backed)
    instead of a plain dict — enables batch fetching in one
    DB query rather than N individual lookups.
    """

    def __init__(
        self,
        vector_store   : BaseVectorStore = None,
        embedder       : BaseEmbedder    = None,
        top_k          : int             = TOP_K,
        rrf_k          : int             = RRF_K,
        dense_weight   : float           = 1.0,
        sparse_weight  : float           = 1.0,
        deduplicate    : bool            = True,
        score_threshold: float           = 0.0,
        parent_store   : ParentStore     = None,
        bm25_path      : str             = None,
    ):
        self.embedder        = embedder or EmbedderFactory.get("huggingface")
        self.store           = vector_store or QdrantVectorStore(embedder=self.embedder)
        self.top_k           = top_k
        self.rrf_k           = rrf_k
        self.dense_weight    = dense_weight
        self.sparse_weight   = sparse_weight
        self.deduplicate     = deduplicate
        self.score_threshold = score_threshold
        self.parent_store    = parent_store   # ParentStore instance or None

        # BM25Store loads from disk automatically on init
        from pathlib import Path
        from config import settings
        default_bm25_path = str(Path(settings.qdrant_path).parent / "bm25.pkl")
        self.bm25 = BM25Store(path=bm25_path or default_bm25_path)

        print(
            f"  [HYBRID] Ready. "
            f"top_k={top_k} | rrf_k={rrf_k} | "
            f"dense_weight={dense_weight} | sparse_weight={sparse_weight}"
        )
        if parent_store:
            print(f"  [HYBRID] Parent store attached ({len(parent_store)} entries)")

    # ── INDEX ─────────────────────────────────────────────

    def index_chunks(self, chunks: list[dict]) -> None:
        """
        Replace the entire BM25 index with a new set of chunks.
        Use for initial build or full rebuild only.
        For incremental ingest use add_chunks().
        """
        self.bm25.build(chunks)

    def add_chunks(self, chunks: list[dict]) -> None:
        """
        Incrementally add chunks to BM25 without wiping existing index.
        Called by rag_service after each upload.
        """
        self.bm25.add(chunks)

    # ── CORE RETRIEVAL ────────────────────────────────────

    def retrieve(
        self,
        query        : str,
        top_k        : int  = None,
        filter_field : str  = None,
        filter_value : str  = None,
    ) -> RetrievalResult:
        """
        Full hybrid retrieval pipeline.

        Args:
            query        : search string
            top_k        : override instance default
            filter_field : Qdrant payload field to filter on (e.g. "source")
            filter_value : value to match  (e.g. "sales_report.pdf")

        Returns:
            RetrievalResult with parent-expanded chunks
        """
        k       = top_k or self.top_k
        fetch_k = max(k * 3, 20)   # over-fetch for RRF to work with

        # ── 1. Embed query once ────────────────────────────
        q_vec = self.embedder.embed_text(query)

        # ── 2. Dense search ───────────────────────────────
        if filter_field and filter_value:
            dense_results = self.store.search_with_filter(
                query_vector = q_vec,
                filter_by    = filter_field,
                filter_val   = filter_value,
                top_k        = fetch_k,
            )
        else:
            dense_results = self.store.search(
                query_vector = q_vec,
                top_k        = fetch_k,
            )

        # ── 3. BM25 search ────────────────────────────────
        sparse_results = self.bm25.search(query=query, top_k=fetch_k)

        # ── 4. RRF fusion ─────────────────────────────────
        fused = reciprocal_rank_fusion(
            dense_results  = dense_results,
            sparse_results = sparse_results,
            k              = self.rrf_k,
            dense_weight   = self.dense_weight,
            sparse_weight  = self.sparse_weight,
        )

        if self.score_threshold > 0:
            fused = [r for r in fused if r["score"] >= self.score_threshold]

        if self.deduplicate:
            fused = self._deduplicate(fused)

        fused = fused[:k]

        # ── 5. Parent expansion ───────────────────────────
        if self.parent_store:
            fused = self._expand_to_parents(fused)

        return RetrievalResult(fused)

    # ── PARENT EXPANSION ──────────────────────────────────

    def _expand_to_parents(self, chunks: list[dict]) -> list[dict]:
        """
        Replace child chunks with their parent (larger) chunks.

        Uses ParentStore.get_batch() — one SQLite query for all
        parent IDs instead of N individual dict lookups.

        Each parent is only included once even if multiple children
        from the same parent were retrieved.

        Child metadata (score, source, page, section) is preserved
        on the parent so citations still show the exact location.

        Falls back to child content if parent_id not in store.
        """
        parent_ids = [
            c.get("parent_id", "") for c in chunks
            if c.get("parent_id")
        ]

        # Single batch query
        parent_map   = self.parent_store.get_batch(parent_ids) if parent_ids else {}
        expanded     : list[dict] = []
        seen_parents : set        = set()

        for child in chunks:
            parent_id = child.get("parent_id", "")

            if parent_id and parent_id in parent_map:
                if parent_id in seen_parents:
                    continue
                seen_parents.add(parent_id)

                parent = parent_map[parent_id]
                merged = {
                    # Parent content — larger, more context for LLM
                    "content"     : parent["content"],
                    # Child metadata — for citations and scoring
                    "score"       : child.get("score",        0.0),
                    "rrf_score"   : child.get("rrf_score",    0.0),
                    "source"      : child.get("source",       parent.get("source", "")),
                    "page"        : child.get("page",         parent.get("page")),
                    "type"        : child.get("type",         parent.get("type", "text")),
                    "heading"     : child.get("heading",      parent.get("heading", "")),
                    "section_path": child.get("section_path", parent.get("section_path", "")),
                    "parent_id"   : parent_id,
                    "image_path"  : child.get("image_path", ""),
                }
                expanded.append(merged)
            else:
                expanded.append(child)

        return expanded

    # ── HELPERS ───────────────────────────────────────────

    @staticmethod
    def _deduplicate(chunks: list[dict]) -> list[dict]:
        """Remove chunks with identical content. Keep first (highest RRF score)."""
        seen  : set        = set()
        unique: list[dict] = []
        for chunk in chunks:
            content = chunk.get("content", "").strip()
            if content not in seen:
                seen.add(content)
                unique.append(chunk)
        return unique

    # ── CONVENIENCE ───────────────────────────────────────

    def get_context(self, query: str, **kwargs) -> str:
        return self.retrieve(query, **kwargs).to_context_string()

    def get_info(self) -> dict:
        return {
            "type"          : "HybridRetriever",
            "top_k"         : self.top_k,
            "rrf_k"         : self.rrf_k,
            "dense_weight"  : self.dense_weight,
            "sparse_weight" : self.sparse_weight,
            "deduplicate"   : self.deduplicate,
            "bm25_docs"     : len(self.bm25),
            "parent_entries": len(self.parent_store) if self.parent_store else 0,
            "vector_store"  : self.store.get_stats(),
        }


__all__ = ["reciprocal_rank_fusion", "HybridRetriever"]