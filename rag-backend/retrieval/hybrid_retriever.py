# retrieval/hybrid_retriever.py
#
# CHANGES:
#   - _expand_to_parents() now reads parent_content directly from the
#     chunk's Qdrant payload instead of doing a SQLite batch lookup.
#   - ParentStore dependency completely removed.
#   - parent_store param removed from __init__() — kept as ignored kwarg
#     for backward compatibility so existing call sites don't crash.
#   - Everything else unchanged: RRF, BM25Store, deduplication.

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.bm25_store      import BM25Store
from retrieval.naive_retriever import RetrievalResult
from vectorstore.qdrant_store  import QdrantVectorStore, BaseVectorStore
from embeddings.embedder       import BaseEmbedder, EmbedderFactory
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
    rrf_scores: dict[str, float] = {}
    chunk_map : dict[str, dict]  = {}

    def _key(chunk: dict) -> str:
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
      1. Embed query once
      2. Dense search on Qdrant
      3. BM25 keyword search
      4. RRF fusion + deduplication
      5. Trim to top_k
      6. Parent expansion — reads parent_content from Qdrant payload
         (no separate SQLite lookup needed)
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
        bm25_path      : str             = None,
        # parent_store kept as ignored kwarg for backward compat
        parent_store                     = None,
    ):
        self.embedder        = embedder or EmbedderFactory.get("huggingface")
        self.store           = vector_store or QdrantVectorStore(embedder=self.embedder)
        self.top_k           = top_k
        self.rrf_k           = rrf_k
        self.dense_weight    = dense_weight
        self.sparse_weight   = sparse_weight
        self.deduplicate     = deduplicate
        self.score_threshold = score_threshold

        from pathlib import Path
        from config import settings
        default_bm25_path = str(Path(settings.qdrant_path).parent / "bm25.pkl")
        self.bm25 = BM25Store(path=bm25_path or default_bm25_path)

        print(
            f"  [HYBRID] Ready. "
            f"top_k={top_k} | rrf_k={rrf_k} | "
            f"dense={dense_weight} | sparse={sparse_weight} | "
            f"parent_expansion=inline"
        )

    # ── INDEX ─────────────────────────────────────────────

    def index_chunks(self, chunks: list[dict]) -> None:
        self.bm25.build(chunks)

    def add_chunks(self, chunks: list[dict]) -> None:
        self.bm25.add(chunks)

    # ── CORE RETRIEVAL ────────────────────────────────────

    def retrieve(
        self,
        query        : str,
        top_k        : int  = None,
        filter_field : str  = None,
        filter_value : str  = None,
    ) -> RetrievalResult:
        k       = top_k or self.top_k
        fetch_k = max(k * 3, 20)

        # 1. Embed query
        q_vec = self.embedder.embed_text(query)

        # 2. Dense search
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

        # 3. BM25 search
        sparse_results = self.bm25.search(query=query, top_k=fetch_k)

        # 4. RRF fusion
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

        # 5. Parent expansion — inline, no DB needed
        fused = self._expand_to_parents(fused)

        return RetrievalResult(fused)

    # ── PARENT EXPANSION (inline) ─────────────────────────

    def _expand_to_parents(self, chunks: list[dict]) -> list[dict]:
        """
        Replace each child's content with its parent_content if available.

        parent_content is stored directly on the Qdrant payload by
        HierarchicalChunker — no SQLite lookup needed.

        Each unique parent is only included once (dedup by parent_id).
        Falls back to child content if parent_content not in payload.
        """
        expanded     : list[dict] = []
        seen_parents : set        = set()

        for child in chunks:
            parent_id      = child.get("parent_id", "")
            parent_content = child.get("parent_content", "")

            if parent_content and parent_id:
                # Deduplicate: if multiple children share a parent, use parent once
                if parent_id in seen_parents:
                    continue
                seen_parents.add(parent_id)

                # Replace child content with the richer parent content
                merged = {k: v for k, v in child.items()}
                merged["content"] = parent_content
                expanded.append(merged)
            else:
                # No parent_content available — use child as-is
                expanded.append(child)

        return expanded

    # ── HELPERS ───────────────────────────────────────────

    @staticmethod
    def _deduplicate(chunks: list[dict]) -> list[dict]:
        seen  : set        = set()
        unique: list[dict] = []
        for chunk in chunks:
            content = chunk.get("content", "").strip()
            if content not in seen:
                seen.add(content)
                unique.append(chunk)
        return unique

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
            "parent_mode"   : "inline_payload",
        }


__all__ = ["reciprocal_rank_fusion", "HybridRetriever"]