# retrieval/hybrid_retriever.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rank_bm25 import BM25Okapi
from vectorstore.qdrant_store import QdrantVectorStore, BaseVectorStore
from embeddings.embedder import BaseEmbedder, EmbedderFactory
from retrieval.naive_retriever import RetrievalResult
from config import TOP_K


# ─────────────────────────────────────────
# BM25 INDEX
# ─────────────────────────────────────────

class BM25Index:
    """
    Lightweight in-memory BM25 keyword index.
    Built from the same chunks stored in the vector store.

    BM25 (Best Match 25) is a classical keyword ranking function —
    great at exact term matching where dense vectors may miss.
    Complement to semantic/dense search.
    """

    def __init__(self):
        self._chunks : list[dict] = []
        self._bm25   : BM25Okapi  = None

    def build(self, chunks: list[dict]) -> None:
        """
        Build BM25 index from a list of chunk dicts.
        Each chunk must have a 'content' key.
        """
        self._chunks = chunks
        tokenized    = [c["content"].lower().split() for c in chunks]
        self._bm25   = BM25Okapi(tokenized)
        print(f"  [BM25] Index built with {len(chunks)} documents.")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Return top_k chunks ranked by BM25 score.
        Returns same dict format as vector store search results.
        """
        if self._bm25 is None or not self._chunks:
            return []

        scores  = self._bm25.get_scores(query.lower().split())
        # pair each chunk with its BM25 score, sort descending
        ranked  = sorted(
            enumerate(scores),
            key    = lambda x: x[1],
            reverse= True
        )[:top_k]

        results = []
        for idx, score in ranked:
            chunk = self._chunks[idx].copy()
            chunk["score"] = round(float(score), 4)
            results.append(chunk)
        return results

    def __len__(self) -> int:
        return len(self._chunks)


# ─────────────────────────────────────────
# RECIPROCAL RANK FUSION
# ─────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_results  : list[dict],
    sparse_results : list[dict],
    k              : int   = 60,
    dense_weight   : float = 1.0,
    sparse_weight  : float = 1.0,
) -> list[dict]:
    """
    Fuse dense and sparse ranked lists using Reciprocal Rank Fusion (RRF).

    RRF formula: score(d) = Σ weight / (k + rank(d))
    where rank is 1-based position in each list.

    k=60 is the standard default (from the original RRF paper).
    Robust to score-scale differences — no normalization needed.

    Returns merged list sorted by RRF score descending,
    with rrf_score added to each chunk dict.
    """
    rrf_scores: dict[str, float] = {}
    chunk_map : dict[str, dict]  = {}

    def _key(chunk: dict) -> str:
        """Stable identity key for a chunk."""
        return chunk.get("content", "").strip()[:200]

    # ── Dense rankings ──
    for rank, chunk in enumerate(dense_results, start=1):
        key = _key(chunk)
        rrf_scores[key]  = rrf_scores.get(key, 0.0) + dense_weight / (k + rank)
        chunk_map[key]   = chunk

    # ── Sparse (BM25) rankings ──
    for rank, chunk in enumerate(sparse_results, start=1):
        key = _key(chunk)
        rrf_scores[key]  = rrf_scores.get(key, 0.0) + sparse_weight / (k + rank)
        chunk_map[key]   = chunk

    # ── Sort by fused RRF score ──
    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

    fused = []
    for key in sorted_keys:
        chunk             = chunk_map[key].copy()
        chunk["rrf_score"]= round(rrf_scores[key], 6)
        chunk["score"]    = chunk["rrf_score"]      # unify score field
        fused.append(chunk)

    return fused


# ─────────────────────────────────────────
# HYBRID RETRIEVER
# ─────────────────────────────────────────

class HybridRetriever:
    """
    Hybrid retriever: Dense (cosine similarity) + Sparse (BM25 keyword)
    fused with Reciprocal Rank Fusion (RRF).

    Why hybrid?
    - Dense search is great at semantic similarity ("revenue" ≈ "income")
    - BM25 is great at exact keyword match ("Q1 2024", proper nouns, IDs)
    - RRF combines both lists without needing score normalization

    Usage:
        retriever = HybridRetriever(vector_store=store)
        retriever.index_chunks(chunks)          # build BM25 index
        result = retriever.retrieve("Q1 revenue")
        print(result.to_context_string())
    """

    def __init__(
        self,
        vector_store   : BaseVectorStore = None,
        embedder       : BaseEmbedder    = None,
        top_k          : int             = TOP_K,
        rrf_k          : int             = 60,
        dense_weight   : float           = 1.0,
        sparse_weight  : float           = 1.0,
        deduplicate    : bool            = True,
        score_threshold: float           = 0.0,
    ):
        self.store          = vector_store or QdrantVectorStore(
            embedder=embedder or EmbedderFactory.get("huggingface")
        )
        self.top_k          = top_k
        self.rrf_k          = rrf_k
        self.dense_weight   = dense_weight
        self.sparse_weight  = sparse_weight
        self.deduplicate    = deduplicate
        self.score_threshold= score_threshold
        self.bm25           = BM25Index()

        print(f"  [HYBRID] HybridRetriever ready.")
        print(f"  [HYBRID] top_k={top_k} | rrf_k={rrf_k} | "
              f"dense_weight={dense_weight} | sparse_weight={sparse_weight}")

    # ── INDEX ────────────────────────────────

    def index_chunks(self, chunks: list[dict]) -> None:
        """
        Load chunks into the BM25 index.
        Call this after ingesting documents into the vector store
        so both indexes stay in sync.
        """
        self.bm25.build(chunks)

    # ── CORE RETRIEVAL ───────────────────────

    def retrieve(
        self,
        query   : str,
        top_k   : int = None,
    ) -> RetrievalResult:
        """
        Run dense + BM25 search, fuse with RRF, return RetrievalResult.

        Args:
            query : the search string
            top_k : override instance default

        Returns:
            RetrievalResult — same interface as NaiveRetriever
        """
        k = top_k or self.top_k

        # Fetch more candidates than needed so RRF has enough to fuse
        fetch_k = max(k * 3, 20)

        # ── Dense search ──
        dense_results  = self.store.search(query=query, top_k=fetch_k)

        # ── Sparse BM25 search ──
        sparse_results = self.bm25.search(query=query, top_k=fetch_k)

        # ── RRF fusion ──
        fused = reciprocal_rank_fusion(
            dense_results  = dense_results,
            sparse_results = sparse_results,
            k              = self.rrf_k,
            dense_weight   = self.dense_weight,
            sparse_weight  = self.sparse_weight,
        )

        # ── Score threshold ──
        if self.score_threshold > 0:
            fused = [r for r in fused if r["score"] >= self.score_threshold]

        # ── Deduplication ──
        if self.deduplicate:
            fused = self._deduplicate(fused)

        # ── Trim to top_k ──
        fused = fused[:k]

        return RetrievalResult(fused)

    def _deduplicate(self, chunks: list[dict]) -> list[dict]:
        """Remove chunks with identical content, keep first (highest RRF score)."""
        seen   = set()
        unique = []
        for chunk in chunks:
            content = chunk.get("content", "").strip()
            if content not in seen:
                seen.add(content)
                unique.append(chunk)
        return unique

    # ── CONVENIENCE ──────────────────────────

    def get_context(self, query: str, **kwargs) -> str:
        """One-liner: retrieve and return formatted context string."""
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
            "vector_store"  : self.store.get_stats(),
        }


__all__ = ["BM25Index", "reciprocal_rank_fusion", "HybridRetriever"]