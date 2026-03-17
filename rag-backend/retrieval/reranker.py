# retrieval/reranker.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import CrossEncoder
from retrieval.naive_retriever import RetrievalResult


# ─────────────────────────────────────────
# CROSS-ENCODER RERANKER
# ─────────────────────────────────────────

class Reranker:
    """
    Cross-encoder reranker — rescores retrieved chunks against the query.

    Why rerank?
    - Bi-encoders (used in retrieval) embed query and doc separately → fast but imprecise
    - Cross-encoders see (query, doc) together → slower but much more accurate
    - Standard RAG pattern: retrieve top-20 cheaply, rerank to top-5 accurately

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
      ✅ Fast  ✅ Free  ✅ Runs offline  ✅ Strong on passage ranking
      384-dim, trained on MS MARCO passage ranking dataset

    Usage:
        retriever = HybridRetriever(...)
        reranker  = Reranker()

        result    = retriever.retrieve(query, top_k=20)   # fetch wide
        reranked  = reranker.rerank(query, result, top_k=5)  # rerank narrow
        context   = reranked.to_context_string()
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name : str   = DEFAULT_MODEL,
        batch_size : int   = 16,
        max_length : int   = 512,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        print(f"  [RERANKER] Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(
            model_name,
            max_length = max_length,
        )
        print(f"  [RERANKER] ✅ Ready!")

    # ── CORE RERANK ──────────────────────────

    def rerank(
        self,
        query        : str,
        retrieval    : RetrievalResult,
        top_k        : int  = 5,
        score_threshold: float = None,
    ) -> RetrievalResult:
        """
        Rerank a RetrievalResult using the cross-encoder.

        Args:
            query           : original search query
            retrieval       : output from any retriever (Naive or Hybrid)
            top_k           : how many chunks to keep after reranking
            score_threshold : optional min cross-encoder score to keep

        Returns:
            New RetrievalResult sorted by cross-encoder score,
            with 'rerank_score' and updated 'score' fields.
        """
        chunks = retrieval.get_chunks()

        if not chunks:
            print("  [RERANKER] No chunks to rerank.")
            return RetrievalResult([])

        # Build (query, passage) pairs for cross-encoder
        pairs = [(query, c["content"]) for c in chunks]

        # Score all pairs — cross-encoder sees query+doc together
        scores = self.model.predict(
            pairs,
            batch_size       = self.batch_size,
            show_progress_bar= len(pairs) > 20,   # only show bar for large sets
        )

        # Attach rerank scores to chunks
        scored_chunks = []
        for chunk, score in zip(chunks, scores):
            c = chunk.copy()
            c["rerank_score"]  = round(float(score), 4)
            c["retrieval_score"] = c.get("score", 0.0)   # preserve original score
            c["score"]         = c["rerank_score"]        # unify score field
            scored_chunks.append(c)

        # Sort by cross-encoder score descending
        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Apply score threshold if set
        if score_threshold is not None:
            scored_chunks = [
                c for c in scored_chunks
                if c["rerank_score"] >= score_threshold
            ]

        # Trim to top_k
        scored_chunks = scored_chunks[:top_k]

        print(
            f"  [RERANKER] Reranked {len(chunks)} → kept top {len(scored_chunks)}"
        )

        return RetrievalResult(scored_chunks)

    # ── CONVENIENCE ──────────────────────────

    def rerank_chunks(
        self,
        query  : str,
        chunks : list[dict],
        top_k  : int = 5,
    ) -> list[dict]:
        """
        Rerank raw chunk dicts directly (no RetrievalResult wrapper needed).
        Returns sorted list of chunk dicts.
        """
        return self.rerank(
            query     = query,
            retrieval = RetrievalResult(chunks),
            top_k     = top_k,
        ).get_chunks()

    def get_info(self) -> dict:
        return {
            "type"      : "Reranker",
            "model"     : self.model_name,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
        }


__all__ = ["Reranker"]