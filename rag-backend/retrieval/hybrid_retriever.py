# retrieval/hybrid_retriever.py
#
# CHANGES vs original:
#   - BM25Index (in-memory) replaced with BM25Store (persistent, disk-backed)
#     Old index was lost on restart → silently degraded to dense-only retrieval
#   - Parent expansion added: retrieved child chunks are swapped for their
#     parent (1200-char) versions before passing to the LLM
#     This is the core benefit of HierarchicalChunker
#   - retrieve() now accepts optional metadata filter (filter_field, filter_value)
#     e.g. filter to a specific source file or content type
#   - store.search() receives pre-embedded vector (applies BGE query prefix)
#   - reciprocal_rank_fusion unchanged (still the paper's formula)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.bm25_store     import BM25Store
from retrieval.naive_retriever import RetrievalResult
from vectorstore.qdrant_store  import QdrantVectorStore, BaseVectorStore
from embeddings.embedder       import BaseEmbedder, EmbedderFactory
from config                    import TOP_K, RRF_K


# ─────────────────────────────────────────────────────────
# RECIPROCAL RANK FUSION  (unchanged from original)
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
        """Stable identity key for a chunk — first 200 chars of content."""
        return chunk.get("content", "").strip()[:200]

    # ── Dense rankings ──
    for rank, chunk in enumerate(dense_results, start=1):
        key              = _key(chunk)
        rrf_scores[key]  = rrf_scores.get(key, 0.0) + dense_weight / (k + rank)
        chunk_map[key]   = chunk

    # ── Sparse (BM25) rankings ──
    for rank, chunk in enumerate(sparse_results, start=1):
        key              = _key(chunk)
        rrf_scores[key]  = rrf_scores.get(key, 0.0) + sparse_weight / (k + rank)
        if key not in chunk_map:
            chunk_map[key] = chunk

    # ── Sort by fused RRF score ──
    sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    fused: list[dict] = []
    for key in sorted_keys:
        chunk             = chunk_map[key].copy()
        chunk["rrf_score"]= round(rrf_scores[key], 6)
        chunk["score"]    = chunk["rrf_score"]      # unify score field
        fused.append(chunk)

    return fused


# ─────────────────────────────────────────────────────────
# HYBRID RETRIEVER
# ─────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Hybrid retriever: Dense (cosine) + Sparse (BM25) fused with RRF.

    ADDITIONS vs original:
      1. BM25Store (persistent) instead of BM25Index (in-memory)
         Hybrid retrieval now works correctly after restart.

      2. Parent expansion (optional, enabled when parent_store is provided)
         After RRF fusion, child chunks are swapped for their parent (1200-char)
         versions. The LLM receives full context; retrieval remains precise.

      3. Metadata filtering (optional)
         retrieve(query, filter_field="source", filter_value="report.pdf")
         Pre-filters Qdrant before vector search — much faster + more relevant
         when you know which file the question is about.

      4. Pre-embedded vector passed to store.search()
         Ensures BGE query prefix is applied exactly once.

    Why hybrid?
      Dense search: great at semantic similarity ("revenue" ≈ "income")
      BM25:         great at exact keyword match ("Q1 2024", proper nouns)
      RRF:          merges both without needing score normalisation

    Usage:
        retriever = HybridRetriever(store, embedder, parent_store=parents)
        retriever.index_chunks(chunks)     # load BM25 index
        result    = retriever.retrieve("Q1 revenue")
        context   = result.to_context_string()
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
        parent_store   : dict            = None,   # NEW: {parent_id: parent_dict}
        bm25_path      : str             = None,   # NEW: optional override
    ):
        self.embedder       = embedder    or EmbedderFactory.get("huggingface")
        self.store          = vector_store or QdrantVectorStore(embedder=self.embedder)
        self.top_k          = top_k
        self.rrf_k          = rrf_k
        self.dense_weight   = dense_weight
        self.sparse_weight  = sparse_weight
        self.deduplicate    = deduplicate
        self.score_threshold= score_threshold
        self.parent_store   = parent_store or {}

        # Persistent BM25 — loads from disk automatically
        from config import BM25_PATH
        self.bm25 = BM25Store(path=bm25_path or BM25_PATH)

        print(
            f"  [HYBRID] HybridRetriever ready. "
            f"top_k={top_k} | rrf_k={rrf_k} | "
            f"dense_weight={dense_weight} | sparse_weight={sparse_weight}"
        )
        if parent_store:
            print(f"  [HYBRID] Parent store: {len(parent_store)} entries")

    # ── INDEX ─────────────────────────────────────────────

    def index_chunks(self, chunks: list[dict]) -> None:
        """
        Load chunks into the BM25 index and persist to disk.
        Call this after ingesting documents into the vector store
        so both indexes stay in sync.

        For incremental ingest (Upload new files), use:
            bm25.add(new_chunks)   ← appends without wiping existing
        """
        self.bm25.build(chunks)

    def add_chunks(self, chunks: list[dict]) -> None:
        """
        Incrementally add chunks to the BM25 index without wiping it.
        Use this for 'Upload new files' mode.
        """
        self.bm25.add(chunks)

    # ── CORE RETRIEVAL ────────────────────────────────────

    def retrieve(
        self,
        query        : str,
        top_k        : int  = None,
        filter_field : str  = None,   # NEW: e.g. "source"
        filter_value : str  = None,   # NEW: e.g. "report.pdf"
    ) -> RetrievalResult:
        """
        Full hybrid retrieval pipeline:
          1. Embed query (applies BGE query prefix)
          2. Dense search (with optional metadata filter)
          3. BM25 keyword search
          4. RRF fusion + deduplication
          5. Parent expansion (if parent_store provided)

        Args:
            query        : search string
            top_k        : override instance default
            filter_field : Qdrant payload field to filter on (e.g. "source")
            filter_value : value to match (e.g. "sales_report.pdf")

        Returns:
            RetrievalResult with parent-expanded chunks
        """
        k       = top_k or self.top_k
        fetch_k = max(k * 3, 20)   # fetch more candidates for RRF to work with

        # ── 1. Embed query ──────────────────────────────────
        q_vec = self.embedder.embed_text(query)

        # ── 2. Dense search (with optional metadata filter) ─
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

        # ── 3. BM25 keyword search ───────────────────────────
        sparse_results = self.bm25.search(query=query, top_k=fetch_k)

        # ── 4. RRF fusion ────────────────────────────────────
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

        # ── 5. Parent expansion ──────────────────────────────
        if self.parent_store:
            fused = self._expand_to_parents(fused)

        return RetrievalResult(fused)

    # ── PARENT EXPANSION ──────────────────────────────────

    def _expand_to_parents(self, chunks: list[dict]) -> list[dict]:
        """
        Replace child chunks with their parent (larger) chunks.

        Each parent is only included once even if multiple children
        from the same parent were retrieved.

        Child metadata (score, source, page, section) is preserved
        on the parent chunk so citations still show the exact location.

        Falls back to child content if parent_id not found in store.
        """
        expanded    : list[dict] = []
        seen_parents: set        = set()

        for child in chunks:
            parent_id = child.get("parent_id", "")

            if parent_id and parent_id in self.parent_store:
                if parent_id in seen_parents:
                    continue   # already included this parent
                seen_parents.add(parent_id)

                parent = self.parent_store[parent_id]
                # Merge: parent content + child retrieval metadata
                merged = {
                    # Parent content — larger, more context for LLM
                    "content"     : parent["content"],
                    # Child metadata — for citations and scoring
                    "score"       : child.get("score", 0.0),
                    "rrf_score"   : child.get("rrf_score", 0.0),
                    "source"      : child.get("source",   parent.get("source", "")),
                    "page"        : child.get("page",     parent.get("page")),
                    "type"        : child.get("type",     parent.get("type", "text")),
                    "heading"     : child.get("heading",  parent.get("heading", "")),
                    "section_path": child.get("section_path", parent.get("section_path", "")),
                    "parent_id"   : parent_id,
                    "image_path"  : child.get("image_path", ""),
                }
                expanded.append(merged)
            else:
                # No parent found — use child as-is (graceful fallback)
                expanded.append(child)

        return expanded

    # ── HELPERS ───────────────────────────────────────────

    @staticmethod
    def _deduplicate(chunks: list[dict]) -> list[dict]:
        """Remove chunks with identical content, keep first (highest RRF score)."""
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
            "parent_entries": len(self.parent_store),
            "vector_store"  : self.store.get_stats(),
        }


__all__ = ["BM25Store", "reciprocal_rank_fusion", "HybridRetriever"]