# retrieval/bm25_store.py
#
# NEW FILE — replaces the in-memory BM25Index that was embedded inside
# hybrid_retriever.py in the original codebase.
#
# WHY THIS EXISTS:
#   Original BM25Index was in-memory only. Every restart silently degraded
#   hybrid retrieval to dense-only with no warning — the BM25 half of RRF
#   simply returned [] because the index was empty after reload.
#
#   This class persists both the chunk list and the BM25Okapi object to disk
#   using pickle. On startup it loads automatically, so hybrid retrieval
#   works correctly across sessions without re-ingesting.
#
# USAGE:
#   bm25 = BM25Store()
#   bm25.add(chunks)          # add + rebuild + save
#   results = bm25.search(query, top_k=20)
#   bm25.reset()              # wipe index + file

import os
import sys
import pickle
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BM25_PATH

from rank_bm25 import BM25Okapi


class BM25Store:
    """
    Persistent BM25 sparse index.

    Persists to disk so hybrid retrieval works correctly after restart.
    Append-safe: add() extends the corpus, rebuilds, and saves atomically.

    Interface matches BM25Index from original hybrid_retriever.py so
    HybridRetriever can use either with no code changes.
    """

    def __init__(self, path: str = BM25_PATH):
        self.path            = path
        self._chunks: list[dict] = []
        self._bm25: BM25Okapi    = None
        self._load()

    # ── persistence ───────────────────────────────────────

    def _load(self) -> None:
        """Load index from disk if it exists."""
        if not Path(self.path).exists():
            print("  [BM25] No saved index — will build on first ingest")
            return
        try:
            with open(self.path, "rb") as f:
                data         = pickle.load(f)
            self._chunks = data.get("chunks", [])
            self._bm25   = data.get("bm25")
            print(f"  [BM25] Loaded {len(self._chunks)} docs from disk")
        except Exception as e:
            print(f"  [BM25] Load failed ({e}) — starting fresh")
            self._chunks = []
            self._bm25   = None

    def _save(self) -> None:
        """Persist index to disk."""
        try:
            with open(self.path, "wb") as f:
                pickle.dump({"chunks": self._chunks, "bm25": self._bm25}, f)
        except Exception as e:
            print(f"  [BM25] Save failed: {e}")

    def _rebuild(self) -> None:
        """Rebuild BM25Okapi from current chunk list."""
        if not self._chunks:
            self._bm25 = None
            return
        tokenized  = [c["content"].lower().split() for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)

    # ── write ─────────────────────────────────────────────

    def build(self, chunks: list[dict]) -> None:
        """
        Replace the entire index with a new set of chunks.
        Mirrors the original BM25Index.build() interface.
        Saves to disk after rebuild.
        """
        self._chunks = chunks
        self._rebuild()
        self._save()
        print(f"  [BM25] Index built with {len(chunks)} documents.")

    def add(self, chunks: list[dict]) -> None:
        """
        Append chunks to the existing index, rebuild, and persist.
        Use this for incremental ingest (Upload new files mode).
        """
        if not chunks:
            return
        self._chunks.extend(chunks)
        self._rebuild()
        self._save()
        print(f"  [BM25] Index now has {len(self._chunks)} docs")

    def reset(self) -> None:
        """Wipe the in-memory index and delete the file."""
        self._chunks = []
        self._bm25   = None
        if Path(self.path).exists():
            Path(self.path).unlink()
        print("  [BM25] Index reset")

    # ── read ──────────────────────────────────────────────

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Return top_k chunks ranked by BM25 score.
        Returns same dict format as vector store search results.
        Scores of 0 are excluded (no keyword overlap at all).
        """
        if not self._bm25 or not self._chunks:
            return []

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results: list[dict] = []
        for idx, score in ranked[:top_k]:
            if score <= 0:
                continue
            chunk          = self._chunks[idx].copy()
            chunk["score"] = round(float(score), 4)
            results.append(chunk)

        return results

    def __len__(self) -> int:
        return len(self._chunks)


__all__ = ["BM25Store"]