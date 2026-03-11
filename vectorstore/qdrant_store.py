# vectorstore/qdrant_store.py

import os
import sys
import uuid
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct, Filter,
    FieldCondition, MatchValue
)
from embeddings.embedder import BaseEmbedder, EmbedderFactory
from config import QDRANT_PATH, QDRANT_COLLECTION, EMBEDDING_DIM


# ─────────────────────────────────────────
# BASE VECTOR STORE
# ─────────────────────────────────────────

class BaseVectorStore:
    """
    Abstract base class for all vector store implementations.
    Qdrant and Pinecone both inherit from this.
    """

    def __init__(self, embedder: BaseEmbedder = None):
        self.embedder   = embedder or EmbedderFactory.get("huggingface")
        self.collection = None

    def add_documents(self, chunks: list[dict]) -> None:
        raise NotImplementedError

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        raise NotImplementedError

    def delete_collection(self) -> None:
        raise NotImplementedError

    def get_stats(self) -> dict:
        raise NotImplementedError


# ─────────────────────────────────────────
# QDRANT VECTOR STORE
# ─────────────────────────────────────────

class QdrantVectorStore(BaseVectorStore):
    """
    Local Qdrant vector store — persists to disk.
    No API key needed, runs fully offline.

    Storage: ./qdrant_local_db/
    Each document chunk becomes one Qdrant Point with:
      - vector : 384-dim embedding
      - payload: content + all metadata (source, page, type...)

    NOTE: Requires qdrant-client >= 1.7 (uses query_points API).
    """

    def __init__(
        self,
        embedder        : BaseEmbedder = None,
        collection_name : str          = QDRANT_COLLECTION,
        embedding_dim   : int          = EMBEDDING_DIM,
        path            : str          = QDRANT_PATH
    ):
        super().__init__(embedder)
        self.collection     = collection_name
        self.embedding_dim  = embedding_dim
        self.path           = path

        print(f"\n  [QDRANT] Connecting to local DB at: {path}")
        self.client = QdrantClient(path=path)
        self._ensure_collection()

    # ── SETUP ────────────────────────────────

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist yet."""
        existing = [c.name for c in self.client.get_collections().collections]

        if self.collection not in existing:
            self.client.create_collection(
                collection_name = self.collection,
                vectors_config  = VectorParams(
                    size     = self.embedding_dim,
                    distance = Distance.COSINE
                )
            )
            print(f"  [QDRANT] Created new collection: '{self.collection}'")
        else:
            print(f"  [QDRANT] Using existing collection: '{self.collection}'")

    def reset_collection(self) -> None:
        """Wipe and recreate the collection — useful for re-ingestion."""
        self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name = self.collection,
            vectors_config  = VectorParams(
                size     = self.embedding_dim,
                distance = Distance.COSINE
            )
        )
        print(f"  [QDRANT] Collection reset: '{self.collection}'")

    # ── WRITE ────────────────────────────────

    def add_documents(self, chunks: list[dict]) -> None:
        """
        Embed all chunks and upsert into Qdrant.
        Each chunk dict must have at least a 'content' key.
        All other keys become searchable payload metadata.
        """
        if not chunks:
            print("  [QDRANT] No chunks to add.")
            return

        texts   = [c["content"] for c in chunks]
        vectors = self.embedder.embed_documents(texts)

        points = []
        for chunk, vector in zip(chunks, vectors):
            payload = {k: v for k, v in chunk.items()}
            points.append(
                PointStruct(
                    id      = str(uuid.uuid4()),
                    vector  = vector,
                    payload = payload
                )
            )

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name = self.collection,
                points          = batch
            )

        print(f"  [QDRANT] ✅ Added {len(points)} vectors to '{self.collection}'")

    # ── READ ─────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Embed the query and find top_k most similar chunks.
        Returns list of dicts with content, score, and metadata.

        Uses query_points() — the current API for qdrant-client >= 1.7.
        (client.search() was removed in v1.7)
        """
        query_vector = self.embedder.embed_text(query)

        results = self.client.query_points(
            collection_name = self.collection,
            query           = query_vector,
            limit           = top_k,
            with_payload    = True
        ).points                                    # ← .points unwraps the response

        return [
            {
                "content" : r.payload.get("content", ""),
                "score"   : round(r.score, 4),
                "source"  : r.payload.get("source", "unknown"),
                "page"    : r.payload.get("page", None),
                "type"    : r.payload.get("type", "text"),
            }
            for r in results
        ]

    def search_with_filter(
        self,
        query     : str,
        filter_by : str,
        filter_val: str,
        top_k     : int = 5
    ) -> list[dict]:
        """
        Search with metadata filtering.
        Example: search only within a specific source file or type.

        Usage:
            store.search_with_filter("revenue", "source", "sales.csv")
            store.search_with_filter("chart",   "type",   "table")
        """
        query_vector = self.embedder.embed_text(query)

        results = self.client.query_points(
            collection_name = self.collection,
            query           = query_vector,
            query_filter    = Filter(
                must=[FieldCondition(
                    key   = filter_by,
                    match = MatchValue(value=filter_val)
                )]
            ),
            limit        = top_k,
            with_payload = True
        ).points

        return [
            {
                "content": r.payload.get("content", ""),
                "score"  : round(r.score, 4),
                "source" : r.payload.get("source", "unknown"),
                "page"   : r.payload.get("page", None),
                "type"   : r.payload.get("type", "text"),
            }
            for r in results
        ]

    # ── STATS ────────────────────────────────

    def get_stats(self) -> dict:
        """Return info about the current collection."""
        info = self.client.get_collection(self.collection)
        # points_count can lag on local storage — fall back to vectors_count
        total = info.points_count or info.vectors_count or 0
        return {
            "collection"   : self.collection,
            "total_vectors": total,
            "dimensions"   : self.embedding_dim,
            "distance"     : "cosine",
            "storage_path" : self.path
        }

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection)
        print(f"  [QDRANT] Deleted collection: '{self.collection}'")


__all__ = ["BaseVectorStore", "QdrantVectorStore"]