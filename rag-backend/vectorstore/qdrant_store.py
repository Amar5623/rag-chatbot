# vectorstore/qdrant_store.py
#
# CHANGES vs original:
#   - search() and search_with_filter() now accept a pre-embedded query_vector
#     instead of a raw query string.
#     REASON: embedding must happen in the embedder (which applies BGE query
#     prefix). If the store embeds the string itself it bypasses the prefix.
#   - list_sources() added — returns distinct source filenames in the collection
#     Used by app.py to show indexed files after a restart
#   - reset_collection() renamed to reset() for consistency (old name kept as alias)
#   - Everything else unchanged: BaseVectorStore, QdrantVectorStore, all fields

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
from config              import QDRANT_PATH, QDRANT_COLLECTION, EMBEDDING_DIM


# ─────────────────────────────────────────────────────────
# BASE VECTOR STORE
# ─────────────────────────────────────────────────────────

class BaseVectorStore:
    def __init__(self, embedder: BaseEmbedder = None):
        self.embedder   = embedder or EmbedderFactory.get("huggingface")
        self.collection = None

    def add_documents(self, chunks: list[dict]) -> None:
        raise NotImplementedError

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        raise NotImplementedError

    def delete_collection(self) -> None:
        raise NotImplementedError

    def get_stats(self) -> dict:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────
# QDRANT VECTOR STORE
# ─────────────────────────────────────────────────────────

class QdrantVectorStore(BaseVectorStore):
    """
    Local Qdrant vector store — persists to disk.
    No API key needed, runs fully offline.

    Storage: ./qdrant_local_db/
    Each document chunk becomes one Qdrant Point with:
      - vector  : 384-dim embedding (BGE-small)
      - payload : content + ALL metadata (source, page, type,
                  heading, section_path, image_path,
                  parent_id, chunk_index, total_chunks)

    CHANGE vs original:
      search() and search_with_filter() accept a pre-embedded query_vector
      instead of a raw string. The embedding step (with BGE prefix) happens
      in the retriever, not here.
    """

    def __init__(
        self,
        embedder        : BaseEmbedder = None,
        collection_name : str          = QDRANT_COLLECTION,
        embedding_dim   : int          = EMBEDDING_DIM,
        path            : str          = QDRANT_PATH
    ):
        super().__init__(embedder)
        self.collection    = collection_name
        self.embedding_dim = embedding_dim
        self.path          = path

        print(f"\n  [QDRANT] Connecting to local DB at: {path}")
        self.client = QdrantClient(path=path)
        self._ensure_collection()

    # ── SETUP ─────────────────────────────────────────────

    def _ensure_collection(self) -> None:
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

    def reset(self) -> None:
        """Wipe and recreate the collection."""
        self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name = self.collection,
            vectors_config  = VectorParams(
                size     = self.embedding_dim,
                distance = Distance.COSINE
            )
        )
        print(f"  [QDRANT] Collection reset: '{self.collection}'")

    # Keep old name as alias for backward compatibility
    def reset_collection(self) -> None:
        self.reset()

    # ── WRITE ─────────────────────────────────────────────

    def add_documents(self, chunks: list[dict]) -> None:
        """
        Embed all chunks and upsert into Qdrant.
        The entire chunk dict becomes the payload — all metadata included.
        """
        if not chunks:
            print("  [QDRANT] No chunks to add.")
            return

        texts   = [c["content"] for c in chunks]
        vectors = self.embedder.embed_documents(texts)

        points: list[PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            payload = {k: v for k, v in chunk.items()}
            points.append(
                PointStruct(
                    id      = str(uuid.uuid4()),
                    vector  = vector,
                    payload = payload
                )
            )

        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name = self.collection,
                points          = points[i : i + batch_size]
            )

        print(f"  [QDRANT] ✅ Added {len(points)} vectors to '{self.collection}'")

    # ── PAYLOAD HELPER ────────────────────────────────────

    @staticmethod
    def _payload_to_dict(r) -> dict:
        """
        Convert a Qdrant search result point into a clean dict.
        Returns ALL metadata fields for full pipeline compatibility.

        Fields returned:
            content       — text content of the chunk
            score         — cosine similarity score
            source        — original filename
            page          — page number
            type          — text / table / image / bullet / heading
            heading       — section heading
            section_path  — full breadcrumb "Ch1 > 1.2 > Results"  (NEW)
            image_path    — absolute path to image on disk
            parent_id     — key into parent_store for context expansion  (NEW)
            chunk_index   — position within parent
            total_chunks  — total children of this parent
        """
        p = r.payload
        return {
            "content"      : p.get("content",      ""),
            "score"        : round(r.score, 4),
            "source"       : p.get("source",       "unknown"),
            "page"         : p.get("page",         None),
            "type"         : p.get("type",         "text"),
            "heading"      : p.get("heading",      ""),
            "section_path" : p.get("section_path", ""),
            "image_path"   : p.get("image_path",   ""),
            "parent_id"    : p.get("parent_id",    ""),
            "chunk_index"  : p.get("chunk_index",  None),
            "total_chunks" : p.get("total_chunks", None),
        }

    # ── READ ──────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],   # CHANGED: was (query: str)
        top_k       : int = 5,
    ) -> list[dict]:
        """
        Find top_k most similar chunks by cosine similarity.

        CHANGE: now accepts a pre-embedded vector, not a raw string.
        The calling retriever is responsible for embedding with the correct
        BGE query prefix before calling this method.
        """
        results = self.client.query_points(
            collection_name = self.collection,
            query           = query_vector,
            limit           = top_k,
            with_payload    = True
        ).points

        return [self._payload_to_dict(r) for r in results]

    def search_with_filter(
        self,
        query_vector: list[float],   # CHANGED: was (query: str)
        filter_by   : str,
        filter_val  : str,
        top_k       : int = 5
    ) -> list[dict]:
        """
        Search with metadata filtering — narrows retrieval to a specific
        subset of the knowledge base before running vector search.

        CHANGE: now accepts a pre-embedded vector, not a raw string.

        Usage examples:
            store.search_with_filter(vec, "source",  "sales.csv")
            store.search_with_filter(vec, "type",    "table")
            store.search_with_filter(vec, "heading", "Introduction")
        """
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

        return [self._payload_to_dict(r) for r in results]

    # ── STATS ─────────────────────────────────────────────

    def get_stats(self) -> dict:
        info  = self.client.get_collection(self.collection)
        total = info.points_count or info.vectors_count or 0
        return {
            "collection"   : self.collection,
            "total_vectors": total,
            "dimensions"   : self.embedding_dim,
            "distance"     : "cosine",
            "storage_path" : self.path
        }

    def count(self) -> int:
        """Return vector count — convenience for app.py."""
        try:
            return self.client.count(collection_name=self.collection).count
        except Exception:
            return 0

    def list_sources(self) -> list[str]:
        """
        Return distinct source filenames in the collection.
        Used by app.py to restore indexed_files list after restart.
        """
        try:
            result = self.client.scroll(
                collection_name = self.collection,
                limit           = 10_000,
                with_payload    = ["source"],
                with_vectors    = False,
            )
            sources = {pt.payload.get("source", "") for pt in result[0]}
            return sorted(s for s in sources if s)
        except Exception:
            return []

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection)
        print(f"  [QDRANT] Deleted collection: '{self.collection}'")


__all__ = ["BaseVectorStore", "QdrantVectorStore"]