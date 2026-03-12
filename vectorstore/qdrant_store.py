# vectorstore/qdrant_store.py
# IMPROVED:
#   - search() and search_with_filter() now return ALL payload metadata:
#       heading, image_path, chunk_index, total_chunks
#   - _payload_to_dict() helper centralises this so it's consistent everywhere

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
      - vector  : 384-dim embedding
      - payload : content + ALL metadata (source, page, type,
                  heading, image_path, chunk_index, total_chunks)
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

    # ── SETUP ────────────────────────────────

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

    def reset_collection(self) -> None:
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
        The entire chunk dict becomes the payload — all metadata included.
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

    # ── PAYLOAD HELPER ───────────────────────

    @staticmethod
    def _payload_to_dict(r) -> dict:
        """
        Convert a Qdrant search result point into a clean dict.

        IMPROVED: returns ALL metadata fields so retrieval, reranking,
        and the UI have full context without separate DB lookups.

        Fields returned:
            content      — the text content
            score        — cosine similarity score
            source       — original filename
            page         — page number
            type         — text / table / image / csv / xlsx
            heading      — section heading detected by pdf_loader
            image_path   — absolute path to image on disk (images only)
            chunk_index  — position of this chunk within its document
            total_chunks — how many chunks this document has total
        """
        p = r.payload
        return {
            "content"      : p.get("content", ""),
            "score"        : round(r.score, 4),
            "source"       : p.get("source", "unknown"),
            "page"         : p.get("page", None),
            "type"         : p.get("type", "text"),
            "heading"      : p.get("heading", ""),
            "image_path"   : p.get("image_path", ""),
            "chunk_index"  : p.get("chunk_index", None),
            "total_chunks" : p.get("total_chunks", None),
        }

    # ── READ ─────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Embed the query and find top_k most similar chunks.
        Returns full metadata for each result via _payload_to_dict().
        """
        query_vector = self.embedder.embed_text(query)

        results = self.client.query_points(
            collection_name = self.collection,
            query           = query_vector,
            limit           = top_k,
            with_payload    = True
        ).points

        return [self._payload_to_dict(r) for r in results]

    def search_with_filter(
        self,
        query      : str,
        filter_by  : str,
        filter_val : str,
        top_k      : int = 5
    ) -> list[dict]:
        """
        Search with metadata filtering — narrows retrieval to a specific
        subset of the knowledge base before running vector search.

        Usage examples:
            # Only search within a specific file
            store.search_with_filter("revenue", "source", "sales.csv")

            # Only search table chunks
            store.search_with_filter("Q3 breakdown", "type", "table")

            # Only search within a specific section
            store.search_with_filter("steps", "heading", "Installation")
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

        return [self._payload_to_dict(r) for r in results]

    # ── STATS ────────────────────────────────

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

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection)
        print(f"  [QDRANT] Deleted collection: '{self.collection}'")


__all__ = ["BaseVectorStore", "QdrantVectorStore"]