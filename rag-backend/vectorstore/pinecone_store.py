# vectorstore/pinecone_store.py
# IMPROVED — matches QdrantVectorStore metadata exactly:
#   - add_documents() stores heading, image_path, chunk_index, total_chunks
#   - search() and search_with_filter() return all fields via _metadata_to_dict()
#   - _metadata_to_dict() is the Pinecone equivalent of Qdrant's _payload_to_dict()
#
# NOTE: Pinecone metadata values must be str/int/float/bool/list only.
#       image_path and heading are stored as strings (empty string = not present).
#       chunk_index and total_chunks are stored as int (-1 = not present).

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinecone import Pinecone, ServerlessSpec
from embeddings.embedder import BaseEmbedder, EmbedderFactory
from config import PINECONE_API_KEY, PINECONE_INDEX, PINECONE_CLOUD, PINECONE_REGION, EMBEDDING_DIM


class PineconeVectorStore:
    """
    Cloud-based Pinecone vector store.
    Requires PINECONE_API_KEY in .env

    Differences vs Qdrant:
      ✅ Managed cloud — no local setup
      ✅ Scales to millions of vectors
      ❌ Needs internet + API key
      ❌ Free tier has limits (1 index, 2GB)

    Metadata stored per chunk (matches QdrantVectorStore exactly):
        content      — text content (capped at 1000 chars — Pinecone limit)
        source       — original filename
        page         — page number
        type         — text / table / image / csv / xlsx
        heading      — section heading detected by pdf_loader (or "")
        image_path   — absolute path to image on disk (or "")
        chunk_index  — position within source document (-1 if not set)
        total_chunks — total chunks for source document (-1 if not set)
    """

    def __init__(
        self,
        embedder        : BaseEmbedder = None,
        index_name      : str          = PINECONE_INDEX,
        embedding_dim   : int          = EMBEDDING_DIM,
        cloud           : str          = PINECONE_CLOUD,
        region          : str          = PINECONE_REGION,
    ):
        self.embedder      = embedder or EmbedderFactory.get("huggingface")
        self.index_name    = index_name
        self.embedding_dim = embedding_dim

        print(f"\n  [PINECONE] Connecting to Pinecone cloud...")
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self._ensure_index(cloud, region)
        self.index = self.pc.Index(self.index_name)
        print(f"  [PINECONE] ✅ Ready! Index: '{self.index_name}'")

    # ── SETUP ────────────────────────────────

    def _ensure_index(self, cloud: str, region: str) -> None:
        existing = [i.name for i in self.pc.list_indexes()]
        if self.index_name not in existing:
            print(f"  [PINECONE] Creating new index: '{self.index_name}'...")
            self.pc.create_index(
                name      = self.index_name,
                dimension = self.embedding_dim,
                metric    = "cosine",
                spec      = ServerlessSpec(cloud=cloud, region=region)
            )
            import time
            while not self.pc.describe_index(self.index_name).status["ready"]:
                print("  [PINECONE] Waiting for index to be ready...")
                time.sleep(2)
            print(f"  [PINECONE] Index created!")
        else:
            print(f"  [PINECONE] Using existing index: '{self.index_name}'")

    def reset_index(self) -> None:
        self.index.delete(delete_all=True)
        print(f"  [PINECONE] Index '{self.index_name}' cleared.")

    # ── WRITE ────────────────────────────────

    def add_documents(self, chunks: list[dict]) -> None:
        """
        Embed all chunks and upsert into Pinecone.

        IMPROVED: stores all metadata fields so search results have
        the same structure as QdrantVectorStore — heading, image_path,
        chunk_index, total_chunks are all preserved.

        Pinecone metadata rules:
          - Values must be str / int / float / bool / list
          - Total metadata per vector should stay under ~40KB
          - content is capped at 1000 chars to respect this limit
        """
        if not chunks:
            print("  [PINECONE] No chunks to add.")
            return

        texts   = [c["content"] for c in chunks]
        vectors = self.embedder.embed_documents(texts)

        records = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            metadata = {
                # Core fields
                "content"      : str(chunk.get("content", ""))[:1000],
                "source"       : str(chunk.get("source", "unknown")),
                "page"         : int(chunk.get("page") or 0),
                "type"         : str(chunk.get("type", "text")),
                # NEW: heading + image_path from improved pdf_loader
                "heading"      : str(chunk.get("heading", "")),
                "image_path"   : str(chunk.get("image_path", "")),
                # NEW: positional metadata from improved chunker
                "chunk_index"  : int(chunk.get("chunk_index")  if chunk.get("chunk_index")  is not None else -1),
                "total_chunks" : int(chunk.get("total_chunks") if chunk.get("total_chunks") is not None else -1),
            }
            records.append({
                "id"      : f"chunk_{i}_{hash(chunk['content']) % 100000}",
                "values"  : vector,
                "metadata": metadata,
            })

        # Pinecone upsert limit = 100 vectors per batch
        batch_size = 100
        for i in range(0, len(records), batch_size):
            self.index.upsert(vectors=records[i:i + batch_size])

        print(f"  [PINECONE] ✅ Added {len(records)} vectors to '{self.index_name}'")

    # ── METADATA HELPER ──────────────────────

    @staticmethod
    def _metadata_to_dict(r) -> dict:
        """
        Convert a Pinecone search result into a clean dict.
        Mirrors QdrantVectorStore._payload_to_dict() exactly so the
        rest of the pipeline (retriever → reranker → chain → UI)
        is fully interchangeable between the two stores.

        Fields returned:
            content      — text content
            score        — cosine similarity score
            source       — original filename
            page         — page number
            type         — text / table / image / csv / xlsx
            heading      — section heading (or "")
            image_path   — absolute path to saved image (or "")
            chunk_index  — position within document (-1 if unknown)
            total_chunks — total chunks for document (-1 if unknown)
        """
        m = r.metadata
        return {
            "content"      : m.get("content", ""),
            "score"        : round(r.score, 4),
            "source"       : m.get("source", "unknown"),
            "page"         : m.get("page", None),
            "type"         : m.get("type", "text"),
            "heading"      : m.get("heading", ""),
            "image_path"   : m.get("image_path", ""),
            "chunk_index"  : m.get("chunk_index", -1),
            "total_chunks" : m.get("total_chunks", -1),
        }

    # ── READ ─────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Embed the query and find top_k most similar chunks.
        Returns full metadata via _metadata_to_dict() —
        same structure as QdrantVectorStore.search().
        """
        query_vector = self.embedder.embed_text(query)

        results = self.index.query(
            vector          = query_vector,
            top_k           = top_k,
            include_metadata= True,
        )

        return [self._metadata_to_dict(r) for r in results.matches]

    def search_with_filter(
        self,
        query      : str,
        filter_by  : str,
        filter_val : str,
        top_k      : int = 5,
    ) -> list[dict]:
        """
        Search with metadata filtering.

        Usage examples:
            store.search_with_filter("revenue",      "source",  "sales.csv")
            store.search_with_filter("Q3 breakdown", "type",    "table")
            store.search_with_filter("steps",        "heading", "Installation")
        """
        query_vector = self.embedder.embed_text(query)

        results = self.index.query(
            vector          = query_vector,
            top_k           = top_k,
            include_metadata= True,
            filter          = {filter_by: {"$eq": filter_val}},
        )

        return [self._metadata_to_dict(r) for r in results.matches]

    # ── STATS ────────────────────────────────

    def get_stats(self) -> dict:
        stats = self.index.describe_index_stats()
        return {
            "index"        : self.index_name,
            "total_vectors": stats.total_vector_count,
            "dimensions"   : self.embedding_dim,
            "distance"     : "cosine",
            "provider"     : "pinecone-cloud",
        }

    def delete_index(self) -> None:
        self.pc.delete_index(self.index_name)
        print(f"  [PINECONE] Deleted index: '{self.index_name}'")