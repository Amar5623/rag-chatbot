# vectorstore/pinecone_store.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinecone import Pinecone, ServerlessSpec
from embeddings.embedder import BaseEmbedder, EmbedderFactory
from config import PINECONE_API_KEY, EMBEDDING_DIM


class PineconeVectorStore:
    """
    Cloud-based Pinecone vector store.
    Requires PINECONE_API_KEY in .env

    Differences vs Qdrant:
      ✅ Managed cloud — no local setup
      ✅ Scales to millions of vectors
      ❌ Needs internet + API key
      ❌ Free tier has limits (1 index, 2GB)

    Same interface as QdrantVectorStore so they're interchangeable.
    """

    def __init__(
        self,
        embedder        : BaseEmbedder = None,
        index_name      : str          = "rag-chatbot",
        embedding_dim   : int          = EMBEDDING_DIM,
        cloud           : str          = "aws",
        region          : str          = "us-east-1"
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
        """Create index if it doesn't exist yet."""
        existing = [i.name for i in self.pc.list_indexes()]

        if self.index_name not in existing:
            print(f"  [PINECONE] Creating new index: '{self.index_name}'...")
            self.pc.create_index(
                name      = self.index_name,
                dimension = self.embedding_dim,
                metric    = "cosine",
                spec      = ServerlessSpec(cloud=cloud, region=region)
            )
            # Wait for index to be ready
            import time
            while not self.pc.describe_index(self.index_name).status["ready"]:
                print("  [PINECONE] Waiting for index to be ready...")
                time.sleep(2)
            print(f"  [PINECONE] Index created!")
        else:
            print(f"  [PINECONE] Using existing index: '{self.index_name}'")

    def reset_index(self) -> None:
        """Wipe all vectors from the index."""
        self.index.delete(delete_all=True)
        print(f"  [PINECONE] Index '{self.index_name}' cleared.")

    # ── WRITE ────────────────────────────────

    def add_documents(self, chunks: list[dict]) -> None:
        """
        Embed all chunks and upsert into Pinecone.
        Pinecone expects: (id, vector, metadata) tuples.
        """
        if not chunks:
            print("  [PINECONE] No chunks to add.")
            return

        texts   = [c["content"] for c in chunks]
        vectors = self.embedder.embed_documents(texts)

        # Build upsert records
        records = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            # Pinecone metadata values must be str/int/float/bool/list
            metadata = {
                "content": chunk.get("content", "")[:1000], # Pinecone limits metadata size
                "source" : str(chunk.get("source", "unknown")),
                "page"   : int(chunk.get("page",   0)),
                "type"   : str(chunk.get("type",   "text")),
            }
            records.append({
                "id"     : f"chunk_{i}_{hash(chunk['content']) % 100000}",
                "values" : vector,
                "metadata": metadata
            })

        # Upsert in batches of 100 (Pinecone limit)
        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            self.index.upsert(vectors=batch)

        print(f"  [PINECONE] ✅ Added {len(records)} vectors to '{self.index_name}'")

    # ── READ ─────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Embed the query and find top_k most similar chunks.
        Returns same format as QdrantVectorStore.search()
        so the rest of the pipeline is interchangeable.
        """
        query_vector = self.embedder.embed_text(query)

        results = self.index.query(
            vector          = query_vector,
            top_k           = top_k,
            include_metadata= True
        )

        return [
            {
                "content": r.metadata.get("content", ""),
                "score"  : round(r.score, 4),
                "source" : r.metadata.get("source", "unknown"),
                "page"   : r.metadata.get("page",   None),
                "type"   : r.metadata.get("type",   "text"),
            }
            for r in results.matches
        ]

    def search_with_filter(
        self,
        query     : str,
        filter_by : str,
        filter_val: str,
        top_k     : int = 5
    ) -> list[dict]:
        """
        Search with metadata filter.
        Example: only search within a specific file type or source.
        """
        query_vector = self.embedder.embed_text(query)

        results = self.index.query(
            vector          = query_vector,
            top_k           = top_k,
            include_metadata= True,
            filter          = {filter_by: {"$eq": filter_val}}
        )

        return [
            {
                "content": r.metadata.get("content", ""),
                "score"  : round(r.score, 4),
                "source" : r.metadata.get("source", "unknown"),
                "page"   : r.metadata.get("page",   None),
                "type"   : r.metadata.get("type",   "text"),
            }
            for r in results.matches
        ]

    # ── STATS ────────────────────────────────

    def get_stats(self) -> dict:
        """Return info about the current Pinecone index."""
        stats = self.index.describe_index_stats()
        return {
            "index"        : self.index_name,
            "total_vectors": stats.total_vector_count,
            "dimensions"   : self.embedding_dim,
            "distance"     : "cosine",
            "provider"     : "pinecone-cloud"
        }

    def delete_index(self) -> None:
        self.pc.delete_index(self.index_name)
        print(f"  [PINECONE] Deleted index: '{self.index_name}'")