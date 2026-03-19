# vectorstore/pinecone_store.py
#
# CHANGES vs original:
#   - search() and search_with_filter() now accept query_vector (pre-embedded)
#     instead of raw query string — matches QdrantVectorStore interface exactly
#   - delete_by_source(filename) added — deletes all vectors for a file
#   - count() added — returns total vector count
#   - list_sources() added — returns distinct source filenames
#   - parent_content stored in metadata (inline parent, same as Qdrant)
#   - reset_collection() alias added for interface parity with Qdrant
#
# SWITCHING: set VECTOR_STORE=pinecone in .env.
# Both stores are now fully interchangeable — same method signatures,
# same return dict structure, same delete interface.

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinecone import Pinecone, ServerlessSpec
from embeddings.embedder import BaseEmbedder, EmbedderFactory
from config import PINECONE_API_KEY, PINECONE_INDEX, PINECONE_CLOUD, PINECONE_REGION, EMBEDDING_DIM


class PineconeVectorStore:
    """
    Cloud Pinecone vector store.

    Fully interface-compatible with QdrantVectorStore:
      - add_documents(chunks)
      - search(query_vector, top_k)
      - search_with_filter(query_vector, filter_by, filter_val, top_k)
      - delete_by_source(filename) → int
      - count() → int
      - list_sources() → list[str]
      - reset() / reset_collection()

    Switch from Qdrant to Pinecone with one .env change:
        VECTOR_STORE=pinecone
        PINECONE_API_KEY=your-key-here
    """

    # Pinecone metadata value limit per field
    _CONTENT_LIMIT = 1000

    def __init__(
        self,
        embedder      : BaseEmbedder = None,
        index_name    : str          = PINECONE_INDEX,
        embedding_dim : int          = EMBEDDING_DIM,
        cloud         : str          = PINECONE_CLOUD,
        region        : str          = PINECONE_REGION,
    ):
        self.embedder      = embedder or EmbedderFactory.get("huggingface")
        self.index_name    = index_name
        self.embedding_dim = embedding_dim

        print(f"\n  [PINECONE] Connecting to Pinecone cloud...")
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self._ensure_index(cloud, region)
        self.index = self.pc.Index(self.index_name)
        print(f"  [PINECONE] ✅ Ready! Index: '{self.index_name}'")

    # ── SETUP ────────────────────────────────────────────────

    def _ensure_index(self, cloud: str, region: str) -> None:
        existing = [i.name for i in self.pc.list_indexes()]
        if self.index_name not in existing:
            print(f"  [PINECONE] Creating index '{self.index_name}'...")
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

    def reset(self) -> None:
        """Wipe all vectors from the index."""
        self.index.delete(delete_all=True)
        print(f"  [PINECONE] Index '{self.index_name}' cleared.")

    # Keep alias for interface parity with QdrantVectorStore
    def reset_collection(self) -> None:
        self.reset()

    # ── WRITE ────────────────────────────────────────────────

    def add_documents(self, chunks: list[dict]) -> None:
        """
        Embed all chunks and upsert into Pinecone.
        Stores parent_content inline (same as Qdrant) — capped at 4000 chars
        since parent chunks are larger than child chunks.
        """
        if not chunks:
            print("  [PINECONE] No chunks to add.")
            return

        texts   = [c["content"] for c in chunks]
        vectors = self.embedder.embed_documents(texts)

        records = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            metadata = {
                "content"       : str(chunk.get("content",        ""))[:self._CONTENT_LIMIT],
                "parent_content": str(chunk.get("parent_content", ""))[:4000],
                "source"        : str(chunk.get("source",         "unknown")),
                "page"          : int(chunk.get("page") or 0),
                "type"          : str(chunk.get("type",           "text")),
                "heading"       : str(chunk.get("heading",        "")),
                "section_path"  : str(chunk.get("section_path",   "")),
                "image_path"    : str(chunk.get("image_path",     "")),
                "parent_id"     : str(chunk.get("parent_id",      "")),
                "chunk_index"   : int(chunk.get("chunk_index")  if chunk.get("chunk_index")  is not None else -1),
                "total_chunks"  : int(chunk.get("total_chunks") if chunk.get("total_chunks") is not None else -1),
            }
            records.append({
                "id"      : f"chunk_{i}_{hash(chunk.get('content','')) % 10_000_000}",
                "values"  : vector,
                "metadata": metadata,
            })

        batch_size = 100
        for i in range(0, len(records), batch_size):
            self.index.upsert(vectors=records[i:i + batch_size])

        print(f"  [PINECONE] ✅ Upserted {len(records)} vectors to '{self.index_name}'")

    # ── DELETE ───────────────────────────────────────────────

    def delete_by_source(self, filename: str) -> int:
        """
        Delete all vectors whose metadata source == filename.
        Pinecone doesn't return a deletion count directly,
        so we count before and after.
        Returns number of vectors deleted.
        """
        before = self.count()

        # Pinecone delete with metadata filter
        self.index.delete(
            filter={"source": {"$eq": filename}}
        )

        after   = self.count()
        deleted = max(before - after, 0)
        print(f"  [PINECONE] Deleted ~{deleted} vectors for source='{filename}'")
        return deleted

    # ── METADATA HELPER ──────────────────────────────────────

    @staticmethod
    def _metadata_to_dict(r) -> dict:
        """
        Convert a Pinecone result into a clean dict.
        Identical structure to QdrantVectorStore._payload_to_dict()
        so the rest of the pipeline is fully interchangeable.
        """
        m = r.metadata
        return {
            "content"       : m.get("content",        ""),
            "score"         : round(r.score, 4),
            "source"        : m.get("source",         "unknown"),
            "page"          : m.get("page",           None),
            "type"          : m.get("type",           "text"),
            "heading"       : m.get("heading",        ""),
            "section_path"  : m.get("section_path",   ""),
            "image_path"    : m.get("image_path",     ""),
            "parent_id"     : m.get("parent_id",      ""),
            "parent_content": m.get("parent_content", ""),
            "chunk_index"   : m.get("chunk_index",    -1),
            "total_chunks"  : m.get("total_chunks",   -1),
        }

    # ── READ ─────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],   # pre-embedded — matches Qdrant interface
        top_k       : int = 5,
    ) -> list[dict]:
        results = self.index.query(
            vector           = query_vector,
            top_k            = top_k,
            include_metadata = True,
        )
        return [self._metadata_to_dict(r) for r in results.matches]

    def search_with_filter(
        self,
        query_vector: list[float],   # pre-embedded — matches Qdrant interface
        filter_by   : str,
        filter_val  : str,
        top_k       : int = 5,
    ) -> list[dict]:
        results = self.index.query(
            vector           = query_vector,
            top_k            = top_k,
            include_metadata = True,
            filter           = {filter_by: {"$eq": filter_val}},
        )
        return [self._metadata_to_dict(r) for r in results.matches]

    # ── STATS ────────────────────────────────────────────────

    def count(self) -> int:
        try:
            stats = self.index.describe_index_stats()
            return stats.total_vector_count or 0
        except Exception:
            return 0

    def list_sources(self) -> list[str]:
        """
        Pinecone doesn't support a native distinct-values query.
        We fetch a large batch of vectors and collect unique sources.
        For small-to-medium indexes this is fine.
        For very large indexes (100k+ vectors) consider maintaining
        a separate source registry.
        """
        try:
            # Fetch up to 10k random vectors with only source metadata
            results = self.index.query(
                vector           = [0.0] * self.embedding_dim,
                top_k            = 10_000,
                include_metadata = True,
                include_values   = False,
            )
            sources = {
                r.metadata.get("source", "")
                for r in results.matches
                if r.metadata.get("source")
            }
            return sorted(sources)
        except Exception:
            return []

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