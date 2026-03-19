# services/rag_service.py
#
# CHANGES:
#   - _build_vector_store()  — VECTOR_STORE=qdrant|pinecone
#   - _build_embedder()      — EMBEDDER=huggingface|ollama
#   - _build_llm()           — LLM_PROVIDER=groq|ollama
#   - All three read from settings — pure .env switches, zero code changes
#   - _build_chain() uses _build_llm() instead of hardcoded "groq"

import threading
from pathlib import Path

from cachetools import TTLCache

from config import settings
from embeddings.embedder        import EmbedderFactory
from generation.groq_llm        import LLMFactory
from retrieval.bm25_store       import BM25Store
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.naive_retriever  import NaiveRetriever
from retrieval.reranker         import Reranker
from vectorstore.qdrant_store   import QdrantVectorStore
from chains.rag_chain           import RAGChain


# ── Singletons ────────────────────────────────────────────────
_embedder:     object = None
_reranker:     object = None
_vector_store: object = None
_bm25_store:   object = None

# ── Session cache ──────────────────────────────────────────────
_sessions: TTLCache = TTLCache(maxsize=settings.session_max, ttl=settings.session_ttl)
_lock = threading.Lock()

# ── Background task registry ──────────────────────────────────
_tasks: dict = {}


# ── Factories ─────────────────────────────────────────────────

def _build_vector_store(embedder):
    """
    VECTOR_STORE=qdrant   → QdrantVectorStore  (default, local, no API key)
    VECTOR_STORE=pinecone → PineconeVectorStore (cloud, needs PINECONE_API_KEY)
    """
    provider = settings.vector_store.lower().strip()

    if provider == "pinecone":
        if not settings.pinecone_api_key:
            raise RuntimeError(
                "VECTOR_STORE=pinecone but PINECONE_API_KEY is not set in .env"
            )
        from vectorstore.pinecone_store import PineconeVectorStore
        print(f"  [SERVICE] Vector store : Pinecone cloud (index='{settings.pinecone_index}')")
        return PineconeVectorStore(embedder=embedder)

    print(f"  [SERVICE] Vector store : Local Qdrant (path='{settings.qdrant_path}')")
    return QdrantVectorStore(embedder=embedder)


def _build_embedder():
    """
    EMBEDDER=huggingface → HuggingFaceEmbedder (default, local BGE model)
    EMBEDDER=ollama      → OllamaEmbedder      (local Ollama, needs ollama serve)
    """
    provider = settings.embedder.lower().strip()
    print(f"  [SERVICE] Embedder     : {provider}")
    return EmbedderFactory.get(provider)


def _build_llm():
    """
    LLM_PROVIDER=groq   → GroqLLM   (default, cloud, needs GROQ_API_KEY)
    LLM_PROVIDER=ollama → OllamaLLM (local, needs ollama serve)
    """
    provider = settings.llm_provider.lower().strip()

    if provider == "ollama":
        # Import here so Ollama registers itself into the factory
        import generation.ollama_llm  # noqa: F401
        print(f"  [SERVICE] LLM          : Ollama local (model='{settings.ollama_model}')")
        return LLMFactory.get("ollama")

    if provider == "groq" and not settings.groq_api_key:
        raise RuntimeError(
            "LLM_PROVIDER=groq but GROQ_API_KEY is not set in .env"
        )

    print(f"  [SERVICE] LLM          : Groq cloud (model='{settings.groq_model}')")
    return LLMFactory.get("groq")


def _build_chunker():
    """
    CHUNKER=hierarchical (default) → HierarchicalChunker
    CHUNKER=recursive              → RecursiveChunker
    CHUNKER=fixed                  → FixedSizeChunker
    """
    from ingestion.chunker import ChunkerFactory
    strategy = settings.chunker.lower().strip()
    print(f"  [SERVICE] Chunker      : {strategy}")
    return ChunkerFactory.get(strategy)


# ── Startup ───────────────────────────────────────────────────

async def startup() -> None:
    """Called once at FastAPI startup. Initialise all singletons."""
    global _embedder, _reranker, _vector_store, _bm25_store

    data_dir = Path(settings.qdrant_path).parent
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n  [SERVICE] Initialising singletons...")
    _embedder     = _build_embedder()
    _reranker     = Reranker()
    _vector_store = _build_vector_store(_embedder)
    _bm25_store   = BM25Store(path=str(data_dir / "bm25.pkl"))
    print("  [SERVICE] ✅ All singletons ready\n")


# ── Session management ────────────────────────────────────────

def get_or_create_session(
    session_id: str,
    retriever_choice: str = "hybrid",
) -> RAGChain:
    with _lock:
        if session_id not in _sessions:
            _sessions[session_id] = _build_chain(retriever_choice)
        return _sessions[session_id]


def _build_chain(retriever_choice: str = "hybrid") -> RAGChain:
    # LLM is built fresh per session so each user has isolated history
    llm = _build_llm()

    if retriever_choice == "hybrid":
        retriever = HybridRetriever(
            vector_store = _vector_store,
            embedder     = _embedder,
            top_k        = settings.top_k,
        )
        if _bm25_store and _bm25_store._chunks:
            retriever.index_chunks(_bm25_store._chunks)
    else:
        retriever = NaiveRetriever(
            vector_store = _vector_store,
            embedder     = _embedder,
            top_k        = settings.top_k,
        )

    return RAGChain(
        llm            = llm,
        vector_store   = _vector_store,
        retriever      = retriever,
        reranker       = _reranker,
        use_reranker   = True,
        retrieve_top_k = settings.top_k,
        rerank_top_k   = 5,
        cite_sources   = True,
    )


def clear_session(session_id: str) -> None:
    with _lock:
        _sessions.pop(session_id, None)


# ── Per-file deletion ─────────────────────────────────────────

def delete_file_from_stores(filename: str) -> dict:
    vectors_deleted = _vector_store.delete_by_source(filename)
    bm25_deleted    = _bm25_store.delete_by_source(filename)

    with _lock:
        for chain in _sessions.values():
            if hasattr(chain.retriever, "bm25"):
                chain.retriever.bm25 = _bm25_store

    return {
        "vectors_deleted": vectors_deleted,
        "bm25_deleted"   : bm25_deleted,
    }


# ── Chunker accessor (used by ingest.py) ──────────────────────

def get_chunker():
    """Returns a fresh chunker instance based on current CHUNKER setting."""
    return _build_chunker()


# ── Singleton accessors ───────────────────────────────────────

def get_vector_store():
    return _vector_store

def get_bm25_store() -> BM25Store:
    return _bm25_store

def get_parent_store():
    return None

def get_embedder():
    return _embedder


# ── Task registry ─────────────────────────────────────────────

def set_task(task_id, status, progress=0, message="", result=None):
    _tasks[task_id] = {
        "status":   status,
        "progress": progress,
        "message":  message,
        "result":   result or {},
    }

def get_task(task_id: str) -> dict | None:
    return _tasks.get(task_id)