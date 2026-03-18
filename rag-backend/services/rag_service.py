# services/rag_service.py
# All shared singletons + session lifecycle.
# One instance of each heavy object (embedder, store, BM25, parents, reranker).
# Sessions are RAGChain instances cached with TTL eviction.

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
from utils.parent_store         import ParentStore
from chains.rag_chain           import RAGChain


# ── Singletons ────────────────────────────────────────────────
_embedder:     object = None
_reranker:     object = None
_vector_store: object = None
_bm25_store:   object = None
_parent_store: object = None

# ── Session cache: max 100 sessions, evicted after 1 hour idle ─
_sessions: TTLCache = TTLCache(maxsize=settings.session_max, ttl=settings.session_ttl)
_lock = threading.Lock()

# ── Background task registry ──────────────────────────────────
_tasks: dict = {}


async def startup() -> None:
    """Called once at FastAPI startup. Initialise all singletons."""
    global _embedder, _reranker, _vector_store, _bm25_store, _parent_store

    data_dir = Path(settings.qdrant_path).parent
    data_dir.mkdir(parents=True, exist_ok=True)

    _embedder     = EmbedderFactory.get("huggingface")
    _reranker     = Reranker()
    _vector_store = QdrantVectorStore(embedder=_embedder)
    _bm25_store   = BM25Store(path=str(data_dir / "bm25.pkl"))
    _parent_store = ParentStore(path=str(data_dir / "parents.db"))

    print("  [SERVICE] All singletons ready")


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
    llm = LLMFactory.get("groq")

    if retriever_choice == "hybrid":
        retriever = HybridRetriever(
            vector_store = _vector_store,
            embedder     = _embedder,
            top_k        = settings.top_k,
            parent_store = _parent_store,
        )
        # Reload BM25 index into the retriever from the persisted store
        if _bm25_store and _bm25_store._chunks:
            retriever.index_chunks(_bm25_store._chunks)
    else:
        retriever = NaiveRetriever(
            vector_store = _vector_store,
            embedder     = _embedder,
            top_k        = settings.top_k,
            parent_store = _parent_store,
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


# ── Singleton accessors ───────────────────────────────────────

def get_vector_store() -> QdrantVectorStore:
    return _vector_store

def get_bm25_store() -> BM25Store:
    return _bm25_store

def get_parent_store() -> ParentStore:
    return _parent_store

def get_embedder():
    return _embedder


# ── Task registry ─────────────────────────────────────────────

def set_task(
    task_id: str,
    status: str,
    progress: int = 0,
    message: str = "",
    result: dict = None,
) -> None:
    _tasks[task_id] = {
        "status":   status,
        "progress": progress,
        "message":  message,
        "result":   result or {},
    }

def get_task(task_id: str) -> dict | None:
    return _tasks.get(task_id)