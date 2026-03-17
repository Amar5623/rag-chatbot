# services.py
# App-wide singleton state and helper functions.
# api.py imports from here — keeping routes clean of business logic.

import os
import pickle
from pathlib import Path

from config import (
    BM25_PATH, PARENT_STORE_PATH, TOP_K, GROQ_API_KEY,
)
from embeddings.embedder        import EmbedderFactory
from vectorstore.qdrant_store   import QdrantVectorStore
from retrieval.bm25_store       import BM25Store
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker         import Reranker
from generation.groq_llm        import LLMFactory, ChatHistory
from chains.rag_chain           import RAGChain, RAG_SYSTEM_PROMPT
from ingestion.pdf_loader       import PDFLoader
from ingestion.csv_loader       import CSVLoader
from ingestion.xlsx_loader      import XLSXLoader
from ingestion.text_loader      import TextLoader
from ingestion.chunker          import HierarchicalChunker


# ─────────────────────────────────────────────────────────
# SINGLETON STATE
# ─────────────────────────────────────────────────────────

class AppState:
    """
    Single instance shared across all requests.
    All mutable fields are set during lifespan startup in api.py.
    """
    embedder     : object             = None
    store        : QdrantVectorStore  = None
    bm25         : BM25Store          = None
    parent_store : dict               = {}
    reranker     : Reranker           = None
    chain        : RAGChain           = None
    chunker      : HierarchicalChunker= None
    sessions     : dict               = {}   # {session_id: ChatHistory}


# Module-level singleton — imported by api.py and routes use it directly
state = AppState()


# ─────────────────────────────────────────────────────────
# STARTUP INITIALISER
# ─────────────────────────────────────────────────────────

def initialise() -> None:
    """
    Load all heavy resources once at startup.
    Called from the FastAPI lifespan context manager in api.py.
    """
    print("\n=== Initialising RAG API ===")
    state.embedder     = EmbedderFactory.get("huggingface")
    state.store        = QdrantVectorStore(embedder=state.embedder)
    state.bm25         = BM25Store(path=BM25_PATH)
    state.parent_store = load_parent_store()
    state.reranker     = Reranker()
    state.chunker      = HierarchicalChunker()
    rebuild_chain()
    print(
        f"\n=== Ready — {state.store.count()} vectors | "
        f"{len(state.bm25)} BM25 docs | "
        f"{len(state.parent_store)} parents ===\n"
    )


# ─────────────────────────────────────────────────────────
# PARENT STORE PERSISTENCE
# ─────────────────────────────────────────────────────────

def load_parent_store() -> dict:
    """Load parent store from disk. Returns empty dict if not found."""
    if Path(PARENT_STORE_PATH).exists():
        try:
            with open(PARENT_STORE_PATH, "rb") as f:
                data = pickle.load(f)
            print(f"  [SERVICES] Loaded {len(data)} parents from disk")
            return data
        except Exception as e:
            print(f"  [SERVICES] Parent store load failed: {e}")
    return {}


def save_parent_store() -> None:
    """Persist current parent store to disk."""
    try:
        with open(PARENT_STORE_PATH, "wb") as f:
            pickle.dump(state.parent_store, f)
    except Exception as e:
        print(f"  [SERVICES] Parent store save failed: {e}")


# ─────────────────────────────────────────────────────────
# CHAIN MANAGEMENT
# ─────────────────────────────────────────────────────────

def rebuild_chain() -> None:
    """
    Rebuild the shared RAGChain after parent_store changes.
    Must be called after every ingest so the retriever has fresh parents.
    """
    retriever = HybridRetriever(
        vector_store = state.store,
        embedder     = state.embedder,
        top_k        = TOP_K,
        parent_store = state.parent_store,
    )
    state.chain = RAGChain(
        llm           = LLMFactory.get("groq"),
        vector_store  = state.store,
        retriever     = retriever,
        reranker      = state.reranker,
        use_reranker  = True,
        retrieve_top_k= TOP_K,
        rerank_top_k  = 5,
        cite_sources  = True,
        parent_store  = state.parent_store,
    )


# ─────────────────────────────────────────────────────────
# SESSION MANAGEMENT
# ─────────────────────────────────────────────────────────

def get_or_create_session(session_id: str) -> ChatHistory:
    """Return existing ChatHistory for session, or create a fresh one."""
    if session_id not in state.sessions:
        state.sessions[session_id] = ChatHistory(
            system_prompt = RAG_SYSTEM_PROMPT,
            max_turns     = 8,
        )
    return state.sessions[session_id]


# ─────────────────────────────────────────────────────────
# INGESTION
# ─────────────────────────────────────────────────────────

# Supported file type → loader class
_LOADERS = {
    ".pdf" : PDFLoader,
    ".csv" : CSVLoader,
    ".xlsx": XLSXLoader,
    ".txt" : TextLoader,
}

SUPPORTED_EXTENSIONS = set(_LOADERS.keys())


def ingest_file(tmp_path: str, filename: str) -> tuple[list[dict], dict]:
    """
    Parse and chunk a single uploaded file.

    Args:
        tmp_path : path to the temp file on disk
        filename : original filename (used for source metadata)

    Returns:
        (children, parents) — children go to Qdrant+BM25, parents to disk
    """
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in _LOADERS:
        raise ValueError(f"Unsupported file type: {ext}")

    docs = _LOADERS[ext](tmp_path).load()
    for d in docs:
        d["source"] = filename

    children, parents = state.chunker.chunk_hierarchical(docs)
    for c in children:
        c["source"] = filename

    return children, parents


# ─────────────────────────────────────────────────────────
# MISC
# ─────────────────────────────────────────────────────────

def has_kb() -> bool:
    """True if the knowledge base contains at least one vector."""
    return state.store.count() > 0


def is_groq_configured() -> bool:
    return bool(GROQ_API_KEY)


__all__ = [
    "AppState", "state",
    "initialise",
    "load_parent_store", "save_parent_store",
    "rebuild_chain",
    "get_or_create_session",
    "ingest_file", "SUPPORTED_EXTENSIONS",
    "has_kb", "is_groq_configured",
]