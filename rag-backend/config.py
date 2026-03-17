import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
HF_TOKEN         = os.getenv("HF_TOKEN")

# ── LLM ───────────────────────────────────────────────────
GROQ_MODEL       = "llama-3.3-70b-versatile"
OLLAMA_MODEL     = "llama3.2"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# ── Embedding ─────────────────────────────────────────────
# BGE-small-en-v1.5: same 22MB / 384-dim as MiniLM, ~10% better MTEB retrieval
# BGE uses asymmetric prompting: different prefix for queries vs documents
EMBEDDING_MODEL  = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM    = 384

# ── Qdrant ────────────────────────────────────────────────
QDRANT_PATH       = "./qdrant_local_db"
QDRANT_COLLECTION = "rag_docs"

# ── Pinecone (kept for compatibility, not recommended) ────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = "rag-chatbot"
PINECONE_CLOUD   = "aws"
PINECONE_REGION  = "us-east-1"

# ── Hierarchical Chunking ─────────────────────────────────
# child  → embedded into Qdrant  (precise retrieval)
# parent → stored on disk        (rich context for LLM)
CHILD_CHUNK_SIZE    = 300
CHILD_CHUNK_OVERLAP = 50
PARENT_CHUNK_SIZE   = 1200
PARENT_CHUNK_OVERLAP= 100

# Legacy flat chunking (kept for FixedSizeChunker / RecursiveChunker)
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100

# ── Retrieval ─────────────────────────────────────────────
TOP_K        = 20   # fetch candidates
RERANK_TOP_K = 5    # keep after rerank
RRF_K        = 60   # RRF constant (paper default)

# ── BM25 persistence ──────────────────────────────────────
BM25_PATH          = "./bm25_index.pkl"
PARENT_STORE_PATH  = "./parent_store.pkl"

# ── Memory ────────────────────────────────────────────────
MAX_TURNS = 8   # sliding window — 8 turns = 16 messages

# ── Retrieval quality gate ────────────────────────────────
# Cross-encoder rerank score below this → "not in documents" fallback
MIN_RERANK_SCORE = -5.0

# ── PDF ───────────────────────────────────────────────────
IMAGES_DIR            = "./extracted_images"
# No Tesseract — removed entirely. Page context used for image semantics.

# ── Dev ───────────────────────────────────────────────────
RESET_ON_START = False