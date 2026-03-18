from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).parent


class Settings(BaseSettings):
    # LLM
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    ollama_model: str = "llama3.2"
    max_turns: int = 20

    # Qdrant
    qdrant_path: str = str(BASE_DIR / "data" / "qdrant")
    qdrant_collection: str = "rag_docs"

    # Embeddings
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768
    hf_token: str = ""

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50
    child_chunk_size: int = 300
    child_chunk_overlap: int = 30
    parent_chunk_size: int = 1200
    parent_chunk_overlap: int = 100

    # Retrieval
    top_k: int = 20
    rrf_k: int = 60
    min_rerank_score: float = 0.1

    # Sessions
    session_max: int = 100
    session_ttl: int = 3600

    # Pinecone (optional, not wired by default)
    pinecone_api_key: str = ""
    pinecone_index: str = "rag-index"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# ── Legacy constants ───────────────────────────────────────────
QDRANT_PATH          = settings.qdrant_path
QDRANT_COLLECTION    = settings.qdrant_collection
EMBEDDING_DIM        = settings.embedding_dim
TOP_K                = settings.top_k
RRF_K                = settings.rrf_k
MIN_RERANK_SCORE     = settings.min_rerank_score
CHUNK_SIZE           = settings.chunk_size
CHUNK_OVERLAP        = settings.chunk_overlap
CHILD_CHUNK_SIZE     = settings.child_chunk_size
CHILD_CHUNK_OVERLAP  = settings.child_chunk_overlap
PARENT_CHUNK_SIZE    = settings.parent_chunk_size
PARENT_CHUNK_OVERLAP = settings.parent_chunk_overlap
GROQ_MODEL           = settings.groq_model
GROQ_API_KEY         = settings.groq_api_key        # ← ADD
MAX_TURNS            = settings.max_turns            # ← ADD
OLLAMA_MODEL         = settings.ollama_model
PINECONE_API_KEY     = settings.pinecone_api_key
PINECONE_INDEX       = settings.pinecone_index
PINECONE_CLOUD       = settings.pinecone_cloud
PINECONE_REGION      = settings.pinecone_region
EMBEDDING_MODEL      = settings.embedding_model
OLLAMA_EMBED_MODEL   = settings.ollama_model
HF_TOKEN             = settings.hf_token
BM25_PATH            = str(Path(settings.qdrant_path).parent / "bm25.pkl")
IMAGES_DIR           = str(BASE_DIR / "data" / "images")

# Ensure directories exist
Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)