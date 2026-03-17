# embeddings/embedder.py
#
# CHANGES vs original:
#   - HuggingFaceEmbedder now uses BAAI/bge-small-en-v1.5 (default)
#     Same 22MB / 384-dim as MiniLM, ~10% better on MTEB retrieval benchmarks
#   - BGE asymmetric retrieval: embed_text() (queries) uses a special prefix
#     embed_documents() (indexing) uses no prefix — this is intentional
#   - normalize_embeddings=True for cosine similarity stability
#   - OllamaEmbedder kept as-is for compatibility
#   - EmbedderFactory unchanged

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, OLLAMA_EMBED_MODEL, HF_TOKEN

# ── HuggingFace Hub login ─────────────────────────────────
if HF_TOKEN:
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN)
        os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
        print("  [EMBEDDER] HuggingFace Hub login successful.")
    except Exception as e:
        print(f"  [EMBEDDER] HuggingFace Hub login failed: {e}")

# BGE asymmetric retrieval prefix — used for QUERIES only, not documents
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ─────────────────────────────────────────────────────────
# BASE EMBEDDER
# ─────────────────────────────────────────────────────────

class BaseEmbedder:
    """
    Abstract base class for all embedding strategies.
    Every embedder must implement embed_text and embed_documents.

    NOTE ON BGE ASYMMETRIC RETRIEVAL:
    BGE models are trained with different prompts for queries vs documents.
    Subclasses that use BGE should apply _BGE_QUERY_PREFIX to embed_text()
    but NOT to embed_documents(). This is the key upgrade from MiniLM.
    """

    def __init__(self):
        self.model_name    = "base"
        self.embedding_dim = None

    def embed_text(self, text: str) -> list[float]:
        """Embed a single query string → vector."""
        raise NotImplementedError("Subclasses must implement embed_text()")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document strings → list of vectors."""
        raise NotImplementedError("Subclasses must implement embed_documents()")

    def get_info(self) -> dict:
        return {
            "model"        : self.model_name,
            "embedding_dim": self.embedding_dim,
        }


# ─────────────────────────────────────────────────────────
# STRATEGY 1 — HuggingFace (Local, Fast)
# Default: BAAI/bge-small-en-v1.5
# ─────────────────────────────────────────────────────────

class HuggingFaceEmbedder(BaseEmbedder):
    """
    Uses sentence-transformers locally.
    No API key needed. Runs fully offline after first download.

    Default model: BAAI/bge-small-en-v1.5
      ✅ 22MB (same as MiniLM)   ✅ 384 dims (same as MiniLM)
      ✅ ~10% better MTEB score  ✅ Asymmetric retrieval support

    BGE ASYMMETRIC RETRIEVAL:
      embed_text()      → applies query prefix (used for search queries)
      embed_documents() → no prefix          (used for indexing chunks)
      This gives better recall than symmetric embedding.
    """

    # Set to "" to disable prefix (e.g. for non-BGE models like MiniLM)
    QUERY_PREFIX = _BGE_QUERY_PREFIX

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        super().__init__()
        self.model_name = model_name

        # Disable BGE prefix if model isn't a BGE variant
        if "bge" not in model_name.lower():
            self.QUERY_PREFIX = ""

        print(f"  [EMBEDDER] Loading HuggingFace model: {model_name}")
        self._model        = SentenceTransformer(model_name)
        self.embedding_dim = self._model.get_sentence_embedding_dimension()
        print(f"  [EMBEDDER] Ready! Dimensions: {self.embedding_dim}")

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single search query.
        BGE models get a special query prefix for asymmetric retrieval.
        """
        query = self.QUERY_PREFIX + text.strip() if self.QUERY_PREFIX else text.strip()
        return self._model.encode(
            query,
            convert_to_numpy    = True,
            normalize_embeddings= True,
        ).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed document chunks — no query prefix.
        BGE asymmetric: documents are embedded as-is.
        """
        if not texts:
            return []
        print(f"  [EMBEDDER] Embedding {len(texts)} chunks...")
        vectors = self._model.encode(
            texts,
            batch_size          = 32,
            show_progress_bar   = len(texts) > 50,
            convert_to_numpy    = True,
            normalize_embeddings= True,
        )
        return vectors.tolist()


# ─────────────────────────────────────────────────────────
# STRATEGY 2 — Ollama (Local, nomic-embed)
# ─────────────────────────────────────────────────────────

class OllamaEmbedder(BaseEmbedder):
    """
    Uses Ollama's nomic-embed-text model locally.
    Requires Ollama to be running: `ollama serve`
    ✅ Higher quality   ✅ Free   ❌ Needs Ollama running
    Default model: nomic-embed-text → 768 dimensions
    """

    QUERY_PREFIX = ""   # Ollama models handle this internally

    def __init__(self, model_name: str = OLLAMA_EMBED_MODEL):
        super().__init__()
        self.model_name    = model_name
        self.embedding_dim = 768
        print(f"  [EMBEDDER] Using Ollama model: {model_name}")
        from langchain_ollama import OllamaEmbeddings
        self._model = OllamaEmbeddings(model=model_name)

    def embed_text(self, text: str) -> list[float]:
        return self._model.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        print(f"  [EMBEDDER] Embedding {len(texts)} chunks via Ollama...")
        return self._model.embed_documents(texts)


# ─────────────────────────────────────────────────────────
# EMBEDDER FACTORY
# ─────────────────────────────────────────────────────────

class EmbedderFactory:
    """
    Returns the right embedder based on provider name.

    Usage:
        embedder = EmbedderFactory.get("huggingface")   # BGE-small (default)
        embedder = EmbedderFactory.get("ollama")
    """

    PROVIDERS: dict[str, type[BaseEmbedder]] = {
        "huggingface": HuggingFaceEmbedder,
        "ollama"     : OllamaEmbedder,
    }

    @staticmethod
    def get(provider: str = "huggingface", **kwargs) -> BaseEmbedder:
        provider = provider.lower().strip()
        if provider not in EmbedderFactory.PROVIDERS:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Choose from: {list(EmbedderFactory.PROVIDERS.keys())}"
            )
        return EmbedderFactory.PROVIDERS[provider](**kwargs)

    @staticmethod
    def available_providers() -> list[str]:
        return list(EmbedderFactory.PROVIDERS.keys())


__all__ = [
    "BaseEmbedder",
    "HuggingFaceEmbedder",
    "OllamaEmbedder",
    "EmbedderFactory",
]