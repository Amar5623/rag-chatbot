# embeddings/embedder.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaEmbeddings
from config import EMBEDDING_MODEL, OLLAMA_EMBED_MODEL, HF_TOKEN

# ── HuggingFace Hub login (only if token is provided) ──────────────────────────
if HF_TOKEN:
    try:
        from huggingface_hub import login          # ← was missing, caused NameError
        login(token=HF_TOKEN)
        os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
        print("  [EMBEDDER] HuggingFace Hub login successful.")
    except Exception as e:
        print(f"  [EMBEDDER] HuggingFace Hub login failed: {e}")


# ─────────────────────────────────────────
# BASE EMBEDDER
# ─────────────────────────────────────────

class BaseEmbedder:
    """
    Abstract base class for all embedding strategies.
    Every embedder must implement embed_text and embed_documents.
    """

    def __init__(self):
        self.model_name    = "base"
        self.embedding_dim = None

    def embed_text(self, text: str) -> list[float]:
        """Embed a single string → vector."""
        raise NotImplementedError("Subclasses must implement embed_text()")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings → list of vectors."""
        raise NotImplementedError("Subclasses must implement embed_documents()")

    def get_info(self) -> dict:
        return {
            "model"        : self.model_name,
            "embedding_dim": self.embedding_dim,
        }


# ─────────────────────────────────────────
# STRATEGY 1 — HuggingFace (Local, Fast)
# ─────────────────────────────────────────

class HuggingFaceEmbedder(BaseEmbedder):
    """
    Uses sentence-transformers locally.
    No API key needed. Runs fully offline.
    ✅ Fast  ✅ Free  ✅ No internet needed
    Default model: all-MiniLM-L6-v2 → 384 dimensions
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        super().__init__()
        self.model_name = model_name
        print(f"  [EMBEDDER] Loading HuggingFace model: {model_name}")
        self.model         = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"  [EMBEDDER] Ready! Dimensions: {self.embedding_dim}")

    def embed_text(self, text: str) -> list[float]:
        """Embed a single query or document."""
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        print(f"  [EMBEDDER] Embedding {len(texts)} chunks...")
        vectors = self.model.encode(
            texts,
            batch_size        = 32,
            show_progress_bar = True,
            convert_to_numpy  = True,
        )
        return vectors.tolist()


# ─────────────────────────────────────────
# STRATEGY 2 — Ollama (Local, nomic-embed)
# ─────────────────────────────────────────

class OllamaEmbedder(BaseEmbedder):
    """
    Uses Ollama's nomic-embed-text model locally.
    Requires Ollama to be running: `ollama serve`
    ✅ Higher quality than MiniLM  ✅ Free  ❌ Needs Ollama running
    Default model: nomic-embed-text → 768 dimensions
    """

    def __init__(self, model_name: str = OLLAMA_EMBED_MODEL):
        super().__init__()
        self.model_name    = model_name
        self.embedding_dim = 768
        print(f"  [EMBEDDER] Using Ollama model: {model_name}")
        self.model = OllamaEmbeddings(model=model_name)

    def embed_text(self, text: str) -> list[float]:
        return self.model.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        print(f"  [EMBEDDER] Embedding {len(texts)} chunks via Ollama...")
        return self.model.embed_documents(texts)


# ─────────────────────────────────────────
# EMBEDDER FACTORY
# ─────────────────────────────────────────

class EmbedderFactory:
    """
    Returns the right embedder based on provider name.

    Usage:
        embedder = EmbedderFactory.get("huggingface")
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