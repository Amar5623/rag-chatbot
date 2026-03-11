# ingestion/chunker.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL


# ─────────────────────────────────────────
# BASE CHUNKER
# ─────────────────────────────────────────

class BaseChunker:
    """
    Abstract base class for all chunking strategies.
    Every chunker takes raw text and returns list of string chunks.
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy_name = "base"

    def chunk(self, text: str) -> list[str]:
        """Override in subclasses."""
        raise NotImplementedError("Subclasses must implement chunk()")

    def chunk_documents(self, docs: list[dict]) -> list[dict]:
        """
        Takes a list of loader chunks (dicts with 'content')
        and re-chunks each one, preserving metadata.
        """
        result = []
        for doc in docs:
            # Tables, images, csv, xlsx → don't re-chunk, keep as-is
            if doc["type"] in ["table", "image", "csv", "xlsx"]:
                result.append(doc)
                continue

            # Re-chunk text content
            sub_chunks = self.chunk(doc["content"])
            for i, sub in enumerate(sub_chunks):
                new_doc = doc.copy()
                new_doc["content"]      = sub
                new_doc["chunk_index"]  = i
                new_doc["strategy"]     = self.strategy_name
                result.append(new_doc)

        return result

    def get_stats(self, chunks: list[str]) -> dict:
        """Return stats about the chunks produced."""
        if not chunks:
            return {}
        lengths = [len(c) for c in chunks]
        return {
            "strategy"   : self.strategy_name,
            "total_chunks": len(chunks),
            "avg_length" : round(sum(lengths) / len(lengths)),
            "min_length" : min(lengths),
            "max_length" : max(lengths),
        }


# ─────────────────────────────────────────
# STRATEGY 1 — FIXED SIZE CHUNKING
# ─────────────────────────────────────────

class FixedSizeChunker(BaseChunker):
    """
    Splits text into fixed character-length chunks with overlap.
    Simplest strategy — good baseline.
    ✅ Fast  ❌ May split mid-sentence
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        super().__init__(chunk_size, chunk_overlap)
        self.strategy_name = "fixed_size"
        self.splitter = CharacterTextSplitter(
            chunk_size    = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            separator     = "\n"
        )

    def chunk(self, text: str) -> list[str]:
        chunks = self.splitter.split_text(text)
        return [c.strip() for c in chunks if c.strip()]


# ─────────────────────────────────────────
# STRATEGY 2 — RECURSIVE CHUNKING (BEST)
# ─────────────────────────────────────────

class RecursiveChunker(BaseChunker):
    """
    Tries to split on paragraphs → sentences → words → characters.
    Preserves natural language boundaries as much as possible.
    ✅ Best for most RAG use cases  ✅ Respects sentence boundaries
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        super().__init__(chunk_size, chunk_overlap)
        self.strategy_name = "recursive"
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size    = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            separators    = ["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

    def chunk(self, text: str) -> list[str]:
        chunks = self.splitter.split_text(text)
        return [c.strip() for c in chunks if c.strip()]


# ─────────────────────────────────────────
# STRATEGY 3 — SEMANTIC CHUNKING (ADVANCED)
# ─────────────────────────────────────────

class SemanticChunkerWrapper(BaseChunker):
    """
    Groups sentences by semantic similarity using embeddings.
    Splits only when the topic/meaning shifts significantly.
    ✅ Best quality  ❌ Slower (uses embeddings during chunking)
    """

    def __init__(self):
        super().__init__()
        self.strategy_name = "semantic"
        print("  [SEMANTIC] Loading embedding model for chunking...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.splitter = SemanticChunker(
            embeddings              = embeddings,
            breakpoint_threshold_type = "percentile"  # splits on biggest meaning shifts
        )

    def chunk(self, text: str) -> list[str]:
        chunks = self.splitter.split_text(text)
        return [c.strip() for c in chunks if c.strip()]


# ─────────────────────────────────────────
# CHUNKER FACTORY — pick strategy by name
# ─────────────────────────────────────────

class ChunkerFactory:
    """
    Factory class — returns the right chunker based on strategy name.
    Makes it easy to switch strategies from the Streamlit UI.
    """

    STRATEGIES = {
        "fixed"    : FixedSizeChunker,
        "recursive": RecursiveChunker,
        "semantic" : SemanticChunkerWrapper,
    }

    @staticmethod
    def get(strategy: str = "recursive", **kwargs) -> BaseChunker:
        """
        Usage:
            chunker = ChunkerFactory.get("recursive", chunk_size=500)
        """
        strategy = strategy.lower()
        if strategy not in ChunkerFactory.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from: {list(ChunkerFactory.STRATEGIES.keys())}"
            )

        cls = ChunkerFactory.STRATEGIES[strategy]

        # SemanticChunker takes no size args
        if strategy == "semantic":
            return cls()
        return cls(**kwargs)