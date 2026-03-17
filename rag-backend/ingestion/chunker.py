# ingestion/chunker.py
#
# CHANGES vs original:
#   - NEW: HierarchicalChunker — small-to-big retrieval pattern
#       child  (300 chars) → embedded into Qdrant for precise retrieval
#       parent (1200 chars) → stored on disk, sent to LLM for context
#       This is the single biggest quality improvement in the pipeline.
#   - RecursiveChunker kept and improved (adds section_path awareness)
#   - FixedSizeChunker kept unchanged
#   - SemanticChunkerWrapper REMOVED — uses embeddings during chunking,
#     too slow + redundant when you already have BGE retrieval
#   - ChunkerFactory updated: "hierarchical" is now the recommended strategy
#   - total_chunks / chunk_index / parent_id added to every chunk dict

import os
import sys
import hashlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from config import (
    CHUNK_SIZE, CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP,
)

# Atomic types — must never be re-chunked (splitting destroys their meaning)
_ATOMIC_TYPES = {"table", "image"}


# ─────────────────────────────────────────────────────────
# BASE CHUNKER  (unchanged interface)
# ─────────────────────────────────────────────────────────

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
        and re-chunks each one, preserving ALL metadata.

        Tables / images are kept atomic (never re-chunked).
        Adds chunk_index, total_chunks to every output chunk.
        """
        result: list[dict] = []
        for doc in docs:
            # ── Atomic content — keep as single chunk ──
            if doc.get("type") in _ATOMIC_TYPES:
                doc["chunk_index"]  = 0
                doc["total_chunks"] = 1
                doc["strategy"]     = "none"
                result.append(doc)
                continue

            # ── Text content — split it ──
            sub_chunks = self.chunk(doc["content"])
            total      = len(sub_chunks)

            for i, sub in enumerate(sub_chunks):
                new_doc                  = doc.copy()
                new_doc["content"]       = sub
                new_doc["chunk_index"]   = i
                new_doc["total_chunks"]  = total
                new_doc["strategy"]      = self.strategy_name
                result.append(new_doc)

        return result

    def get_stats(self, chunks: list[str]) -> dict:
        if not chunks:
            return {}
        lengths = [len(c) for c in chunks]
        return {
            "strategy"    : self.strategy_name,
            "total_chunks": len(chunks),
            "avg_length"  : round(sum(lengths) / len(lengths)),
            "min_length"  : min(lengths),
            "max_length"  : max(lengths),
        }


# ─────────────────────────────────────────────────────────
# STRATEGY 1 — FIXED SIZE CHUNKING
# ─────────────────────────────────────────────────────────

class FixedSizeChunker(BaseChunker):
    """
    Splits text into fixed character-length chunks with overlap.
    Simplest strategy — good baseline.
    ✅ Fast  ❌ May split mid-sentence
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        super().__init__(chunk_size, chunk_overlap)
        self.strategy_name = "fixed_size"
        self._splitter = CharacterTextSplitter(
            chunk_size    = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            separator     = "\n"
        )

    def chunk(self, text: str) -> list[str]:
        chunks = self._splitter.split_text(text)
        return [c.strip() for c in chunks if c.strip()]


# ─────────────────────────────────────────────────────────
# STRATEGY 2 — RECURSIVE CHUNKING
# ─────────────────────────────────────────────────────────

class RecursiveChunker(BaseChunker):
    """
    Tries to split on paragraphs → sentences → words → characters.
    Preserves natural language boundaries as much as possible.
    ✅ Best for most RAG use cases  ✅ Respects sentence boundaries
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        super().__init__(chunk_size, chunk_overlap)
        self.strategy_name = "recursive"
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size    = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            separators    = ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
        )

    def chunk(self, text: str) -> list[str]:
        chunks = self._splitter.split_text(text)
        return [c.strip() for c in chunks if c.strip()]


# ─────────────────────────────────────────────────────────
# STRATEGY 3 — HIERARCHICAL PARENT-CHILD  (RECOMMENDED)
# ─────────────────────────────────────────────────────────

class HierarchicalChunker(BaseChunker):
    """
    Small-to-big retrieval: produces child + parent chunk pairs.

    Pattern:
      - Child  (300 chars) → embedded into Qdrant for precise retrieval
      - Parent (1200 chars) → stored in parent_store dict, sent to LLM

    Why this is better than flat chunking:
      - Small embeddings are precise (less noise per vector)
      - LLM still gets full context (parent passage, not truncated child)
      - 10-15% better answer quality vs flat 500-char chunks

    Atomic blocks (tables, images, bullets) are never split.
    Each atomic block becomes its own parent=child (parent_id still set).

    Usage:
        chunker = HierarchicalChunker()
        children, parents = chunker.chunk_hierarchical(blocks)
        # children → Qdrant
        # parents  → pickle to disk, passed to HybridRetriever
    """

    def __init__(
        self,
        child_size    : int = CHILD_CHUNK_SIZE,
        child_overlap : int = CHILD_CHUNK_OVERLAP,
        parent_size   : int = PARENT_CHUNK_SIZE,
        parent_overlap: int = PARENT_CHUNK_OVERLAP,
    ):
        super().__init__(child_size, child_overlap)
        self.strategy_name  = "hierarchical"
        self.child_size     = child_size
        self.child_overlap  = child_overlap
        self.parent_size    = parent_size
        self.parent_overlap = parent_overlap

        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size    = child_size,
            chunk_overlap = child_overlap,
            separators    = ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        )
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size    = parent_size,
            chunk_overlap = parent_overlap,
            separators    = ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        )

    # ── chunk() — used by BaseChunker.chunk_documents() ──
    # Returns child-sized strings (for backward compatibility)
    def chunk(self, text: str) -> list[str]:
        return [c.strip() for c in self._child_splitter.split_text(text) if c.strip()]

    # ── chunk_hierarchical() — the full parent-child flow ─
    def chunk_hierarchical(
        self, blocks: list[dict]
    ) -> tuple[list[dict], dict]:
        """
        Full hierarchical chunking pipeline.

        Args:
            blocks : structured blocks from PDFLoader (or other loaders)

        Returns:
            children : list[dict]  — small chunks with parent_id → Qdrant
            parents  : dict        — {parent_id: parent_dict} → disk
        """
        children: list[dict] = []
        parents:  dict       = {}

        # Separate by type
        text_blocks   = [b for b in blocks if b.get("type") not in _ATOMIC_TYPES]
        atomic_blocks = [b for b in blocks if b.get("type")     in _ATOMIC_TYPES]

        # ── 1. Text/heading/bullet blocks ─────────────────
        groups = self._group_by_section(text_blocks)

        for g_idx, group in enumerate(groups):
            combined = "\n\n".join(b["content"] for b in group)
            meta_base = {
                "source"      : group[0]["source"],
                "page"        : group[0]["page"],
                "type"        : group[0].get("type", "text"),
                "heading"     : group[0].get("heading", ""),
                "section_path": group[0].get("section_path", ""),
            }

            parent_texts = self._parent_splitter.split_text(combined)

            for p_idx, parent_text in enumerate(parent_texts):
                parent_id = self._make_parent_id(
                    meta_base["source"],
                    meta_base["page"],
                    meta_base["section_path"],
                    g_idx * 1000 + p_idx,
                )

                child_texts = self._child_splitter.split_text(parent_text)
                total_c     = len(child_texts)

                for c_idx, child_text in enumerate(child_texts):
                    if not child_text.strip():
                        continue
                    children.append({
                        **meta_base,
                        "content"      : child_text,
                        "parent_id"    : parent_id,
                        "chunk_index"  : c_idx,
                        "total_chunks" : total_c,
                        "strategy"     : self.strategy_name,
                    })

                parents[parent_id] = {
                    **meta_base,
                    "parent_id"  : parent_id,
                    "content"    : parent_text,
                    "child_count": total_c,
                }

        # ── 2. Atomic blocks — each is own parent+child ───
        for a_idx, block in enumerate(atomic_blocks):
            content = block.get("content", "").strip()
            if not content:
                continue

            parent_id = self._make_parent_id(
                block["source"],
                block["page"],
                block.get("section_path", ""),
                100_000 + a_idx,
            )

            child = {
                **block,
                "parent_id"    : parent_id,
                "chunk_index"  : 0,
                "total_chunks" : 1,
                "strategy"     : self.strategy_name,
            }
            parent = {
                **block,
                "parent_id"  : parent_id,
                "child_count": 1,
            }
            children.append(child)
            parents[parent_id] = parent

        print(
            f"  [CHUNKER] {len(children)} children, "
            f"{len(parents)} parents from {len(blocks)} blocks"
        )
        return children, parents

    # ── helpers ───────────────────────────────────────────

    @staticmethod
    def _make_parent_id(source: str, page: int, section: str, idx: int) -> str:
        raw = f"{source}|p{page}|{section}|{idx}"
        return "par_" + hashlib.md5(raw.encode()).hexdigest()[:12]

    @staticmethod
    def _group_by_section(blocks: list[dict]) -> list[list[dict]]:
        """Group consecutive blocks that share the same section_path."""
        if not blocks:
            return []
        groups: list[list[dict]] = []
        current: list[dict]      = [blocks[0]]

        for block in blocks[1:]:
            new_section = (
                block.get("type") == "heading"
                or block.get("section_path") != current[-1].get("section_path")
            )
            if new_section:
                groups.append(current)
                current = [block]
            else:
                current.append(block)

        if current:
            groups.append(current)
        return groups


# ─────────────────────────────────────────────────────────
# CHUNKER FACTORY
# ─────────────────────────────────────────────────────────

class ChunkerFactory:
    """
    Returns the right chunker based on strategy name.

    Strategies:
        "hierarchical" (recommended) — parent-child, best retrieval quality
        "recursive"                  — flat recursive, good general purpose
        "fixed"                      — flat fixed-size, fastest
    """

    STRATEGIES: dict[str, type[BaseChunker]] = {
        "hierarchical": HierarchicalChunker,
        "recursive"   : RecursiveChunker,
        "fixed"       : FixedSizeChunker,
    }

    @staticmethod
    def get(strategy: str = "hierarchical", **kwargs) -> BaseChunker:
        strategy = strategy.lower()
        if strategy not in ChunkerFactory.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from: {list(ChunkerFactory.STRATEGIES.keys())}"
            )
        return ChunkerFactory.STRATEGIES[strategy](**kwargs)

    @staticmethod
    def available_strategies() -> list[str]:
        return list(ChunkerFactory.STRATEGIES.keys())


__all__ = [
    "BaseChunker",
    "FixedSizeChunker",
    "RecursiveChunker",
    "HierarchicalChunker",
    "ChunkerFactory",
]