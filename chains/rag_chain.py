# chains/rag_chain.py
# IMPROVED:
#   - ChainResponse now carries image_paths from retrieved image chunks
#   - get_images() method lets app.py display images alongside answers
#   - heading metadata shown in citations for better source context

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.naive_retriever  import NaiveRetriever, RetrievalResult
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker         import Reranker
from generation.groq_llm        import BaseLLM, ChatHistory, LLMFactory
from vectorstore.qdrant_store   import QdrantVectorStore, BaseVectorStore
from embeddings.embedder        import EmbedderFactory
from config import TOP_K


# ─────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions \
based on provided context documents.

Rules:
- Answer ONLY from the provided context. Do not use outside knowledge.
- If the context does not contain enough information, say: \
"I don't have enough information in the provided documents to answer that."
- Be concise and precise.
- When relevant, mention which source your answer comes from.
- If the context contains image descriptions or OCR text, reference them naturally."""

RAG_USER_TEMPLATE = """\
Context:
{context}

Question: {question}"""

RAG_USER_TEMPLATE_WITH_CITATIONS = """\
Context:
{context}

Question: {question}

After your answer, list the sources you used on a new line starting with "Sources:"."""


# ─────────────────────────────────────────
# CHAIN RESPONSE
# ─────────────────────────────────────────

class ChainResponse:
    """
    Wraps the full output of a RAG chain call.

    IMPROVED vs original:
      - get_images() returns list of image paths from retrieved image chunks
        so app.py can display the actual images in the response
      - get_citations() now includes heading for richer source display
    """

    def __init__(
        self,
        answer    : str,
        retrieval : RetrievalResult,
        question  : str,
        model     : str,
        usage     : dict = None,
    ):
        self.answer    = answer
        self.retrieval = retrieval
        self.question  = question
        self.model     = model
        self.usage     = usage or {}

    def get_answer(self) -> str:
        return self.answer

    def get_citations(self) -> list[dict]:
        """
        Returns citation dicts with source, page, heading, type.
        Heading lets the UI show which section the answer came from.
        """
        citations = []
        for chunk in self.retrieval.get_chunks():
            citations.append({
                "source" : chunk.get("source", "unknown"),
                "page"   : chunk.get("page", "?"),
                "heading": chunk.get("heading", ""),
                "type"   : chunk.get("type", "text"),
            })
        return citations

    def get_images(self) -> list[str]:
        """
        NEW: Return absolute paths of any images retrieved as context.
        app.py uses this to display images alongside the LLM answer.

        Only returns paths that actually exist on disk.
        """
        paths = []
        for chunk in self.retrieval.get_chunks():
            if chunk.get("type") == "image":
                img_path = chunk.get("image_path", "")
                if img_path and os.path.exists(img_path):
                    paths.append(img_path)
        return paths

    def has_images(self) -> bool:
        return len(self.get_images()) > 0

    def format_citations(self) -> str:
        """Plain-text formatted citations string."""
        lines = []
        for c in self.get_citations():
            heading_part = f" [{c['heading']}]" if c.get("heading") else ""
            lines.append(f"  • {c['source']} (p.{c['page']}){heading_part}")
        return "Sources:\n" + "\n".join(lines) if lines else ""

    def get_chunks(self) -> list[dict]:
        return self.retrieval.get_chunks()

    def get_context(self) -> str:
        return self.retrieval.to_context_string()

    def __str__(self) -> str:
        parts = [self.answer]
        cites = self.format_citations()
        if cites:
            parts.append("\n" + cites)
        return "\n".join(parts)

    def __repr__(self) -> str:
        return (
            f"ChainResponse("
            f"model={self.model}, "
            f"chunks={len(self.retrieval)}, "
            f"images={len(self.get_images())}, "
            f"tokens={self.usage.get('total_tokens', '?')})"
        )


# ─────────────────────────────────────────
# RAG CHAIN
# ─────────────────────────────────────────

class RAGChain:
    """
    Full RAG pipeline: Retrieve → Rerank → Generate.

    IMPROVED vs original:
      - ChainResponse.get_images() exposes image paths to the UI
      - Citations include heading metadata for section-level attribution
      - Memory: sliding window (10 turns) + entity memory (whole session)
    """

    def __init__(
        self,
        llm           : BaseLLM        = None,
        vector_store  : BaseVectorStore = None,
        retriever                       = None,
        reranker      : Reranker        = None,
        use_reranker  : bool            = True,
        retrieve_top_k: int             = 20,
        rerank_top_k  : int             = 5,
        cite_sources  : bool            = True,
        llm_provider  : str             = "groq",
    ):
        # ── LLM ──────────────────────────────
        self.llm = llm or LLMFactory.get(llm_provider)
        self.llm.set_system_prompt(RAG_SYSTEM_PROMPT)

        # ── Vector store ──────────────────────
        embedder   = EmbedderFactory.get("huggingface")
        self.store = vector_store or QdrantVectorStore(embedder=embedder)

        # ── Retriever ─────────────────────────
        if retriever is not None:
            self.retriever = retriever
        else:
            self.retriever = HybridRetriever(
                vector_store = self.store,
                embedder     = embedder,
                top_k        = retrieve_top_k,
            )

        # ── Reranker ──────────────────────────
        self.use_reranker = use_reranker
        self.reranker     = reranker or (Reranker() if use_reranker else None)
        self.rerank_top_k = rerank_top_k

        # ── Settings ──────────────────────────
        self.retrieve_top_k = retrieve_top_k
        self.cite_sources   = cite_sources

        # ── Memory ────────────────────────────
        self.history = self.llm.history   # sliding window + entity memory

        print(f"\n  [RAG CHAIN] ✅ Ready!")
        print(f"  [RAG CHAIN] LLM       : {self.llm.model_name}")
        print(f"  [RAG CHAIN] Retriever : {type(self.retriever).__name__}")
        print(f"  [RAG CHAIN] Reranker  : {'✅' if use_reranker else '❌'}")
        print(f"  [RAG CHAIN] Citations : {'✅' if cite_sources else '❌'}")

    # ── INDEXING ─────────────────────────────

    def index_documents(self, chunks: list[dict]) -> None:
        self.store.add_documents(chunks)
        if hasattr(self.retriever, "index_chunks"):
            self.retriever.index_chunks(chunks)
        print(f"  [RAG CHAIN] Indexed {len(chunks)} chunks.")

    # ── RETRIEVAL ────────────────────────────

    def _retrieve(self, question: str) -> RetrievalResult:
        retrieval = self.retriever.retrieve(question)
        if self.use_reranker and self.reranker and len(retrieval) > 0:
            retrieval = self.reranker.rerank(
                query     = question,
                retrieval = retrieval,
                top_k     = self.rerank_top_k,
            )
        return retrieval

    # ── PROMPT BUILDING ──────────────────────

    def _build_prompt(self, question: str, context: str) -> str:
        template = (
            RAG_USER_TEMPLATE_WITH_CITATIONS
            if self.cite_sources
            else RAG_USER_TEMPLATE
        )
        return template.format(context=context, question=question)

    # ── ASK (blocking) ───────────────────────

    def ask(self, question: str, top_k: int = None, cite_sources: bool = None) -> ChainResponse:
        retrieval = self._retrieve(question)
        context   = retrieval.to_context_string()

        if not context.strip():
            answer = "I don't have enough information in the provided documents to answer that."
            return ChainResponse(
                answer=answer, retrieval=retrieval,
                question=question, model=self.llm.model_name,
            )

        prompt = self._build_prompt(question, context)
        result = self.llm.generate(prompt=prompt, history=self.history)

        return ChainResponse(
            answer    = result["content"],
            retrieval = retrieval,
            question  = question,
            model     = result["model"],
            usage     = result["usage"],
        )

    # ── STREAM (generator) ───────────────────

    def stream(self, question: str, cite_sources: bool = None):
        """
        Streaming RAG pipeline.

        Yields:
            str           — text tokens as they stream
            ChainResponse — final item with full metadata + image paths
        """
        retrieval = self._retrieve(question)
        context   = retrieval.to_context_string()

        if not context.strip():
            answer = "I don't have enough information in the provided documents to answer that."
            yield answer
            yield ChainResponse(
                answer=answer, retrieval=retrieval,
                question=question, model=self.llm.model_name,
            )
            return

        prompt     = self._build_prompt(question, context)
        full_reply = []
        usage      = {}

        for chunk in self.llm.stream(prompt=prompt, history=self.history):
            if isinstance(chunk, str):
                full_reply.append(chunk)
                yield chunk
            else:
                usage = chunk.get("usage", {})

        yield ChainResponse(
            answer    = "".join(full_reply),
            retrieval = retrieval,
            question  = question,
            model     = self.llm.model_name,
            usage     = usage,
        )

    # ── MEMORY ───────────────────────────────

    def reset_memory(self) -> None:
        """Clear both sliding window AND entity memory."""
        self.llm.reset_history()
        print("  [RAG CHAIN] Memory fully cleared.")

    def get_history(self) -> list[dict]:
        return self.history.to_messages()

    # ── INFO ─────────────────────────────────

    def get_info(self) -> dict:
        return {
            "llm"            : self.llm.get_info(),
            "retriever"      : type(self.retriever).__name__,
            "reranker"       : self.reranker.get_info() if self.reranker else None,
            "retrieve_top_k" : self.retrieve_top_k,
            "rerank_top_k"   : self.rerank_top_k,
            "cite_sources"   : self.cite_sources,
            "history_turns"  : len(self.history),
            "entity_facts"   : len(self.history.entity_memory),
            "vector_store"   : self.store.get_stats(),
        }


__all__ = ["RAGChain", "ChainResponse"]