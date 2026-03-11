# chains/rag_chain.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.naive_retriever import NaiveRetriever, RetrievalResult
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from generation.groq_llm import BaseLLM, ChatHistory, LLMFactory
from vectorstore.qdrant_store import QdrantVectorStore, BaseVectorStore
from embeddings.embedder import EmbedderFactory
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
- When relevant, mention which source your answer comes from."""

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
    Contains the answer, retrieved chunks, and citations.
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
        return self.retrieval.get_citations()

    def format_citations(self) -> str:
        return self.retrieval.format_citations()

    def get_chunks(self) -> list[dict]:
        return self.retrieval.get_chunks()

    def get_context(self) -> str:
        return self.retrieval.to_context_string()

    def __str__(self) -> str:
        parts = [self.answer]
        citations = self.format_citations()
        if citations:
            parts.append("\n" + citations)
        return "\n".join(parts)

    def __repr__(self) -> str:
        return (
            f"ChainResponse("
            f"model={self.model}, "
            f"chunks={len(self.retrieval)}, "
            f"tokens={self.usage.get('total_tokens', '?')})"
        )


# ─────────────────────────────────────────
# RAG CHAIN
# ─────────────────────────────────────────

class RAGChain:
    """
    Full RAG pipeline: Retrieve → Rerank → Generate.

    Default pipeline:
        HybridRetriever (dense + BM25 + RRF)
            → Reranker (cross-encoder)
                → LLM (Groq or Ollama)

    Features:
        ✅ Multi-turn conversation memory
        ✅ Source citations in answers
        ✅ Streaming responses
        ✅ Configurable retriever and LLM
        ✅ Optional reranking

    Usage:
        chain = RAGChain()
        chain.index_documents(chunks)

        # Single turn
        response = chain.ask("What was Q1 revenue?")
        print(response)

        # Streaming
        for chunk in chain.stream("What was Q1 revenue?"):
            print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        llm           : BaseLLM        = None,
        vector_store  : BaseVectorStore = None,
        retriever                       = None,
        reranker      : Reranker        = None,
        use_reranker  : bool            = True,
        retrieve_top_k: int             = 20,    # fetch wide before rerank
        rerank_top_k  : int             = 5,     # keep narrow after rerank
        cite_sources  : bool            = True,
        llm_provider  : str             = "groq",
    ):
        # ── LLM ──────────────────────────────
        self.llm = llm or LLMFactory.get(llm_provider)
        self.llm.set_system_prompt(RAG_SYSTEM_PROMPT)

        # ── Vector store ──────────────────────
        embedder     = EmbedderFactory.get("huggingface")
        self.store   = vector_store or QdrantVectorStore(embedder=embedder)

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

        # ── Conversation memory ───────────────
        # Shared ChatHistory — persists across ask() calls
        self.history = self.llm.history

        print(f"\n  [RAG CHAIN] ✅ Ready!")
        print(f"  [RAG CHAIN] LLM       : {self.llm.model_name}")
        print(f"  [RAG CHAIN] Retriever : {type(self.retriever).__name__}")
        print(f"  [RAG CHAIN] Reranker  : {'✅' if use_reranker else '❌'}")
        print(f"  [RAG CHAIN] Citations : {'✅' if cite_sources else '❌'}")

    # ── INDEXING ─────────────────────────────

    def index_documents(self, chunks: list[dict]) -> None:
        """
        Add document chunks to both the vector store and BM25 index.
        Call this once after ingesting your documents.

        Args:
            chunks : list of dicts with at least a 'content' key
        """
        self.store.add_documents(chunks)

        # Build BM25 index if retriever supports it
        if hasattr(self.retriever, "index_chunks"):
            self.retriever.index_chunks(chunks)

        print(f"  [RAG CHAIN] Indexed {len(chunks)} chunks.")

    # ── RETRIEVAL ────────────────────────────

    def _retrieve(self, question: str) -> RetrievalResult:
        """Run retrieval + optional reranking."""
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
        """Format the user prompt with context injected."""
        template = (
            RAG_USER_TEMPLATE_WITH_CITATIONS
            if self.cite_sources
            else RAG_USER_TEMPLATE
        )
        return template.format(context=context, question=question)

    # ── ASK (blocking) ───────────────────────

    def ask(
        self,
        question      : str,
        top_k         : int  = None,
        cite_sources  : bool = None,
    ) -> ChainResponse:
        """
        Full RAG pipeline — blocking, returns complete ChainResponse.

        Args:
            question     : user question
            top_k        : override rerank_top_k for this call
            cite_sources : override chain default for this call

        Returns:
            ChainResponse with answer, chunks, citations, usage
        """
        cite    = cite_sources if cite_sources is not None else self.cite_sources
        rerank_k = top_k or self.rerank_top_k

        # ── Step 1: Retrieve + Rerank ──
        retrieval = self._retrieve(question)
        context   = retrieval.to_context_string()

        if not context.strip():
            answer = "I don't have enough information in the provided documents to answer that."
            return ChainResponse(
                answer    = answer,
                retrieval = retrieval,
                question  = question,
                model     = self.llm.model_name,
            )

        # ── Step 2: Build prompt ──
        prompt = self._build_prompt(question, context)

        # ── Step 3: Generate ──
        result = self.llm.generate(
            prompt  = prompt,
            history = self.history,
        )

        return ChainResponse(
            answer    = result["content"],
            retrieval = retrieval,
            question  = question,
            model     = result["model"],
            usage     = result["usage"],
        )

    # ── STREAM ───────────────────────────────

    def stream(
        self,
        question     : str,
        cite_sources : bool = None,
    ):
        """
        Streaming RAG pipeline — yields text chunks then a final ChainResponse.

        Usage:
            for chunk in chain.stream("What was Q1 revenue?"):
                if isinstance(chunk, str):
                    print(chunk, end="", flush=True)
                else:
                    print()
                    print(chunk.format_citations())

        Yields:
            str          — text tokens as they stream
            ChainResponse — final item with full metadata
        """
        cite = cite_sources if cite_sources is not None else self.cite_sources

        # ── Step 1: Retrieve + Rerank ──
        retrieval = self._retrieve(question)
        context   = retrieval.to_context_string()

        if not context.strip():
            answer = "I don't have enough information in the provided documents to answer that."
            yield answer
            yield ChainResponse(
                answer    = answer,
                retrieval = retrieval,
                question  = question,
                model     = self.llm.model_name,
            )
            return

        # ── Step 2: Build prompt ──
        prompt = self._build_prompt(question, context)

        # ── Step 3: Stream ──
        full_reply = []
        usage      = {}

        for chunk in self.llm.stream(prompt=prompt, history=self.history):
            if isinstance(chunk, str):
                full_reply.append(chunk)
                yield chunk
            else:
                usage = chunk.get("usage", {})

        # ── Final ChainResponse ──
        yield ChainResponse(
            answer    = "".join(full_reply),
            retrieval = retrieval,
            question  = question,
            model     = self.llm.model_name,
            usage     = usage,
        )

    # ── MEMORY ───────────────────────────────

    def reset_memory(self) -> None:
        """Clear conversation history, keep system prompt and chain config."""
        self.llm.reset_history()
        print("  [RAG CHAIN] Conversation memory cleared.")

    def get_history(self) -> list[dict]:
        """Return raw message history list."""
        return self.history.to_messages()

    # ── INFO ─────────────────────────────────

    def get_info(self) -> dict:
        return {
            "llm"          : self.llm.get_info(),
            "retriever"    : type(self.retriever).__name__,
            "reranker"     : self.reranker.get_info() if self.reranker else None,
            "retrieve_top_k": self.retrieve_top_k,
            "rerank_top_k" : self.rerank_top_k,
            "cite_sources" : self.cite_sources,
            "history_turns": len(self.history),
            "vector_store" : self.store.get_stats(),
        }


__all__ = ["RAGChain", "ChainResponse"]