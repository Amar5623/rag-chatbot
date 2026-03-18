# chains/rag_chain.py

import os
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.naive_retriever  import NaiveRetriever, RetrievalResult
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker         import Reranker
from generation.groq_llm        import BaseLLM, ChatHistory, LLMFactory
from vectorstore.qdrant_store   import QdrantVectorStore, BaseVectorStore
from embeddings.embedder        import EmbedderFactory
from config                     import TOP_K, MIN_RERANK_SCORE


# ─────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """\
You are a precise document assistant. Answer questions based on the provided context documents AND the ongoing conversation history.

Rules:
1. For factual claims, answer strictly from the provided context. Do not invent facts not in the context.
2. For follow-up questions referencing previous turns, use the conversation history — this is expected and correct.
3. Be concise and direct. No padding, no "certainly!" or "great question!"
4. Do NOT write a 'Sources:' or 'References:' section — citations are handled separately by the system.
5. Preserve technical terminology exactly as it appears in the source.
6. Always format tabular data as a markdown table using | col | col | syntax."""

RAG_USER_TEMPLATE = """\
Context:
{context}

Question: {question}"""

GENERAL_FALLBACK_PROMPT = """\
You are a helpful assistant. The user asked a question but the provided documents do not contain relevant information to answer it.

Rules:
1. Start your response with one short sentence noting the documents didn't cover this topic.
2. Then answer from your general knowledge if you have sufficient knowledge on the topic.
3. If you don't have general knowledge on it either, say so honestly.
4. Be concise. No padding."""

CHITCHAT_SYSTEM_PROMPT = """\
You are a helpful document assistant. You help users understand their uploaded documents.
For casual conversation, respond naturally and briefly.
If the user shares personal info like their name, acknowledge it warmly and remember it for the conversation.
You can briefly mention you can answer questions about their uploaded documents, but don't be repetitive about it."""

NO_KB_RESPONSE = (
    "No documents are loaded yet. "
    "Please upload files using the sidebar before asking questions."
)


# ─────────────────────────────────────────────────────────
# QUERY ROUTER
# ─────────────────────────────────────────────────────────

class QueryRouter:
    """
    Rule-based query classifier — zero latency, no LLM call.

    Three classes:
        CHITCHAT → greetings, thanks, goodbyes, name sharing, small talk
        DOCUMENT → everything else (safe default)
        GENERAL  → used internally when doc retrieval fails

    All classes route through the LLM — no canned string responses.
    The class only decides WHICH system prompt and pipeline to use.

    Design choice: errs on the side of DOCUMENT when uncertain.
    Better to attempt retrieval and fall back gracefully than to
    skip a real document question.
    """

    CHITCHAT = "chitchat"
    DOCUMENT = "document"
    GENERAL  = "general"

    _PATTERNS = [
        r"^\s*(hi+|hello+|hey+|howdy|greetings)\s*[!.?]*\s*$",
        r"^\s*good\s*(morning|afternoon|evening|day|night)\s*[!.?]*\s*$",
        r"^\s*how are you\b.*$",
        r"^\s*(thanks?|thank\s*you|thx|ty|cheers)\s*[!.?]*\s*$",
        r"^\s*(bye|goodbye|see\s*ya?|cya|later|take\s*care)\s*[!.?]*\s*$",
        r"^\s*who\s+are\s+you\s*[?!.]*\s*$",
        r"^\s*what\s+(can|do)\s+you\s+(do|help(\s+with)?)\s*[?!.]*\s*$",
        r"^\s*help\s*[?!.]*\s*$",
        r"^\s*(what('?s| is) (up|new))\s*[?!.]*\s*$",
        r"^\s*tell me (a )?joke\s*[?!.]*\s*$",
        r"^\s*my name is\b.*$",
        r"^\s*i('?m| am)\s+\w+\s*[!.?]*\s*$",
        r"^\s*call me\s+\w+\s*[!.?]*\s*$",
        r"^\s*nice\s*(work|job|one)\s*[!.?]*\s*$",
    ]

    _COMPILED = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]

    @classmethod
    def classify(cls, question: str) -> str:
        q = question.strip()
        for pattern in cls._COMPILED:
            if pattern.match(q):
                return cls.CHITCHAT
        # Very short with no question mark → likely chitchat
        if len(q.split()) <= 2 and "?" not in q:
            return cls.CHITCHAT
        return cls.DOCUMENT


# ─────────────────────────────────────────────────────────
# CHAIN RESPONSE
# ─────────────────────────────────────────────────────────

class ChainResponse:
    """
    Wraps the full output of a RAG chain call.
    query_type is one of: "document", "chitchat", "general"
    Citations and images are only meaningful when query_type == "document".
    """

    def __init__(
        self,
        answer     : str,
        retrieval  : RetrievalResult,
        question   : str,
        model      : str,
        usage      : dict = None,
        query_type : str  = "document",
    ):
        self.answer     = answer
        self.retrieval  = retrieval
        self.question   = question
        self.model      = model
        self.usage      = usage or {}
        self.query_type = query_type

    def get_answer(self) -> str:
        return self.answer

    def get_citations(self) -> list[dict]:
        """Returns citation dicts. Only populated when query_type == 'document'."""
        citations: list[dict] = []
        for chunk in self.retrieval.get_chunks():
            citations.append({
                "source"      : chunk.get("source", "unknown"),
                "page"        : chunk.get("page", "?"),
                "heading"     : chunk.get("heading", ""),
                "section_path": chunk.get("section_path", ""),
                "type"        : chunk.get("type", "text"),
            })
        return citations

    def get_images(self) -> list[str]:
        """Return absolute paths of retrieved images. Only populated for document answers."""
        return self.retrieval.get_images()

    def has_images(self) -> bool:
        return len(self.get_images()) > 0

    def format_citations(self) -> str:
        lines: list[str] = []
        for c in self.get_citations():
            section = c.get("section_path") or c.get("heading") or ""
            section_str = f" [{section}]" if section else ""
            lines.append(f"  • {c['source']} (p.{c['page']}){section_str}")
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
            f"query_type={self.query_type}, "
            f"chunks={len(self.retrieval)}, "
            f"images={len(self.get_images())}, "
            f"tokens={self.usage.get('total_tokens', '?')})"
        )


# ─────────────────────────────────────────────────────────
# RAG CHAIN
# ─────────────────────────────────────────────────────────

class RAGChain:
    """
    Full RAG pipeline: Route → Retrieve → Rerank → Generate.

    Query priority:
      1. Chitchat  → LLM with chitchat system prompt (no retrieval)
      2. No KB     → LLM with general knowledge prompt
      3. Document  → retrieve from KB
                     if context weak → LLM with general knowledge fallback
                     if context good → full RAG answer with citations
    """

    def __init__(
        self,
        llm           : BaseLLM        = None,
        vector_store  : BaseVectorStore = None,
        retriever                       = None,
        reranker      : Reranker        = None,
        use_reranker  : bool            = True,
        retrieve_top_k: int             = TOP_K,
        rerank_top_k  : int             = 5,
        cite_sources  : bool            = True,
        llm_provider  : str             = "groq",
    ):
        # ── LLM ───────────────────────────────────────────
        self.llm = llm or LLMFactory.get(llm_provider)
        self.llm.set_system_prompt(RAG_SYSTEM_PROMPT)

        # ── Vector store ──────────────────────────────────
        embedder   = EmbedderFactory.get("huggingface")
        self.store = vector_store or QdrantVectorStore(embedder=embedder)

        # ── Retriever ─────────────────────────────────────
        if retriever is not None:
            self.retriever = retriever
        else:
            self.retriever = HybridRetriever(
                vector_store = self.store,
                embedder     = embedder,
                top_k        = retrieve_top_k,
            )

        # ── Reranker ──────────────────────────────────────
        self.use_reranker = use_reranker
        self.reranker     = reranker or (Reranker() if use_reranker else None)
        self.rerank_top_k = rerank_top_k

        # ── Settings ──────────────────────────────────────
        self.retrieve_top_k = retrieve_top_k
        self.cite_sources   = cite_sources

        # ── Memory ────────────────────────────────────────
        self.history    = self.llm.history
        self._last_type = QueryRouter.DOCUMENT

        print(f"\n  [RAG CHAIN] ✅ Ready!")
        print(f"  [RAG CHAIN] LLM       : {self.llm.model_name}")
        print(f"  [RAG CHAIN] Retriever : {type(self.retriever).__name__}")
        print(f"  [RAG CHAIN] Reranker  : {'✅' if use_reranker else '❌'}")
        print(f"  [RAG CHAIN] Router    : ✅ (LLM-routed, no canned responses)")
        print(f"  [RAG CHAIN] Citations : {'✅' if cite_sources else '❌'}")

    # ── INDEXING ──────────────────────────────────────────

    def index_documents(self, chunks: list[dict]) -> None:
        self.store.add_documents(chunks)
        if hasattr(self.retriever, "index_chunks"):
            self.retriever.index_chunks(chunks)
        print(f"  [RAG CHAIN] Indexed {len(chunks)} chunks.")

    # ── RETRIEVAL ─────────────────────────────────────────

    def _retrieve(self, question: str) -> RetrievalResult:
        retrieval = self.retriever.retrieve(question)
        if self.use_reranker and self.reranker and len(retrieval) > 0:
            retrieval = self.reranker.rerank(
                query     = question,
                retrieval = retrieval,
                top_k     = self.rerank_top_k,
            )
        return retrieval

    # ── PROMPT BUILDING ───────────────────────────────────

    def _build_prompt(self, question: str, context: str) -> str:
        return RAG_USER_TEMPLATE.format(context=context, question=question)

    # ── QUERY EXPANSION ───────────────────────────────────

    def _expand_query(self, question: str) -> str:
        """
        Query rewriting using conversation context.
        Rewrites vague follow-ups into clean standalone search queries.
        Used ONLY for retrieval — original question goes to the LLM.
        """
        recent_turns = [
            m["content"] for m in self.history._turns
            if m["role"] == "user"
        ][-3:]
        history_ctx = ""
        if recent_turns:
            history_ctx = "\nRecent conversation context:\n" + "\n".join(
                f"- {t}" for t in recent_turns
            )

        prompt = (
            f"Rewrite the following question as a clear, self-contained search query "
            f"that would retrieve relevant passages from a document.{history_ctx}\n\n"
            f"Original question: {question}\n\n"
            f"Rules:\n"
            f"- Fix spelling mistakes\n"
            f"- Expand abbreviations and pronouns using the conversation context\n"
            f"- Make it a complete standalone query (no pronouns like 'it', 'they', 'that')\n"
            f"- Keep it concise — one sentence\n"
            f"- Return ONLY the rewritten query, nothing else"
        )

        try:
            resp = self.llm.client.chat.completions.create(
                model       = self.llm.model_name,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = 80,
                temperature = 0.1,
            )
            expanded = resp.choices[0].message.content.strip().strip('"').strip("'")
            if expanded:
                print(f"  [QUERY EXPAND] '{question[:40]}' → '{expanded[:60]}'")
                return expanded
        except Exception as e:
            print(f"  [QUERY EXPAND] Failed: {e}")

        return question

    # ── HELPER: stream through LLM with a given system prompt ─

    def _stream_with_prompt(self, system_prompt: str, user_prompt: str):
        """
        Internal helper. Sets system prompt, streams LLM response,
        restores RAG system prompt, returns (full_reply, usage).
        Yields str tokens during streaming.
        """
        self.llm.set_system_prompt(system_prompt)
        full_reply: list[str] = []
        usage: dict = {}
        for chunk in self.llm.stream(
            prompt   = user_prompt,
            history  = self.history,
            store_as = user_prompt,
        ):
            if isinstance(chunk, str):
                full_reply.append(chunk)
                yield chunk
            else:
                usage = chunk.get("usage", {})
        self.llm.set_system_prompt(RAG_SYSTEM_PROMPT)
        yield {"_meta": True, "full_reply": "".join(full_reply), "usage": usage}

    # ── ASK (blocking) ────────────────────────────────────

    def ask(self, question: str, top_k: int = None, has_kb: bool = True) -> ChainResponse:
        """
        Blocking RAG pipeline.
        Priority: chitchat → no_kb → doc retrieval → general fallback
        """
        query_type = QueryRouter.classify(question)
        self._last_type = query_type

        # ── 1. Chitchat ───────────────────────────────────
        if query_type == QueryRouter.CHITCHAT:
            self.llm.set_system_prompt(CHITCHAT_SYSTEM_PROMPT)
            result = self.llm.generate(
                prompt   = question,
                history  = self.history,
                store_as = question,
            )
            self.llm.set_system_prompt(RAG_SYSTEM_PROMPT)
            return ChainResponse(
                answer     = result["content"],
                retrieval  = RetrievalResult([]),
                question   = question,
                model      = result["model"],
                usage      = result["usage"],
                query_type = QueryRouter.CHITCHAT,
            )

        # ── 2. No KB ──────────────────────────────────────
        if not has_kb:
            self.llm.set_system_prompt(GENERAL_FALLBACK_PROMPT)
            result = self.llm.generate(
                prompt   = question,
                history  = self.history,
                store_as = question,
            )
            self.llm.set_system_prompt(RAG_SYSTEM_PROMPT)
            return ChainResponse(
                answer     = result["content"],
                retrieval  = RetrievalResult([]),
                question   = question,
                model      = result["model"],
                usage      = result["usage"],
                query_type = QueryRouter.GENERAL,
            )

        # ── 3. Retrieve ───────────────────────────────────
        expanded  = self._expand_query(question)
        retrieval = self._retrieve(expanded)
        context   = retrieval.to_context_string()

        # ── 4. Weak context → general knowledge fallback ──
        if not context.strip() or (
            self.use_reranker
            and retrieval.best_score() < MIN_RERANK_SCORE
        ):
            self.llm.set_system_prompt(GENERAL_FALLBACK_PROMPT)
            result = self.llm.generate(
                prompt   = question,
                history  = self.history,
                store_as = question,
            )
            self.llm.set_system_prompt(RAG_SYSTEM_PROMPT)
            return ChainResponse(
                answer     = result["content"],
                retrieval  = RetrievalResult([]),   # no citations on fallback
                question   = question,
                model      = result["model"],
                usage      = result["usage"],
                query_type = QueryRouter.GENERAL,
            )

        # ── 5. Full RAG ───────────────────────────────────
        self.llm.set_system_prompt(RAG_SYSTEM_PROMPT)
        prompt = self._build_prompt(question, context)
        result = self.llm.generate(
            prompt   = prompt,
            history  = self.history,
            store_as = question,
        )
        return ChainResponse(
            answer     = result["content"],
            retrieval  = retrieval,
            question   = question,
            model      = result["model"],
            usage      = result["usage"],
            query_type = QueryRouter.DOCUMENT,
        )

    # ── STREAM (generator) ────────────────────────────────

    def stream(self, question: str, has_kb: bool = True):
        """
        Streaming RAG pipeline.
        Priority: chitchat → no_kb → doc retrieval → general fallback

        Yields:
            str           — text tokens as they stream
            ChainResponse — final item with full metadata
        """
        query_type = QueryRouter.classify(question)
        self._last_type = query_type

        # ── 1. Chitchat ───────────────────────────────────
        if query_type == QueryRouter.CHITCHAT:
            self.llm.set_system_prompt(CHITCHAT_SYSTEM_PROMPT)
            full_reply: list[str] = []
            usage: dict = {}
            for chunk in self.llm.stream(
                prompt   = question,
                history  = self.history,
                store_as = question,
            ):
                if isinstance(chunk, str):
                    full_reply.append(chunk)
                    yield chunk
                else:
                    usage = chunk.get("usage", {})
            self.llm.set_system_prompt(RAG_SYSTEM_PROMPT)
            yield ChainResponse(
                answer     = "".join(full_reply),
                retrieval  = RetrievalResult([]),
                question   = question,
                model      = self.llm.model_name,
                usage      = usage,
                query_type = QueryRouter.CHITCHAT,
            )
            return

        # ── 2. No KB ──────────────────────────────────────
        if not has_kb:
            self.llm.set_system_prompt(GENERAL_FALLBACK_PROMPT)
            full_reply = []
            usage = {}
            for chunk in self.llm.stream(
                prompt   = question,
                history  = self.history,
                store_as = question,
            ):
                if isinstance(chunk, str):
                    full_reply.append(chunk)
                    yield chunk
                else:
                    usage = chunk.get("usage", {})
            self.llm.set_system_prompt(RAG_SYSTEM_PROMPT)
            yield ChainResponse(
                answer     = "".join(full_reply),
                retrieval  = RetrievalResult([]),
                question   = question,
                model      = self.llm.model_name,
                usage      = usage,
                query_type = QueryRouter.GENERAL,
            )
            return

        # ── 3. Retrieve ───────────────────────────────────
        expanded  = self._expand_query(question)
        retrieval = self._retrieve(expanded)
        context   = retrieval.to_context_string()

        # ── 4. Weak context → general knowledge fallback ──
        if not context.strip() or (
            self.use_reranker
            and retrieval.best_score() < MIN_RERANK_SCORE
        ):
            self.llm.set_system_prompt(GENERAL_FALLBACK_PROMPT)
            full_reply = []
            usage = {}
            for chunk in self.llm.stream(
                prompt   = question,
                history  = self.history,
                store_as = question,
            ):
                if isinstance(chunk, str):
                    full_reply.append(chunk)
                    yield chunk
                else:
                    usage = chunk.get("usage", {})
            self.llm.set_system_prompt(RAG_SYSTEM_PROMPT)
            yield ChainResponse(
                answer     = "".join(full_reply),
                retrieval  = RetrievalResult([]),   # no citations on fallback
                question   = question,
                model      = self.llm.model_name,
                usage      = usage,
                query_type = QueryRouter.GENERAL,
            )
            return

        # ── 5. Full RAG ───────────────────────────────────
        self.llm.set_system_prompt(RAG_SYSTEM_PROMPT)
        prompt     = self._build_prompt(question, context)
        full_reply = []
        usage      = {}
        for chunk in self.llm.stream(
            prompt   = prompt,
            history  = self.history,
            store_as = question,
        ):
            if isinstance(chunk, str):
                full_reply.append(chunk)
                yield chunk
            else:
                usage = chunk.get("usage", {})

        yield ChainResponse(
            answer     = "".join(full_reply),
            retrieval  = retrieval,
            question   = question,
            model      = self.llm.model_name,
            usage      = usage,
            query_type = QueryRouter.DOCUMENT,
        )

    # ── MEMORY ────────────────────────────────────────────

    def reset_memory(self) -> None:
        """Clear both sliding window AND rolling summary."""
        self.llm.reset_history()
        print("  [RAG CHAIN] Memory fully cleared.")

    def get_history(self) -> list[dict]:
        return self.history.to_messages()

    # ── INFO ──────────────────────────────────────────────

    def get_info(self) -> dict:
        return {
            "llm"            : self.llm.get_info(),
            "retriever"      : type(self.retriever).__name__,
            "reranker"       : self.reranker.get_info() if self.reranker else None,
            "retrieve_top_k" : self.retrieve_top_k,
            "rerank_top_k"   : self.rerank_top_k,
            "cite_sources"   : self.cite_sources,
            "history_turns"  : len(self.history),
            "parent_store"   : len(self.retriever.parent_store) if hasattr(self.retriever, "parent_store") and self.retriever.parent_store else 0,
            "vector_store"   : self.store.get_stats(),
            "last_query_type": self._last_type,
        }


__all__ = ["QueryRouter", "ChainResponse", "RAGChain"]