# chains/rag_chain.py
#
# CHANGES vs original:
#   - QueryRouter class added — rule-based, zero latency classifier
#     Chitchat → instant canned response, no retrieval
#     Document query → full pipeline
#     This prevents greetings from burning retrieval + LLM time
#   - RAGChain.ask() and stream() check router first
#   - Fallback chain added:
#     If reranker's best score < MIN_RERANK_SCORE → "not found in documents"
#     instead of hallucinating an answer from empty context
#   - parent_store wiring: RAGChain constructor accepts parent_store dict
#     and passes it to the retriever
#   - ChainResponse unchanged — same get_citations(), get_images(), usage
#   - get_info() now exposes query_type from last call

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
# PROMPT TEMPLATES  (unchanged)
# ─────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """\
You are a precise document assistant. You answer questions based on the provided context documents AND the ongoing conversation history.

Rules:
1. For factual claims about documents, answer from the provided context. Do not invent facts not in the context.
2. For follow-up questions that reference previous turns (e.g. "what about that?", "you mentioned X earlier", \
"my name is Y" from earlier in conversation), use the conversation history — this is expected and correct.
3. If neither the context nor the conversation history contains sufficient information, say:
   "I don't have enough information in the provided documents to answer this."
4. Be concise and direct. No padding, no "certainly!" or "great question!"
5. Do NOT write a 'Sources:' or 'References:' section at the end of your answer. Source citations are handled separately by the system. Do not list filenames or page numbers at the bottom.
6. Preserve technical terminology exactly as it appears in the source.
7. Always format tabular data as a markdown table using | col | col | syntax."""

RAG_USER_TEMPLATE = """\
Context:
{context}

Question: {question}"""


NOT_FOUND_RESPONSE = (
    "I don't have enough information in the provided documents to answer this. "
    "Please make sure the relevant document is uploaded and indexed."
)

NO_KB_RESPONSE = (
    "No documents are loaded yet. "
    "Please upload files using the sidebar before asking questions."
)


# ─────────────────────────────────────────────────────────
# QUERY ROUTER  (NEW)
# ─────────────────────────────────────────────────────────

class QueryRouter:
    """
    Rule-based query classifier — zero latency, no LLM call.

    Two classes:
        CHITCHAT  → greetings, thanks, goodbyes, small talk
        DOCUMENT  → everything else (safe default)

    Design choice: errs on the side of DOCUMENT when uncertain.
    Better to retrieve and say "not found" than to skip a real question.
    """

    CHITCHAT = "chitchat"
    DOCUMENT = "document"

    # fmt: off
    _PATTERNS = [
        r"^\s*(hi+|hello+|hey+|howdy|greetings)\s*[!.?]*\s*$",
        r"^\s*good\s*(morning|afternoon|evening|day|night)\s*[!.?]*\s*$",
        r"^\s*how are you\b.*$",
        r"^\s*(thanks?|thank\s*you|thx|ty|cheers)\s*[!.?]*\s*$",
        r"^\s*(bye|goodbye|see\s*ya?|cya|later|take\s*care)\s*[!.?]*\s*$",
        r"^\s*(ok+|okay|sure|got\s*it|understood|alright|cool|nice|great|awesome)\s*[!.?]*\s*$",
        r"^\s*(yes|yeah|yep|yup|no|nope|nah)\s*[!.?]*\s*$",
        r"^\s*who\s+are\s+you\s*[?!.]*\s*$",
        r"^\s*what\s+(can|do)\s+you\s+(do|help(\s+with)?)\s*[?!.]*\s*$",
        r"^\s*help\s*[?!.]*\s*$",
        r"^\s*(what('?s| is) (up|new))\s*[?!.]*\s*$",
        r"^\s*tell me (a )?joke\s*[?!.]*\s*$",
        r"^\s*nice\s*(work|job|one)\s*[!.?]*\s*$",
    ]
    # fmt: on

    _COMPILED = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]

    _RESPONSES = {
        "greeting": (
            "Hey! I'm your document assistant. "
            "Upload some PDFs and ask me anything about them."
        ),
        "thanks"  : "You're welcome! Anything else about your documents?",
        "bye"     : "Goodbye! Come back anytime.",
        "help"    : (
            "I can answer questions about your uploaded documents. "
            "Just ask me anything — I'll search through the content and give you a precise answer with sources."
        ),
        "who"     : (
            "I'm a RAG assistant — I read your documents and answer questions "
            "based strictly on their content."
        ),
        "default" : "Got it! Feel free to ask me anything about your documents.",
    }

    @classmethod
    def classify(cls, question: str) -> str:
        q = question.strip()

        for pattern in cls._COMPILED:
            if pattern.match(q):
                return cls.CHITCHAT

        # Very short + no question mark → likely chitchat
        words = q.split()
        if len(words) <= 2 and "?" not in q:
            return cls.CHITCHAT

        return cls.DOCUMENT

    @classmethod
    def get_chitchat_response(cls, question: str) -> str:
        q = question.lower()
        if any(w in q for w in ("hi", "hello", "hey", "howdy", "morning", "afternoon", "evening")):
            return cls._RESPONSES["greeting"]
        if any(w in q for w in ("thanks", "thank", "thx", "ty")):
            return cls._RESPONSES["thanks"]
        if any(w in q for w in ("bye", "goodbye", "cya", "later")):
            return cls._RESPONSES["bye"]
        if "help" in q:
            return cls._RESPONSES["help"]
        if "who are you" in q or "what can you do" in q:
            return cls._RESPONSES["who"]
        return cls._RESPONSES["default"]


# ─────────────────────────────────────────────────────────
# CHAIN RESPONSE  (unchanged interface)
# ─────────────────────────────────────────────────────────

class ChainResponse:
    """
    Wraps the full output of a RAG chain call.

    Unchanged vs original — same get_citations(), get_images(), usage.
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
        """Returns citation dicts with source, page, heading, section_path, type."""
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
        """Return absolute paths of any retrieved images that exist on disk."""
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

    CHANGES vs original:
      - QueryRouter: chitchat is handled instantly without retrieval
      - parent_store: passed to retriever for small-to-big expansion
      - Fallback: if reranker best score < MIN_RERANK_SCORE → not-found response
        prevents hallucination when context is empty or irrelevant
      - has_kb parameter on ask()/stream() for clean "no documents" path
      - ChainResponse gains query_type field
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
        parent_store  : dict            = None,   # {parent_id: parent_dict}
    ):
        # ── LLM ───────────────────────────────────────────
        self.llm = llm or LLMFactory.get(llm_provider)
        self.llm.set_system_prompt(RAG_SYSTEM_PROMPT)

        # ── Vector store ──────────────────────────────────
        embedder   = EmbedderFactory.get("huggingface")
        self.store = vector_store or QdrantVectorStore(embedder=embedder)

        # ── Parent store ──────────────────────────────────
        self.parent_store = parent_store or {}

        # ── Retriever ─────────────────────────────────────
        if retriever is not None:
            self.retriever = retriever
            # Inject parent_store into existing retriever if it supports it
            if hasattr(self.retriever, "parent_store") and self.parent_store:
                self.retriever.parent_store = self.parent_store
        else:
            self.retriever = HybridRetriever(
                vector_store = self.store,
                embedder     = embedder,
                top_k        = retrieve_top_k,
                parent_store = self.parent_store,
            )

        # ── Reranker ──────────────────────────────────────
        self.use_reranker = use_reranker
        self.reranker     = reranker or (Reranker() if use_reranker else None)
        self.rerank_top_k = rerank_top_k

        # ── Settings ──────────────────────────────────────
        self.retrieve_top_k = retrieve_top_k
        self.cite_sources   = cite_sources

        # ── Memory ────────────────────────────────────────
        self.history   = self.llm.history
        self._last_type= QueryRouter.DOCUMENT

        print(f"\n  [RAG CHAIN] ✅ Ready!")
        print(f"  [RAG CHAIN] LLM       : {self.llm.model_name}")
        print(f"  [RAG CHAIN] Retriever : {type(self.retriever).__name__}")
        print(f"  [RAG CHAIN] Reranker  : {'✅' if use_reranker else '❌'}")
        print(f"  [RAG CHAIN] Router    : ✅ (chitchat bypass active)")
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

    # ── HYDE QUERY EXPANSION ──────────────────────────────

    def _expand_query(self, question: str) -> str:
        """
        HyDE (Hypothetical Document Embedding) query expansion.

        Always-on. Uses conversation history as context so vague follow-ups
        like "wat about last yr?" or "what did it say about rev?" are
        rewritten into clean, self-contained search queries.

        The expanded query is used ONLY for retrieval — the original
        question is still passed to the LLM for generation so the
        answer stays natural.

        Returns the expanded query string (or original if expansion fails).
        """
        # Build a compact conversation context (last 3 user turns only)
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
                print(f"  [HYDE] '{question[:40]}' → '{expanded[:60]}'")
                return expanded
        except Exception as e:
            print(f"  [HYDE] Expansion failed: {e}")

        return question

    # ── ASK (blocking) ────────────────────────────────────

    def ask(
        self,
        question  : str,
        top_k     : int  = None,
        has_kb    : bool = True,
    ) -> ChainResponse:
        """
        Blocking RAG pipeline with router + fallback.

        Args:
            question : user's question
            has_kb   : set False if no documents are indexed yet
        """
        # ── 1. Router ──────────────────────────────────────
        query_type = QueryRouter.classify(question)
        self._last_type = query_type

        if query_type == QueryRouter.CHITCHAT:
            response = QueryRouter.get_chitchat_response(question)
            return ChainResponse(
                answer    = response,
                retrieval = RetrievalResult([]),
                question  = question,
                model     = self.llm.model_name,
                query_type= QueryRouter.CHITCHAT,
            )

        # ── 2. No KB guard ─────────────────────────────────
        if not has_kb:
            return ChainResponse(
                answer    = NO_KB_RESPONSE,
                retrieval = RetrievalResult([]),
                question  = question,
                model     = self.llm.model_name,
                query_type= QueryRouter.DOCUMENT,
            )

        # ── 3. Expand query + retrieve + rerank ────────────
        expanded  = self._expand_query(question)
        retrieval = self._retrieve(expanded)
        context   = retrieval.to_context_string()

        # ── 4. Fallback if context is empty or low quality ─
        if not context.strip() or (
            self.use_reranker
            and retrieval.best_score() < MIN_RERANK_SCORE
        ):
            self.history.add_user(question)
            self.history.add_assistant(NOT_FOUND_RESPONSE)
            return ChainResponse(
                answer    = NOT_FOUND_RESPONSE,
                retrieval = retrieval,
                question  = question,
                model     = self.llm.model_name,
                query_type= QueryRouter.DOCUMENT,
            )

        # ── 5. Generate ────────────────────────────────────
        prompt = self._build_prompt(question, context)
        result = self.llm.generate(
            prompt    = prompt,
            history   = self.history,
            store_as  = question,   # store clean question, not full prompt+context
        )

        return ChainResponse(
            answer    = result["content"],
            retrieval = retrieval,
            question  = question,
            model     = result["model"],
            usage     = result["usage"],
            query_type= QueryRouter.DOCUMENT,
        )

    # ── STREAM (generator) ────────────────────────────────

    def stream(self, question: str, has_kb: bool = True):
        """
        Streaming RAG pipeline with router + fallback.

        Yields:
            str           — text tokens as they stream
            ChainResponse — final item with full metadata
        """
        # ── 1. Router ──────────────────────────────────────
        query_type = QueryRouter.classify(question)
        self._last_type = query_type

        if query_type == QueryRouter.CHITCHAT:
            response = QueryRouter.get_chitchat_response(question)
            yield response
            yield ChainResponse(
                answer    = response,
                retrieval = RetrievalResult([]),
                question  = question,
                model     = self.llm.model_name,
                query_type= QueryRouter.CHITCHAT,
            )
            return

        # ── 2. No KB guard ─────────────────────────────────
        if not has_kb:
            yield NO_KB_RESPONSE
            yield ChainResponse(
                answer    = NO_KB_RESPONSE,
                retrieval = RetrievalResult([]),
                question  = question,
                model     = self.llm.model_name,
                query_type= QueryRouter.DOCUMENT,
            )
            return

        # ── 3. Expand query + retrieve + rerank ────────────
        expanded  = self._expand_query(question)
        retrieval = self._retrieve(expanded)
        context   = retrieval.to_context_string()

        # ── 4. Fallback ────────────────────────────────────
        if not context.strip() or (
            self.use_reranker
            and retrieval.best_score() < MIN_RERANK_SCORE
        ):
            self.history.add_user(question)
            self.history.add_assistant(NOT_FOUND_RESPONSE)
            yield NOT_FOUND_RESPONSE
            yield ChainResponse(
                answer    = NOT_FOUND_RESPONSE,
                retrieval = retrieval,
                question  = question,
                model     = self.llm.model_name,
                query_type= QueryRouter.DOCUMENT,
            )
            return

        # ── 5. Generate (streaming) ────────────────────────
        prompt     = self._build_prompt(question, context)
        full_reply: list[str] = []
        usage:      dict      = {}

        for chunk in self.llm.stream(
            prompt   = prompt,
            history  = self.history,
            store_as = question,   # store clean question, not full prompt+context
        ):
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
            query_type= QueryRouter.DOCUMENT,
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
            "llm"           : self.llm.get_info(),
            "retriever"     : type(self.retriever).__name__,
            "reranker"      : self.reranker.get_info() if self.reranker else None,
            "retrieve_top_k": self.retrieve_top_k,
            "rerank_top_k"  : self.rerank_top_k,
            "cite_sources"  : self.cite_sources,
            "history_turns" : len(self.history),
            "entity_facts"  : len(self.history.entity_memory),  # compat
            "parent_store"  : len(self.parent_store),
            "vector_store"  : self.store.get_stats(),
            "last_query_type": self._last_type,
        }


__all__ = ["QueryRouter", "ChainResponse", "RAGChain"]