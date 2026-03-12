# generation/groq_llm.py
# IMPROVED:
#   - ChatHistory now has TWO memory layers:
#       1. Sliding window  — keeps last N turns (prevents context overflow)
#       2. Entity memory   — extracts & persists key facts across the whole session
#   - Entity memory is injected into every prompt so facts from turn 1
#     are still available at turn 100 even after the window slides past them

import os
import sys
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL


# ─────────────────────────────────────────
# ENTITY MEMORY
# ─────────────────────────────────────────

class EntityMemory:
    """
    Lightweight key-fact store that persists across the entire session,
    even after the sliding window has discarded older turns.

    HOW IT WORKS
    ------------
    After each assistant reply, we run a tiny regex scan over the
    conversation looking for common patterns:
        - Named entities  (capitalised noun phrases)
        - Numeric facts   (numbers with units or labels)
        - Dates           (YYYY, DD/MM/YYYY, "January 2024", etc.)

    These are stored in a dict and prepended to every system prompt
    so the LLM always has the session's key facts available.

    WHY NOT USE AN LLM TO EXTRACT ENTITIES?
    ----------------------------------------
    It would work better but adds latency + token cost on every turn.
    For a local project, regex extraction is free and fast enough.
    In production (enterprise RAG), use an LLM extraction step.
    """

    def __init__(self):
        self._facts: dict[str, str] = {}   # entity → last seen value

    # ── EXTRACTION ───────────────────────────

    def update_from_text(self, text: str) -> None:
        """Scan text for facts and update the store."""
        self._extract_numbers(text)
        self._extract_dates(text)
        self._extract_named_entities(text)

    def _extract_numbers(self, text: str) -> None:
        """Capture patterns like '$2.3M', '40%', '1,200 units'."""
        patterns = [
            r"\$[\d,]+(?:\.\d+)?[MBKmkb]?",        # $2.3M, $1,200
            r"\d+(?:\.\d+)?%",                        # 40%, 3.5%
            r"\b\d[\d,]*\s+(?:units|users|customers|employees|records)\b",
        ]
        for pat in patterns:
            for match in re.findall(pat, text):
                # use match itself as key (deduplicates exact figures)
                self._facts[match] = match

    def _extract_dates(self, text: str) -> None:
        """Capture year references and common date formats."""
        patterns = [
            r"\b(20\d{2})\b",                                      # 2024
            r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",               # 01/03/2024
            r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+20\d{2})\b",
        ]
        for pat in patterns:
            for match in re.findall(pat, text):
                self._facts[f"date:{match}"] = match

    def _extract_named_entities(self, text: str) -> None:
        """
        Very lightweight NER: capitalised sequences of 1-3 words
        that aren't common sentence starters.
        """
        SKIP = {"I", "The", "A", "An", "This", "That", "These", "It",
                 "He", "She", "They", "We", "You", "In", "On", "At",
                 "For", "From", "To", "Of", "And", "Or", "But"}
        pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b"
        for match in re.findall(pattern, text):
            if match not in SKIP and len(match) > 3:
                self._facts[match] = match

    # ── READ ─────────────────────────────────

    def to_prompt_block(self) -> str:
        """
        Format stored facts as a compact block to prepend to system prompt.
        Returns empty string if no facts collected yet.
        """
        if not self._facts:
            return ""
        facts_str = ", ".join(sorted(set(self._facts.values()))[:30])  # cap at 30
        return f"\n\n[Session facts: {facts_str}]"

    def clear(self) -> None:
        self._facts = {}

    def __len__(self) -> int:
        return len(self._facts)

    def get_all(self) -> dict:
        return dict(self._facts)


# ─────────────────────────────────────────
# CHAT HISTORY  (sliding window + entity)
# ─────────────────────────────────────────

class ChatHistory:
    """
    Manages conversation memory with TWO layers:

    Layer 1 — Sliding Window
        Keeps only the last `max_turns` user+assistant pairs.
        Prevents the context window from overflowing in long sessions.
        Older turns are silently dropped.

    Layer 2 — Entity Memory
        Scans every assistant reply for key facts (numbers, dates,
        named entities) and stores them persistently for the whole session.
        Even after turn 1 slides out of the window, facts from it survive.

    The system prompt is always pinned at index 0.
    Entity facts are appended to the system prompt on every to_messages() call.
    """

    def __init__(self, system_prompt: str = None, max_turns: int = 10):
        self.max_turns    = max_turns          # sliding window size
        self._system      = system_prompt
        self._turns: list[dict] = []
        self.entity_memory = EntityMemory()    # ← NEW: persistent fact store

    def set_system(self, prompt: str) -> None:
        self._system = prompt

    def add_user(self, content: str) -> None:
        self._turns.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: str) -> None:
        self._turns.append({"role": "assistant", "content": content})
        # ── Extract entities from every assistant reply ──
        self.entity_memory.update_from_text(content)

    def _trim(self) -> None:
        """
        Sliding window — drop oldest pairs when limit exceeded.
        Always trims in pairs (user+assistant) to keep history coherent.
        """
        max_messages = self.max_turns * 2
        if len(self._turns) > max_messages:
            self._turns = self._turns[-max_messages:]

    def to_messages(self) -> list[dict]:
        """
        Return full message list ready for the LLM API.

        Structure:
            [system + entity_facts]  ← always present
            [turn N-max ... turn N]  ← sliding window of recent turns
        """
        messages = []
        if self._system:
            # Append entity memory block to system prompt
            system_content = self._system + self.entity_memory.to_prompt_block()
            messages.append({"role": "system", "content": system_content})
        messages.extend(self._turns)
        return messages

    def clear(self) -> None:
        """Wipe turn history AND entity memory (full reset)."""
        self._turns = []
        self.entity_memory.clear()

    def clear_turns_only(self) -> None:
        """Wipe only turn history, preserve entity memory."""
        self._turns = []

    def __len__(self) -> int:
        return len(self._turns)


# ─────────────────────────────────────────
# BASE LLM
# ─────────────────────────────────────────

class BaseLLM:
    """
    Abstract base class for all LLM providers.
    Every LLM must implement generate() and stream().
    """

    def __init__(self):
        self.model_name = "base"

    def generate(self, prompt, system_prompt=None, history=None, **kwargs) -> dict:
        raise NotImplementedError("Subclasses must implement generate()")

    def stream(self, prompt, system_prompt=None, history=None, **kwargs):
        raise NotImplementedError("Subclasses must implement stream()")

    def get_info(self) -> dict:
        return {"model": self.model_name, "provider": "base"}


# ─────────────────────────────────────────
# GROQ LLM
# ─────────────────────────────────────────

class GroqLLM(BaseLLM):
    """
    Groq Cloud LLM — extremely fast inference via GroqChip.
    Requires GROQ_API_KEY in .env

    ✅ Very fast (LPU inference)  ✅ OpenAI-compatible API
    ✅ Free tier available        ❌ Needs internet + API key

    Default model : llama-3.3-70b-versatile
    Memory        : sliding window (max_turns=10) + entity memory
    """

    AVAILABLE_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile",
        "gemma2-9b-it",
    ]

    DEFAULT_SYSTEM = (
        "You are a helpful AI assistant. Answer questions clearly and concisely "
        "based on the context provided. If you don't know something, say so."
    )

    def __init__(
        self,
        model_name    : str   = GROQ_MODEL,
        system_prompt : str   = None,
        temperature   : float = 0.7,
        max_tokens    : int   = 1024,
        max_turns     : int   = 10,        # CHANGED: 20 → 10 (safer default)
    ):
        super().__init__()
        self.model_name    = model_name
        self.temperature   = temperature
        self.max_tokens    = max_tokens
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM

        print(f"  [GROQ] Initializing model: {model_name}")
        self.client  = Groq(api_key=GROQ_API_KEY)
        self.history = ChatHistory(
            system_prompt = self.system_prompt,
            max_turns     = max_turns,      # sliding window
        )
        print(f"  [GROQ] ✅ Ready! Memory: sliding window ({max_turns} turns) + entity")

    # ── HELPERS ──────────────────────────────

    def _build_messages(self, prompt, system_prompt=None, history=None) -> list[dict]:
        active_history = history or self.history
        if system_prompt:
            active_history.set_system(system_prompt)
        active_history.add_user(prompt)
        return active_history.to_messages()

    # ── GENERATE (blocking) ──────────────────

    def generate(
        self,
        prompt        : str,
        system_prompt : str         = None,
        history       : ChatHistory  = None,
        temperature   : float       = None,
        max_tokens    : int         = None,
    ) -> dict:
        messages = self._build_messages(prompt, system_prompt, history)

        response = self.client.chat.completions.create(
            model       = self.model_name,
            messages    = messages,
            temperature = temperature or self.temperature,
            max_tokens  = max_tokens  or self.max_tokens,
            stream      = False,
        )

        content        = response.choices[0].message.content
        active_history = history or self.history
        active_history.add_assistant(content)

        usage = {
            "prompt_tokens"    : response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens"     : response.usage.total_tokens,
        }
        return {"content": content, "model": self.model_name, "usage": usage}

    # ── STREAM (generator) ───────────────────

    def stream(
        self,
        prompt        : str,
        system_prompt : str         = None,
        history       : ChatHistory  = None,
        temperature   : float       = None,
        max_tokens    : int         = None,
    ):
        messages = self._build_messages(prompt, system_prompt, history)

        response_stream = self.client.chat.completions.create(
            model       = self.model_name,
            messages    = messages,
            temperature = temperature or self.temperature,
            max_tokens  = max_tokens  or self.max_tokens,
            stream      = True,
        )

        full_reply = []
        usage_data = None

        for chunk in response_stream:
            if hasattr(chunk, "usage") and chunk.usage is not None:
                usage_data = {
                    "prompt_tokens"    : chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens"     : chunk.usage.total_tokens,
                }
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                full_reply.append(delta.content)
                yield delta.content

        complete_reply = "".join(full_reply)
        active_history = history or self.history
        active_history.add_assistant(complete_reply)

        yield {"model": self.model_name, "usage": usage_data or {}}

    # ── CONVENIENCE ──────────────────────────

    def chat(self, message: str, **kwargs) -> str:
        return self.generate(message, **kwargs)["content"]

    def reset_history(self) -> None:
        """Full reset — clears turns AND entity memory."""
        self.history.clear()
        print(f"  [GROQ] Conversation history and entity memory cleared.")

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt
        self.history.set_system(prompt)

    def get_info(self) -> dict:
        return {
            "model"        : self.model_name,
            "provider"     : "groq-cloud",
            "temperature"  : self.temperature,
            "max_tokens"   : self.max_tokens,
            "history_len"  : len(self.history),
            "entity_count" : len(self.history.entity_memory),
        }


# ─────────────────────────────────────────
# LLM FACTORY
# ─────────────────────────────────────────

class LLMFactory:
    PROVIDERS: dict = {"groq": GroqLLM}

    @staticmethod
    def get(provider: str = "groq", **kwargs) -> BaseLLM:
        provider = provider.lower().strip()
        if provider not in LLMFactory.PROVIDERS:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Choose from: {list(LLMFactory.PROVIDERS.keys())}"
            )
        return LLMFactory.PROVIDERS[provider](**kwargs)

    @staticmethod
    def available_providers() -> list[str]:
        return list(LLMFactory.PROVIDERS.keys())


__all__ = ["EntityMemory", "ChatHistory", "BaseLLM", "GroqLLM", "LLMFactory"]