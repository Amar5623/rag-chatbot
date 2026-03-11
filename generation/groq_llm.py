# generation/groq_llm.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL


# ─────────────────────────────────────────
# MESSAGE HISTORY HELPER
# ─────────────────────────────────────────

class ChatHistory:
    """
    Manages a rolling conversation history.
    Keeps system prompt pinned at index 0.
    Optionally caps history to avoid exceeding context window.
    """

    def __init__(self, system_prompt: str = None, max_turns: int = 20):
        self.max_turns    = max_turns          # max user+assistant pairs to keep
        self._system      = system_prompt
        self._turns: list[dict] = []           # only user/assistant messages

    def set_system(self, prompt: str) -> None:
        self._system = prompt

    def add_user(self, content: str) -> None:
        self._turns.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: str) -> None:
        self._turns.append({"role": "assistant", "content": content})

    def _trim(self) -> None:
        """Keep only the last max_turns * 2 messages (pairs)."""
        max_messages = self.max_turns * 2
        if len(self._turns) > max_messages:
            self._turns = self._turns[-max_messages:]

    def to_messages(self) -> list[dict]:
        """Return full message list ready for the API."""
        messages = []
        if self._system:
            messages.append({"role": "system", "content": self._system})
        messages.extend(self._turns)
        return messages

    def clear(self) -> None:
        """Wipe turn history but keep system prompt."""
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

    def generate(
        self,
        prompt        : str,
        system_prompt : str        = None,
        history       : ChatHistory = None,
        **kwargs
    ) -> dict:
        """
        Single-turn or multi-turn generation.
        Returns: { "content": str, "usage": dict, "model": str }
        """
        raise NotImplementedError("Subclasses must implement generate()")

    def stream(
        self,
        prompt        : str,
        system_prompt : str        = None,
        history       : ChatHistory = None,
        **kwargs
    ):
        """
        Stream response tokens as a generator.
        Yields str chunks; last item is a dict with usage stats.
        """
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
    Features      : streaming, chat history, system prompt, token tracking
    """

    # Models available on Groq (as of 2025)
    AVAILABLE_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
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
        max_turns     : int   = 20,
    ):
        super().__init__()
        self.model_name   = model_name
        self.temperature  = temperature
        self.max_tokens   = max_tokens
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM

        print(f"  [GROQ] Initializing model: {model_name}")
        self.client  = Groq(api_key=GROQ_API_KEY)
        self.history = ChatHistory(
            system_prompt = self.system_prompt,
            max_turns     = max_turns
        )
        print(f"  [GROQ] ✅ Ready!")

    # ── HELPERS ──────────────────────────────

    def _build_messages(
        self,
        prompt        : str,
        system_prompt : str        = None,
        history       : ChatHistory = None,
    ) -> list[dict]:
        """
        Build the messages list for the API call.
        Priority: explicit history arg > self.history > fresh context
        """
        active_history = history or self.history

        # Override system prompt if caller provided one
        if system_prompt:
            active_history.set_system(system_prompt)

        active_history.add_user(prompt)
        return active_history.to_messages()

    # ── GENERATE (blocking) ──────────────────

    def generate(
        self,
        prompt        : str,
        system_prompt : str        = None,
        history       : ChatHistory = None,
        temperature   : float      = None,
        max_tokens    : int        = None,
    ) -> dict:
        """
        Blocking generation — waits for full response.

        Returns:
            {
                "content" : str,         ← the reply text
                "model"   : str,         ← model used
                "usage"   : {
                    "prompt_tokens"    : int,
                    "completion_tokens": int,
                    "total_tokens"     : int,
                }
            }
        """
        messages = self._build_messages(prompt, system_prompt, history)

        response = self.client.chat.completions.create(
            model       = self.model_name,
            messages    = messages,
            temperature = temperature or self.temperature,
            max_tokens  = max_tokens  or self.max_tokens,
            stream      = False,
        )

        content = response.choices[0].message.content

        # Save assistant reply into history
        active_history = history or self.history
        active_history.add_assistant(content)

        usage = {
            "prompt_tokens"    : response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens"     : response.usage.total_tokens,
        }

        return {
            "content": content,
            "model"  : self.model_name,
            "usage"  : usage,
        }

    # ── STREAM (generator) ───────────────────

    def stream(
        self,
        prompt        : str,
        system_prompt : str        = None,
        history       : ChatHistory = None,
        temperature   : float      = None,
        max_tokens    : int        = None,
    ):
        """
        Streaming generation — yields text chunks as they arrive.

        Usage:
            for chunk in llm.stream("Tell me about Q1 revenue"):
                if isinstance(chunk, str):
                    print(chunk, end="", flush=True)
                else:
                    print("\\nUsage:", chunk["usage"])  # final dict

        Yields:
            str  — each text token/chunk
            dict — final item: { "usage": {...}, "model": str }
        """
        messages = self._build_messages(prompt, system_prompt, history)

        response_stream = self.client.chat.completions.create(
            model       = self.model_name,
            messages    = messages,
            temperature = temperature or self.temperature,
            max_tokens  = max_tokens  or self.max_tokens,
            stream      = True,
        )

        full_reply    = []
        usage_data    = None

        for chunk in response_stream:
            # Token usage arrives in the final chunk
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

        # Save complete reply into history after stream ends
        complete_reply = "".join(full_reply)
        active_history = history or self.history
        active_history.add_assistant(complete_reply)

        # Yield final metadata dict
        yield {
            "model": self.model_name,
            "usage": usage_data or {},
        }

    # ── CONVENIENCE ──────────────────────────

    def chat(self, message: str, **kwargs) -> str:
        """Simple one-liner: send message, get back just the text."""
        return self.generate(message, **kwargs)["content"]

    def reset_history(self) -> None:
        """Clear conversation history, keep system prompt."""
        self.history.clear()
        print(f"  [GROQ] Conversation history cleared.")

    def set_system_prompt(self, prompt: str) -> None:
        """Update system prompt mid-conversation."""
        self.system_prompt = prompt
        self.history.set_system(prompt)

    def get_info(self) -> dict:
        return {
            "model"      : self.model_name,
            "provider"   : "groq-cloud",
            "temperature": self.temperature,
            "max_tokens" : self.max_tokens,
            "history_len": len(self.history),
        }


# ─────────────────────────────────────────
# LLM FACTORY
# ─────────────────────────────────────────

class LLMFactory:
    """
    Returns the right LLM based on provider name.
    Mirrors EmbedderFactory for consistency.

    Usage:
        llm = LLMFactory.get("groq")
        llm = LLMFactory.get("groq", model_name="llama-3.1-8b-instant", temperature=0.3)
    """

    PROVIDERS: dict[str, type[BaseLLM]] = {
        "groq": GroqLLM,
        # "ollama": OllamaLLM,   ← will be added in ollama_llm.py
    }

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


__all__ = [
    "ChatHistory",
    "BaseLLM",
    "GroqLLM",
    "LLMFactory",
]