# generation/ollama_llm.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ollama
from generation.groq_llm import BaseLLM, ChatHistory, LLMFactory
from config import OLLAMA_MODEL


class OllamaLLM(BaseLLM):
    """
    Local Ollama LLM — runs fully offline via Ollama daemon.
    Requires: `ollama serve` running + model pulled locally.

    ✅ Free  ✅ Private  ✅ No internet needed
    ❌ Slower than Groq  ❌ Needs Ollama running

    Default model : llama3.2  (set in config.py)
    Features      : streaming, chat history, system prompt, token tracking
    """

    DEFAULT_SYSTEM = (
        "You are a helpful AI assistant. Answer questions clearly and concisely "
        "based on the context provided. If you don't know something, say so."
    )

    def __init__(
        self,
        model_name    : str   = OLLAMA_MODEL,
        system_prompt : str   = None,
        temperature   : float = 0.7,
        max_tokens    : int   = 1024,
        max_turns     : int   = 20,
    ):
        super().__init__()
        self.model_name    = model_name
        self.temperature   = temperature
        self.max_tokens    = max_tokens
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM

        print(f"  [OLLAMA] Initializing model: {model_name}")
        self._check_model()
        self.history = ChatHistory(
            system_prompt = self.system_prompt,
            max_turns     = max_turns
        )
        print(f"  [OLLAMA] ✅ Ready!")

    # ── SETUP ────────────────────────────────

    def _check_model(self) -> None:
        """Verify the model is pulled locally. Print warning if not found."""
        try:
            local_models = [m.model for m in ollama.list().models]
            # model names may have :latest suffix
            names = [m.split(":")[0] for m in local_models]
            if self.model_name.split(":")[0] not in names:
                print(
                    f"  [OLLAMA] ⚠️  Model '{self.model_name}' not found locally. "
                    f"Run: ollama pull {self.model_name}"
                )
            else:
                print(f"  [OLLAMA] Model found locally ✅")
        except Exception as e:
            print(f"  [OLLAMA] ⚠️  Could not connect to Ollama daemon: {e}")
            print(f"  [OLLAMA]    Make sure `ollama serve` is running.")

    # ── HELPERS ──────────────────────────────

    def _build_messages(
        self,
        prompt        : str,
        system_prompt : str         = None,
        history       : ChatHistory = None,
    ) -> list[dict]:
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
        history       : ChatHistory = None,
        temperature   : float       = None,
        max_tokens    : int         = None,
    ) -> dict:
        """
        Blocking generation — waits for full response.

        Returns:
            {
                "content" : str,
                "model"   : str,
                "usage"   : {
                    "prompt_tokens"    : int,
                    "completion_tokens": int,
                    "total_tokens"     : int,
                }
            }
        """
        messages = self._build_messages(prompt, system_prompt, history)

        response = ollama.chat(
            model   = self.model_name,
            messages= messages,
            options = {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens  or self.max_tokens,
            },
            stream  = False,
        )

        content = response.message.content

        active_history = history or self.history
        active_history.add_assistant(content)

        usage = {
            "prompt_tokens"    : response.prompt_eval_count or 0,
            "completion_tokens": response.eval_count        or 0,
            "total_tokens"     : (response.prompt_eval_count or 0)
                                + (response.eval_count        or 0),
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
        system_prompt : str         = None,
        history       : ChatHistory = None,
        temperature   : float       = None,
        max_tokens    : int         = None,
    ):
        """
        Streaming generation — yields text chunks as they arrive.

        Yields:
            str  — each text chunk
            dict — final item: { "usage": {...}, "model": str }
        """
        messages = self._build_messages(prompt, system_prompt, history)

        response_stream = ollama.chat(
            model   = self.model_name,
            messages= messages,
            options = {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens  or self.max_tokens,
            },
            stream  = True,
        )

        full_reply = []
        usage_data = {}

        for chunk in response_stream:
            text = chunk.message.content
            if text:
                full_reply.append(text)
                yield text

            # Final chunk carries usage stats
            if chunk.done:
                usage_data = {
                    "prompt_tokens"    : chunk.prompt_eval_count or 0,
                    "completion_tokens": chunk.eval_count        or 0,
                    "total_tokens"     : (chunk.prompt_eval_count or 0)
                                       + (chunk.eval_count        or 0),
                }

        active_history = history or self.history
        active_history.add_assistant("".join(full_reply))

        yield {
            "model": self.model_name,
            "usage": usage_data,
        }

    # ── CONVENIENCE ──────────────────────────

    def chat(self, message: str, **kwargs) -> str:
        """Simple one-liner: send message, get back just the text."""
        return self.generate(message, **kwargs)["content"]

    def reset_history(self) -> None:
        self.history.clear()
        print(f"  [OLLAMA] Conversation history cleared.")

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt
        self.history.set_system(prompt)

    def get_info(self) -> dict:
        return {
            "model"      : self.model_name,
            "provider"   : "ollama-local",
            "temperature": self.temperature,
            "max_tokens" : self.max_tokens,
            "history_len": len(self.history),
        }


# ─────────────────────────────────────────
# REGISTER INTO FACTORY
# ─────────────────────────────────────────

# Plug OllamaLLM into the shared factory so both providers
# are accessible from one place: LLMFactory.get("ollama")
LLMFactory.PROVIDERS["ollama"] = OllamaLLM


__all__ = ["OllamaLLM"]