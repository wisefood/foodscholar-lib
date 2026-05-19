"""Concrete LLMClient adapters, one per provider.

Each class implements the `LLMClient` protocol — a `model_id` attribute and a
`generate(prompt, max_tokens) -> str` method. SDKs are lazy-imported inside
`__init__` so the core package never hard-depends on them; install them via
the `[llm]` extra. API keys come from environment variables, never config.

All adapters are deliberately thin: a single non-streaming text completion.
The linker's `llm_select` tier and Layer C card generation only need that.
"""

from __future__ import annotations

import os


def _require_env(var: str, provider: str) -> str:
    key = os.environ.get(var)
    if not key:
        raise RuntimeError(
            f"{provider} requires the {var} environment variable to be set."
        )
    return key


class AnthropicClient:
    """Anthropic Claude. Needs `ANTHROPIC_API_KEY`; install `foodscholar[llm]`."""

    def __init__(self, model: str = "claude-sonnet-4-6", *, timeout_s: float = 30.0) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "the 'anthropic' package is required for AnthropicClient. "
                "Install with: pip install 'foodscholar[llm]'"
            ) from e
        self.model_id = model
        self._client = anthropic.Anthropic(
            api_key=_require_env("ANTHROPIC_API_KEY", "Anthropic"),
            timeout=timeout_s,
        )

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        msg = self._client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        return "".join(parts)


class OpenAIClient:
    """OpenAI. Needs `OPENAI_API_KEY`; install `foodscholar[llm]`."""

    def __init__(self, model: str = "gpt-4.1", *, timeout_s: float = 30.0) -> None:
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "the 'openai' package is required for OpenAIClient. "
                "Install with: pip install 'foodscholar[llm]'"
            ) from e
        self.model_id = model
        self._client = openai.OpenAI(
            api_key=_require_env("OPENAI_API_KEY", "OpenAI"),
            timeout=timeout_s,
        )

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""


class GroqClient:
    """Groq (fast Llama/Mixtral inference). Needs `GROQ_API_KEY`; install `foodscholar[llm]`."""

    def __init__(
        self, model: str = "llama-3.3-70b-versatile", *, timeout_s: float = 30.0
    ) -> None:
        try:
            import groq
        except ImportError as e:
            raise ImportError(
                "the 'groq' package is required for GroqClient. "
                "Install with: pip install 'foodscholar[llm]'"
            ) from e
        self.model_id = model
        self._client = groq.Groq(
            api_key=_require_env("GROQ_API_KEY", "Groq"),
            timeout=timeout_s,
        )

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""


class GeminiClient:
    """Google Gemini. Needs `GEMINI_API_KEY`; install `foodscholar[llm]`."""

    def __init__(self, model: str = "gemini-2.0-flash", *, timeout_s: float = 30.0) -> None:
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "the 'google-genai' package is required for GeminiClient. "
                "Install with: pip install 'foodscholar[llm]'"
            ) from e
        self.model_id = model
        self._client = genai.Client(api_key=_require_env("GEMINI_API_KEY", "Gemini"))

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        from google.genai import types

        resp = self._client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=max_tokens),
        )
        return resp.text or ""


class OllamaClient:
    """Local Ollama. No API key — needs a running Ollama daemon. Install `foodscholar[llm]`."""

    def __init__(
        self,
        model: str = "llama3.1",
        *,
        host: str = "http://localhost:11434",
        timeout_s: float = 60.0,
    ) -> None:
        try:
            import ollama
        except ImportError as e:
            raise ImportError(
                "the 'ollama' package is required for OllamaClient. "
                "Install with: pip install 'foodscholar[llm]'"
            ) from e
        self.model_id = model
        self._client = ollama.Client(host=host, timeout=timeout_s)

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        resp = self._client.generate(
            model=self.model_id,
            prompt=prompt,
            options={"num_predict": max_tokens},
        )
        return resp.get("response", "")
