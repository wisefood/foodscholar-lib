"""Concrete LLMClient adapters, one per provider.

Each class implements the `LLMClient` protocol: a `model_id` attribute, a
`generate(prompt, max_tokens) -> str` method, and a `generate_json(prompt,
schema, max_tokens) -> dict` method that returns schema-conforming JSON via the
provider's native structured-output mode where one exists.

SDKs are lazy-imported inside `__init__` so the core package never hard-depends
on them; install via the `[llm]` extra. API keys can be supplied either via
the explicit `api_key=` constructor argument (typically forwarded from
`cfg.llm.primary.api_key`) or via the provider's standard environment
variable. The explicit value wins when both are set.
"""

from __future__ import annotations

import json
import os
import re

# JSON sometimes arrives wrapped in ```json ... ``` fences or with leading
# prose. This pulls the first balanced {...} object out as a fallback.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _resolve_secret(explicit: str | None, env_var: str, provider: str) -> str:
    """Return the secret, preferring `explicit` then `os.environ[env_var]`."""
    if explicit:
        return explicit
    value = os.environ.get(env_var)
    if not value:
        raise RuntimeError(
            f"{provider} requires an API key. Provide it via cfg.llm.primary.api_key "
            f"(or .fallbacks[*].api_key) or set the {env_var} environment variable."
        )
    return value


def _parse_json_object(text: str) -> dict[str, object]:
    """Parse `text` as a JSON object, tolerating fences / surrounding prose."""
    text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_OBJECT_RE.search(text)
        if m is None:
            raise ValueError(f"LLM did not return JSON: {text[:200]!r}") from None
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError(f"LLM returned non-object JSON: {type(obj).__name__}")
    return obj


class AnthropicClient:
    """Anthropic Claude. Needs `ANTHROPIC_API_KEY` or an explicit `api_key`."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        *,
        api_key: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "the 'anthropic' package is required for AnthropicClient. "
                "Install with: pip install 'foodscholar[llm]'"
            ) from e
        self.model_id = model
        self._client = anthropic.Anthropic(
            api_key=_resolve_secret(api_key, "ANTHROPIC_API_KEY", "Anthropic"),
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

    def generate_json(
        self, prompt: str, schema: dict[str, object], max_tokens: int = 1024
    ) -> dict[str, object]:
        # Anthropic has no JSON-schema response mode; instruct + parse.
        full = (
            f"{prompt}\n\nRespond with ONLY a JSON object matching this schema "
            f"(no prose, no markdown fences):\n{json.dumps(schema)}"
        )
        return _parse_json_object(self.generate(full, max_tokens=max_tokens))


class OpenAIClient:
    """OpenAI. Needs `OPENAI_API_KEY` or an explicit `api_key`."""

    def __init__(
        self,
        model: str = "gpt-4.1",
        *,
        api_key: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "the 'openai' package is required for OpenAIClient. "
                "Install with: pip install 'foodscholar[llm]'"
            ) from e
        self.model_id = model
        self._client = openai.OpenAI(
            api_key=_resolve_secret(api_key, "OPENAI_API_KEY", "OpenAI"),
            timeout=timeout_s,
        )

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    def generate_json(
        self, prompt: str, schema: dict[str, object], max_tokens: int = 1024
    ) -> dict[str, object]:
        resp = self._client.chat.completions.create(
            model=self.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema, "strict": False},
            },
        )
        return _parse_json_object(resp.choices[0].message.content or "")


class GroqClient:
    """Groq (fast Llama/Mixtral inference). Needs `GROQ_API_KEY` or `api_key`."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        *,
        api_key: str | None = None,
        timeout_s: float = 30.0,
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
            api_key=_resolve_secret(api_key, "GROQ_API_KEY", "Groq"),
            timeout=timeout_s,
        )

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    def generate_json(
        self, prompt: str, schema: dict[str, object], max_tokens: int = 1024
    ) -> dict[str, object]:
        full = f"{prompt}\n\nRespond with JSON matching this schema:\n{json.dumps(schema)}"
        resp = self._client.chat.completions.create(
            model=self.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": full}],
            response_format={"type": "json_object"},
        )
        return _parse_json_object(resp.choices[0].message.content or "")


class GeminiClient:
    """Google Gemini. Needs `GEMINI_API_KEY` or an explicit `api_key`."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        *,
        api_key: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "the 'google-genai' package is required for GeminiClient. "
                "Install with: pip install 'foodscholar[llm]'"
            ) from e
        self.model_id = model
        self._client = genai.Client(
            api_key=_resolve_secret(api_key, "GEMINI_API_KEY", "Gemini")
        )

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        from google.genai import types

        resp = self._client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=max_tokens),
        )
        return resp.text or ""

    def generate_json(
        self, prompt: str, schema: dict[str, object], max_tokens: int = 1024
    ) -> dict[str, object]:
        from google.genai import types

        resp = self._client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        return _parse_json_object(resp.text or "")


class OllamaClient:
    """Local Ollama. No API key — needs a running Ollama daemon. Install `foodscholar[llm]`."""

    def __init__(
        self,
        model: str = "llama3.1",
        *,
        host: str = "http://localhost:11434",
        timeout_s: float = 60.0,
        api_key: str | None = None,  # accepted but ignored — daemon has no auth
    ) -> None:
        try:
            import ollama
        except ImportError as e:
            raise ImportError(
                "the 'ollama' package is required for OllamaClient. "
                "Install with: pip install 'foodscholar[llm]'"
            ) from e
        _ = api_key  # accepted for uniform constructor signature; ollama has no auth
        self.model_id = model
        self._client = ollama.Client(host=host, timeout=timeout_s)

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        resp = self._client.generate(
            model=self.model_id,
            prompt=prompt,
            options={"num_predict": max_tokens},
        )
        return resp.get("response", "")

    def generate_json(
        self, prompt: str, schema: dict[str, object], max_tokens: int = 1024
    ) -> dict[str, object]:
        resp = self._client.generate(
            model=self.model_id,
            prompt=prompt,
            format=schema,
            options={"num_predict": max_tokens},
        )
        return _parse_json_object(resp.get("response", ""))
