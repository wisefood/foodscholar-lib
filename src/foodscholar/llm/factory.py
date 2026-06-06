"""Construct an LLMClient (with fallback chain) from configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from foodscholar.llm.fallback import FallbackLLMClient
from foodscholar.llm.providers import (
    AnthropicClient,
    GeminiClient,
    GroqClient,
    OllamaClient,
    OpenAIClient,
    OpenRouterClient,
)

if TYPE_CHECKING:
    from foodscholar.config import LLMConfig, ProviderConfig
    from foodscholar.storage.protocols import LLMClient

# Provider name → adapter class. Used by build_llm and by config validation.
PROVIDERS = {
    "anthropic": AnthropicClient,
    "openai": OpenAIClient,
    "openrouter": OpenRouterClient,
    "groq": GroqClient,
    "gemini": GeminiClient,
    "ollama": OllamaClient,
}


def _build_one(spec: ProviderConfig, *, timeout_s: float) -> LLMClient:
    cls = PROVIDERS.get(spec.provider)
    if cls is None:
        raise ValueError(
            f"unknown LLM provider {spec.provider!r}; "
            f"expected one of {sorted(PROVIDERS)}"
        )
    kwargs: dict[str, object] = {"timeout_s": timeout_s, "api_key": spec.api_key}
    if spec.host:
        # `host` overloads to the ollama daemon URL and the openrouter base_url.
        if spec.provider == "ollama":
            kwargs["host"] = spec.host
        elif spec.provider == "openrouter":
            kwargs["base_url"] = spec.host
    return cls(spec.model, **kwargs)  # type: ignore[arg-type]


def build_llm(config: LLMConfig) -> LLMClient:
    """Build the LLM client described by `cfg.llm`.

    Returns the primary client directly when there are no fallbacks, otherwise
    a `FallbackLLMClient` chaining primary → fallbacks in order. Adapter
    construction is lazy in its SDK import but eager in client setup, so a
    misconfigured provider fails fast here rather than mid-pipeline.
    """
    primary = _build_one(config.primary, timeout_s=config.timeout_s)
    if not config.fallbacks:
        return primary
    chain: list[LLMClient] = [primary]
    chain.extend(_build_one(fb, timeout_s=config.timeout_s) for fb in config.fallbacks)
    return FallbackLLMClient(chain)
