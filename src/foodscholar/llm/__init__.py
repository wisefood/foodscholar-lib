"""Pluggable LLM clients.

A provider-agnostic `LLMClient` layer: one thin adapter per provider, all
implementing the `foodscholar.storage.protocols.LLMClient` protocol, plus a
`FallbackLLMClient` that chains them. Construction is YAML-driven via
`build_llm(cfg.llm)`.

Providers: Anthropic, OpenAI, OpenRouter, Gemini, Groq, Ollama. Each lazy-imports
its SDK (gated by the `[llm]` extra) and reads its API key from the environment —
never from config files.
"""

from foodscholar.llm.factory import PROVIDERS, build_llm
from foodscholar.llm.fallback import FallbackLLMClient
from foodscholar.llm.providers import (
    AnthropicClient,
    GeminiClient,
    GroqClient,
    OllamaClient,
    OpenAIClient,
    OpenRouterClient,
)

__all__ = [
    "PROVIDERS",
    "AnthropicClient",
    "FallbackLLMClient",
    "GeminiClient",
    "GroqClient",
    "OllamaClient",
    "OpenAIClient",
    "OpenRouterClient",
    "build_llm",
]
