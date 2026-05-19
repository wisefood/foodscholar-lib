"""Tests for the pluggable LLM layer: FallbackLLMClient + build_llm factory.

Provider adapters (Anthropic/OpenAI/Groq/Gemini/Ollama) are thin SDK wrappers;
their SDKs aren't installed in the default test env, so we test the parts that
carry logic — the fallback chain and the factory — with fake clients, plus the
adapters' lazy-import error path.
"""

from __future__ import annotations

import pytest

from foodscholar.config import LLMConfig, ProviderConfig
from foodscholar.llm import build_llm
from foodscholar.llm.factory import PROVIDERS
from foodscholar.llm.fallback import AllLLMClientsFailedError, FallbackLLMClient
from foodscholar.storage.protocols import LLMClient


class _OKClient:
    def __init__(self, model_id: str, reply: str = "ok") -> None:
        self.model_id = model_id
        self._reply = reply
        self.calls = 0

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        self.calls += 1
        return self._reply

    def generate_json(self, prompt, schema, max_tokens=1024):  # type: ignore[no-untyped-def]
        self.calls += 1
        return {"from": self.model_id}


class _FailClient:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.calls = 0

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        self.calls += 1
        raise RuntimeError(f"{self.model_id} is down")

    def generate_json(self, prompt, schema, max_tokens=1024):  # type: ignore[no-untyped-def]
        self.calls += 1
        raise RuntimeError(f"{self.model_id} is down")


# ------------------------------------------------------------ FallbackLLMClient


def test_fallback_implements_protocol() -> None:
    fb = FallbackLLMClient([_OKClient("a")])
    assert isinstance(fb, LLMClient)


def test_fallback_requires_at_least_one_client() -> None:
    with pytest.raises(ValueError, match="at least one"):
        FallbackLLMClient([])


def test_fallback_uses_primary_when_it_works() -> None:
    primary = _OKClient("primary", reply="from-primary")
    secondary = _OKClient("secondary", reply="from-secondary")
    fb = FallbackLLMClient([primary, secondary])
    assert fb.generate("hi") == "from-primary"
    assert primary.calls == 1
    assert secondary.calls == 0


def test_fallback_falls_through_on_error() -> None:
    primary = _FailClient("primary")
    secondary = _OKClient("secondary", reply="rescued")
    fb = FallbackLLMClient([primary, secondary])
    assert fb.generate("hi") == "rescued"
    assert primary.calls == 1
    assert secondary.calls == 1


def test_fallback_tries_all_in_order() -> None:
    a, b = _FailClient("a"), _FailClient("b")
    c = _OKClient("c", reply="third-time-lucky")
    fb = FallbackLLMClient([a, b, c])
    assert fb.generate("hi") == "third-time-lucky"
    assert a.calls == b.calls == c.calls == 1


def test_fallback_raises_when_all_fail() -> None:
    fb = FallbackLLMClient([_FailClient("a"), _FailClient("b")])
    with pytest.raises(AllLLMClientsFailedError, match="every LLM client"):
        fb.generate("hi")


def test_fallback_model_id_describes_chain() -> None:
    fb = FallbackLLMClient([_OKClient("groq-llama"), _OKClient("ollama-llama")])
    assert fb.model_id == "fallback(groq-llama,ollama-llama)"


def test_fallback_primary_property() -> None:
    primary = _OKClient("p")
    fb = FallbackLLMClient([primary, _OKClient("s")])
    assert fb.primary is primary


# ------------------------------------------------- FallbackLLMClient.generate_json


def test_fallback_generate_json_uses_primary() -> None:
    fb = FallbackLLMClient([_OKClient("primary"), _OKClient("secondary")])
    assert fb.generate_json("p", {}) == {"from": "primary"}


def test_fallback_generate_json_falls_through() -> None:
    fb = FallbackLLMClient([_FailClient("primary"), _OKClient("secondary")])
    assert fb.generate_json("p", {}) == {"from": "secondary"}


def test_fallback_generate_json_raises_when_all_fail() -> None:
    fb = FallbackLLMClient([_FailClient("a"), _FailClient("b")])
    with pytest.raises(AllLLMClientsFailedError, match="generate_json"):
        fb.generate_json("p", {})


# --------------------------------------------------------------- build_llm


def test_provider_registry_has_all_five() -> None:
    assert set(PROVIDERS) == {"anthropic", "openai", "groq", "gemini", "ollama"}


def test_build_llm_unknown_provider_in_factory() -> None:
    # ProviderConfig itself validates the literal, but the factory's guard is
    # the last line of defence — exercise it directly with a forced value.
    from foodscholar.llm.factory import _build_one

    class _FakeSpec:
        provider = "no-such-provider"
        model = "x"
        host = None

    with pytest.raises(ValueError, match="unknown LLM provider"):
        _build_one(_FakeSpec(), timeout_s=10.0)  # type: ignore[arg-type]


def test_build_llm_missing_sdk_raises_import_error() -> None:
    """With no LLM SDKs installed, building a real provider should raise a
    clear ImportError pointing at the [llm] extra — not an obscure failure."""
    cfg = LLMConfig(primary=ProviderConfig(provider="groq", model="llama-3.3-70b-versatile"))
    try:
        import groq  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match=r"foodscholar\[llm\]"):
            build_llm(cfg)
    else:
        pytest.skip("groq SDK is installed — ImportError path not exercised")


def test_provider_config_rejects_unknown_provider() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ProviderConfig(provider="mystery", model="x")  # type: ignore[arg-type]


def test_llm_config_defaults() -> None:
    cfg = LLMConfig(primary=ProviderConfig(provider="ollama", model="llama3.1"))
    assert cfg.fallbacks == []
    assert cfg.timeout_s == 30.0
    assert cfg.max_retries == 2


# ---------------------------------------------------------- _parse_json_object


def test_parse_json_object_plain() -> None:
    from foodscholar.llm.providers import _parse_json_object

    assert _parse_json_object('{"a": 1}') == {"a": 1}


def test_parse_json_object_strips_code_fences() -> None:
    from foodscholar.llm.providers import _parse_json_object

    fenced = '```json\n{"a": 1}\n```'
    assert _parse_json_object(fenced) == {"a": 1}


def test_parse_json_object_extracts_from_prose() -> None:
    from foodscholar.llm.providers import _parse_json_object

    noisy = 'Here is the result: {"a": 1} — hope that helps!'
    assert _parse_json_object(noisy) == {"a": 1}


def test_parse_json_object_rejects_non_object() -> None:
    from foodscholar.llm.providers import _parse_json_object

    with pytest.raises(ValueError, match="non-object"):
        _parse_json_object("[1, 2, 3]")


def test_parse_json_object_rejects_garbage() -> None:
    from foodscholar.llm.providers import _parse_json_object

    with pytest.raises(ValueError, match="did not return JSON"):
        _parse_json_object("no json here at all")
