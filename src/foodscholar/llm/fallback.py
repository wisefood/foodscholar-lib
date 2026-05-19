"""FallbackLLMClient — an ordered chain of LLM clients with fail-through.

Calls the primary client; on any exception (timeout, rate limit, auth error,
service down) it logs and tries the next client in the chain. Raises only if
every client fails. Satisfies the `LLMClient` protocol, so it drops in
anywhere a single client would.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from foodscholar.logging import get_logger

if TYPE_CHECKING:
    from foodscholar.storage.protocols import LLMClient

_log = get_logger("foodscholar.llm.fallback")


class AllLLMClientsFailedError(RuntimeError):
    """Raised when every client in a FallbackLLMClient chain has failed."""


class FallbackLLMClient:
    """Tries each client in order; falls through to the next on any error.

    `model_id` reports the chain so artifacts record what was *attempted*; the
    actually-used client for a given call is logged at `info` level.
    """

    def __init__(self, clients: list[LLMClient]) -> None:
        if not clients:
            raise ValueError("FallbackLLMClient needs at least one client")
        self._clients = clients
        self.model_id = "fallback(" + ",".join(c.model_id for c in clients) + ")"

    @property
    def primary(self) -> LLMClient:
        return self._clients[0]

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        errors: list[str] = []
        for i, client in enumerate(self._clients):
            try:
                result = client.generate(prompt, max_tokens=max_tokens)
                if i > 0:
                    _log.info(
                        "llm.fallback_used",
                        used=client.model_id,
                        rank=i,
                        failed=errors,
                    )
                return result
            except Exception as e:
                msg = f"{client.model_id}: {type(e).__name__}: {e}"
                errors.append(msg)
                _log.warning("llm.client_failed", client=client.model_id, error=str(e))
        raise AllLLMClientsFailedError(
            "every LLM client in the fallback chain failed: " + " | ".join(errors)
        )
