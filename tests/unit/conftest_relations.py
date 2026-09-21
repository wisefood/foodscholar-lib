"""Shared stubs for the relations tests (imported, not auto-collected)."""

from __future__ import annotations

from foodscholar.io.chunk import EntityLink


class StubLLM:
    """Returns canned JSON per call, in order. Records the schemas it was given."""

    model_id = "stub-llm"

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.schemas: list[dict] = []
        self.prompts: list[str] = []

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        return ""

    def generate_json(self, prompt, schema, max_tokens=1024):
        self.schemas.append(schema)
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("StubLLM ran out of canned responses")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class RaisingLLM:
    model_id = "raising-llm"

    def __init__(self, error: Exception, *, fail_on: int = 0) -> None:
        self._error = error
        self._fail_on = fail_on
        self.calls = 0

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        return ""

    def generate_json(self, prompt, schema, max_tokens=1024):
        current = self.calls
        self.calls += 1
        if current == self._fail_on:
            raise self._error
        return {"entities": ["alpha", "beta"]}


class StubLinker:
    """`surface.lower() -> (ontology_id, confidence)`; counts batch calls."""

    linker_id = "stub-linker"

    def __init__(self, table: dict[str, tuple[str, float]]) -> None:
        self.table = table
        self.batch_calls = 0

    def link(self, mention):
        hit = self.table.get(mention.text.lower())
        if hit is None:
            return None
        return EntityLink(
            mention=mention,
            ontology_id=hit[0],
            confidence=hit[1],
            method="dense",
            linker_version=self.linker_id,
        )

    def link_many(self, mentions):
        self.batch_calls += 1
        return [self.link(m) for m in mentions]
