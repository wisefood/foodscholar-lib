"""Prompt templates for relation extraction.

`entities.txt` and `relations.txt` are copied **verbatim** from
kggen_extended: they are the benchmarked artifact, so treat them as fixtures
rather than prose to improve.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

PROMPT_VERSION = "kggen-relations-v1"

_DIR = Path(__file__).parent


@cache
def _load(name: str) -> str:
    return (_DIR / name).read_text(encoding="utf-8")


def entities_system_prompt() -> str:
    return _load("entities.txt")


def relations_system_prompt() -> str:
    return _load("relations.txt")


def entities_prompt(text: str) -> str:
    return (
        f"{entities_system_prompt()}\n\n"
        "Here is the text to extract entities from:\n\n"
        f"<article>\n{text}\n</article>\n"
    )


def relations_prompt(text: str, entities: list[str]) -> str:
    entity_block = "\n".join(f"- {e}" for e in entities)
    return (
        f"{relations_system_prompt()}\n\n"
        "Here is the list of entities that were previously extracted from the "
        "source text:\n\n"
        f"<entities>\n{entity_block}\n</entities>\n\n"
        "Here is the source text to analyze:\n\n"
        f"<text>\n{text}\n</text>\n"
    )


__all__ = [
    "PROMPT_VERSION",
    "entities_prompt",
    "entities_system_prompt",
    "relations_prompt",
    "relations_system_prompt",
]
