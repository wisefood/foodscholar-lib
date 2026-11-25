"""Gold statements: the hand-extracted evidence every chunker is scored against.

The neutral format is one JSON object per line with ``statement_id``, ``text``
and, when known, ``source_id``, ``page`` (1-based physical PDF page),
``section`` and free ``qualifiers``. :func:`load_fskg_statements` reads the FSKG
pilot workbooks into it; another domain's gold set needs only another loader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from foodscholar.chunking._xlsx import read_xlsx, records
from foodscholar.chunking.registry import SourceRegistry


class GoldStatement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement_id: str
    text: str
    source_id: str | None = None
    page: int | None = None
    """1-based physical PDF page."""
    printed_page: str | None = None
    section: str | None = None
    statement_type: str | None = None
    qualifiers: dict[str, Any] = Field(default_factory=dict)
    """Population, quantity, action… whatever the gold set records."""


_FSKG_QUALIFIERS = ("Dimension", "Action", "Quantity", "Population", "Food entities")


def load_fskg_statements(path: str | Path, registry: SourceRegistry | None = None) -> list[GoldStatement]:
    """Read the ``Statements`` sheet of an FSKG pilot extraction workbook.

    Statements are mapped to sources through the registry's statement-id
    ranges; without a registry, ``source_id`` stays None and scoring falls back
    to matching against every chunk.
    """
    sheets = read_xlsx(path)
    if "Statements" not in sheets:
        raise ValueError(f"{path}: no 'Statements' sheet (found {sorted(sheets)})")
    out: list[GoldStatement] = []
    for row in records(sheets["Statements"]):
        sid = str(row.get("Statement ID") or "").strip()
        text = str(row.get("Statement text") or "").strip()
        if not sid or not text:
            continue
        page = row.get("Page (PDF)")
        out.append(
            GoldStatement(
                statement_id=sid,
                text=text,
                source_id=registry.source_for_statement(sid) if registry else None,
                page=int(page) if isinstance(page, (int, float)) else _first_int(page),
                printed_page=str(row["Page (printed)"]) if row.get("Page (printed)") not in (None, "") else None,
                section=str(row.get("Section") or "").strip() or None,
                statement_type=str(row.get("Statement type") or "").strip() or None,
                qualifiers={k: row[k] for k in _FSKG_QUALIFIERS if row.get(k) not in (None, "", "n/a")},
            )
        )
    return out


def _first_int(value: Any) -> int | None:
    digits = "".join(ch if ch.isdigit() else " " for ch in str(value or "")).split()
    return int(digits[0]) if digits else None


def load_gold_jsonl(path: str | Path) -> list[GoldStatement]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [GoldStatement.model_validate_json(line) for line in lines if line.strip()]


def save_gold_jsonl(statements: list[GoldStatement], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(s.model_dump_json() + "\n" for s in statements), encoding="utf-8")
    return out


def gold_to_dicts(statements: list[GoldStatement]) -> list[dict[str, Any]]:
    return [json.loads(s.model_dump_json()) for s in statements]
