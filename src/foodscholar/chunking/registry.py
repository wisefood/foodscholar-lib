"""The source registry: document-level metadata, from a catalogue, once.

Heads get titles, licences, languages, content page ranges and printed-page
offsets from here instead of from per-run CSVs and inline page sets. The core
schema is domain-free; anything else a catalogue carries (country, population
group, issuer, fiscal year…) is kept verbatim in :attr:`SourceRecord.extra`.

Loaders:

- :meth:`SourceRegistry.from_xlsx` reads the FSKG catalogue
  (``FSKG_Sources_Extracted.xlsx``): one sheet per source family, a header row,
  free-text page notes such as *"book content on PDF pages 43-1190; printed
  page = PDF page - 42"* parsed into numbers;
- :meth:`SourceRegistry.from_records` takes plain dicts with the same column
  names, so a CSV or JSON catalogue for another domain needs no new code.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from foodscholar.chunking._xlsx import read_xlsx, records

# Catalogue column → field. Matching is case-insensitive.
_COLUMNS = {
    "source id": "source_id",
    "title": "title",
    "short name": "short_name",
    "language": "language",
    "license id": "licence_id",
    "licence id": "licence_id",
    "license summary": "licence_summary",
    "licence summary": "licence_summary",
    "local copy": "local_copy",
    "statement id range": "statement_range",
    "url": "url",
    "type": "catalogue_type",
    "source type": "source_type",
}

NUTRITION_TYPE_RULES: tuple[tuple[str, str], ...] = (
    ("SRC-TB-", "textbook"),
    ("SRC-JN-", "abstract"),
    ("SRC-", "guide"),
)
"""Source-id prefix → source type, first match wins."""

_LANGUAGE_CODES = {
    "english": "en", "hungarian": "hu", "slovene": "sl", "slovenian": "sl",
    "german": "de", "french": "fr", "spanish": "es", "italian": "it", "greek": "el",
}
_EN_FILE_HINTS = ("angol", "english", "-en.", "_en.", "-en-", "_en_", "-en_", "_en-")

_CONTENT_RE = re.compile(r"content(?:\s+on)?\s+PDF(?:\s+pages)?\s+(\d+)\s*[-–]\s*(\d+)", re.I)  # noqa: RUF001
_OFFSET_RE = re.compile(r"printed\s+(?:page|number)\s*=\s*PDF\s+page(?:\s*([-+−])\s*(\d+))?", re.I)  # noqa: RUF001
_FOOTER_RE = re.compile(r"footer\s+numbers\s+(\d+)\s*[-–]\s*\d+\s+on\s+PDF\s+pages\s+(\d+)", re.I)  # noqa: RUF001
_NO_PRINTED_RE = re.compile(r"no\s+printed\s+(?:page\s+)?numbers", re.I)
_RANGE_RE = re.compile(r"^\s*([A-Za-z]+)(\d+)\s*[-–—]\s*([A-Za-z]+)(\d+)\s*$")  # noqa: RUF001


def _norm_name(text: str) -> str:
    """NFC + casefold: macOS and some zips store accented names decomposed."""
    return unicodedata.normalize("NFC", text).casefold()


class SourceRecord(BaseModel):
    """One catalogued source document (or edition pair)."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    source_type: str
    title: str | None = None
    short_name: str | None = None
    languages: list[str] = Field(default_factory=list)
    """ISO 639-1 codes, in catalogue order (``["hu", "en"]`` for a bilingual pair)."""
    licence_id: str | None = None
    licence_summary: str | None = None
    local_copies: list[str] = Field(default_factory=list)
    """Paths as catalogued, relative to the catalogue's root (``Sources/…``)."""
    content_pages: tuple[int, int] | None = None
    """Physical pages holding the body of the book, when catalogued."""
    printed_page_offset: int | None = None
    """``printed = physical + offset``; None when unknown."""
    has_printed_page_numbers: bool | None = None
    statement_range: tuple[str, int, int] | None = None
    """``(prefix, first, last)`` of the gold statements extracted from this source."""
    url: str | None = None
    page_note: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_edition_pair(self) -> bool:
        return len(self.local_copies) > 1

    def printed_label(self, page: int) -> str | None:
        """The printed page label for a physical page, or None — never invented."""
        if self.has_printed_page_numbers is False or self.printed_page_offset is None:
            return None
        printed = page + self.printed_page_offset
        return str(printed) if printed >= 1 else None

    def in_content(self, page: int) -> bool:
        if self.content_pages is None:
            return True
        lo, hi = self.content_pages
        return lo <= page <= hi

    def language_for(self, filename: str) -> str | None:
        """Which language a local copy is in. Bilingual pairs are told apart by filename."""
        if len(self.languages) <= 1:
            return self.languages[0] if self.languages else None
        name = _norm_name(PurePosixPath(filename).name)
        if "en" in self.languages and any(h in name for h in _EN_FILE_HINTS):
            return "en"
        return next((lang for lang in self.languages if lang != "en"), self.languages[0])


class SourceRegistry:
    """Lookup by source id, by file path and by gold-statement id."""

    def __init__(self, records_: Iterable[SourceRecord]) -> None:
        self.records: dict[str, SourceRecord] = {}
        for r in records_:
            if r.source_id in self.records:
                raise ValueError(f"duplicate source id {r.source_id!r} in registry")
            self.records[r.source_id] = r
        self._by_path: dict[str, str] = {}
        self._by_name: dict[str, str] = {}
        self._by_dir: dict[str, set[str]] = {}
        for r in self.records.values():
            for copy in r.local_copies:
                self._by_path[_norm_name(copy)] = r.source_id
                self._by_name.setdefault(_norm_name(PurePosixPath(copy).name), r.source_id)
                parent = str(PurePosixPath(_norm_name(copy)).parent)
                if parent not in ("", "."):
                    self._by_dir.setdefault(parent, set()).add(r.source_id)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.records.values())

    def get(self, source_id: str) -> SourceRecord:
        try:
            return self.records[source_id]
        except KeyError:
            raise KeyError(f"source {source_id!r} is not in the registry") from None

    def resolve(self, path: str | Path) -> SourceRecord | None:
        """Find a file's record by catalogued path suffix, then by file name.

        Then, for a file catalogued by a sibling (a journal's records JSON next
        to its catalogued manifest), by its directory when exactly one record
        lives there. Finally, names that a legacy pipeline decorated (for
        example ``OK_Ireland_<stem>_unknown_all.pdf``) by stem containment,
        again only when exactly one record matches.
        """
        posix = _norm_name(str(path).replace("\\", "/"))
        for catalogued, sid in self._by_path.items():
            if posix.endswith(catalogued) or catalogued.endswith(posix):
                return self.records[sid]
        name = PurePosixPath(posix).name
        if name in self._by_name:
            return self.records[self._by_name[name]]
        parent = str(PurePosixPath(posix).parent)
        dir_hits = {sid for d, sids in self._by_dir.items() if parent.endswith(d) for sid in sids}
        if len(dir_hits) == 1:
            return self.records[dir_hits.pop()]
        stem = PurePosixPath(name).stem
        hits = {
            sid for cat_name, sid in self._by_name.items()
            if len(PurePosixPath(cat_name).stem) >= 8 and PurePosixPath(cat_name).stem in stem
        }
        return self.records[hits.pop()] if len(hits) == 1 else None

    def source_for_statement(self, statement_id: str) -> str | None:
        """Map a gold statement id (``C0042``) to its source via the catalogued ranges."""
        m = re.match(r"^\s*([A-Za-z]+)(\d+)\s*$", statement_id)
        if not m:
            return None
        prefix, number = m.group(1).upper(), int(m.group(2))
        for r in self.records.values():
            if r.statement_range is None:
                continue
            p, lo, hi = r.statement_range
            if p == prefix and lo <= number <= hi:
                return r.source_id
        return None

    # -------------------------------------------------------------- loaders

    @classmethod
    def from_records(
        cls,
        rows: Iterable[Mapping[str, Any]],
        *,
        type_rules: Sequence[tuple[str, str]] = NUTRITION_TYPE_RULES,
    ) -> SourceRegistry:
        return cls(r for r in (_record_from_row(row, type_rules) for row in rows) if r is not None)

    @classmethod
    def from_xlsx(
        cls,
        path: str | Path,
        *,
        type_rules: Sequence[tuple[str, str]] = NUTRITION_TYPE_RULES,
    ) -> SourceRegistry:
        """Read every sheet whose header row has a ``Source ID`` column."""
        rows: list[dict[str, Any]] = []
        for sheet, data in read_xlsx(path).items():
            header_row = next(
                (i for i, r in enumerate(data[:10]) if any(str(v).strip().lower() == "source id" for v in r if v)),
                None,
            )
            if header_row is None:
                continue
            for rec in records(data, header_row=header_row):
                rec.setdefault("_sheet", sheet)
                rows.append(rec)
        return cls.from_records(rows, type_rules=type_rules)

    @classmethod
    def load(cls, path: str | Path) -> SourceRegistry:
        """Load by extension: ``.xlsx``, ``.csv`` or ``.json`` (a list of rows)."""
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix == ".xlsx":
            return cls.from_xlsx(p)
        if suffix == ".csv":
            import csv

            with p.open(newline="", encoding="utf-8") as fh:
                return cls.from_records(csv.DictReader(fh))
        if suffix == ".json":
            import json

            return cls.from_records(json.loads(p.read_text(encoding="utf-8")))
        raise ValueError(f"unsupported registry format: {p.name} (want .xlsx, .csv or .json)")


def _record_from_row(row: Mapping[str, Any], type_rules: Sequence[tuple[str, str]]) -> SourceRecord | None:
    fields: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    page_note: str | None = None
    for key, value in row.items():
        if key is None:
            continue
        k = str(key).strip()
        low = k.lower()
        if low in _COLUMNS:
            fields[_COLUMNS[low]] = value
        elif low.startswith("pages (") and "catalogue" not in low:
            page_note = str(value) if value not in (None, "") else None
        elif not k.startswith("_") and value not in (None, "", "-"):
            extra[k] = value
    source_id = str(fields.get("source_id") or "").strip()
    if not source_id or source_id.lower() == "total":
        return None
    source_type = str(fields.get("source_type") or "").strip() or next(
        (t for prefix, t in type_rules if source_id.upper().startswith(prefix.upper())), "unknown"
    )
    copies = [c.strip() for c in str(fields.get("local_copy") or "").split(";") if c.strip()]
    return SourceRecord(
        source_id=source_id,
        source_type=source_type,
        title=_clean(fields.get("title")),
        short_name=_clean(fields.get("short_name")),
        languages=_languages(fields.get("language")),
        licence_id=_clean(fields.get("licence_id")),
        licence_summary=_clean(fields.get("licence_summary")),
        local_copies=copies,
        content_pages=_content_pages(page_note),
        printed_page_offset=_printed_offset(page_note),
        has_printed_page_numbers=_has_printed(page_note),
        statement_range=_statement_range(fields.get("statement_range")),
        url=_clean(fields.get("url")),
        page_note=page_note,
        extra=extra,
    )


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s and s != "-" else None


def _languages(value: Any) -> list[str]:
    out: list[str] = []
    for part in re.split(r"[+,/;&]| and ", str(value or "")):
        code = _LANGUAGE_CODES.get(part.strip().lower())
        if code and code not in out:
            out.append(code)
    return out


def _content_pages(note: str | None) -> tuple[int, int] | None:
    m = _CONTENT_RE.search(note or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def _printed_offset(note: str | None) -> int | None:
    if not note:
        return None
    m = _OFFSET_RE.search(note)
    if m:
        if not m.group(1):
            return 0
        n = int(m.group(2))
        return n if m.group(1) == "+" else -n
    m = _FOOTER_RE.search(note)
    if m:
        return int(m.group(1)) - int(m.group(2))
    if _NO_PRINTED_RE.search(note):
        return 0  # the page *is* the PDF page; labels are still withheld below
    return None


def _has_printed(note: str | None) -> bool | None:
    if not note:
        return None
    if _NO_PRINTED_RE.search(note):
        return False
    if _OFFSET_RE.search(note) or _FOOTER_RE.search(note) or "footer numbers" in note.lower():
        return True
    return None


def _statement_range(value: Any) -> tuple[str, int, int] | None:
    m = _RANGE_RE.match(str(value or ""))
    if not m or m.group(1).upper() != m.group(3).upper():
        return None
    return (m.group(1).upper(), int(m.group(2)), int(m.group(4)))
