"""The record head: a file of short structured records → one IR per record.

Genre ``record``: journal abstracts here, but nothing in it is specific to
nutrition — field names come from :class:`~foodscholar.chunking.config.RecordHeadOptions`.

Each record becomes one document whose ``source_id`` is its DOI (or
``<collection>:<record id>``), whose title becomes the context header, and
whose text becomes one paragraph — or, for a structured abstract, one
paragraph per labelled section. The assembler then keeps a record whole when it
fits the budget and cuts at section boundaries when it does not; every part
shares the record's ``parent_id``, so ``expand(chunk_id, "section")`` returns
the whole abstract.

Structured abstracts arrive in three shapes, all handled:

- ``Purpose of review: While …`` — label, colon;
- ``Background Two previous meta-analyses …`` — label, space, capital;
- ``Purpose of reviewThe first stages …`` — label glued to the text.

A record counts as structured only when it *starts* with a known label and has
at least two of them, so an unstructured abstract that happens to begin with
"Objective measures…" stays one paragraph. Section text is rendered as
``Label: text`` whatever the input shape; only separators change, never words.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from foodscholar.chunking.ir import DocumentIR, DocumentMeta, Element, Paragraph
from foodscholar.logging import get_logger

if TYPE_CHECKING:
    from foodscholar.chunking.config import ChunkingConfig, RecordHeadOptions
    from foodscholar.chunking.registry import SourceRecord

_log = get_logger("foodscholar.chunking.heads.record")

HEAD_VERSION = "record/1"

SECTION_TYPES: dict[str, str] = {
    # introduction
    "background": "introduction", "backgrounds": "introduction", "context": "introduction",
    "introduction": "introduction", "objective": "introduction", "objectives": "introduction",
    "aim": "introduction", "aims": "introduction", "purpose": "introduction",
    "purpose of review": "introduction", "purpose of the review": "introduction",
    "rationale": "introduction", "scope": "introduction", "scope and approach": "introduction",
    "importance": "introduction",
    # methods
    "methods": "methods", "method": "methods", "methodology": "methods",
    "materials and methods": "methods", "design": "methods", "setting": "methods",
    "settings": "methods", "participants": "methods", "subjects": "methods",
    "data sources": "methods", "data source": "methods", "data extraction": "methods",
    "data analysis": "methods", "data analyses": "methods", "data synthesis": "methods",
    "study selection": "methods", "interventions": "methods", "intervention": "methods",
    "measurements": "methods", "exposure": "methods", "exposures": "methods",
    "main outcome measures": "methods", "outcome measures": "methods",
    "evidence acquisition": "methods", "eligibility criteria": "methods",
    "information sources": "methods", "search methods": "methods",
    "selection criteria": "methods", "risk of bias": "methods", "approach": "methods",
    # results
    "results": "results", "result": "results", "findings": "results",
    "main findings": "results", "recent findings": "results", "recent finding": "results",
    "key findings": "results",
    "evidence synthesis": "results", "synthesis of results": "results", "outcomes": "results",
    # conclusion
    "conclusion": "conclusion", "conclusions": "conclusion",
    "conclusions and relevance": "conclusion", "conclusion and relevance": "conclusion",
    "summary": "conclusion", "implications": "conclusion",
    "implications for practice": "conclusion", "key findings and conclusions": "conclusion",
    "significance": "conclusion",
    # discussion
    "discussion": "discussion", "limitations": "discussion",
    # other
    "registration": "other", "systematic review registration": "other",
    "trial registration": "other", "funding": "other",
}

_LABEL_ALTS = "|".join(re.escape(k) for k in sorted(SECTION_TYPES, key=len, reverse=True))
# A label starts the text or a sentence, begins with a capital, and is followed
# by a colon, by whitespace and a capital/digit, or directly by a capital or
# digit (``Recent findings156 studies…``).
_LABEL_RE = re.compile(
    r"(?:^|(?<=[.?!)\]]\s)|(?<=[.?!)\]]))"
    r"(?=[A-Z])(?P<label>(?i:" + _LABEL_ALTS + r"))"
    r"(?:\s*:\s*|\s+(?=[A-Z0-9(\[\"“])|(?=[A-Z0-9]))"
)


def split_sections(text: str) -> list[tuple[str | None, str]]:
    """``[(label, body), …]`` for a structured abstract, ``[(None, text)]`` otherwise."""
    text = " ".join(text.split())
    matches = list(_LABEL_RE.finditer(text))
    if len(matches) < 2 or matches[0].start() != 0:
        return [(None, text)]
    out: list[tuple[str | None, str]] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end].strip()
        if body:
            out.append((m.group("label"), body))
    return out if len(out) >= 2 else [(None, text)]


def section_type(label: str | None) -> str | None:
    return SECTION_TYPES.get(label.lower(), "other") if label else None


def _load(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    data = json.loads(raw)
    if isinstance(data, dict):
        data = data.get("records", [])
    if not isinstance(data, list):
        raise ValueError(f"{path.name}: expected a JSON list of records")
    return data


def _keywords(value: Any) -> str | None:
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value if v)
    value = " ".join(str(value or "").split())
    return value or None


# Record keys renamed on the way into chunk metadata, where the plain names
# already mean something else (``pages`` is a chunk's page set, ``type`` the
# source type, ``id`` the chunk id).
_RENAMED = {"id": "record_id", "type": "article_type", "pages": "article_pages"}


class RecordHead:
    genre = "record"

    def extract(
        self,
        path: Path,
        *,
        record: SourceRecord | None,
        config: ChunkingConfig,
        source_type: str,
    ) -> list[DocumentIR]:
        opts = config.profile_for(source_type).record
        collection_id = record.source_id if record else path.stem
        collection = (record.short_name or record.title) if record else None
        language = record.languages[0] if record and record.languages else None
        documents: list[DocumentIR] = []
        skipped = 0
        for r in _load(path):
            doc = self._document(r, opts, path, source_type, collection_id, collection, language, record)
            if doc is None:
                skipped += 1
                continue
            documents.append(doc)
        _log.info("record_head.file_done", file=path.name, documents=len(documents), skipped_empty=skipped)
        return documents

    def _document(
        self,
        r: dict[str, Any],
        opts: RecordHeadOptions,
        path: Path,
        source_type: str,
        collection_id: str,
        collection: str | None,
        language: str | None,
        record: SourceRecord | None,
    ) -> DocumentIR | None:
        text = " ".join(str(r.get(opts.text_field) or "").split())
        rid = str(r.get(opts.id_field) or "")
        if not text:
            _log.info("record_head.skip_empty", file=path.name, record_id=rid)
            return None
        doi = str(r.get(opts.doi_field) or "").strip() or None
        title = " ".join(str(r.get(opts.title_field) or "").split()) or None
        keywords = _keywords(r.get(opts.keywords_field))
        lang = (str(r[opts.language_field]) if opts.language_field and r.get(opts.language_field) else None) or language
        label = title
        if opts.header == "title_keywords" and keywords:
            label = f"{title} | Keywords: {keywords}" if title else f"Keywords: {keywords}"

        extra: dict[str, Any] = {"collection_id": collection_id}
        if collection:
            extra["journal"] = collection
        skip = {opts.text_field, opts.title_field, opts.licence_field, opts.keywords_field, opts.doi_field}
        for key, value in r.items():
            if key in skip or value in (None, "", []):
                continue
            extra[_RENAMED.get(key, key)] = value
        if doi:
            extra["doi"] = doi
        if keywords:
            extra["keywords"] = keywords

        sections = split_sections(text) if opts.structured_sections else [(None, text)]
        elements: list[Element] = [
            Paragraph(
                id=f"s{i}",
                text=f"{lab}: {body}" if lab else body,
                section_label=lab,
                section_type=section_type(lab),
                language=lang,
            )
            for i, (lab, body) in enumerate(sections)
        ]
        digest = hashlib.sha1(json.dumps(r, sort_keys=True, ensure_ascii=False).encode(), usedforsecurity=False).hexdigest()[:16]
        return DocumentIR(
            meta=DocumentMeta(
                source_id=doi or f"{collection_id}:{rid}",
                source_type=source_type,
                genre=self.genre,
                document_version_id=digest,
                extraction_run_id=HEAD_VERSION,
                title=title,
                short_name=label,
                file=path.name,
                language=lang,
                licence=str(r.get(opts.licence_field) or "").strip() or (record.licence_id if record else None),
                extra=extra,
            ),
            elements=elements,
        )
