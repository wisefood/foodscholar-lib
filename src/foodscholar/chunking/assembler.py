"""The structure-first assembler: a :class:`DocumentIR` in, chunks out.

Order of operations, each of which changes the output:

1. **Sections from headings.** A heading stack gives every element its *full*
   section path (``("Unit 2", "Carbohydrates", "Digestion")``), not only the
   top-level heading. Furniture (running headers, page numbers) is dropped.
2. **Atomic units.** A paragraph, a whole list, a table with its caption, a
   figure with its caption, a box — each is one unit. Nothing below a unit is
   ever split unless the unit alone exceeds the budget.
3. **Pack within a section, never across, in balanced parts.** A section that
   fits stays whole unless it is well over ``target_tokens``; one that does not
   is cut into the fewest near-equal parts the budget allows, so no part is a
   stray tail. The budget is ``max_tokens`` of *embedded* text — context header,
   content and the tokenizer's special tokens — which is what the embedder
   would otherwise truncate.
4. **Split only what does not fit**: a paragraph at sentence boundaries, a list
   between items, a table between rows with the header repeated, and a single
   over-long sentence by words as the last resort.
5. **Merge fragments.** A section's short tail joins the previous chunk; then
   adjacent chunks that are both under ``min_tokens`` and share a parent
   section merge, labelled with the common path.
6. **No overlap by default.** Neighbours are reached through ``ordinal`` and
   the previous/next links instead (see :mod:`foodscholar.chunking.identity`
   and the chunk store's ``expand()``). ``overlap_sentences`` is an ablation.

The assembler never calls a model. Token counting and sentence splitting are
injected (:mod:`foodscholar.chunking.protocols`).
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from itertools import groupby
from typing import Any

from foodscholar.chunking.config import AssemblyConfig
from foodscholar.chunking.ir import DocumentIR, DocumentMeta, Region
from foodscholar.chunking.protocols import SentenceSplitter, TokenCounter
from foodscholar.chunking.text import RegexSentenceSplitter, normalize_space
from foodscholar.chunking.window import has_meaningful_text
from foodscholar.logging import get_logger

_log = get_logger("foodscholar.chunking.assembler")

SectionPath = tuple[str, ...]

_JOIN = "\n\n"
_LIST_JOIN = "\n"
# A header longer than this share of the budget is shortened, keeping the
# document label and the deepest levels.
_MAX_HEADER_SHARE = 0.25


@dataclass
class AssembledChunk:
    """One chunk, with everything needed to cite, navigate and re-identify it."""

    source_id: str
    ordinal: int
    section_path: SectionPath
    parent_id: str
    element_ids: list[str]
    source_text: str
    """The faithful passage — what a citation displays."""
    context_header: str
    """Deterministic context, embedded with the text but never shown as a quote."""
    regions: list[Region]
    """One per page the chunk touches."""
    content_tokens: int
    total_tokens: int
    """Tokens of `retrieval_text` including special tokens. Never above the budget."""
    unit_kinds: list[str] = field(default_factory=list)
    figure_ids: list[str] = field(default_factory=list)
    table_ids: list[str] = field(default_factory=list)
    section_labels: list[str] = field(default_factory=list)
    """Labelled record sections the chunk contains, verbatim and in order."""
    section_types: list[str] = field(default_factory=list)
    """The same sections, normalised. One entry means the chunk is that section."""
    language: str | None = None
    overlap_tokens: int = 0
    # Set by `foodscholar.chunking.identity.assign_identity`.
    chunk_version_id: str = ""
    chunk_id: str = ""
    supersedes: list[str] = field(default_factory=list)
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None

    @property
    def retrieval_text(self) -> str:
        """What gets embedded and indexed: header, newline, passage."""
        return f"{self.context_header}\n{self.source_text}" if self.context_header else self.source_text

    @property
    def pages(self) -> tuple[int, ...]:
        return tuple(sorted({r.page for r in self.regions}))

    @property
    def page_start(self) -> int | None:
        return self.pages[0] if self.pages else None

    @property
    def page_end(self) -> int | None:
        return self.pages[-1] if self.pages else None


@dataclass
class _Unit:
    kind: str
    text: str
    element_ids: list[str]
    regions: list[Region]
    section_path: SectionPath
    tokens: int
    language: str | None = None
    figure_ids: list[str] = field(default_factory=list)
    table_ids: list[str] = field(default_factory=list)
    items: list[_Unit] = field(default_factory=list)
    """List items, for a list unit — split points."""
    section_label: str | None = None
    section_type: str | None = None
    table: Any = None
    """`(TableData, caption)` for a table unit — row split points."""


def section_id(source_id: str, section_path: Sequence[str]) -> str:
    """Stable id of a section within a document — a chunk's `parent_id`."""
    payload = "\x1f".join([source_id, *section_path]).encode()
    return "sec-" + hashlib.sha1(payload, usedforsecurity=False).hexdigest()[:16]


class Assembler:
    """Packs one document's elements into chunks. Stateless between documents."""

    def __init__(
        self,
        config: AssemblyConfig | None = None,
        *,
        token_counter: TokenCounter,
        sentence_splitter: SentenceSplitter | None = None,
    ) -> None:
        self.config = config or AssemblyConfig()
        self.tokens = token_counter
        self.splitter = sentence_splitter or RegexSentenceSplitter()

    # ------------------------------------------------------------------ api

    def assemble(self, ir: DocumentIR) -> list[AssembledChunk]:
        units = self._units(ir)
        groups: list[list[AssembledChunk]] = []
        for path, run in groupby(units, key=lambda u: u.section_path):
            groups.extend(self._pack_section(ir.meta, path, list(run)))
        if self.config.merge_small_sections:
            groups = self._merge_small(ir.meta, groups)
        groups = [self._enforce_budget(ir.meta, g) for g in groups]
        flat = [c for g in groups for c in g]
        for i, c in enumerate(flat):
            c.ordinal = i
        return flat

    # --------------------------------------------------------------- units

    def _units(self, ir: DocumentIR) -> list[_Unit]:
        ids = {el.id for el in ir.elements}
        captions = {
            el.target_id: el
            for el in ir.elements
            if el.kind == "caption" and el.target_id and el.target_id in ids
        }
        paired = {c.id for c in captions.values()}
        stack: list[tuple[int, str]] = []
        units: list[_Unit] = []
        pending_list: _Unit | None = None

        def path() -> SectionPath:
            return tuple(text for _, text in stack)

        def flush_list() -> None:
            nonlocal pending_list
            if pending_list is not None:
                pending_list.text = _LIST_JOIN.join(i.text for i in pending_list.items)
                pending_list.tokens = sum(i.tokens for i in pending_list.items)
                units.append(pending_list)
                pending_list = None

        for el in ir.elements:
            if el.kind == "furniture" or el.id in paired:
                continue
            if el.kind == "heading":
                flush_list()
                title = normalize_space(el.text)
                if not title:
                    continue
                while stack and stack[-1][0] >= el.level:
                    stack.pop()
                stack.append((el.level, title))
                continue
            if el.kind == "list_item":
                text = el.text.strip()
                if not has_meaningful_text(text):
                    continue
                marker = el.marker or ("1." if el.ordered else "-")
                item = self._unit("list_item", f"{'  ' * el.depth}{marker} {text}", [el], path())
                if pending_list is None or pending_list.section_path != item.section_path:
                    flush_list()
                    pending_list = _Unit("list", "", [], [], item.section_path, 0, el.language)
                pending_list.items.append(item)
                pending_list.element_ids.append(el.id)
                pending_list.regions.extend(el.regions)
                continue
            flush_list()
            unit = self._element_unit(el, captions.get(el.id), path())
            if unit is not None and has_meaningful_text(unit.text):
                units.append(unit)
        flush_list()
        return units

    def _element_unit(self, el: Any, caption: Any, section: SectionPath) -> _Unit | None:
        sources = [el] if caption is None else [el, caption]
        if el.kind == "table":
            cap = el.caption or (caption.text if caption is not None else None)
            body = el.table.to_markdown() or el.text
            text = _JOIN.join(p for p in (cap, body) if p)
            unit = self._unit("table", text, sources, section)
            unit.table = (el.table, cap)
            unit.table_ids = [el.asset_id or el.id]
            return unit
        if el.kind == "figure":
            cap = el.caption or (caption.text if caption is not None else None)
            parts = [cap, el.description, el.text]
            seen: list[str] = []
            for p in parts:
                if p and p.strip() and p.strip() not in seen:
                    seen.append(p.strip())
            unit = self._unit("figure", _JOIN.join(seen), sources, section)
            unit.figure_ids = [el.asset_id or el.id]
            return unit
        if el.kind == "box":
            text = _JOIN.join(p for p in (el.title, el.text) if p)
            return self._unit("box", text, sources, section)
        if el.kind == "paragraph":
            unit = self._unit("paragraph", el.text.strip(), sources, section)
            unit.section_label, unit.section_type = el.section_label, el.section_type
            return unit
        if el.kind in ("caption", "footnote"):
            return self._unit(el.kind, el.text.strip(), sources, section)
        return None

    def _unit(self, kind: str, text: str, elements: Sequence[Any], section: SectionPath) -> _Unit:
        return _Unit(
            kind=kind,
            text=text,
            element_ids=[e.id for e in elements],
            regions=[r for e in elements for r in e.regions],
            section_path=section,
            tokens=self.tokens.count(text),
            language=next((e.language for e in elements if e.language), None),
        )

    # ------------------------------------------------------------- packing

    def _header(self, meta: DocumentMeta, path: SectionPath) -> str:
        if not self.config.context_header:
            return ""
        label = meta.short_name or meta.title or meta.source_id
        levels = list(path)
        header = " > ".join([label, *levels])
        limit = int(self.config.max_tokens * _MAX_HEADER_SHARE)
        # Drop the shallowest levels first: the deepest one names the passage.
        while self.tokens.count(header) > limit and len(levels) > 1:
            levels.pop(0)
            header = " > ".join([label, "…", *levels])
        return header

    def _available(self, header: str) -> int:
        return self.config.max_tokens - self.tokens.special_tokens - (self.tokens.count(header) if header else 0)

    def _pack_section(self, meta: DocumentMeta, path: SectionPath, units: list[_Unit]) -> list[list[AssembledChunk]]:
        cfg = self.config
        header = self._header(meta, path)
        available = max(self._available(header), 16)
        pieces: list[_Unit] = []
        for u in units:
            pieces.extend(self._fit(u, available) if u.tokens > available else [u])

        groups = balanced_groups([u.tokens for u in pieces], available, cfg.target_tokens)
        batches = [pieces[a:b] for a, b in groups]
        # A short tail rejoins the previous batch when the budget allows.
        if len(batches) >= 2:
            tail = sum(u.tokens for u in batches[-1])
            prev = sum(u.tokens for u in batches[-2])
            if tail < cfg.min_tokens and prev + tail <= available:
                batches[-2].extend(batches.pop())

        out: list[list[AssembledChunk]] = []
        previous: AssembledChunk | None = None
        for batch in batches:
            chunk = self._make_chunk(meta, path, header, batch)
            if previous is not None and cfg.overlap_sentences > 0:
                self._apply_overlap(previous, chunk, available)
            out.append([chunk])
            previous = chunk
        return out

    def _fit(self, unit: _Unit, available: int) -> list[_Unit]:
        """Split an over-budget unit at its natural boundaries."""
        if unit.kind == "list" and len(unit.items) > 1:
            return self._regroup(unit, unit.items, available, _LIST_JOIN, kind="list")
        if unit.kind == "table" and unit.table is not None:
            rows = self._table_rows(unit, available)
            if len(rows) > 1:
                return rows
        sentences = self.splitter.split(unit.text, unit.language)
        if len(sentences) > 1:
            parts = [replace(unit, text=s, tokens=self.tokens.count(s), items=[], table=None) for s in sentences]
            return self._regroup(unit, parts, available, " ", kind=unit.kind)
        return self._hard_split(unit, available)

    def _regroup(self, parent: _Unit, parts: list[_Unit], available: int, joiner: str, *, kind: str) -> list[_Unit]:
        pieces = [q for p in parts for q in (self._hard_split(p, available) if p.tokens > available else [p])]
        out: list[_Unit] = []
        for a, b in balanced_groups([p.tokens for p in pieces], available, self.config.target_tokens):
            cur = pieces[a:b]
            text = joiner.join(p.text for p in cur)
            ids = list(dict.fromkeys(i for p in cur for i in p.element_ids))
            regs = [r for p in cur for r in p.regions] if kind == "list" else list(parent.regions)
            out.append(replace(parent, kind=kind, text=text, tokens=sum(p.tokens for p in cur),
                               element_ids=ids or list(parent.element_ids), regions=regs, items=[], table=None))
        return out

    def _table_rows(self, unit: _Unit, available: int) -> list[_Unit]:
        data, caption = unit.table
        body = data.body if data.header_rows else data.cells[1:]
        if not body:
            return [unit]
        prefix = f"{caption}{_JOIN}" if caption else ""
        out: list[_Unit] = []
        start = 0
        while start < len(body):
            end = start + 1
            while end < len(body):
                trial = prefix + data.to_markdown(body[start : end + 1])
                if self.tokens.count(trial) > min(available, self.config.target_tokens):
                    break
                end += 1
            text = prefix + data.to_markdown(body[start:end])
            out.append(replace(unit, text=text, tokens=self.tokens.count(text), table=None))
            start = end
        return out

    def _hard_split(self, unit: _Unit, available: int) -> list[_Unit]:
        """Last resort: cut one over-long sentence by words."""
        words = unit.text.split()
        out: list[_Unit] = []
        cur: list[str] = []
        cur_tokens = 0
        for w in words:
            t = self.tokens.count(w)
            if cur and cur_tokens + t > available:
                text = " ".join(cur)
                out.append(replace(unit, text=text, tokens=self.tokens.count(text), items=[], table=None))
                cur, cur_tokens = [], 0
            cur.append(w)
            cur_tokens += t
        if cur:
            text = " ".join(cur)
            out.append(replace(unit, text=text, tokens=self.tokens.count(text), items=[], table=None))
        return out or [unit]

    def _make_chunk(self, meta: DocumentMeta, path: SectionPath, header: str, batch: list[_Unit]) -> AssembledChunk:
        text = _JOIN.join(u.text for u in batch)
        content = self.tokens.count(text)
        total = self.tokens.special_tokens + self.tokens.count(f"{header}\n{text}" if header else text)
        languages = Counter(u.language for u in batch if u.language)
        return AssembledChunk(
            source_id=meta.source_id,
            ordinal=-1,
            section_path=path,
            parent_id=section_id(meta.source_id, path),
            element_ids=list(dict.fromkeys(i for u in batch for i in u.element_ids)),
            source_text=text,
            context_header=header,
            regions=_regions_per_page([r for u in batch for r in u.regions]),
            content_tokens=content,
            total_tokens=total,
            unit_kinds=[u.kind for u in batch],
            figure_ids=list(dict.fromkeys(f for u in batch for f in u.figure_ids)),
            table_ids=list(dict.fromkeys(t for u in batch for t in u.table_ids)),
            section_labels=list(dict.fromkeys(u.section_label for u in batch if u.section_label)),
            section_types=list(dict.fromkeys(u.section_type for u in batch if u.section_type)),
            language=languages.most_common(1)[0][0] if languages else meta.language,
        )

    def _apply_overlap(self, previous: AssembledChunk, chunk: AssembledChunk, available: int) -> None:
        sentences = self.splitter.split(previous.source_text, previous.language)
        tail: list[str] = []
        tokens = 0
        for s in reversed(sentences[-self.config.overlap_sentences :]):
            t = self.tokens.count(s)
            if tokens + t > self.config.overlap_max_tokens:
                break
            tail.insert(0, s)
            tokens += t
        if not tail or chunk.content_tokens + tokens > available:
            return
        chunk.source_text = " ".join(tail) + _JOIN + chunk.source_text
        chunk.content_tokens = self.tokens.count(chunk.source_text)
        chunk.total_tokens = self.tokens.special_tokens + self.tokens.count(chunk.retrieval_text)
        chunk.overlap_tokens = tokens

    # ------------------------------------------------------------- merging

    def _merge_small(self, meta: DocumentMeta, groups: list[list[AssembledChunk]]) -> list[list[AssembledChunk]]:
        chunks = [c for g in groups for c in g]
        if not chunks:
            return groups
        out: list[AssembledChunk] = [chunks[0]]
        for nxt in chunks[1:]:
            cur = out[-1]
            common = _common_prefix(cur.section_path, nxt.section_path)
            same_top = bool(common) or (not cur.section_path and not nxt.section_path)
            small = cur.content_tokens < self.config.min_tokens and nxt.content_tokens < self.config.min_tokens
            if small and same_top and not nxt.overlap_tokens:
                merged = self._merge_pair(meta, cur, nxt, common)
                if merged is not None:
                    out[-1] = merged
                    continue
            out.append(nxt)
        return [[c] for c in out]

    def _merge_pair(self, meta: DocumentMeta, a: AssembledChunk, b: AssembledChunk, path: SectionPath) -> AssembledChunk | None:
        header = self._header(meta, path)
        if a.section_path == b.section_path:
            text = a.source_text + _JOIN + b.source_text
        else:
            # The header now names only the common section, so each side keeps
            # its own subsection title in the passage, as printed.
            def titled(c: AssembledChunk) -> str:
                sub = " > ".join(c.section_path[len(path):])
                return f"{sub}{_JOIN}{c.source_text}" if sub else c.source_text

            text = titled(a) + _JOIN + titled(b)
        total = self.tokens.special_tokens + self.tokens.count(f"{header}\n{text}" if header else text)
        if total > self.config.max_tokens:
            return None
        return replace(
            a,
            section_path=path,
            parent_id=section_id(meta.source_id, path),
            element_ids=list(dict.fromkeys([*a.element_ids, *b.element_ids])),
            source_text=text,
            context_header=header,
            regions=_regions_per_page([*a.regions, *b.regions]),
            content_tokens=self.tokens.count(text),
            total_tokens=total,
            unit_kinds=[*a.unit_kinds, *b.unit_kinds],
            figure_ids=list(dict.fromkeys([*a.figure_ids, *b.figure_ids])),
            table_ids=list(dict.fromkeys([*a.table_ids, *b.table_ids])),
            section_labels=list(dict.fromkeys([*a.section_labels, *b.section_labels])),
            section_types=list(dict.fromkeys([*a.section_types, *b.section_types])),
        )

    # ------------------------------------------------------------- budget

    def _enforce_budget(self, meta: DocumentMeta, group: list[AssembledChunk]) -> list[AssembledChunk]:
        """Re-check against the real encoding; split further if a tokenizer was not additive."""
        out: list[AssembledChunk] = []
        for c in group:
            if c.total_tokens <= self.config.max_tokens:
                out.append(c)
                continue
            _log.warning("assembler.budget_recheck_split", source_id=meta.source_id, tokens=c.total_tokens)
            available = max(self._available(c.context_header) - 8, 16)
            unit = _Unit("paragraph", c.source_text, c.element_ids, c.regions, c.section_path, c.content_tokens, c.language)
            if len(c.section_types) == 1:
                unit.section_type = c.section_types[0]
                unit.section_label = c.section_labels[0] if c.section_labels else None
            for piece in self._fit(unit, available):
                out.append(self._make_chunk(meta, c.section_path, c.context_header, [piece]))
        return out


def balanced_groups(sizes: Sequence[int], available: int, target: int) -> list[tuple[int, int]]:
    """Cut a run of units into contiguous groups of near-equal size.

    The number of groups is the larger of what the hard budget forces
    (``ceil(total / available)``) and what the soft target suggests
    (``round(total / target)``), so a run that fits and is near the target stays
    whole, and a run that must be cut is cut evenly instead of leaving a stray
    tail. No group exceeds `available` unless a single unit does. Returns
    ``(start, end)`` index pairs.
    """
    if not sizes:
        return []
    total = sum(sizes)
    k = max(1, math.ceil(total / max(available, 1)), round(total / max(target, 1)))
    goal = total / k
    groups: list[tuple[int, int]] = []
    start, cur = 0, 0
    for i, size in enumerate(sizes):
        if i > start:
            over = cur + size > available
            closer_without = cur >= goal or abs(goal - cur) <= abs(cur + size - goal)
            if over or (closer_without and len(groups) < k - 1):
                groups.append((start, i))
                start, cur = i, 0
        cur += size
    groups.append((start, len(sizes)))
    return groups


def _common_prefix(a: SectionPath, b: SectionPath) -> SectionPath:
    out: list[str] = []
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        out.append(x)
    return tuple(out)


def _regions_per_page(regions: Sequence[Region]) -> list[Region]:
    """Collapse element regions to one per page: the union box where boxes exist."""
    by_page: dict[int, list[Region]] = {}
    for r in regions:
        by_page.setdefault(r.page, []).append(r)
    out: list[Region] = []
    for page in sorted(by_page):
        regs = by_page[page]
        boxed = [r for r in regs if r.bbox is not None]
        if not boxed:
            out.append(Region(page=page, page_size=regs[0].page_size))
            continue
        space = boxed[0].coordinate_space
        same = [r for r in boxed if r.coordinate_space == space]
        x0 = min(r.bbox[0] for r in same)  # type: ignore[index]
        y0 = min(r.bbox[1] for r in same)  # type: ignore[index]
        x1 = max(r.bbox[2] for r in same)  # type: ignore[index]
        y1 = max(r.bbox[3] for r in same)  # type: ignore[index]
        out.append(Region(page=page, bbox=(x0, y0, x1, y1), coordinate_space=space, page_size=same[0].page_size))
    return out
