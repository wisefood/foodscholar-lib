"""The three-stage chunking harness.

Scores any pipeline's chunks against gold statements, in three separate
stages so a loss is blamed on the stage that caused it:

1. **Extraction coverage** — does the extracted document (the IR) still
   contain the evidence? Needs the IR; skipped otherwise.
2. **Assembly coverage** — does one chunk contain the statement (*single*),
   or one chunk plus its next neighbour (*bundle*)? On the right page and in
   the right section?
3. **Retrieval coverage** — does a retriever, given a query, return a chunk
   that supports the query's evidence within the first *k* results, or within
   a fixed budget of retrieved tokens? The token budget keeps small chunks and
   overlap from gaming *k*.

Input is the neutral chunk format (:class:`ChunkRecord`, JSONL with ``doc_id``
and ``text`` required), so the library's legacy window, Docling's shipped
chunker, LangChain or a commercial parser are all scored the same way.
Nothing here calls a model.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

from foodscholar.chunking.evaluation.gold import GoldStatement
from foodscholar.chunking.evaluation.matching import PassageIndex, Support, content_tokens
from foodscholar.chunking.registry import SourceRegistry

DEFAULT_THRESHOLD = 0.8

# ------------------------------------------------------------------ records


@dataclass
class ChunkRecord:
    """One chunk in the neutral format."""

    chunk_id: str
    doc_id: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    section_path: tuple[str, ...] = ()
    parent_id: str | None = None
    ordinal: int | None = None
    tokens: int | None = None
    """Tokens of the embedded text, when the producer reports them."""
    collection_id: str | None = None
    """For records: the catalogued collection (a journal) the document belongs
    to. Gold statements catalogued per collection match through it."""

    @property
    def token_estimate(self) -> int:
        return self.tokens if self.tokens is not None else max(1, len(self.text) // 4)


def _int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _from_metadata(chunk_id: str, text: str, meta: dict[str, Any], registry: SourceRegistry | None, fallback_doc: str) -> ChunkRecord:
    doc = str(meta.get("source_id") or meta.get("file") or meta.get("doi") or meta.get("DOI") or fallback_doc)
    if registry is not None and not meta.get("source_id"):
        rec = registry.resolve(doc)
        if rec is not None:
            doc = rec.source_id
    path = meta.get("section_path") or ([meta["heading"]] if meta.get("heading") else [])
    start = _int(meta.get("page_start", meta.get("page_number")))
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc,
        text=text,
        page_start=start,
        page_end=_int(meta.get("page_end")),
        section=(path[-1] if path else None),
        section_path=tuple(str(p) for p in path),
        parent_id=meta.get("parent_id"),
        ordinal=_int(meta.get("ordinal")),
        tokens=_int(meta.get("tokens")),
        collection_id=meta.get("collection_id") or None,
    )


def load_chunk_records(path: str | Path, *, registry: SourceRegistry | None = None) -> list[ChunkRecord]:
    """Read chunks from neutral JSONL, corpus CSV(s) or a stored-chunk parquet snapshot.

    A directory reads every ``*.csv`` in it. Document ids are mapped to
    registry source ids where the registry recognises the file, so legacy
    chunks (keyed by decorated file names) score against the same gold.
    """
    p = Path(path)
    if p.is_dir():
        out: list[ChunkRecord] = []
        for f in sorted(p.glob("*.csv")):
            out.extend(load_chunk_records(f, registry=registry))
        return _fill_ordinals(out)
    suffix = p.suffix.lower()
    records: list[ChunkRecord] = []
    if suffix == ".jsonl":
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            doc = str(d["doc_id"])
            if registry is not None and doc not in registry.records:
                hit = registry.resolve(doc)
                doc = hit.source_id if hit else doc
            cid = str(d.get("chunk_id") or f"{doc}#{len(records)}")
            records.append(_from_metadata(cid, str(d["text"]), {**d, "source_id": doc}, None, doc))
    elif suffix == ".csv":
        csv.field_size_limit(10 * 1024 * 1024)
        with p.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                meta = ast.literal_eval(row.get("chunk_metadata") or "{}") or {}
                records.append(_from_metadata(row["chunk_id"], row["chunk_text"], meta, registry, p.stem))
    elif suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(p, columns=["chunk_id", "text", "source_doc_id", "source_metadata"])  # type: ignore[no-untyped-call]
        for row in table.to_pylist():
            meta = row["source_metadata"]
            meta = json.loads(meta) if isinstance(meta, str) and meta else (meta or {})
            records.append(_from_metadata(row["chunk_id"], row["text"], meta, registry, row["source_doc_id"]))
    else:
        raise ValueError(f"unsupported chunk file: {p.name} (want .jsonl, .csv, .parquet or a directory)")
    return _fill_ordinals(records)


def _fill_ordinals(records: list[ChunkRecord]) -> list[ChunkRecord]:
    """Input order stands in for reading order where the producer gave none."""
    counters: Counter[str] = Counter()
    for r in records:
        if r.ordinal is None:
            r.ordinal = counters[r.doc_id]
        counters[r.doc_id] = max(counters[r.doc_id], r.ordinal) + 1
    return records


# ------------------------------------------------------------- outcomes


@dataclass
class StatementOutcome:
    statement_id: str
    source_id: str | None
    scored: bool
    """False when the pipeline produced no chunk for the statement's source."""
    extraction: float | None = None
    straddles_page_break: bool | None = None
    best_score: float = 0.0
    best_chunk_id: str | None = None
    single: bool = False
    bundle: bool = False
    numbers_lost: bool = False
    page_ok: bool | None = None
    section_ok: bool | None = None


@dataclass
class EvaluationReport:
    label: str
    threshold: float
    totals: dict[str, Any]
    per_source: dict[str, dict[str, Any]]
    hygiene: dict[str, Any]
    retrieval: dict[str, Any] = field(default_factory=dict)
    outcomes: list[StatementOutcome] = field(default_factory=list)

    def to_dict(self, *, with_outcomes: bool = False) -> dict[str, Any]:
        d = {
            "label": self.label,
            "threshold": self.threshold,
            "totals": self.totals,
            "per_source": self.per_source,
            "hygiene": self.hygiene,
            "retrieval": self.retrieval,
        }
        if with_outcomes:
            d["outcomes"] = [asdict(o) for o in self.outcomes]
        return d

    def to_markdown(self) -> str:
        t = self.totals
        lines = [
            f"### {self.label}",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Statements scored / total | {t['n_scored']} / {t['n_statements']} |",
        ]
        for key, name in (
            ("extraction", "Extraction coverage"),
            ("assembly_single", "Supported by one chunk"),
            ("assembly_bundle", "Supported by a chunk + its next neighbour"),
            ("page_accuracy", "Gold page inside the chunk's page span"),
            ("section_accuracy", "Gold section matches the chunk's section"),
            ("numbers_lost", "Words found but a number missing"),
            ("straddle_single", "Page-straddling statements kept in one chunk"),
        ):
            if t.get(key) is not None:
                lines.append(f"| {name} | {t[key]:.1%} |")
        h = self.hygiene
        lines += [
            f"| Chunks | {h['n_chunks']} |",
            f"| Tokens p50 / p95 / max | {h['tokens_p50']} / {h['tokens_p95']} / {h['tokens_max']}{' (estimated)' if h['estimated'] else ''} |",
            f"| Chunks under {h['min_tokens']} / over {h['max_tokens']} tokens | {h['share_under_min']:.1%} / {h['share_over_max']:.1%} |",
            f"| Duplicate chunk texts | {h['duplicates']} |",
        ]
        for key, value in sorted(self.retrieval.items()):
            if isinstance(value, float):
                lines.append(f"| Retrieval {key} | {value:.1%} |")
        if t.get("missing_sources"):
            lines.append(f"| Sources with no chunks | {', '.join(t['missing_sources'])} |")
        if t.get("gold_page_offsets"):
            shifted = ", ".join(f"{k} {v:+d}" for k, v in t["gold_page_offsets"].items())
            lines.append(f"| Gold pages corrected | {shifted} |")
        return "\n".join(lines)


# ---------------------------------------------------------------- stages


def _section_match(gold: str | None, path: Sequence[str]) -> bool | None:
    if not gold or not path:
        return None
    g = set(content_tokens(gold))
    if not g:
        return None
    for level in path:
        c = set(content_tokens(level))
        if not c:
            continue
        if g <= c or c <= g or len(g & c) / len(g | c) >= 0.5:
            return True
    return False


def _page_ok(page: int | None, rec: ChunkRecord) -> bool | None:
    if page is None or rec.page_start is None:
        return None
    end = rec.page_end if rec.page_end is not None else rec.page_start
    return rec.page_start <= page <= end


def _windows(texts: list[str], size: int = 3) -> list[str]:
    return [" ".join(texts[i : i + size]) for i in range(max(1, len(texts) - size + 1))] if texts else []


def evaluate(
    gold: Sequence[GoldStatement],
    chunks: Sequence[ChunkRecord],
    *,
    label: str = "pipeline",
    threshold: float = DEFAULT_THRESHOLD,
    ir_texts: dict[str, list[str]] | None = None,
    ir_pages: dict[str, dict[int, str]] | None = None,
    min_tokens: int = 120,
    max_tokens: int = 512,
) -> EvaluationReport:
    """Score `chunks` against `gold`.

    `ir_texts` maps a source id to its extracted element texts in reading
    order (extraction stage); `ir_pages` maps a source id to per-page text
    (page-straddle subset). Both are optional.
    """
    by_doc: dict[str, list[int]] = defaultdict(list)
    by_collection: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(chunks):
        by_doc[c.doc_id].append(i)
        if c.collection_id and c.collection_id != c.doc_id:
            by_collection[c.collection_id].append(i)
    index = PassageIndex([c.text for c in chunks])
    next_of: dict[int, int] = {}
    for idxs in by_doc.values():
        ordered = sorted(idxs, key=lambda i: chunks[i].ordinal or 0)
        next_of.update(pairwise(ordered))

    ir_index = {sid: PassageIndex(_windows(texts)) for sid, texts in (ir_texts or {}).items()}
    page_index = {sid: (sorted(pages), PassageIndex([pages[p] for p in sorted(pages)])) for sid, pages in (ir_pages or {}).items()}

    outcomes: list[StatementOutcome] = []
    for g in gold:
        if g.source_id:
            cand = by_doc.get(g.source_id, []) or by_collection.get(g.source_id, [])
        else:
            cand = list(range(len(chunks)))
        out = StatementOutcome(statement_id=g.statement_id, source_id=g.source_id, scored=bool(cand))
        if g.source_id in ir_index:
            idx = ir_index[g.source_id]
            out.extraction = max((idx.support(g.text, i).score for i in range(len(idx.counts))), default=0.0)
        if g.source_id in page_index and g.page is not None:
            pages, pidx = page_index[g.source_id]
            if g.page in pages and (g.page + 1) in pages:
                i = pages.index(g.page)
                alone = max(pidx.support(g.text, i).score, pidx.support(g.text, i + 1).score)
                both = pidx.support_union(g.text, [i, i + 1]).score
                out.straddles_page_break = alone < threshold <= both
        if cand:
            best_i, best = -1, Support(0.0, False)
            for i in cand:
                s = index.support(g.text, i)
                if (s.numbers_ok, s.score) > (best.numbers_ok, best.score):
                    best_i, best = i, s
            out.best_score = best.score
            out.best_chunk_id = chunks[best_i].chunk_id if best_i >= 0 else None
            out.single = best.supported(threshold)
            out.numbers_lost = best.score >= threshold and not best.numbers_ok
            out.bundle = out.single or any(
                index.support_union(g.text, [i, next_of[i]]).supported(threshold) for i in cand if i in next_of
            )
            if out.single:
                out.page_ok = _page_ok(g.page, chunks[best_i])
                out.section_ok = _section_match(g.section, chunks[best_i].section_path)
        outcomes.append(out)

    return EvaluationReport(
        label=label,
        threshold=threshold,
        totals=_totals(outcomes, gold),
        per_source={sid: _totals([o for o in outcomes if o.source_id == sid]) for sid in sorted({o.source_id or "?" for o in outcomes})},
        hygiene=hygiene(chunks, min_tokens=min_tokens, max_tokens=max_tokens),
        outcomes=outcomes,
    )


def _rate(values: Iterable[bool | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _totals(outcomes: list[StatementOutcome], gold: Sequence[GoldStatement] | None = None) -> dict[str, Any]:
    scored = [o for o in outcomes if o.scored]
    straddle = [o for o in scored if o.straddles_page_break]
    totals: dict[str, Any] = {
        "n_statements": len(outcomes),
        "n_scored": len(scored),
        "extraction": _rate((o.extraction >= DEFAULT_THRESHOLD) if o.extraction is not None else None for o in outcomes),
        "assembly_single": _rate(o.single for o in scored),
        "assembly_bundle": _rate(o.bundle for o in scored),
        "page_accuracy": _rate(o.page_ok for o in scored),
        "section_accuracy": _rate(o.section_ok for o in scored),
        "numbers_lost": _rate(o.numbers_lost for o in scored),
        "n_straddling": len(straddle),
        "straddle_single": _rate(o.single for o in straddle),
        "mean_best_score": statistics.fmean([o.best_score for o in scored]) if scored else None,
    }
    missing = Counter(o.source_id for o in outcomes if not o.scored and o.source_id)
    if missing:
        totals["missing_sources"] = [f"{sid} ({n})" for sid, n in sorted(missing.items())]
    return totals


# ----------------------------------------------------------------- hygiene


def hygiene(chunks: Sequence[ChunkRecord], *, min_tokens: int = 120, max_tokens: int = 512) -> dict[str, Any]:
    if not chunks:
        return {"n_chunks": 0, "tokens_p50": 0, "tokens_p95": 0, "tokens_max": 0, "share_under_min": 0.0,
                "share_over_max": 0.0, "duplicates": 0, "estimated": True, "min_tokens": min_tokens, "max_tokens": max_tokens}
    toks = sorted(c.token_estimate for c in chunks)
    norm = Counter(" ".join(c.text.split()).lower() for c in chunks)
    return {
        "n_chunks": len(chunks),
        "n_documents": len({c.doc_id for c in chunks}),
        "tokens_p50": int(statistics.median(toks)),
        "tokens_p95": toks[min(len(toks) - 1, int(0.95 * len(toks)))],
        "tokens_max": toks[-1],
        "share_under_min": sum(t < min_tokens for t in toks) / len(toks),
        "share_over_max": sum(t > max_tokens for t in toks) / len(toks),
        "duplicates": sum(n - 1 for n in norm.values() if n > 1),
        "estimated": any(c.tokens is None for c in chunks),
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
    }


# --------------------------------------------------------------- retrieval


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    evidence: tuple[str, ...] = ()
    """Statement ids whose support makes a chunk relevant."""
    evidence_docs: tuple[str, ...] = ()
    """Document ids any of whose chunks is relevant — for queries written about
    a whole short document, such as an abstract."""


def load_queries_jsonl(path: str | Path) -> list[Query]:
    """Queries, one JSON object per line: ``query_id``, ``text``, and
    ``evidence`` (statement ids) and/or ``evidence_docs`` (document ids)."""
    out: list[Query] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out.append(
            Query(
                query_id=str(d["query_id"]),
                text=str(d["text"]),
                evidence=tuple(d.get("evidence") or ()),
                evidence_docs=tuple(d.get("evidence_docs") or ()),
            )
        )
    return out


def statement_queries(gold: Sequence[GoldStatement]) -> list[Query]:
    """Each statement as its own query — a **leaky smoke test**, not an evaluation.

    The query shares its wording with the evidence, so it measures whether
    retrieval can find the passage at all. Real query sets are generated from
    held-out source evidence and reviewed.
    """
    return [Query(g.statement_id, g.text, (g.statement_id,)) for g in gold]


class BM25Retriever:
    """Okapi BM25 over the harness's content tokens. A deterministic baseline."""

    def __init__(self, chunks: Sequence[ChunkRecord], *, k1: float = 1.2, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.docs = [Counter(content_tokens(c.text)) for c in self.chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, d in enumerate(self.docs):
            for t in d:
                df[t] += 1
                self.postings[t].append(i)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def __call__(self, query: str, k: int) -> list[str]:
        scores: dict[int, float] = defaultdict(float)
        for t in set(content_tokens(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.postings[t]:
                tf = self.docs[i][t]
                norm = 1 - self.b + self.b * (self.lengths[i] / self.avg if self.avg else 1)
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        ranked = sorted(scores, key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i].chunk_id for i in ranked]


def evaluate_retrieval(
    queries: Sequence[Query],
    gold: Sequence[GoldStatement],
    chunks: Sequence[ChunkRecord],
    *,
    retriever: Callable[[str, int], list[str]] | None = None,
    ks: Sequence[int] = (5, 10, 20),
    token_budgets: Sequence[int] = (1024, 2048),
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Evidence recall at k and at fixed retrieved-token budgets, plus MRR.

    Relevance is decided by evidence support, not by chunk id, so pipelines with
    different chunk boundaries are compared on the same truth.
    """
    if not queries:
        return {}
    retrieve = retriever or BM25Retriever(chunks)
    text_of = {g.statement_id: g.text for g in gold}
    by_id = {c.chunk_id: c for c in chunks}
    index = PassageIndex([c.text for c in chunks])
    pos = {c.chunk_id: i for i, c in enumerate(chunks)}
    depth = max([*ks, 50])
    hits_at = {k: 0 for k in ks}
    hits_budget = {b: 0 for b in token_budgets}
    rr = 0.0
    for q in queries:
        ranked = [cid for cid in retrieve(q.text, depth) if cid in by_id]
        docs = set(q.evidence_docs)
        first = next(
            (
                rank
                for rank, cid in enumerate(ranked)
                if by_id[cid].doc_id in docs
                or any(index.support(text_of[e], pos[cid]).supported(threshold) for e in q.evidence if e in text_of)
            ),
            None,
        )
        if first is not None:
            rr += 1.0 / (first + 1)
            for k in ks:
                hits_at[k] += first < k
            spent = sum(by_id[cid].token_estimate for cid in ranked[: first + 1])
            for b in token_budgets:
                hits_budget[b] += spent <= b
    n = len(queries)
    out = {f"recall@{k}": hits_at[k] / n for k in ks}
    out.update({f"recall@{b}tok": hits_budget[b] / n for b in token_budgets})
    out["mrr"] = rr / n
    return out
