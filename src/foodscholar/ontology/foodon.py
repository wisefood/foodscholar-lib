"""FoodOn loader.

Reads a FoodOn `.owl` (or any pronto-supported format) into a list of
`OntologyTerm` objects with ancestors materialized transitively. Optionally
caches the result to Parquet, keyed on the source file's size + mtime so the
cache invalidates the moment FoodOn is updated on disk.

Per BRIEF §2 we run pronto with `import_depth=0` — FoodOn only, no MONDO,
no ChEBI. Those are deferred to v2.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from foodscholar.io.ontology import OntologyTerm

_log = logging.getLogger(__name__)


def load_ontology(
    path: str | Path,
    *,
    cache_path: str | Path | None = None,
    include_imports: bool = False,
    use_cache: bool = True,
) -> list[OntologyTerm]:
    """Load an ontology from `path` into a list of OntologyTerm.

    Reads from the Parquet `cache_path` if it exists and matches the source
    file's size+mtime; otherwise loads via pronto and writes the cache.
    Pure function — no global state.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"ontology file not found: {src}")

    cache = Path(cache_path) if cache_path else None
    cache_meta = cache.with_suffix(cache.suffix + ".meta.json") if cache else None

    if use_cache and cache and cache.exists() and cache_meta and cache_meta.exists():
        if _cache_is_fresh(src, cache_meta):
            _log.info("ontology.cache_hit path=%s", cache)
            return _read_cache(cache)
        _log.info("ontology.cache_stale path=%s — rebuilding", cache)

    terms = _load_via_pronto(src, include_imports=include_imports)

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        _write_cache(terms, cache)
        assert cache_meta is not None
        _write_cache_meta(src, cache_meta)
        _log.info("ontology.cached path=%s n_terms=%d", cache, len(terms))

    return terms


def _load_via_pronto(src: Path, *, include_imports: bool) -> list[OntologyTerm]:
    try:
        import pronto
    except ImportError as e:
        raise ImportError(
            "the 'pronto' package is required to load ontology files. "
            "Install with: pip install 'foodscholar[ontology]'"
        ) from e

    ont = pronto.Ontology(str(src), import_depth=-1 if include_imports else 0)

    terms: list[OntologyTerm] = []
    for t in ont.terms():
        if t.id is None or t.name is None:
            continue
        # pronto returns the term itself in superclasses/subclasses — filter it out.
        ancestors = sorted(s.id for s in t.superclasses() if s.id != t.id)
        parents = sorted(s.id for s in t.superclasses(distance=1) if s.id != t.id)
        exact = tuple(s.description for s in t.synonyms if s.scope == "EXACT")
        related = tuple(s.description for s in t.synonyms if s.scope == "RELATED")
        terms.append(
            OntologyTerm(
                id=t.id,
                label=t.name,
                synonyms=exact,
                related_synonyms=related,
                parent_ids=tuple(parents),
                ancestor_ids=tuple(ancestors),
                obsolete=bool(t.obsolete),
            )
        )
    return terms


# --------------------------------------------------------------------- caching


def _cache_is_fresh(src: Path, meta_path: Path) -> bool:
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return False
    stat = src.stat()
    return meta.get("source_size") == stat.st_size and meta.get("source_mtime") == stat.st_mtime


def _write_cache_meta(src: Path, meta_path: Path) -> None:
    stat = src.stat()
    meta_path.write_text(
        json.dumps({"source_size": stat.st_size, "source_mtime": stat.st_mtime})
    )


def _read_cache(cache_path: Path) -> list[OntologyTerm]:
    import pyarrow.parquet as pq

    table = pq.read_table(cache_path)
    rows = table.to_pylist()
    return [_row_to_term(r) for r in rows]


def _write_cache(terms: list[OntologyTerm], cache_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [_term_to_row(t) for t in terms]
    if not rows:
        # Empty ontology — write an empty file with the right schema.
        rows = [
            {
                "id": "",
                "label": "",
                "synonyms": [],
                "related_synonyms": [],
                "parent_ids": [],
                "ancestor_ids": [],
                "obsolete": False,
            }
        ]
        table = pa.Table.from_pylist(rows).slice(0, 0)
    else:
        table = pa.Table.from_pylist(rows)
    pq.write_table(table, cache_path)


def _term_to_row(t: OntologyTerm) -> dict[str, Any]:
    return {
        "id": t.id,
        "label": t.label,
        "synonyms": list(t.synonyms),
        "related_synonyms": list(t.related_synonyms),
        "parent_ids": list(t.parent_ids),
        "ancestor_ids": list(t.ancestor_ids),
        "obsolete": t.obsolete,
    }


def _row_to_term(row: dict[str, Any]) -> OntologyTerm:
    return OntologyTerm(
        id=row["id"],
        label=row["label"],
        synonyms=tuple(row.get("synonyms") or ()),
        related_synonyms=tuple(row.get("related_synonyms") or ()),
        parent_ids=tuple(row.get("parent_ids") or ()),
        ancestor_ids=tuple(row.get("ancestor_ids") or ()),
        obsolete=bool(row.get("obsolete", False)),
    )
