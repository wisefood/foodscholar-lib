"""Surface and predicate deduplication, provenance-preserving.

Ported from kggen's `utils/deduplicate.py`. Two properties of that code are
essential and one is a bug:

**Essential — provenance unions.** When two surfaces collapse, their chunk sets
are merged rather than overwritten. A triple extracted from three chunks that
dedup merges with a variant from two more has five chunks. Losing that would
discard the one thing this pipeline exists for.

**Essential — the normalization.** NFKC, then per-token singularization via
`inflect.singular_noun` ("whole grains" -> "whole grain"), *then* semantic
hashing. The singularization is undocumented upstream but does much of the
collapsing.

**The bug — nondeterminism.** `DeduplicateList.deduplicate` iterates a `set`
and assigns ``items_map[singular] = item`` (last write wins), then hands
``list(set)`` to semhash. Python randomizes string hashing per process, so
whenever two surfaces share a singular form the canonical pick varies between
runs in different processes. We sort every collection before it influences the
outcome, which makes the result reproducible under any ``PYTHONHASHSEED``.

`semhash` and `inflect` are lazy-imported (the `[relations]` extra). Without
them this degrades to exact normalized matching and logs once — the phase still
runs, which is what keeps the in-memory/unit path working.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from foodscholar.logging import get_logger

_log = get_logger("foodscholar.relations.dedupe")

DEDUPE_VERSION = "semhash-v1"
DEDUPE_VERSION_EXACT = "exact-v1"


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", " ".join(text.split()))


def _singularize(text: str, engine: object | None) -> str:
    if engine is None:
        return text
    tokens: list[str] = []
    for token in text.split():
        try:
            singular = engine.singular_noun(token)  # type: ignore[attr-defined]
        except Exception:
            singular = None
        tokens.append(singular if isinstance(singular, str) and singular else token)
    return " ".join(tokens).strip()


def _inflect_engine(enabled: bool) -> object | None:
    if not enabled:
        return None
    try:
        import inflect
    except ImportError:
        _log.warning(
            "relations.dedupe.no_inflect",
            hint="pip install 'foodscholar[relations]' for plural collapsing",
        )
        return None
    return inflect.engine()


@dataclass
class DedupeResult:
    """`alias -> canonical` for every input surface (identity when unmerged)."""

    canonical: dict[str, str]
    version: str

    def resolve(self, surface: str) -> str:
        return self.canonical.get(surface, surface)

    @property
    def n_merged(self) -> int:
        return sum(1 for k, v in self.canonical.items() if k != v)


def deduplicate_surfaces(
    surfaces: list[str],
    *,
    threshold: float = 0.95,
    singularize: bool = True,
) -> DedupeResult:
    """Collapse near-duplicate surfaces, deterministically.

    `surfaces` is sorted before anything else so the canonical representative
    is a function of the input set, not of this process's hash seed.
    """
    ordered = sorted(set(surfaces))
    if not ordered:
        return DedupeResult(canonical={}, version=DEDUPE_VERSION_EXACT)

    engine = _inflect_engine(singularize)

    # normalized form -> the canonical surface (first in sorted order wins,
    # which is stable across processes)
    norm_to_canonical: dict[str, str] = {}
    surface_to_norm: dict[str, str] = {}
    for surface in ordered:
        norm = _singularize(_normalize(surface), engine).lower()
        surface_to_norm[surface] = norm
        norm_to_canonical.setdefault(norm, surface)

    canonical = {s: norm_to_canonical[surface_to_norm[s]] for s in ordered}
    version = DEDUPE_VERSION_EXACT

    try:
        from semhash import SemHash
    except ImportError:
        _log.info(
            "relations.dedupe.no_semhash",
            hint="pip install 'foodscholar[relations]' for semantic dedup",
            n_surfaces=len(ordered),
        )
        return DedupeResult(canonical=canonical, version=version)

    # Semantic pass over the distinct normalized forms, in sorted order.
    records = sorted(norm_to_canonical)
    try:
        result = SemHash.from_records(records=records).self_deduplicate(
            threshold=threshold
        )
    except Exception as e:
        _log.warning("relations.dedupe.semhash_failed", error=str(e))
        return DedupeResult(canonical=canonical, version=version)

    version = DEDUPE_VERSION
    norm_alias: dict[str, str] = {}
    for duplicate in getattr(result, "duplicates", []):
        record = getattr(duplicate, "record", None)
        others = getattr(duplicate, "duplicates", None)
        if not record or not others:
            continue
        first = others[0]
        target = first[0] if isinstance(first, (tuple, list)) and first else first
        if isinstance(target, str) and target != record:
            norm_alias[record] = target

    # Resolve alias chains (a -> b -> c) with a cycle guard.
    def _root(norm: str) -> str:
        seen: set[str] = set()
        while norm in norm_alias and norm not in seen:
            seen.add(norm)
            norm = norm_alias[norm]
        return norm

    canonical = {
        s: norm_to_canonical[_root(surface_to_norm[s])] for s in ordered
    }
    return DedupeResult(canonical=canonical, version=version)
