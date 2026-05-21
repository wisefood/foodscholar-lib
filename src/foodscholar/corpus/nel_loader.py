"""Load pre-computed NER + NEL output from the validated prototype.

The prototype writes one output CSV per input file with the shape:

    chunk_id, chunk_entities_ner, chunk_uri_nel

Where `chunk_entities_ner` is a semicolon-separated list of surface forms
and `chunk_uri_nel` is a semicolon-separated list of OBO Foundry URIs (or
empty strings for NIL entities) — 1:1 aligned with the entities.

This loader reads those files and turns each row into a
`(list[Mention], list[EntityLink], list[foodon_id])` tuple so the facade
can attach annotations to chunks without re-running GLiNER + HNSW.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterator
from pathlib import Path

from foodscholar.io.chunk import EntityLink, Mention
from foodscholar.logging import get_logger

_log = get_logger("foodscholar.corpus.nel_loader")

# Bumped at module load so large chunk fields don't trip the stdlib default.
csv.field_size_limit(10 * 1024 * 1024)

REQUIRED_COLUMNS = {"chunk_id", "chunk_entities_ner", "chunk_uri_nel"}

# Matches OBO Foundry purls and shortens them to PREFIX:1234567.
_OBO_URI_RE = re.compile(r"^https?://purl\.obolibrary\.org/obo/([A-Z]+)_([0-9A-Za-z]+)$")

NEL_LINKER_VERSION = "prototype-hnsw-biolord-v1"
NEL_NER_VERSION = "prototype-gliner-bio-v1"


def shorten_obo_uri(uri: str) -> str:
    """Normalize an OBO purl to its PREFIX:LOCALID form. Returns the input
    unchanged if it doesn't match the expected pattern (e.g. already short)."""
    uri = uri.strip()
    if not uri:
        return ""
    match = _OBO_URI_RE.match(uri)
    if not match:
        return uri
    return f"{match.group(1)}:{match.group(2)}"


def iter_nel_rows(
    path: str | Path,
) -> Iterator[tuple[str, list[Mention], list[EntityLink], list[str]]]:
    """Yield (chunk_id, mentions, entity_links, foodon_ids) per row.

    `foodon_ids` carries the deduplicated FOODON: ids only — other OBO
    ontologies (CHEBI, GAZ, …) still appear inside `entity_links` but are
    not surfaced through the FoodOn-specific denormalized field.
    """
    p = Path(path)
    with p.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"{p} is missing required columns: {sorted(missing)}")
        for row in reader:
            chunk_id = (row.get("chunk_id") or "").strip()
            if not chunk_id:
                continue
            surfaces = [s for s in (row.get("chunk_entities_ner") or "").split(";")]
            uris = [u for u in (row.get("chunk_uri_nel") or "").split(";")]
            mentions, links, foodon_ids = _row_to_annotations(surfaces, uris)
            yield chunk_id, mentions, links, foodon_ids


def load_nel_dir(path: str | Path) -> dict[str, tuple[list[Mention], list[EntityLink], list[str]]]:
    """Read every CSV under `path` (sorted for determinism) into a single dict
    keyed on chunk_id. Skips files whose schema doesn't match.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"NEL directory not found: {p}")
    out: dict[str, tuple[list[Mention], list[EntityLink], list[str]]] = {}
    files = sorted(p.glob("*.csv")) if p.is_dir() else [p]
    n_files = 0
    for csv_path in files:
        try:
            for chunk_id, mentions, links, foodon_ids in iter_nel_rows(csv_path):
                out[chunk_id] = (mentions, links, foodon_ids)
        except ValueError as e:
            _log.warning("nel_loader.skipped_file", path=str(csv_path), reason=str(e))
            continue
        n_files += 1
    _log.info("nel_loader.loaded", n_files=n_files, n_chunks=len(out))
    return out


def _row_to_annotations(
    surfaces: list[str], uris: list[str]
) -> tuple[list[Mention], list[EntityLink], list[str]]:
    mentions: list[Mention] = []
    links: list[EntityLink] = []
    foodon_ids: list[str] = []

    # When the two columns have different lengths (rare; usually empty trailing
    # field) the shorter one wins to keep the 1:1 alignment intact.
    n = min(len(surfaces), len(uris))
    for i in range(n):
        surface = (surfaces[i] or "").strip()
        if not surface:
            continue
        # The prototype does not store offsets — use 0:len so the Mention
        # still validates. Layer A/retrieval consume entity_type, not offsets.
        mention = Mention(
            text=surface,
            start=0,
            end=len(surface),
            score=1.0,
            ner_model_version=NEL_NER_VERSION,
            entity_type="other",
        )
        mentions.append(mention)

        ontology_id = shorten_obo_uri(uris[i])
        if not ontology_id:
            continue
        links.append(
            EntityLink(
                mention=mention,
                ontology_id=ontology_id,
                confidence=1.0,
                method="dense",
                linker_version=NEL_LINKER_VERSION,
            )
        )
        if ontology_id.startswith("FOODON:") and ontology_id not in foodon_ids:
            foodon_ids.append(ontology_id)

    return mentions, links, foodon_ids
