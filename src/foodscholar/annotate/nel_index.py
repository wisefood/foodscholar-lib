"""Vector index backends for entity linking (NEL).

`NELIndex` is the protocol the linker talks to. Two implementations live here:

  - `HNSWNELIndex` (default) — local `hnswlib` `ip` index over BioLORD (or
    SapBERT / MiniLM / MPNet) embeddings of every FoodOn term. Built on first
    use from the loaded ontology and cached to disk; subsequent runs load
    instantly. Mirrors the validated `gliner.py` prototype.

  - `ElasticNELIndex` — same protocol, backed by Elasticsearch `dense_vector`
    kNN. Stub today (interface frozen); production implementation lands with
    the storage milestone.

Selection happens in the facade via `cfg.annotate.linker.nel_backend`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from foodscholar.io.ontology import OntologyId
from foodscholar.logging import get_logger

if TYPE_CHECKING:
    from foodscholar.ontology import FoodOnAPI

_log = get_logger("foodscholar.annotate.nel_index")

# SentenceTransformer model IDs per supported encoder shortname. The default
# is BioLORD — biomedical synonymy-aware, the prototype's validated choice.
ENCODER_IDS: dict[str, str] = {
    "biolord": "FremyCompany/BioLORD-2023",
    "sapbert": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "minilm": "all-MiniLM-L6-v2",
    "mpnet": "all-mpnet-base-v2",
}


@runtime_checkable
class NELIndex(Protocol):
    """Surface-form → (ontology_id, similarity) lookup over an ontology."""

    backend_id: str

    def link(self, surface: str) -> tuple[OntologyId, float] | None: ...
    def link_batch(
        self, surfaces: list[str]
    ) -> list[tuple[OntologyId, float] | None]: ...


# ---------------------------------------------------------------------- HNSW


def _ontology_signature(ontology: FoodOnAPI) -> str:
    """Deterministic hash over the non-obsolete term ids and labels.

    The cache key derives from this signature plus the encoder name, so a
    re-encode happens iff the ontology *or* the encoder changes. Cheap to
    compute (one SHA over ~30k ids).
    """
    h = hashlib.sha256()
    for term in sorted(
        (t for t in ontology if not t.obsolete), key=lambda t: t.id
    ):
        h.update(term.id.encode("utf-8"))
        h.update(b"\x00")
        h.update(term.label.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _term_text(term: Any) -> str:
    """Indexed text for an ontology term: label + " ; " + synonyms."""
    parts = [term.label]
    parts.extend(term.synonyms)
    return " ; ".join(p for p in parts if p)


class HNSWNELIndex:
    """Local hnswlib `ip` index over per-term embeddings.

    Build-on-first-use, cache-to-disk: if the configured paths exist and the
    sidecar metadata reports a matching signature, the index is `load_index`'d
    in milliseconds. Otherwise the ontology is encoded once with the chosen
    SentenceTransformer, the index is built, and both files are written.
    """

    def __init__(
        self,
        ontology: FoodOnAPI,
        *,
        encoder: str = "biolord",
        top_k: int = 1,
        min_sim: float = 0.70,
        index_path: Path | None = None,
        metadata_path: Path | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        if encoder not in ENCODER_IDS:
            raise ValueError(
                f"unknown NEL encoder {encoder!r} (want one of {sorted(ENCODER_IDS)})"
            )
        self._encoder_name = encoder
        self._encoder_model_id = ENCODER_IDS[encoder]
        self._top_k = max(1, top_k)
        self._min_sim = min_sim
        self._cache: dict[str, tuple[OntologyId, float] | None] = {}
        self.backend_id = f"hnsw-nel({encoder})"

        self._signature = _ontology_signature(ontology)
        index_path, metadata_path = self._resolve_paths(
            index_path, metadata_path, cache_dir
        )
        self._index_path = index_path
        self._metadata_path = metadata_path

        self._encoder: Any = None
        self._index: Any = None
        self._metadata: list[dict[str, str]] = []
        self._dim: int | None = None
        self._build_or_load(ontology)

    # ------------------------------------------------------------------ paths

    def _resolve_paths(
        self,
        index_path: Path | None,
        metadata_path: Path | None,
        cache_dir: Path | None,
    ) -> tuple[Path, Path]:
        if index_path is not None and metadata_path is not None:
            return Path(index_path), Path(metadata_path)
        base = Path(cache_dir or "data")
        stem = f"foodon_hnsw_{self._encoder_name}_{self._signature}"
        return (
            Path(index_path) if index_path else base / f"{stem}.bin",
            Path(metadata_path) if metadata_path else base / f"{stem}.meta.json",
        )

    # ------------------------------------------------------------------ build / load

    def _build_or_load(self, ontology: FoodOnAPI) -> None:
        if self._metadata_path.exists() and self._index_path.exists():
            try:
                meta = json.loads(self._metadata_path.read_text())
                if (
                    meta.get("encoder") == self._encoder_model_id
                    and meta.get("signature") == self._signature
                    and isinstance(meta.get("terms"), list)
                ):
                    self._metadata = meta["terms"]
                    self._dim = int(meta["dim"])
                    self._load_index_only()
                    _log.info(
                        "nel_index.loaded",
                        path=str(self._index_path),
                        n_terms=len(self._metadata),
                    )
                    return
                _log.info(
                    "nel_index.cache_stale",
                    expected_signature=self._signature,
                    cached_signature=meta.get("signature"),
                )
            except Exception as e:
                _log.warning("nel_index.cache_unreadable", error=str(e))

        self._build_fresh(ontology)

    def _load_index_only(self) -> None:
        import hnswlib

        encoder = self._ensure_encoder()
        # Sanity-encode to confirm dim matches the cached index.
        dim = encoder.get_sentence_embedding_dimension()
        if dim != self._dim:
            raise RuntimeError(
                f"NEL cache dim mismatch (cached={self._dim}, encoder={dim})"
            )
        index = hnswlib.Index(space="ip", dim=self._dim)
        index.load_index(str(self._index_path), max_elements=len(self._metadata))
        index.set_ef(max(50, self._top_k * 8))
        self._index = index

    def _build_fresh(self, ontology: FoodOnAPI) -> None:
        import hnswlib
        import numpy as np

        encoder = self._ensure_encoder()
        terms = [t for t in ontology if not t.obsolete]
        if not terms:
            raise ValueError("ontology contains no non-obsolete terms")
        texts = [_term_text(t) for t in terms]
        _log.info(
            "nel_index.building",
            n_terms=len(terms),
            encoder=self._encoder_model_id,
        )
        vectors = encoder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        ).astype(np.float32)
        self._dim = int(vectors.shape[1])

        index = hnswlib.Index(space="ip", dim=self._dim)
        index.init_index(max_elements=len(terms), ef_construction=200, M=32)
        index.add_items(vectors, ids=list(range(len(terms))))
        index.set_ef(max(50, self._top_k * 8))
        self._index = index

        self._metadata = [{"uri": t.id, "label": t.label} for t in terms]

        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        index.save_index(str(self._index_path))
        self._metadata_path.write_text(
            json.dumps(
                {
                    "encoder": self._encoder_model_id,
                    "signature": self._signature,
                    "dim": self._dim,
                    "terms": self._metadata,
                },
                separators=(",", ":"),
            )
        )
        _log.info(
            "nel_index.saved",
            index_path=str(self._index_path),
            metadata_path=str(self._metadata_path),
        )

    def _ensure_encoder(self) -> Any:
        if self._encoder is not None:
            return self._encoder
        try:
            from sentence_transformers import (  # type: ignore[import-not-found]
                SentenceTransformer,
            )
        except ImportError as e:
            raise ImportError(
                "the 'sentence-transformers' package is required for HNSWNELIndex. "
                "Install with: pip install 'foodscholar[annotate]'"
            ) from e
        self._encoder = SentenceTransformer(self._encoder_model_id)
        return self._encoder

    # ------------------------------------------------------------------ NELIndex protocol

    def link(self, surface: str) -> tuple[OntologyId, float] | None:
        if not surface or not surface.strip():
            return None
        key = surface.strip().lower()
        if key in self._cache:
            return self._cache[key]
        [result] = self.link_batch([surface])
        self._cache[key] = result
        return result

    def link_batch(
        self, surfaces: list[str]
    ) -> list[tuple[OntologyId, float] | None]:
        if not surfaces:
            return []

        # Use the surface-form cache to skip repeats — this is the dominant
        # speedup over the prototype's per-surface call (a chunk corpus has
        # massive surface-form repetition).
        keys = [(s.strip().lower() if s else "") for s in surfaces]
        results: list[tuple[OntologyId, float] | None] = [None] * len(surfaces)
        to_encode_idx: list[int] = []
        to_encode_texts: list[str] = []
        for i, (s, k) in enumerate(zip(surfaces, keys, strict=True)):
            if not k:
                results[i] = None
                continue
            if k in self._cache:
                results[i] = self._cache[k]
                continue
            to_encode_idx.append(i)
            to_encode_texts.append(s)

        if not to_encode_idx:
            return results

        import numpy as np

        encoder = self._ensure_encoder()
        vecs = encoder.encode(
            to_encode_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        k = min(self._top_k, len(self._metadata))
        labels_idx, distances = self._index.knn_query(vecs, k=k)

        for row, slot in enumerate(to_encode_idx):
            top1_dist = float(distances[row][0])
            sim = 1.0 - top1_dist
            if sim < self._min_sim:
                hit = None
            else:
                entry = self._metadata[int(labels_idx[row][0])]
                hit = (entry["uri"], sim)
            self._cache[keys[slot]] = hit
            results[slot] = hit
        return results


# ---------------------------------------------------------------------- Elastic


class ElasticNELIndex:
    """Elasticsearch dense_vector kNN backend (opt-in).

    Interface frozen here so the linker can target either backend; the
    implementation lands alongside the full `ElasticChunkStore`.
    """

    backend_id = "elastic-nel"

    def __init__(self, url: str, index: str) -> None:
        self.url = url
        self.index = index
        raise NotImplementedError(
            "ElasticNELIndex is not implemented yet. Use nel_backend='hnsw' "
            "in cfg.annotate.linker until the elastic adapter lands."
        )

    def link(self, surface: str) -> tuple[OntologyId, float] | None:  # pragma: no cover
        raise NotImplementedError

    def link_batch(
        self, surfaces: list[str]
    ) -> list[tuple[OntologyId, float] | None]:  # pragma: no cover
        raise NotImplementedError
