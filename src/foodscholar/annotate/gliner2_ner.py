"""GLiNER2 NER — described label set, batched inference.

Sibling to `gliner_ner.GLinerNER`, implementing the same `NER` protocol so the
two are interchangeable via `cfg.annotate.ner`. Neither replaces the other:
GLiNER-bio remains the default, because the evidence for switching is mixed.

The cross-dataset benchmark that selected GLiNER2 scored it against a
GPT-4o-mini proxy ground truth, where it wins F1 0.729 vs 0.630 by trading
recall (0.806 -> 0.709) for precision (0.435 -> 0.610) — about **28% fewer
mentions per passage**. On the single-passage human-scored evaluation,
GLiNER-bio scores highest (F1 0.909). Layer A support counts and the Layer B
relatedness graph key off mention volume, so the switch must be measured
downstream before any default flips. See the integration brief §3.1.

Unlike GLiNER v1, GLiNER2 takes ``{label: description}`` and the descriptions
are part of the model input — `config._GLINER2_DEFAULT_LABELS` holds them
verbatim from the benchmark and should be treated as a fixture.
"""

from __future__ import annotations

from typing import Any, get_args

from foodscholar.io.chunk import GLINER2_TO_ENTITY_TYPE, EntityType, Mention
from foodscholar.logging import get_logger

_log = get_logger("foodscholar.annotate.gliner2_ner")

PROMPT_VERSION = "gliner2-v1"

_VALID_TYPES: frozenset[str] = frozenset(get_args(EntityType))


def map_label(label: str) -> EntityType:
    """GLiNER2 label -> `Mention.entity_type`.

    Folds the labels that differ only in spelling or case onto the established
    member (`disease` -> `medical condition`, `country` -> `Country`), keeps the
    rest verbatim since they are members of the literal, and falls back to
    `other` for anything unrecognized. Mapping to `other` wholesale would throw
    away the richest part of this model's output — `entity_type` feeds
    `ENTITY_TYPE_TO_FACET`, which is how non-`foods` facets get populated.
    """
    mapped = GLINER2_TO_ENTITY_TYPE.get(label, label)
    return mapped if mapped in _VALID_TYPES else "other"  # type: ignore[return-value]


class GLiner2NER:
    """GLiNER2 NER. Lazy model load; batched inference is the fast path."""

    def __init__(
        self,
        *,
        model_id: str = "fastino/gliner2-large-v1",
        threshold: float = 0.35,
        labels: dict[str, str] | None = None,
        batch_size: int = 16,
        quantize: bool = True,
    ) -> None:
        if not labels:
            raise ValueError("GLiner2NER requires a non-empty {label: description} map")
        self._model_id_raw = model_id
        self._threshold = threshold
        self._labels = dict(labels)
        self._batch_size = batch_size
        self._quantize = quantize
        self._model: Any | None = None
        self.model_id = f"gliner2({model_id};t={threshold};labels={len(self._labels)})"

    # ------------------------------------------------------------------ lazy model

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from gliner2 import GLiNER2  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "the 'gliner2' package is required for GLiner2NER. "
                "Install with: pip install 'foodscholar[annotate]'"
            ) from e
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover - torch ships with the extra
            device = "cpu"
        _log.info("gliner2.loading", model=self._model_id_raw, device=device)
        self._model = GLiNER2.from_pretrained(
            self._model_id_raw,
            map_location=device,
            # Quantization is a CUDA-only path in the reference pipeline.
            quantize=(self._quantize and device == "cuda"),
        )
        return self._model

    # ------------------------------------------------------------------ NER protocol

    def extract(self, text: str) -> list[Mention]:
        if not text or not text.strip():
            return []
        return self.extract_batch([text])[0]

    def extract_batch(self, texts: list[str]) -> list[list[Mention]]:
        """Run GLiNER2 over a batch; falls back to per-text on batch failure."""
        if not texts:
            return []
        model = self._ensure_model()

        idx_nonempty = [i for i, t in enumerate(texts) if t and t.strip()]
        nonempty = [texts[i] for i in idx_nonempty]
        results: list[list[Mention]] = [[] for _ in texts]
        if not nonempty:
            return results

        try:
            batch_raw = model.batch_extract_entities(
                nonempty,
                self._labels,
                batch_size=min(self._batch_size, len(nonempty)),
                threshold=self._threshold,
                include_confidence=True,
                include_spans=True,
            )
        except Exception as e:
            _log.warning("gliner2.batch_failed", error=str(e))
            batch_raw = []
            for text in nonempty:
                try:
                    batch_raw.append(
                        model.extract_entities(
                            text,
                            self._labels,
                            threshold=self._threshold,
                            include_confidence=True,
                            include_spans=True,
                        )
                    )
                except Exception as e2:
                    _log.warning("gliner2.per_text_failed", error=str(e2))
                    batch_raw.append({"entities": {}})

        for slot, raw in zip(idx_nonempty, batch_raw, strict=True):
            results[slot] = self._mentions_from_raw(texts[slot], raw)
        return results

    # ------------------------------------------------------------------ helpers

    def _mentions_from_raw(self, text: str, raw: dict) -> list[Mention]:
        """GLiNER2 returns `{"entities": {label: [ {text, start, end, ...} ]}}`.

        Deduplicated per chunk on the lowercased surface, first occurrence
        winning, minimum length 2 — matching the reference `run_ner_batch`.
        Offsets are used when usable and reconstructed via `str.find` otherwise,
        as `GLinerNER` does; unlike the pre-computed NEL loader, this path has
        real offsets available and should not fall back to `0:len`.
        """
        seen: set[str] = set()
        out: list[Mention] = []
        entities = raw.get("entities", {}) if isinstance(raw, dict) else {}
        if not isinstance(entities, dict):
            return out
        for label, items in entities.items():
            if not isinstance(items, list):
                continue
            entity_type = map_label(str(label))
            for ent in items:
                if not isinstance(ent, dict):
                    continue
                surface_raw = " ".join(str(ent.get("text", "")).split())
                if not surface_raw or len(surface_raw) < 2:
                    continue
                key = surface_raw.lower()
                if key in seen:
                    continue

                start = _as_int(ent.get("start"), -1)
                end = _as_int(ent.get("end"), -1)
                if start < 0 or end <= start or end > len(text):
                    idx = text.find(surface_raw)
                    if idx == -1:
                        continue
                    start, end = idx, idx + len(surface_raw)
                surface = " ".join(text[start:end].split()) or surface_raw

                seen.add(key)
                out.append(
                    Mention(
                        text=surface,
                        start=start,
                        end=end,
                        score=float(ent.get("confidence", ent.get("score", 1.0)) or 1.0),
                        ner_model_version=f"{PROMPT_VERSION}:{self._model_id_raw}",
                        entity_type=entity_type,
                    )
                )
        return out


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
