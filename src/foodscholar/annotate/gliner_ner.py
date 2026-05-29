"""GLiNER-bio NER — fine-tuned model that extracts food/health spans.

Wraps `urchade/gliner_large_bio-v0.1` (the prototype-validated model) behind
the `NER` protocol. The fast path is `extract_batch(texts)` — a single
`GLiNER.inference(batch_size=N)` call — which the annotate runner uses to
amortize GPU/model-load cost across many chunks. The per-text `extract(text)`
exists only to satisfy the protocol; callers should prefer `extract_batch`.

GLiNER returns character offsets directly, so unlike `AgenticNER` no local
`str.find` reconciliation is required. Labels are taken verbatim and assigned
to `Mention.entity_type`; since `EntityType` literally contains GLiNER's
vocabulary, the bridge is a no-op string copy.
"""

from __future__ import annotations

from typing import Any, get_args

from foodscholar.io.chunk import EntityType, Mention
from foodscholar.logging import get_logger

_log = get_logger("foodscholar.annotate.gliner_ner")

PROMPT_VERSION = "gliner-v1"

# Valid Mention.entity_type values — derived from the io contract so this
# module and the Pydantic Literal never drift.
_VALID_TYPES: frozenset[str] = frozenset(get_args(EntityType))


class GLinerNER:
    """GLiNER-bio NER. Lazy model load; batched inference is the fast path."""

    def __init__(
        self,
        *,
        model_id: str = "urchade/gliner_large_bio-v0.1",
        threshold: float = 0.4,
        flat_ner: bool = True,
        labels: list[str] | None = None,
        batch_size: int = 16,
        max_length: int = 2048,
    ) -> None:
        if labels is None or not labels:
            raise ValueError("GLinerNER requires at least one label")
        self._model_id_raw = model_id
        self._threshold = threshold
        self._flat_ner = flat_ner
        self._labels = list(labels)
        self._batch_size = batch_size
        self._max_length = max_length
        self._model: Any | None = None
        # Stamp on each Mention; surfaces in fs.info() via NER.model_id.
        self.model_id = (
            f"gliner({model_id};t={threshold};flat={flat_ner};labels={len(self._labels)})"
        )

    # ------------------------------------------------------------------ lazy model

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from gliner import GLiNER  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "the 'gliner' package is required for GLinerNER. "
                "Install with: pip install 'foodscholar[annotate]'"
            ) from e
        _log.info("gliner.loading", model=self._model_id_raw)
        self._model = GLiNER.from_pretrained(self._model_id_raw)
        return self._model

    # ------------------------------------------------------------------ NER protocol

    def extract(self, text: str) -> list[Mention]:
        if not text or not text.strip():
            return []
        return self.extract_batch([text])[0]

    def extract_batch(self, texts: list[str]) -> list[list[Mention]]:
        """Run GLiNER on a batch of texts. Returns one list of `Mention` per text.

        Falls back to per-text inference if the batch call raises — mirrors
        the prototype's defensive behavior on rare CUDA/OOM hiccups.
        """
        if not texts:
            return []
        model = self._ensure_model()

        # Empty inputs short-circuit but keep positional alignment.
        idx_nonempty = [i for i, t in enumerate(texts) if t and t.strip()]
        nonempty = [texts[i] for i in idx_nonempty]
        results: list[list[Mention]] = [[] for _ in texts]
        if not nonempty:
            return results

        try:
            batch_raw = model.inference(
                nonempty,
                self._labels,
                batch_size=min(self._batch_size, len(nonempty)),
                threshold=self._threshold,
                max_length=self._max_length,
                flat_ner=self._flat_ner,
            )
        except Exception as e:
            _log.warning("gliner.batch_failed", error=str(e))
            batch_raw = []
            for text in nonempty:
                try:
                    batch_raw.append(
                        model.predict_entities(
                            text,
                            self._labels,
                            threshold=self._threshold,
                            max_length=self._max_length,
                            flat_ner=self._flat_ner,
                        )
                    )
                except Exception as e2:
                    _log.warning("gliner.per_text_failed", error=str(e2))
                    batch_raw.append([])

        for slot, raw in zip(idx_nonempty, batch_raw, strict=True):
            results[slot] = self._mentions_from_raw(texts[slot], raw)
        return results

    # ------------------------------------------------------------------ helpers

    def _mentions_from_raw(self, text: str, raw: list[dict]) -> list[Mention]:
        seen: set[tuple[str, int]] = set()
        out: list[Mention] = []
        for ent in raw:
            surface_raw = ent.get("text", "")
            if not isinstance(surface_raw, str) or not surface_raw.strip():
                continue
            start = int(ent.get("start", -1))
            end = int(ent.get("end", -1))
            if start < 0 or end <= start or end > len(text):
                # GLiNER occasionally returns slightly off offsets; locate the
                # surface verbatim and reuse the model's surface form.
                idx = text.find(surface_raw)
                if idx == -1:
                    continue
                start = idx
                end = idx + len(surface_raw)
            surface = " ".join(text[start:end].split())
            key = (surface.lower(), start)
            if key in seen:
                continue
            seen.add(key)
            label = ent.get("label", "other")
            entity_type: EntityType = label if label in _VALID_TYPES else "other"  # type: ignore[assignment]
            score = float(ent.get("score", 1.0))
            out.append(
                Mention(
                    text=surface,
                    start=start,
                    end=end,
                    score=score,
                    ner_model_version=f"{PROMPT_VERSION}:{self._model_id_raw}",
                    entity_type=entity_type,
                )
            )
        return out
