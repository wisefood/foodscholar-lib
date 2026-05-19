"""Embedder adapters.

  - `HashEmbedder` — deterministic toy embedder. Free of model dependencies;
    used as the default for `FoodScholar.in_memory()` and in unit tests.
  - `HFEmbedder` — wraps a sentence-transformers / transformers model.
    Lazy-imports torch + transformers; gated by `[annotate]` extra.
  - `SourceTypeRouter` — picks SPECTER2 vs BGE based on `Chunk.source_type`
    per BRIEF §2. Itself an Embedder (composable).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from foodscholar.io.chunk import SourceType

if TYPE_CHECKING:
    from foodscholar.storage.protocols import Embedder


class HashEmbedder:
    """Deterministic toy embedder: same input → same vector, every time.

    Dimension is configurable. Vectors are bounded in [0, 1] component-wise.
    Good enough for unit tests and notebook demos; not semantically meaningful.
    """

    model_id = "hash-embedder-v0"

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            out.append([h[i % len(h)] / 255.0 for i in range(self._dim)])
        return out


class HFEmbedder:
    """sentence-transformers / transformers backed Embedder.

    Default uses SPECTER2 (BRIEF §2). Construction loads the model. Gated by
    `[annotate]` extra. For unit tests, prefer `HashEmbedder`.
    """

    def __init__(self, model_name: str = "allenai/specter2_base") -> None:
        try:
            from sentence_transformers import (  # type: ignore[import-not-found]
                SentenceTransformer,
            )
        except ImportError as e:
            raise ImportError(
                "the 'sentence-transformers' package is required for HFEmbedder. "
                "Install with: pip install 'foodscholar[annotate]'"
            ) from e

        self.model_id = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [v.tolist() for v in vecs]


class SapBERTEmbedder:
    """SapBERT embedder for entity linking (BRIEF §2).

    `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` is trained on UMLS
    biomedical synonymy — short surface forms of the same concept embed close
    together, which is exactly what the linker's dense tier needs ("EVOO"
    near "olive oil"). SapBERT is a plain `transformers` model, so this adapter
    does explicit CLS-token pooling rather than relying on sentence-transformers.

    Lazy-imports torch + transformers; gated by the `[annotate]` extra. Unit
    tests use `HashEmbedder`; real SapBERT runs behind `@pytest.mark.slow`.
    """

    def __init__(
        self,
        model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        *,
        batch_size: int = 64,
    ) -> None:
        try:
            import torch  # noqa: F401
            from transformers import (  # type: ignore[import-not-found]
                AutoModel,
                AutoTokenizer,
            )
        except ImportError as e:
            raise ImportError(
                "the 'torch' + 'transformers' packages are required for SapBERTEmbedder. "
                "Install with: pip install 'foodscholar[annotate]'"
            ) from e

        self.model_id = model_name
        self._batch_size = batch_size
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()

    @property
    def dim(self) -> int:
        return int(self._model.config.hidden_size)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import torch

        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            with torch.no_grad():
                hidden = self._model(**enc).last_hidden_state
            # SapBERT uses the [CLS] token representation.
            cls = hidden[:, 0, :]
            out.extend(v.tolist() for v in cls)
        return out


class SourceTypeRouter:
    """Routes embedding to SPECTER2 vs BGE per BRIEF §2 dispatch rule.

    Use `embed_chunk(text, source_type)` for source-aware routing, or the
    plain `embed(texts)` for the default (scientific) path so this class
    still satisfies the `Embedder` protocol.
    """

    def __init__(self, scientific: Embedder, general: Embedder) -> None:
        self._scientific = scientific
        self._general = general
        # Pick a stable model_id that reflects both backends.
        self.model_id = f"router(scientific={scientific.model_id};general={general.model_id})"

    @property
    def dim(self) -> int:
        # SPECTER2 = 768, BGE-large = 1024. Per BRIEF §7 these go to separate
        # indexes — the router's "dim" only makes sense for one branch at a
        # time. Return the scientific path's dim; callers using both backends
        # must route through embed_chunk and read each chunk's stamped model.
        return self._scientific.dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._scientific.embed(texts)

    def embed_chunk(self, text: str, source_type: SourceType) -> tuple[list[float], str]:
        backend = self._scientific if source_type == "abstract" else self._general
        [vec] = backend.embed([text])
        return vec, backend.model_id
