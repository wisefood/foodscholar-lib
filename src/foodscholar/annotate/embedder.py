"""Embedder adapters.

  - `HashEmbedder` — deterministic toy embedder. Free of model dependencies;
    used as the default for `FoodScholar.in_memory()` and in unit tests.
  - `HFEmbedder` — wraps a sentence-transformers / transformers model.
    Lazy-imports torch + transformers; gated by `[annotate]` extra.
  - `SapBERTEmbedder` — UMLS-trained surface-form embedder for entity linking.
"""

from __future__ import annotations

import hashlib


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

    Default is BGE-base (BRIEF §2 — the single production chunk embedder).
    Construction loads the model. Gated by `[annotate]` extra. For unit tests,
    prefer `HashEmbedder`.
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5") -> None:
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
