"""Index builder for the tri-hybrid OntoRAG retriever.

Builds three on-disk indexes over a `FoodOnAPI` term set:

  - a Whoosh BM25 index   (lexical: label + synonyms + definition text)
  - a FAISS index of MiniLM embeddings   (general semantic)
  - a FAISS index of SapBERT embeddings  (biomedical synonymy)

All three plus the id-to-row mapping are persisted under one directory so the
expensive build (embedding ~29k FoodOn terms with 2 models) happens once. A
fingerprint of the term set + embedder model ids guards the cache: change
the ontology or a model and the index rebuilds.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from foodscholar.io.ontology import OntologyId
from foodscholar.logging import get_logger

if TYPE_CHECKING:
    from foodscholar.ontology import FoodOnAPI
    from foodscholar.storage.protocols import Embedder

_log = get_logger("foodscholar.annotate.ontorag.index")

# Whoosh schema field names — kept module-level so index + retriever agree.
_F_ID = "term_id"
_F_TEXT = "text"


def _term_text(label: str, synonyms: tuple[str, ...]) -> str:
    """Searchable text for a term: label first, then synonyms."""
    return " ".join([label, *synonyms])


def _fingerprint(ids: list[OntologyId], minilm_id: str, sapbert_id: str) -> str:
    h = hashlib.sha256()
    h.update(f"{minilm_id}|{sapbert_id}".encode())
    for tid in sorted(ids):
        h.update(b"\x00")
        h.update(tid.encode("utf-8"))
    return h.hexdigest()[:16]


@dataclass
class OntoRagIndex:
    """Loaded tri-hybrid index. Held by `OntoRagRetriever`.

    `term_ids` is the row order shared by both FAISS indexes; `whoosh_dir` is
    the Whoosh index directory. `_faiss_minilm` / `_faiss_sapbert` are the
    loaded FAISS index objects.
    """

    index_dir: Path
    term_ids: list[OntologyId]
    whoosh_dir: Path
    _faiss_minilm: object
    _faiss_sapbert: object

    @property
    def size(self) -> int:
        return len(self.term_ids)


def _faiss():
    try:
        import faiss
    except ImportError as e:
        raise ImportError(
            "the 'faiss-cpu' package is required for the OntoRAG retriever. "
            "Install with: pip install 'foodscholar[ontorag]'"
        ) from e
    return faiss


def _whoosh():
    try:
        import whoosh  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "the 'whoosh' package is required for the OntoRAG retriever. "
            "Install with: pip install 'foodscholar[ontorag]'"
        ) from e


def build_index(
    ontology: FoodOnAPI,
    *,
    minilm: Embedder,
    sapbert: Embedder,
    index_dir: str | Path,
    rebuild: bool = False,
) -> OntoRagIndex:
    """Build (or load) the tri-hybrid index for `ontology`.

    `minilm` / `sapbert` are `Embedder`s for the two dense arms. The index is
    persisted under `index_dir`; an existing index whose fingerprint matches
    the term set + embedder ids is loaded instead of rebuilt unless `rebuild`.
    """
    import numpy as np

    _whoosh()
    faiss = _faiss()

    index_dir = Path(index_dir)
    meta_path = index_dir / "meta.json"

    terms = [t for t in ontology if not t.obsolete]
    term_ids = [t.id for t in terms]
    fp = _fingerprint(term_ids, minilm.model_id, sapbert.model_id)

    if not rebuild and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") == fp:
            _log.info("ontorag_index.cache_hit", dir=str(index_dir), n_terms=len(term_ids))
            return _load(index_dir, term_ids)
        _log.info("ontorag_index.cache_stale", dir=str(index_dir))

    _log.info("ontorag_index.building", n_terms=len(term_ids))
    index_dir.mkdir(parents=True, exist_ok=True)
    texts = [_term_text(t.label, t.synonyms) for t in terms]

    # --- Whoosh BM25 index ---
    _build_whoosh(index_dir / "whoosh", term_ids, texts)

    # --- FAISS dense indexes (inner-product on L2-normalized vectors = cosine) ---
    for name, embedder in (("minilm", minilm), ("sapbert", sapbert)):
        vecs = np.asarray(embedder.embed(texts), dtype=np.float32)
        faiss.normalize_L2(vecs)
        idx = faiss.IndexFlatIP(vecs.shape[1])
        idx.add(vecs)
        faiss.write_index(idx, str(index_dir / f"{name}.faiss"))

    (index_dir / "term_ids.txt").write_text("\n".join(term_ids))
    meta_path.write_text(
        json.dumps(
            {"fingerprint": fp, "n_terms": len(term_ids),
             "minilm": minilm.model_id, "sapbert": sapbert.model_id}
        )
    )
    _log.info("ontorag_index.built", dir=str(index_dir))
    return _load(index_dir, term_ids)


def _build_whoosh(whoosh_dir: Path, term_ids: list[OntologyId], texts: list[str]) -> None:
    from whoosh import index as windex
    from whoosh.fields import ID, TEXT, Schema

    whoosh_dir.mkdir(parents=True, exist_ok=True)
    schema = Schema(**{_F_ID: ID(stored=True), _F_TEXT: TEXT})
    ix = windex.create_in(str(whoosh_dir), schema)
    writer = ix.writer()
    for tid, text in zip(term_ids, texts, strict=True):
        writer.add_document(**{_F_ID: tid, _F_TEXT: text})
    writer.commit()


def _load(index_dir: Path, term_ids: list[OntologyId]) -> OntoRagIndex:
    faiss = _faiss()
    # If term_ids weren't passed (pure reload), read them back.
    ids_file = index_dir / "term_ids.txt"
    if not term_ids and ids_file.exists():
        term_ids = ids_file.read_text().splitlines()
    return OntoRagIndex(
        index_dir=index_dir,
        term_ids=term_ids,
        whoosh_dir=index_dir / "whoosh",
        _faiss_minilm=faiss.read_index(str(index_dir / "minilm.faiss")),
        _faiss_sapbert=faiss.read_index(str(index_dir / "sapbert.faiss")),
    )
