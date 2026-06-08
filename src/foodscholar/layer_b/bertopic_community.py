"""BERTopic theme discovery — an alternative to Leiden for Layer B Pass 1.

`run_bertopic(chunk_ids, chunk_store, cfg)` clusters the chunks' cached
embeddings directly (no similarity graph) and returns chunk-id groups, parallel
to `community.run_leiden`. Two clusterers (selected by `cfg.clusterer`):

  - ``hdbscan``: BERTopic's native UMAP + HDBSCAN — discovers the topic count
    from density; the ``-1`` outlier topic is dropped.
  - ``kmeans``: a passthrough reducer + KMeans on the raw BGE vectors (the
    head-to-head notebook recipe) — full coverage, ``n_clusters`` (or auto).

`bertopic` and its deps are lazy-imported (gated by the ``[bertopic]`` extra),
so importing this module is free.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from foodscholar.config import BertopicConfig


def _auto_k(n: int) -> int:
    return max(2, min(12, round(math.sqrt(n / 2))))


def run_bertopic(
    chunk_ids: list[str],
    chunk_store: Any,
    cfg: BertopicConfig,
) -> list[set[str]]:
    """Cluster `chunk_ids` by embedding into topic groups (size-filtered).

    Returns a list of chunk-id sets, each ≥ `cfg.min_topic_size`. The ``-1``
    outlier topic (hdbscan) is never returned. Empty input, too-few-embedded
    chunks, or an all-outlier fit all return ``[]``.
    """
    if not chunk_ids:
        return []

    import numpy as np

    chunks = chunk_store.get_many(list(chunk_ids))
    embedded = [(c.chunk_id, c.embedding) for c in chunks if c.embedding is not None]
    if len(embedded) < cfg.min_topic_size:
        return []

    ids = [cid for cid, _ in embedded]
    M = np.asarray([vec for _, vec in embedded], dtype=np.float32)
    docs = [c.text for c in chunks if c.embedding is not None]

    from bertopic import BERTopic
    from bertopic.dimensionality import BaseDimensionalityReduction
    from bertopic.vectorizers import ClassTfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)

    if cfg.clusterer == "kmeans":
        from sklearn.cluster import KMeans

        k = cfg.n_clusters or _auto_k(len(ids))
        k = max(2, min(k, len(ids) - 1))
        reducer = BaseDimensionalityReduction()  # passthrough — cluster raw vectors
        cluster_model: Any = KMeans(
            n_clusters=k, random_state=cfg.random_state, n_init=10
        )
    else:  # hdbscan — native density clustering on UMAP-reduced vectors
        from hdbscan import HDBSCAN
        from umap import UMAP

        reducer = UMAP(
            n_neighbors=15, n_components=5, min_dist=0.0,
            metric="cosine", random_state=cfg.random_state,
        )
        cluster_model = HDBSCAN(
            min_cluster_size=cfg.min_topic_size, metric="euclidean",
            cluster_selection_method="eom", prediction_data=True,
        )

    topic_model = BERTopic(
        umap_model=reducer,
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,
        calculate_probabilities=False,
        verbose=False,
    )
    try:
        topics = topic_model.fit_transform(docs, embeddings=M)[0]
    except Exception:
        # Degenerate fit (e.g. too few separable points) → no usable themes.
        return []

    groups: dict[int, set[str]] = {}
    for cid, t in zip(ids, topics, strict=True):
        ti = int(t)
        if ti == -1:  # HDBSCAN outlier bucket — never a theme
            continue
        groups.setdefault(ti, set()).add(cid)

    return [g for g in groups.values() if len(g) >= cfg.min_topic_size]
