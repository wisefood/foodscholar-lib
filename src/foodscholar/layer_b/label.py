"""Theme labeling.

Two strategies (per `layer_b_construction_brief.md` §4.5 + the v1 plan):

  - `label_by_keywords(theme_chunks, cfg)` — c-TF-IDF over the union of
    chunks in each theme. Free, deterministic. Returns top-k tokens per
    theme.
  - `label_by_llm(theme_chunks, keywords, llm, cfg)` — for each theme, hand
    the keyword terms + up to 3 sample chunks to the LLM and ask for a 3-5
    word label. One LLM call per theme.

The orchestrator picks one based on `cfg.labeling.strategy`. Keyword is
always computed (cheap, deterministic, gets fed to the LLM as context); LLM
runs on top when `strategy == "llm"` (the v1 default — navigation labels
need to read well).

LLM is injected as an `LLMClient`-protocol object (`fs.llm`); the function
itself is pure logic + I/O via the injected client.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foodscholar.config import LabelingConfig
    from foodscholar.io.chunk import Chunk
    from foodscholar.storage.protocols import LLMClient


_LABEL_PROMPT = """Write a short navigation label (2 to 5 words, lowercase, no
quotes, no punctuation, no explanation) for the following cluster of related
passages from a nutrition knowledge graph.

The label should describe what the passages are ABOUT as a topic phrase, for
example: hydration and fluid intake, fiber-rich whole grains, carbohydrate
counting for diabetes. Do NOT return a single word. Do NOT echo a keyword that
looks like a code or OCR fragment (digits, uppercase IDs, short tokens).

Keywords from the cluster: {keywords}

Sample passages:
1. {chunk_1}
2. {chunk_2}
3. {chunk_3}

Reply with only the label."""


# Token regex for c-TF-IDF garbage filtering. Drops:
#   - tokens containing any digit (OCR codes like "h18567", "1234abc")
#   - tokens of length < 3 (junk like "a", "of" — though sklearn stopwords
#     mostly handles these)
#   - tokens that are pure uppercase length >= 4 (ontology-id leakage)
_GARBAGE_TOKEN = re.compile(r"\d|^.{1,2}$|^[A-Z0-9]{4,}$")


def label_by_keywords(
    theme_chunks: dict[int, list[Chunk]],
    cfg: LabelingConfig,
) -> dict[int, list[str]]:
    """Return `{theme_idx: [top-k discriminative terms]}` via c-TF-IDF.

    Each theme is treated as one "document" (the concatenation of its
    chunks' texts) and TfidfVectorizer runs over all themes' documents.
    Top-k tokens by TF-IDF per theme = the keyword label. English stop words
    are filtered; uni-grams + bigrams.
    """
    if not theme_chunks:
        return {}

    from sklearn.feature_extraction.text import TfidfVectorizer

    theme_ids = sorted(theme_chunks.keys())
    docs = [" ".join(c.text for c in theme_chunks[tid]) for tid in theme_ids]

    vec = TfidfVectorizer(
        max_features=2000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    X = vec.fit_transform(docs)
    terms = vec.get_feature_names_out()
    # Drop OCR codes / ontology-id leakage / 1-2 char fragments before ranking.
    keep_mask = [not _GARBAGE_TOKEN.search(t) for t in terms]
    out: dict[int, list[str]] = {}
    for i, tid in enumerate(theme_ids):
        row = X[i].toarray().ravel()
        order = row.argsort()[::-1]
        kept = [j for j in order if keep_mask[j] and row[j] > 0][: cfg.top_keywords]
        out[tid] = [terms[j] for j in kept]
    return out


def label_by_llm(
    theme_chunks: dict[int, list[Chunk]],
    keywords: dict[int, list[str]],
    llm: LLMClient,
    cfg: LabelingConfig,
) -> dict[int, str]:
    """One LLM call per theme — keyword terms + up to 3 sample chunks → label.

    Strips surrounding quotes (LLMs often add them) and falls back to the
    top keyword if the LLM returns whitespace-only output. Theme texts are
    truncated to 300 chars per chunk to keep the prompt short.
    """
    out: dict[int, str] = {}
    for tid, chunks in theme_chunks.items():
        kws = ", ".join(keywords.get(tid, []))
        # Cycle so we always have 3 slots — single-chunk themes repeat.
        sample = (chunks + chunks + chunks)[:3]
        prompt = _LABEL_PROMPT.format(
            keywords=kws,
            chunk_1=sample[0].text[:300],
            chunk_2=sample[1].text[:300],
            chunk_3=sample[2].text[:300],
        )
        raw = llm.generate(prompt, max_tokens=cfg.llm_max_tokens)
        label = raw.strip().strip('"').strip("'").strip()
        if not label:
            kw_fallback = keywords.get(tid, [])
            label = kw_fallback[0] if kw_fallback else "unlabeled"
        out[tid] = label
    return out
