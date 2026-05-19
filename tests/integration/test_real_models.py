"""End-to-end tests against real ML models.

Marked @pytest.mark.slow — opt-in via `pytest -m slow`. First run downloads
SciFoodNER (~500MB), SPECTER2 (~440MB) and SapBERT (~440MB) into
~/.cache/huggingface; subsequent runs use the cache.

Only runs if `transformers` and `sentence-transformers` are importable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
sentence_transformers = pytest.importorskip("sentence_transformers")


@pytest.mark.slow
def test_scifood_ner_extracts_olive_oil() -> None:
    from foodscholar.annotate.ner import SciFoodNERAdapter

    ner = SciFoodNERAdapter()
    out = ner.extract("Mediterranean diet rich in olive oil reduces cardiovascular risk.")
    assert any("olive" in m.text.lower() for m in out)


@pytest.mark.slow
def test_hf_embedder_specter2_round_trip() -> None:
    from foodscholar.annotate.embedder import HFEmbedder

    e = HFEmbedder("allenai/specter2_base")
    [vec] = e.embed(["A study of the Mediterranean diet."])
    assert e.dim == 768
    assert len(vec) == 768


@pytest.mark.slow
def test_groq_llm_client_round_trip() -> None:
    """Real Groq call — needs GROQ_API_KEY and the `groq` SDK ([llm] extra)."""
    import os

    pytest.importorskip("groq")
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")

    from foodscholar.llm.providers import GroqClient

    client = GroqClient("llama-3.3-70b-versatile")
    reply = client.generate("Reply with exactly the word: ok", max_tokens=8)
    assert "ok" in reply.lower()


@pytest.mark.slow
def test_sapbert_embedder_round_trip() -> None:
    from foodscholar.annotate.embedder import SapBERTEmbedder

    e = SapBERTEmbedder()
    [vec] = e.embed(["olive oil"])
    assert e.dim == 768
    assert len(vec) == 768


@pytest.mark.slow
def test_dense_tier_links_lexically_distinct_synonym() -> None:
    """The dense tier's real strength: linking a mention to a term whose name
    shares NO tokens with it, but means the same thing.

    The mention 'ascorbate' shares no tokens with 'vitamin C' — fuzzy matching
    cannot connect them. SapBERT (trained on UMLS synonymy) embeds 'ascorbate'
    at ~0.76 cosine to the 'vitamin C ascorbic acid' term text, well clear of
    the unrelated baseline (~0.32 vs salmon / olive oil), so the dense tier
    resolves it.

    Honest caveats this test encodes:
      - We use dense_threshold=0.70, not the 0.78 production default: SapBERT
        puts 'ascorbate' near vitamin C, but not overwhelmingly. This is a
        real near-miss, documented rather than hidden.
      - SapBERT does NOT reliably link opaque abbreviations ('EVOO' ~ 'olive
        oil' measures only ~0.46). Those are the LLM tier's job. This asserts
        what the dense tier actually delivers, not what we wish it did.
    """
    from foodscholar.annotate.embedder import SapBERTEmbedder
    from foodscholar.annotate.linker import ThreeTierLinker
    from foodscholar.io.chunk import Mention
    from foodscholar.io.ontology import OntologyTerm
    from foodscholar.ontology import FoodOnAPI

    api = FoodOnAPI(
        [
            OntologyTerm(id="FOODON:V1", label="vitamin C", synonyms=("ascorbic acid",)),
            OntologyTerm(id="FOODON:S1", label="salmon"),
            OntologyTerm(id="FOODON:O1", label="olive oil"),
        ],
        prefix_filter=None,
    )
    linker = ThreeTierLinker(
        api,
        fuzzy_threshold=0.99,          # force fuzzy to miss
        dense_embedder=SapBERTEmbedder(),
        dense_threshold=0.70,          # see docstring — a documented near-miss
    )
    link = linker.link(
        Mention(text="ascorbate", start=0, end=9, score=1.0, ner_model_version="test")
    )
    assert link is not None
    assert link.method == "dense"
    assert link.ontology_id == "FOODON:V1"  # vitamin C
