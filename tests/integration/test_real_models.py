"""End-to-end tests against real ML models.

Marked @pytest.mark.slow — opt-in via `pytest -m slow`. First run downloads
SPECTER2 (~440MB) and SapBERT (~440MB) into ~/.cache/huggingface; subsequent
runs use the cache. Real-provider tests (Groq) also need the relevant API key.

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
def test_hf_embedder_specter2_round_trip() -> None:
    from foodscholar.annotate.embedder import HFEmbedder

    e = HFEmbedder("allenai/specter2_base")
    [vec] = e.embed(["A study of the Mediterranean diet."])
    assert e.dim == 768
    assert len(vec) == 768


def _groq_or_skip():  # type: ignore[no-untyped-def]
    import os

    pytest.importorskip("groq")
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")
    from foodscholar.llm.providers import GroqClient

    return GroqClient("llama-3.3-70b-versatile")


@pytest.mark.slow
def test_groq_llm_client_round_trip() -> None:
    """Real Groq call — needs GROQ_API_KEY and the `groq` SDK ([llm] extra)."""
    client = _groq_or_skip()
    reply = client.generate("Reply with exactly the word: ok", max_tokens=8)
    assert "ok" in reply.lower()


@pytest.mark.slow
def test_groq_generate_json_returns_schema_object() -> None:
    """Real Groq structured-output call via generate_json."""
    client = _groq_or_skip()
    schema = {
        "type": "object",
        "properties": {"color": {"type": "string"}},
        "required": ["color"],
    }
    obj = client.generate_json("What color is a ripe banana? Answer in JSON.", schema)
    assert isinstance(obj, dict)
    assert "color" in obj


@pytest.mark.slow
def test_agentic_ner_with_real_groq() -> None:
    """End-to-end AgenticNER against a real Groq model."""
    client = _groq_or_skip()
    from foodscholar.annotate.agent_ner import AgenticNER

    ner = AgenticNER(client)
    text = "The Mediterranean diet is rich in olive oil and whole grains."
    out = ner.extract(text)
    # The model should find at least one food mention; offsets must be exact.
    assert out, "expected at least one mention"
    for m in out:
        assert text[m.start : m.end] == m.text


@pytest.mark.slow
def test_sapbert_embedder_round_trip() -> None:
    from foodscholar.annotate.embedder import SapBERTEmbedder

    e = SapBERTEmbedder()
    [vec] = e.embed(["olive oil"])
    assert e.dim == 768
    assert len(vec) == 768


@pytest.mark.slow
def test_ontorag_retriever_with_real_embedders(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Tri-hybrid OntoRAG retriever with real MiniLM + SapBERT over the mini
    ontology. Verifies the three arms + RRF merge surface the right term."""
    pytest.importorskip("whoosh")
    pytest.importorskip("faiss")

    from foodscholar.annotate.embedder import HFEmbedder, SapBERTEmbedder
    from foodscholar.annotate.ontorag import OntoRagRetriever, build_index
    from foodscholar.ontology import FoodOnAPI, load_ontology

    api = FoodOnAPI(load_ontology(FIXTURES / "mini_foodon.obo"), prefix_filter=None)
    minilm = HFEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    sapbert = SapBERTEmbedder()
    index = build_index(api, minilm=minilm, sapbert=sapbert, index_dir=tmp_path / "idx")
    retriever = OntoRagRetriever(index, api, minilm=minilm, sapbert=sapbert)

    cands = retriever.retrieve("olive oil", k=5)
    assert cands
    assert any(c.label == "olive oil" for c in cands)


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
        Mention(
            text="ascorbate",
            start=0,
            end=9,
            score=1.0,
            ner_model_version="test",
            entity_type="nutrient",  # food-like → passes the semantic-type gate
        )
    )
    assert link is not None
    assert link.method == "dense"
    assert link.ontology_id == "FOODON:V1"  # vitamin C
