"""End-to-end tests against real ML models.

Marked @pytest.mark.slow — opt-in via `pytest -m slow`. First run downloads:

  - BGE-base (~440MB), BioLORD (~440MB), SapBERT (~440MB)
  - GLiNER bio-v0.1 (~1.5GB)

into ~/.cache/huggingface; subsequent runs use the cache. Real-provider
tests (Groq) also need the relevant API key.

Skip-imports gate ensures CI without ML deps still passes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
sentence_transformers = pytest.importorskip("sentence_transformers")


@pytest.mark.slow
def test_hf_embedder_bge_base_round_trip() -> None:
    from foodscholar.annotate.embedder import HFEmbedder

    e = HFEmbedder("BAAI/bge-base-en-v1.5")
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
def test_gliner_ner_extracts_mentions_with_correct_offsets() -> None:
    """End-to-end GLinerNER: downloads the bio model, runs a single chunk."""
    pytest.importorskip("gliner")
    from foodscholar.annotate.gliner_ner import GLinerNER

    ner = GLinerNER(
        labels=[
            "food",
            "nutrient",
            "dietary pattern",
            "medical condition",
        ],
        threshold=0.4,
        batch_size=2,
    )
    text = "The Mediterranean diet is rich in olive oil and whole grains."
    mentions = ner.extract(text)
    assert mentions, "expected at least one mention"
    # Offsets must always be exact substrings of the source.
    for m in mentions:
        assert text[m.start : m.end] == m.text


@pytest.mark.slow
def test_gliner_ner_batched_path_matches_per_text() -> None:
    """`extract_batch` and `extract` must produce the same mentions per text."""
    pytest.importorskip("gliner")
    from foodscholar.annotate.gliner_ner import GLinerNER

    ner = GLinerNER(
        labels=["food", "dietary pattern"],
        threshold=0.4,
        batch_size=4,
    )
    texts = [
        "The Mediterranean diet is rich in olive oil.",
        "Apples are a common breakfast food.",
    ]
    by_batch = ner.extract_batch(texts)
    by_single = [ner.extract(t) for t in texts]
    assert len(by_batch) == len(by_single) == 2
    for b, s in zip(by_batch, by_single, strict=True):
        # Order-invariant comparison on (text, start, end, entity_type).
        keys_b = sorted((m.text, m.start, m.end, m.entity_type) for m in b)
        keys_s = sorted((m.text, m.start, m.end, m.entity_type) for m in s)
        assert keys_b == keys_s


@pytest.mark.slow
def test_hnsw_nel_index_links_lexically_distinct_synonym(tmp_path: Path) -> None:
    """End-to-end NEL: BioLORD + HNSW links 'ascorbate' to vitamin C.

    'ascorbate' shares no tokens with 'vitamin C' — pure string matching cannot
    connect them. BioLORD (biomedical paraphrase-aware) embeds 'ascorbate' near
    the 'vitamin C ; ascorbic acid' term text, so the HNSW kNN resolves it.
    """
    pytest.importorskip("hnswlib")
    from foodscholar.annotate.linker import HNSWLinker
    from foodscholar.annotate.nel_index import HNSWNELIndex
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
    index = HNSWNELIndex(
        api,
        encoder="biolord",
        min_sim=0.60,  # documented near-miss — see prototype's 0.70 default
        cache_dir=tmp_path,
    )
    linker = HNSWLinker(index, min_sim=0.60)
    link = linker.link(
        Mention(text="ascorbate", start=0, end=9, score=1.0, ner_model_version="test")
    )
    assert link is not None
    assert link.method == "dense"
    assert link.ontology_id == "FOODON:V1"


@pytest.mark.slow
def test_sapbert_embedder_round_trip() -> None:
    """SapBERT remains available for the `nel_encoder='sapbert'` option."""
    from foodscholar.annotate.embedder import SapBERTEmbedder

    e = SapBERTEmbedder()
    [vec] = e.embed(["olive oil"])
    assert e.dim == 768
    assert len(vec) == 768
