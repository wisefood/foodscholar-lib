"""End-to-end tests against real ML models.

Marked @pytest.mark.slow — opt-in via `pytest -m slow`. First run downloads
SciFoodNER (~500MB) and SPECTER2 (~440MB) into ~/.cache/huggingface; subsequent
runs use the cache.

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
