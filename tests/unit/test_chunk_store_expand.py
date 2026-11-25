"""expand(): neighbours by adjacency instead of overlap, on both chunk stores."""

from __future__ import annotations

from typing import Any

import pytest

from foodscholar.chunking import Chunker, Figure, Heading, Paragraph, Region, write_corpus_csv
from foodscholar.corpus.csv_reader import iter_csv_chunks
from foodscholar.io.chunk import Chunk
from foodscholar.storage.memory import InMemoryChunkStore
from foodscholar.storage.protocols import ChunkStore
from tests.unit.chunking_fixtures import COUNTER, config, ir


@pytest.fixture
def store(tmp_path):
    elements = [Heading(id="h1", text="Fats", level=1)]
    elements += [Paragraph(id=f"a{i}", text=f"Fats paragraph {i}. " * 12, regions=[Region(page=1 + i // 2)]) for i in range(4)]
    elements += [Figure(id="fig", asset_id="plate", caption="Figure 1 The plate.", regions=[Region(page=3)])]
    elements += [Heading(id="h2", text="Drinks", level=1)]
    elements += [Paragraph(id=f"b{i}", text=f"Drinks paragraph {i}. " * 12, regions=[Region(page=4)]) for i in range(2)]
    result = Chunker(config(min_tokens=0, target_tokens=40, max_tokens=64, overlap_max_tokens=8), token_counter=COUNTER).assemble(ir(elements, source_id="DOC"))
    path = write_corpus_csv(result.corpus_rows(), tmp_path / "c.csv")
    s = InMemoryChunkStore()
    s.upsert(iter_csv_chunks(path))
    other = Chunk(chunk_id="z", text="other doc", source_doc_id="OTHER", source_type="guide", section_type="guideline",
                  source_metadata={"ordinal": 1, "parent_id": "x"})
    s.upsert([other])
    return s, result


def test_memory_store_still_satisfies_the_protocol():
    assert isinstance(InMemoryChunkStore(), ChunkStore)


def test_window_returns_neighbours_in_reading_order_within_the_document(store):
    s, result = store
    ids = [c.chunk_id for c in result.chunks]
    got = [c.chunk_id for c in s.expand(ids[2], "window", 1)]
    assert got == ids[1:4]
    assert [c.chunk_id for c in s.expand(ids[0], "window", 2)] == ids[:3]
    assert "z" not in [c.chunk_id for c in s.expand(ids[1], "window", 5)]


def test_section_returns_the_whole_parent_section(store):
    s, result = store
    fats = [c.chunk_id for c in result.chunks if c.section_path == ("Fats",)]
    assert [c.chunk_id for c in s.expand(fats[-1], "section")] == fats


def test_page_and_figure_modes(store):
    s, result = store
    on_page_3 = [c.chunk_id for c in result.chunks if 3 in c.pages]
    assert [c.chunk_id for c in s.expand(result.chunks[0].chunk_id, "page", page=3)] == on_page_3
    with_plate = [c.chunk_id for c in result.chunks if "plate" in c.figure_ids]
    assert with_plate and [c.chunk_id for c in s.expand(with_plate[0], "figure")] == with_plate


def test_legacy_chunks_without_adjacency_expand_to_themselves():
    s = InMemoryChunkStore()
    legacy = Chunk(chunk_id="old", text="t", source_doc_id="a.pdf", source_type="guide", section_type="guideline",
                   source_metadata={"file": "a.pdf", "page_number": 4})
    s.upsert([legacy])
    assert [c.chunk_id for c in s.expand("old", "window", 2)] == ["old"]
    assert [c.chunk_id for c in s.expand("old", "page")] == ["old"]
    assert s.expand("missing") == []
    with pytest.raises(ValueError):
        s.expand("old", "sideways")  # type: ignore[arg-type]


class _RecordingES:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs
        self.bodies: list[dict[str, Any]] = []

    def get(self, index: str, id: str) -> dict[str, Any]:
        for d in self.docs:
            if d["chunk_id"] == id:
                return {"found": True, "_source": d}
        return {"found": False}

    def search(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.bodies.append(body)
        filters = body["query"]["bool"]["filter"]
        (field, values), = filters[0]["terms"].items()
        key = field.removeprefix("source_metadata.")
        doc = next((f["term"]["source_doc_id"] for f in filters[1:] if "term" in f), None)
        hits = []
        for d in self.docs:
            v = d["source_metadata"].get(key)
            vals = [str(x) for x in (v if isinstance(v, list) else [v])] if v is not None else []
            if set(vals) & set(values) and (doc is None or d["source_doc_id"] == doc):
                hits.append({"_id": d["chunk_id"], "_source": d})
        return {"hits": {"hits": hits}}


def test_elastic_store_queries_flattened_metadata_as_keyword_strings(store, monkeypatch):
    monkeypatch.setattr("elasticsearch.Elasticsearch", lambda *a, **k: None, raising=True)
    from foodscholar.storage.elastic import ElasticChunkStore

    s, result = store
    es = ElasticChunkStore(url="http://x", index="chunks")
    es._es = _RecordingES([c.model_dump(mode="json") for c in s.scan()])
    ids = [c.chunk_id for c in result.chunks]
    assert [c.chunk_id for c in es.expand(ids[2], "window", 1)] == ids[1:4]
    body = es._es.bodies[-1]
    assert body["query"]["bool"]["filter"][0] == {"terms": {"source_metadata.ordinal": ["1", "2", "3"]}}
    assert body["query"]["bool"]["filter"][1] == {"term": {"source_doc_id": "DOC"}}
    assert isinstance(es, ChunkStore)
