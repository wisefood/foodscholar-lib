"""The v2 structure-first assembler, on synthetic IR with a word-count tokenizer."""

from __future__ import annotations

import pytest

from foodscholar.chunking import (
    Assembler,
    AssemblyConfig,
    Box,
    Caption,
    Chunker,
    Figure,
    Furniture,
    Heading,
    ListItem,
    Paragraph,
    Region,
    Table,
    TableData,
)
from foodscholar.chunking.assembler import section_id
from foodscholar.chunking.text import RegexSentenceSplitter
from tests.unit.chunking_fixtures import COUNTER, SENTENCE, config, ir


def _sents(n: int) -> str:
    return " ".join([SENTENCE] * n)


def _assemble(elements, **cfg):
    doc = ir(elements, short_name="Book")
    return Chunker(config(**cfg), token_counter=COUNTER).assemble(doc).chunks


def test_section_path_is_the_full_heading_stack():
    chunks = _assemble(
        [
            Heading(id="h1", text="Carbohydrates", level=1),
            Heading(id="h2", text="Digestion", level=2),
            Paragraph(id="p1", text=_sents(5)),
            Heading(id="h3", text="Proteins", level=1),
            Paragraph(id="p2", text=_sents(5)),
        ],
        min_tokens=0,
    )
    assert [c.section_path for c in chunks] == [("Carbohydrates", "Digestion"), ("Proteins",)]
    assert chunks[0].context_header == "Book > Carbohydrates > Digestion"
    assert chunks[0].parent_id == section_id("DOC", ("Carbohydrates", "Digestion"))


def test_a_chunk_never_spans_two_sections():
    elements = []
    for s in range(4):
        elements += [Heading(id=f"h{s}", text=f"Section {s}", level=1), Paragraph(id=f"p{s}", text=_sents(12))]
    chunks = _assemble(elements, min_tokens=0)
    for c in chunks:
        assert len({eid[1:] for eid in c.element_ids}) == 1, c.element_ids


def test_budget_counts_header_content_and_special_tokens():
    chunks = _assemble([Heading(id="h", text="A long heading name", level=1), Paragraph(id="p", text=_sents(60))])
    assert len(chunks) > 1
    for c in chunks:
        assert c.total_tokens <= 128
        assert c.total_tokens == COUNTER.special_tokens + COUNTER.count(c.retrieval_text)


def test_long_paragraph_splits_at_sentence_boundaries():
    chunks = _assemble([Paragraph(id="p", text=_sents(40))])
    for c in chunks:
        assert c.source_text.endswith(".")
        assert c.source_text.startswith("Whole")


def test_short_units_pack_together_up_to_the_target():
    chunks = _assemble([Paragraph(id=f"p{i}", text=SENTENCE) for i in range(20)], min_tokens=0)
    assert len(chunks) == 2  # 20 paragraphs x 10 words at target 100 -> 10 + 10
    assert all(len(c.element_ids) >= 1 for c in chunks)
    assert sum(len(c.element_ids) for c in chunks) == 20


def test_list_stays_whole_when_it_fits_and_splits_between_items_when_not():
    items = [ListItem(id=f"l{i}", text=f"Eat food number {i} daily.", marker="•") for i in range(5)]
    whole = _assemble(items)
    assert len(whole) == 1 and whole[0].unit_kinds == ["list"]
    assert whole[0].source_text.splitlines()[0] == "• Eat food number 0 daily."
    many = [ListItem(id=f"l{i}", text=SENTENCE, marker="-") for i in range(30)]
    split = _assemble(many, min_tokens=0)
    assert len(split) > 1
    for c in split:
        assert all(line.startswith("- ") for line in c.source_text.splitlines())


def test_table_splits_by_rows_and_repeats_the_header():
    rows = [["Food", "Protein (g)"]] + [[f"food {i}", str(i)] for i in range(80)]
    chunks = _assemble([Table(id="t", caption="Table 1 Protein", table=TableData(cells=rows, header_rows=1))])
    assert len(chunks) > 1
    for c in chunks:
        lines = c.source_text.splitlines()
        assert lines[0] == "Table 1 Protein"
        assert lines[2] == "| Food | Protein (g) |"
    body = [ln for c in chunks for ln in c.source_text.splitlines() if ln.startswith("| food")]
    assert len(body) == 80


def test_figure_absorbs_its_paired_caption_and_records_the_asset():
    chunks = _assemble(
        [
            Paragraph(id="p", text="The tract is long."),
            Figure(id="fig", asset_id="asset-1", figure_kind="diagram", regions=[Region(page=4)]),
            Caption(id="cap", text="Figure 3.10 The digestive tract.", target_id="fig", regions=[Region(page=5)]),
        ]
    )
    assert len(chunks) == 1
    c = chunks[0]
    assert "Figure 3.10 The digestive tract." in c.source_text
    assert c.figure_ids == ["asset-1"]
    assert {"fig", "cap"} <= set(c.element_ids)
    assert c.pages == (4, 5)  # the caption sat on the next page


def test_a_caption_whose_figure_is_missing_is_kept_as_text():
    chunks = _assemble([Caption(id="cap", text="Figure 9 Orphan caption here.", target_id="nope")])
    assert chunks and "Orphan caption" in chunks[0].source_text


def test_furniture_is_never_chunked():
    chunks = _assemble([Furniture(id="f", text="Chapter 2 | page 17"), Paragraph(id="p", text=SENTENCE)])
    assert all("page 17" not in c.source_text for c in chunks)


def test_box_keeps_its_title():
    chunks = _assemble([Box(id="b", box_kind="tip", title="Top tip", text="Drink water.")])
    assert chunks[0].source_text == "Top tip\n\nDrink water."


def test_small_sibling_sections_merge_under_their_parent_with_titles_kept():
    chunks = _assemble(
        [
            Heading(id="h", text="Fats", level=1),
            Heading(id="a", text="Oils", level=2),
            Paragraph(id="pa", text="Use olive oil."),
            Heading(id="b", text="Spreads", level=2),
            Paragraph(id="pb", text="Choose low-fat spreads."),
        ]
    )
    assert len(chunks) == 1
    c = chunks[0]
    assert c.section_path == ("Fats",)
    assert c.source_text == "Oils\n\nUse olive oil.\n\nSpreads\n\nChoose low-fat spreads."
    assert c.parent_id == section_id("DOC", ("Fats",))


def test_small_sections_never_merge_across_top_level_headings():
    chunks = _assemble(
        [
            Heading(id="a", text="Fats", level=1),
            Paragraph(id="pa", text="Use olive oil."),
            Heading(id="b", text="Drinks", level=1),
            Paragraph(id="pb", text="Drink water."),
        ]
    )
    assert [c.section_path for c in chunks] == [("Fats",), ("Drinks",)]


def test_merging_can_be_switched_off():
    chunks = _assemble(
        [
            Heading(id="h", text="Fats", level=1),
            Heading(id="a", text="Oils", level=2),
            Paragraph(id="pa", text="Use olive oil."),
            Heading(id="b", text="Spreads", level=2),
            Paragraph(id="pb", text="Choose spreads."),
        ],
        merge_small_sections=False,
    )
    assert len(chunks) == 2


def test_a_short_tail_rejoins_the_previous_chunk():
    chunks = _assemble([Paragraph(id=f"p{i}", text=SENTENCE) for i in range(11)])
    # 110 words at target 100: a 10-word tail is under min_tokens=30 and rejoins.
    assert len(chunks) == 1


def test_overlap_is_zero_by_default_and_an_ablation_when_set():
    paras = [Paragraph(id=f"p{i}", text=f"Sentence {i} one. Sentence {i} two. " * 5) for i in range(6)]
    plain = _assemble(paras, min_tokens=0)
    assert all(c.overlap_tokens == 0 for c in plain)
    overlapped = _assemble(paras, min_tokens=0, overlap_sentences=1, overlap_max_tokens=10)
    assert overlapped[0].overlap_tokens == 0
    assert all(c.overlap_tokens > 0 for c in overlapped[1:])
    last = RegexSentenceSplitter().split(plain[0].source_text)[-1]
    assert overlapped[1].source_text.startswith(last)


def test_regions_collapse_to_one_union_box_per_page():
    chunks = _assemble(
        [
            Paragraph(id="a", text="One.", regions=[Region(page=3, bbox=(10, 10, 100, 50))]),
            Paragraph(id="b", text="Two.", regions=[Region(page=3, bbox=(10, 60, 200, 90)), Region(page=4, bbox=(5, 5, 50, 50))]),
        ]
    )
    regions = chunks[0].regions
    assert [r.page for r in regions] == [3, 4]
    assert regions[0].bbox == (10, 10, 200, 90)


def test_ordinals_are_reading_order_and_output_is_deterministic():
    elements = [Paragraph(id=f"p{i}", text=_sents(6)) for i in range(10)]
    a = _assemble(elements, min_tokens=0)
    b = _assemble(elements, min_tokens=0)
    assert [c.ordinal for c in a] == list(range(len(a)))
    assert [(c.chunk_id, c.source_text) for c in a] == [(c.chunk_id, c.source_text) for c in b]


def test_an_over_long_sentence_is_cut_by_words_as_a_last_resort():
    chunks = _assemble([Paragraph(id="p", text="word " * 400)])
    assert len(chunks) >= 4
    assert all(c.total_tokens <= 128 for c in chunks)


def test_header_can_be_disabled():
    chunks = _assemble([Heading(id="h", text="X", level=1), Paragraph(id="p", text="Text.")], context_header=False)
    assert chunks[0].context_header == ""
    assert chunks[0].retrieval_text == "Text."


def test_deep_headers_are_shortened_from_the_top():
    asm = Assembler(AssemblyConfig(max_tokens=40, target_tokens=30, min_tokens=0, overlap_max_tokens=8), token_counter=COUNTER)
    doc = ir([Heading(id=f"h{i}", text=f"Level {i} title words", level=i + 1) for i in range(6)] + [Paragraph(id="p", text="Body.")], short_name="Book")
    c = asm.assemble(doc)[0]
    assert c.context_header.startswith("Book > … >")
    assert c.context_header.endswith("Level 5 title words")
    assert COUNTER.count(c.context_header) <= 10


@pytest.mark.parametrize(
    "kw",
    [dict(max_tokens=0), dict(target_tokens=600), dict(min_tokens=500), dict(overlap_max_tokens=512)],
)
def test_assembly_config_rejects_inconsistent_budgets(kw):
    with pytest.raises(ValueError):
        AssemblyConfig(**kw)
