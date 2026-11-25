"""The source registry parses the FSKG catalogue's free-text page notes."""

from __future__ import annotations

import unicodedata

import pytest

from foodscholar.chunking import SourceRegistry
from foodscholar.chunking._xlsx import read_xlsx
from tests.unit.chunking_fixtures import write_xlsx

HEADER = ["Source ID", "Title", "Short name", "Country", "Language", "Pages (catalogue)", "Pages (PDF/notes)", "License ID", "Local copy", "Statement ID range"]


@pytest.fixture
def catalogue(tmp_path):
    rows = [
        HEADER,
        ["SRC-TB-001", "Human Nutrition", "HN 2020", None, "English", "667", "1,208 PDF pages; book content on PDF pages 43-1190; printed page = PDF page - 42.", "CC-BY", "Sources/Textbooks/Human-Nutrition-2020.pdf", "HN0001-HN4027"],
        ["SRC-TB-004", "Kansas", "KS", None, "English", None, "593 PDF pages; no printed page numbers (page = PDF page).", "CC-BY-NC", "Sources/Textbooks/Kansas State University Human Nutrition.pdf", "KS0001–KS1192"],  # noqa: RUF001 — the catalogue uses an en dash here
        ["SRC-IE-002", "Pyramid", "Pyramid", "Ireland", "English", "24", "24 pages (PDF); footer numbers 1-22 on PDF pages 3-24", "CC-BY", "Sources/Guides/Ireland/pyramid.pdf", "C0262-C0394"],
        ["SRC-HU-001", "Okostanyer", "Plate", "Hungary", "Hungarian + English", "3", "3 pages per language; no printed page numbers", "HU", "Sources/Guides/Hungary/OKOSTANYER_felnott.pdf ; Sources/Guides/Hungary/2021_OKOSTANYER_ANGOL_felnott.pdf", "H0001-H0062"],
        ["SRC-HU-004", "Portions", "Portions", "Hungary", "Hungarian", "5", "5 slide pages", "HU", "Sources/Guides/Hungary/képes-élelmiszer-2000kcal.pdf", "H0130-H0146"],
        ["SRC-JN-001", "ARN", "ARN", None, "English", None, "Abstracts retrieved", "CC-BY-4.0", "Sources/Journals/ARN/ARN_abstract_manifest.xlsx", "AR0001-AR0246"],
        ["Total", None, None, None, None, None, None, None, None, None],
    ]
    return write_xlsx(tmp_path / "cat.xlsx", {"README": [["About this workbook"]], "Sources": rows})


def test_xlsx_reader_reads_inline_strings_numbers_and_gaps(tmp_path):
    path = write_xlsx(tmp_path / "x.xlsx", {"S": [["a", None, 3], [1.5, True]]})
    assert read_xlsx(path) == {"S": [["a", None, 3], [1.5, True]]}


def test_types_languages_and_licences(catalogue):
    reg = SourceRegistry.from_xlsx(catalogue)
    assert len(reg) == 6  # README sheet and the Total row are skipped
    assert {r.source_id: r.source_type for r in reg}["SRC-TB-001"] == "textbook"
    assert reg.get("SRC-JN-001").source_type == "abstract"
    assert reg.get("SRC-IE-002").source_type == "guide"
    assert reg.get("SRC-HU-001").languages == ["hu", "en"]
    assert reg.get("SRC-IE-002").extra["Country"] == "Ireland"


def test_content_range_and_printed_offsets(catalogue):
    reg = SourceRegistry.from_xlsx(catalogue)
    hn = reg.get("SRC-TB-001")
    assert hn.content_pages == (43, 1190)
    assert hn.printed_label(43) == "1" and hn.printed_label(100) == "58"
    assert hn.printed_label(10) is None  # front matter before printed page 1
    assert not hn.in_content(10) and hn.in_content(500)
    assert reg.get("SRC-IE-002").printed_label(3) == "1"  # footer numbers 1-22 on PDF pages 3-24


def test_no_printed_numbers_means_no_invented_labels(catalogue):
    reg = SourceRegistry.from_xlsx(catalogue)
    assert reg.get("SRC-TB-004").has_printed_page_numbers is False
    assert reg.get("SRC-TB-004").printed_label(10) is None


def test_edition_pairs_are_one_source_with_a_language_per_file(catalogue):
    hu = SourceRegistry.from_xlsx(catalogue).get("SRC-HU-001")
    assert hu.is_edition_pair
    assert [hu.language_for(c) for c in hu.local_copies] == ["hu", "en"]


def test_resolve_by_path_name_decomposed_unicode_and_decorated_names(catalogue):
    reg = SourceRegistry.from_xlsx(catalogue)
    assert reg.resolve("/data/Sources/Textbooks/Human-Nutrition-2020.pdf").source_id == "SRC-TB-001"
    nfd = unicodedata.normalize("NFD", "képes-élelmiszer-2000kcal.pdf")
    assert reg.resolve(f"/somewhere/{nfd}").source_id == "SRC-HU-004"
    assert reg.resolve("OK_Ireland_pyramid_unknown_all.pdf") is None  # stem too short to trust
    assert reg.resolve("OK_Hungary_2021_OKOSTANYER_ANGOL_felnott_all.pdf").source_id == "SRC-HU-001"
    assert reg.resolve("unrelated.pdf") is None


def test_statement_ranges_map_gold_ids_to_sources_including_en_dashes(catalogue):
    reg = SourceRegistry.from_xlsx(catalogue)
    assert reg.source_for_statement("C0300") == "SRC-IE-002"
    assert reg.source_for_statement("KS0500") == "SRC-TB-004"
    assert reg.source_for_statement("C9999") is None


def test_generic_records_and_csv_for_another_domain(tmp_path):
    path = tmp_path / "reg.csv"
    path.write_text("Source ID,Source type,Title,Language,Local copy,Issuer\nAR-2025,annual_report,Annual report 2025,English,reports/ar2025.pdf,ACME\n")
    rec = SourceRegistry.load(path).get("AR-2025")
    assert rec.source_type == "annual_report" and rec.extra == {"Issuer": "ACME"}
    with pytest.raises(ValueError, match="duplicate"):
        SourceRegistry.from_records([{"Source ID": "A", "Source type": "x"}, {"Source ID": "A", "Source type": "x"}])
