"""Shared builders for the chunking tests: synthetic IR and tiny .xlsx files.

A plain helper module (imported as ``tests.unit.chunking_fixtures``), not a
pytest conftest, so nothing here runs implicitly.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from foodscholar.chunking import (
    AssemblyConfig,
    ChunkingConfig,
    DocumentIR,
    DocumentMeta,
    SourceProfile,
    WhitespaceTokenCounter,
)

SENTENCE = "Whole grains provide fibre and B vitamins for steady energy."


def meta(source_id: str = "DOC", source_type: str = "textbook", **kw) -> DocumentMeta:
    genre = {"textbook": "long_document", "guide": "leaflet", "abstract": "record"}.get(source_type, "long_document")
    return DocumentMeta(source_id=source_id, source_type=source_type, genre=genre, **kw)


def ir(elements, **meta_kw) -> DocumentIR:
    return DocumentIR(meta=meta(**meta_kw), elements=list(elements))


def config(max_tokens=128, target_tokens=100, min_tokens=30, **assembly_kw) -> ChunkingConfig:
    assembly = AssemblyConfig(max_tokens=max_tokens, target_tokens=target_tokens, min_tokens=min_tokens, **assembly_kw)
    return ChunkingConfig(
        sources={
            "textbook": SourceProfile(genre="long_document", assembly=assembly),
            "guide": SourceProfile(genre="leaflet", assembly=assembly),
            "abstract": SourceProfile(genre="record", assembly=assembly),
        }
    )


COUNTER = WhitespaceTokenCounter()


def write_xlsx(path: Path, sheets: dict[str, list[list[object]]]) -> Path:
    """A minimal valid workbook with inline strings — enough for the stdlib reader."""

    def cell(ref: str, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return f'<c r="{ref}" t="b"><v>{int(value)}</v></c>'
        if isinstance(value, (int, float)):
            return f'<c r="{ref}"><v>{value}</v></c>'
        return f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'

    def col(i: int) -> str:
        s = ""
        i += 1
        while i:
            i, r = divmod(i - 1, 26)
            s = chr(65 + r) + s
        return s

    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    rns = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
        sheet_entries, rels = [], []
        for n, (name, rows) in enumerate(sheets.items(), 1):
            body = "".join(
                f'<row r="{r}">' + "".join(cell(f"{col(c)}{r}", v) for c, v in enumerate(row)) + "</row>"
                for r, row in enumerate(rows, 1)
            )
            z.writestr(f"xl/worksheets/sheet{n}.xml", f"<worksheet {ns}><sheetData>{body}</sheetData></worksheet>")
            sheet_entries.append(f'<sheet name="{escape(name)}" sheetId="{n}" r:id="rId{n}"/>')
            rels.append(f'<Relationship Id="rId{n}" Type="worksheet" Target="worksheets/sheet{n}.xml"/>')
        z.writestr("xl/workbook.xml", f"<workbook {ns} {rns}><sheets>{''.join(sheet_entries)}</sheets></workbook>")
        z.writestr(
            "xl/_rels/workbook.xml.rels",
            f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{"".join(rels)}</Relationships>',
        )
    return path
