"""A minimal, read-only .xlsx reader on the standard library.

Enough for catalogue and gold-statement workbooks: shared and inline strings,
numbers, booleans, sparse cells. No styles, formulas are read as their cached
values, dates stay serial numbers. It exists so the chunker does not need
openpyxl just to read its own source registry.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

_NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}
_REF_RE = re.compile(r"^([A-Z]+)(\d+)$")


def _col_index(letters: str) -> int:
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def _text(node: ET.Element) -> str:
    return "".join(t.text or "" for t in node.iter(f"{{{_NS['m']}}}t"))


def read_xlsx(path: str | Path) -> dict[str, list[list[Any]]]:
    """Return ``{sheet name: rows}``; each row is a list padded to its last cell."""
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            shared = [_text(si) for si in root.findall("m:si", _NS)]

        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        targets = {
            r.get("Id"): r.get("Target", "") for r in rels.findall("pr:Relationship", _NS)
        }
        workbook = ET.fromstring(z.read("xl/workbook.xml"))
        out: dict[str, list[list[Any]]] = {}
        for sheet in workbook.findall("m:sheets/m:sheet", _NS):
            rid = sheet.get(f"{{{_NS['r']}}}id")
            target = targets.get(rid, "")
            member = target.lstrip("/") if target.startswith("/") else f"xl/{target}"
            if member not in names:
                continue
            out[sheet.get("name", member)] = _read_sheet(z.read(member), shared)
        return out


def _read_sheet(xml: bytes, shared: list[str]) -> list[list[Any]]:
    root = ET.fromstring(xml)
    rows: list[list[Any]] = []
    for row in root.findall("m:sheetData/m:row", _NS):
        values: dict[int, Any] = {}
        for pos, cell in enumerate(row.findall("m:c", _NS)):
            ref = cell.get("r", "")
            m = _REF_RE.match(ref)
            col = _col_index(m.group(1)) if m else pos
            values[col] = _cell_value(cell, shared)
        width = max(values, default=-1) + 1
        rows.append([values.get(i) for i in range(width)])
    return rows


def _cell_value(cell: ET.Element, shared: list[str]) -> Any:
    kind = cell.get("t")
    if kind == "inlineStr":
        node = cell.find("m:is", _NS)
        return _text(node) if node is not None else ""
    v = cell.find("m:v", _NS)
    if v is None or v.text is None:
        return None
    raw = v.text
    if kind == "s":
        return shared[int(raw)]
    if kind in ("str", "e"):
        return raw
    if kind == "b":
        return raw == "1"
    try:
        num = float(raw)
    except ValueError:
        return raw
    return int(num) if num.is_integer() else num


def records(rows: list[list[Any]], *, header_row: int = 0) -> list[dict[str, Any]]:
    """Rows → dicts keyed by the header row; blank rows are skipped."""
    if len(rows) <= header_row:
        return []
    header = [str(h).strip() if h is not None else "" for h in rows[header_row]]
    out: list[dict[str, Any]] = []
    for row in rows[header_row + 1 :]:
        if not any(v not in (None, "") for v in row):
            continue
        out.append({h: (row[i] if i < len(row) else None) for i, h in enumerate(header) if h})
    return out
