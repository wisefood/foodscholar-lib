#!/usr/bin/env python3
"""
pdf_page_triage.py
==================

Scan PDFs page by page, convert each page to Markdown, and ask a local Ollama
model whether the page carries content worth chunking for a nutrition
knowledge graph. Pages judged worthless (covers, TOC, lists of figures,
colophons, blank pages, pure bibliographies, committee member lists, ...) are
written out as `removed_pages`.

Input manifest format (one line per PDF, `removed_pages` may be empty):

    filename: be-dietary-recommendations-for-the-belgian-population.pdf | removed_pages: []

A plain list of `*.pdf` paths (one per line) is also accepted.

Output manifest (same format, filled in):

    filename: be-dietary-recommendations-for-the-belgian-population.pdf | removed_pages: [1, 2, 3, 7]

Page numbers are 1-based (use --zero-based to change that).

Quick start
-----------
    pip install pymupdf pymupdf4llm requests
    ollama pull qwen2.5:14b-instruct        # or any instruct model you like

    python pdf_page_triage.py \
        --manifest guides_removed_pages_log.txt \
        --pdf-dir ../data/all \
        --model mistral-small3.2:24b-instruct-2506-q8_0 \
        --out ../chunking/pdfs.filtered.txt \
        --report ./report.json \
        --dump-md ./md_pages

Reruns are cheap: every decision is cached in `.triage_cache.json` keyed by
model + prompt version + page content hash.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import contextlib
import hashlib
import io
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("Missing dependency: pip install requests")


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

PROMPT_VERSION = "v3"

SYSTEM_PROMPT = """\
You are a document triage filter. A pipeline is about to chunk pages of \
nutrition/public-health documents and feed those chunks into a KNOWLEDGE GRAPH \
about nutrition (entities such as nutrients, foods, food groups, population \
groups, health outcomes, intake values, recommendations, evidence grades).

For the page you are given, decide whether it should be KEPT (it contains \
substantive content a knowledge graph could learn from) or DISCARDED (it is \
structural/navigational/administrative filler that would only add noise).

DISCARD a page when it is essentially:
- cover page, title page, half-title, colophon, imprint, legal/copyright notice, ISBN/depot page
- table of contents, list of chapters, list of tables, list of figures, list of abbreviations \
  that is only a symbol/acronym lookup with no nutritional statements
- index, page of running headers only, blank or near-blank page, page with only a section \
  divider title, a page containing only an image caption or a logo
- acknowledgements, preface/foreword with no scientific content, working-group and committee \
  membership lists, author/affiliation lists, conflict-of-interest declarations
- a pure bibliography / reference list (numbered or author-year citations, no claims)
- boilerplate: how-to-cite, disclaimers, contact details, funding statements, appendix \
  cover sheets, questionnaires or methodological forms with no results

KEEP a page when it contains any of:
- dietary recommendations, guidelines, advice, target or limit intakes
- nutrient or food descriptions, functions, sources, deficiency/excess effects
- reference values (AI, AR, PRI, UL, EAR, RDA), intake data, portions, food-group quantities
- population-group specific guidance (children, pregnancy, elderly, athletes ...)
- health outcomes / risk associations, evidence grading, dose-response statements
- data tables or figure text carrying numbers, units and their interpretation
- definitions of nutrition concepts, methodology text that states criteria actually used
- discussion, conclusions, rationale behind a recommendation

Rules:
- Judge only the page in front of you. Mixed pages (a bit of boilerplate plus real content) => KEEP.
- A page that is mostly a table of numeric nutrition values => KEEP, even if the prose is thin.
- A heading plus one substantive sentence => KEEP. A heading alone => DISCARD.
- When genuinely unsure, KEEP. Recall of real content matters more than purity.

Answer with JSON only."""

USER_TEMPLATE = """\
Document: {doc_title}
Page {page_no} of {n_pages}
Signals: {signals}

--- BEGIN PAGE MARKDOWN ---
{page_md}
--- END PAGE MARKDOWN ---

Return JSON: {{"decision": "keep" | "discard", "page_type": "<short label>", "reason": "<one short sentence>"}}"""

RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["keep", "discard"]},
        "page_type": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["decision", "page_type", "reason"],
}


# --------------------------------------------------------------------------- #
# Manifest I/O
# --------------------------------------------------------------------------- #

MANIFEST_RE = re.compile(
    r"^\s*filename\s*:\s*(?P<name>.+?)\s*\|\s*removed_pages\s*:\s*(?P<pages>\[[^\]]*\])\s*$",
    re.IGNORECASE,
)


@dataclass
class ManifestEntry:
    filename: str
    removed_pages: List[int]


def parse_manifest(path: Path) -> List[ManifestEntry]:
    entries: List[ManifestEntry] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = MANIFEST_RE.match(line)
        if m:
            inner = m.group("pages").strip()[1:-1].strip()
            pages = [int(x) for x in re.findall(r"-?\d+", inner)]
            entries.append(ManifestEntry(m.group("name"), pages))
        elif line.lower().endswith(".pdf"):
            entries.append(ManifestEntry(line, []))
        else:
            print(f"[warn] unparsable manifest line skipped: {line!r}", file=sys.stderr)
    return entries


def format_manifest(entries: List[ManifestEntry]) -> str:
    lines = []
    for e in entries:
        pages = ", ".join(str(p) for p in sorted(e.removed_pages))
        lines.append(f"filename: {e.filename} | removed_pages: [{pages}]")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# PDF -> markdown, one page at a time
# --------------------------------------------------------------------------- #

@contextlib.contextmanager
def _quiet_stdout():
    """Silence C-level chatter from MuPDF (it writes straight to fd 1)."""
    try:
        sys.stdout.flush()
        saved = os.dup(1)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.close(devnull)
    except Exception:
        yield
        return
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        try:
            os.dup2(saved, 1)
            os.close(saved)
        except Exception:
            pass


def _load_backend(prefer_llm_md: bool = True):
    """Return (backend_name, extract) where extract: Path -> List[str] of per-page markdown."""
    if prefer_llm_md:
        try:
            import pymupdf4llm  # noqa: F401

            def extract(path: Path) -> List[str]:
                import pymupdf4llm

                # one pass over the document; page_chunks gives us per-page text
                with _quiet_stdout():
                    chunks = pymupdf4llm.to_markdown(
                        str(path),
                        page_chunks=True,
                        write_images=False,
                        embed_images=False,
                        show_progress=False,
                        use_ocr=False,
                    )
                return [(c.get("text") or "") for c in chunks]

            return "pymupdf4llm", extract
        except Exception:
            pass

    try:
        import fitz  # noqa: F401

        def extract(path: Path) -> List[str]:
            import fitz

            with fitz.open(path) as doc:
                return [pg.get_text("text") for pg in doc]

        return "pymupdf-text", extract
    except Exception:
        pass

    try:
        import pdfplumber  # noqa: F401

        def extract(path: Path) -> List[str]:
            import pdfplumber

            with pdfplumber.open(path) as pdf:
                return [(pg.extract_text() or "") for pg in pdf.pages]

        return "pdfplumber", extract
    except Exception:
        pass

    sys.exit(
        "No PDF backend available. Install one of:\n"
        "  pip install pymupdf pymupdf4llm   (best: real markdown, tables)\n"
        "  pip install pdfplumber            (fallback: plain text)"
    )


def page_signals(md: str) -> Dict[str, Any]:
    text = md.strip()
    lines = [l for l in text.splitlines() if l.strip()]
    digits = sum(c.isdigit() for c in text)
    letters = sum(c.isalpha() for c in text)
    dot_leaders = len(re.findall(r"\.{4,}\s*\d+\s*$", text, re.MULTILINE))
    trailing_nums = len(re.findall(r"\s\d{1,3}\s*$", text, re.MULTILINE))
    return {
        "chars": len(text),
        "lines": len(lines),
        "digit_ratio": round(digits / max(len(text), 1), 3),
        "alpha_ratio": round(letters / max(len(text), 1), 3),
        "dot_leader_lines": dot_leaders,
        "lines_ending_in_number": trailing_nums,
        "has_table_pipes": "|" in text,
    }


# --------------------------------------------------------------------------- #
# Ollama client
# --------------------------------------------------------------------------- #

class Ollama:
    def __init__(
        self,
        host: str,
        model: str,
        num_ctx: int = 8192,
        timeout: int = 180,
        retries: int = 3,
        use_schema: bool = True,
    ):
        self.url = host.rstrip("/") + "/api/chat"
        self.model = model
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.retries = retries
        self.use_schema = use_schema
        self.session = requests.Session()
        self._lock = threading.Lock()

    def check(self) -> None:
        base = self.url.rsplit("/api/", 1)[0]
        try:
            r = self.session.get(base + "/api/tags", timeout=10)
            r.raise_for_status()
        except Exception as e:
            sys.exit(f"Cannot reach Ollama at {base}: {e}\nIs `ollama serve` running?")
        names = {m.get("name", "") for m in r.json().get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            print(
                f"[warn] model {self.model!r} not in local list ({', '.join(sorted(names)) or 'none'}). "
                f"Ollama may try to pull it.",
                file=sys.stderr,
            )

    def judge(self, system: str, user: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
        }
        payload["format"] = RESPONSE_SCHEMA if self.use_schema else "json"

        last_err: Optional[str] = None
        for attempt in range(self.retries):
            try:
                r = self.session.post(self.url, json=payload, timeout=self.timeout)
                if r.status_code == 400 and self.use_schema:
                    # older Ollama: JSON-schema format unsupported
                    with self._lock:
                        self.use_schema = False
                    payload["format"] = "json"
                    continue
                r.raise_for_status()
                content = r.json()["message"]["content"]
                return self._parse(content)
            except Exception as e:  # network hiccup, bad JSON, ...
                last_err = str(e)
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Ollama call failed after {self.retries} tries: {last_err}")

    @staticmethod
    def _parse(content: str) -> Dict[str, Any]:
        content = content.strip()
        content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.MULTILINE).strip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if not m:
                raise
            data = json.loads(m.group(0))
        decision = str(data.get("decision", "")).strip().lower()
        if decision not in ("keep", "discard"):
            raise ValueError(f"bad decision value: {data!r}")
        return {
            "decision": decision,
            "page_type": str(data.get("page_type", ""))[:80],
            "reason": str(data.get("reason", ""))[:300],
        }


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #

class Cache:
    def __init__(self, path: Optional[Path]):
        self.path = path
        self.data: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._dirty = False
        if path and path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}

    @staticmethod
    def key(model: str, md: str) -> str:
        h = hashlib.sha256(f"{PROMPT_VERSION}\x00{model}\x00{md}".encode("utf-8"))
        return h.hexdigest()

    def get(self, k: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.data.get(k)

    def put(self, k: str, v: Dict[str, Any]) -> None:
        with self._lock:
            self.data[k] = v
            self._dirty = True

    def flush(self) -> None:
        if self.path and self._dirty:
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)
            self._dirty = False


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #

def process_pdf(
    pdf_path: Path,
    args: argparse.Namespace,
    client: Ollama,
    cache: Cache,
    extract,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    doc_title = pdf_path.stem.replace("-", " ").replace("_", " ")

    # 1. extract every page to markdown (single pass, PDF handles aren't thread-safe)
    pages_md = [(md or "").strip() for md in extract(pdf_path)]
    n_pages = len(pages_md)

    if args.dump_md:
        md_dir = Path(args.dump_md) / pdf_path.stem
        md_dir.mkdir(parents=True, exist_ok=True)
        for i, md in enumerate(pages_md):
            (md_dir / f"page_{i + 1:04d}.md").write_text(md, encoding="utf-8")

    results: List[Optional[Dict[str, Any]]] = [None] * n_pages

    def work(i: int) -> None:
        md = pages_md[i]
        sig = page_signals(md)
        page_no = i + 1

        if sig["chars"] < args.min_chars:
            results[i] = {
                "page": page_no,
                "decision": "discard",
                "page_type": "blank/near-empty",
                "reason": f"only {sig['chars']} chars of text extracted",
                "source": "heuristic",
                "chars": sig["chars"],
            }
            return

        k = cache.key(client.model, md)
        cached = cache.get(k)
        if cached and not args.no_cache:
            results[i] = {**cached, "page": page_no, "source": "cache", "chars": sig["chars"]}
            return

        snippet = md[: args.max_chars]
        if len(md) > args.max_chars:
            snippet += "\n[...truncated...]"
        user = USER_TEMPLATE.format(
            doc_title=doc_title,
            page_no=page_no,
            n_pages=n_pages,
            signals=json.dumps(sig, ensure_ascii=False),
            page_md=snippet,
        )
        try:
            verdict = client.judge(args.system_prompt, user)
            verdict["source"] = "llm"
        except Exception as e:
            print(f"[warn] {pdf_path.name} p{page_no}: LLM failed ({e}) -> keeping", file=sys.stderr)
            verdict = {
                "decision": "keep",
                "page_type": "unknown",
                "reason": f"llm error: {e}",
                "source": "error",
            }
        else:
            cache.put(k, {kk: verdict[kk] for kk in ("decision", "page_type", "reason")})
        results[i] = {**verdict, "page": page_no, "chars": sig["chars"]}

    todo = list(range(n_pages))
    done = 0
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _ in pool.map(work, todo):
            done += 1
            if args.progress:
                print(f"\r  {pdf_path.name}: {done}/{n_pages} pages", end="", file=sys.stderr)
    if args.progress:
        print(file=sys.stderr)
    cache.flush()

    offset = 0 if args.zero_based else 1
    removed = [
        (r["page"] - 1 + offset) for r in results if r and r["decision"] == "discard"  # type: ignore[index]
    ]
    detail = [
        {**r, "page": r["page"] - 1 + offset} for r in results if r  # type: ignore[index]
    ]
    return removed, detail


def resolve_pdf(name: str, pdf_dir: Optional[str]) -> Optional[Path]:
    p = Path(name)
    if p.is_file():
        return p
    if pdf_dir:
        cand = Path(pdf_dir) / p.name
        if cand.is_file():
            return cand
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Per-page PDF triage with a local Ollama model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="input manifest file (filename: X.pdf | removed_pages: [])")
    src.add_argument("--pdf", nargs="+", help="one or more PDF paths instead of a manifest")

    ap.add_argument("--pdf-dir", help="directory to resolve manifest filenames against")
    ap.add_argument("--out", default="manifest.filtered.txt", help="output manifest path")
    ap.add_argument("--report", help="optional JSON report with per-page decisions and reasons")
    ap.add_argument("--dump-md", help="optional directory to write the extracted per-page markdown")

    ap.add_argument("--model", default="qwen2.5:14b-instruct", help="Ollama model tag")
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--workers", type=int, default=1, help="parallel requests (see OLLAMA_NUM_PARALLEL)")

    ap.add_argument("--min-chars", type=int, default=60, help="pages shorter than this are auto-discarded")
    ap.add_argument("--max-chars", type=int, default=6000, help="page markdown sent to the model is truncated here")
    ap.add_argument("--prompt-file", help="file overriding the built-in system prompt")
    ap.add_argument("--cache", default=".triage_cache.json")
    ap.add_argument("--no-cache", action="store_true", help="ignore cached decisions (still writes them)")
    ap.add_argument("--zero-based", action="store_true", help="emit 0-based page numbers")
    ap.add_argument("--plain-text", action="store_true", help="skip pymupdf4llm, use plain text extraction")
    ap.add_argument("--quiet", dest="progress", action="store_false", help="no progress output")
    args = ap.parse_args()

    args.system_prompt = (
        Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else SYSTEM_PROMPT
    )

    if args.manifest:
        entries = parse_manifest(Path(args.manifest))
    else:
        entries = [ManifestEntry(p, []) for p in args.pdf]
    if not entries:
        sys.exit("Nothing to do: no PDFs found in input.")

    backend, extract = _load_backend(prefer_llm_md=not args.plain_text)
    print(f"[info] extraction backend: {backend}", file=sys.stderr)

    client = Ollama(args.host, args.model, args.num_ctx, args.timeout)
    client.check()
    cache = Cache(None if args.no_cache and not args.cache else Path(args.cache))

    report: Dict[str, Any] = {
        "model": args.model,
        "backend": backend,
        "prompt_version": PROMPT_VERSION,
        "page_numbering": "0-based" if args.zero_based else "1-based",
        "documents": [],
    }

    t0 = time.time()
    for entry in entries:
        pdf_path = resolve_pdf(entry.filename, args.pdf_dir)
        if not pdf_path:
            print(f"[error] not found: {entry.filename} (try --pdf-dir)", file=sys.stderr)
            report["documents"].append({"filename": entry.filename, "error": "file not found"})
            continue
        print(f"[info] {pdf_path.name}", file=sys.stderr)
        removed, detail = process_pdf(pdf_path, args, client, cache, extract)
        entry.removed_pages = sorted(set(entry.removed_pages) | set(removed))
        kept = len(detail) - len(removed)
        print(f"[info]   kept {kept}, removed {len(removed)} of {len(detail)}", file=sys.stderr)
        report["documents"].append(
            {
                "filename": entry.filename,
                "n_pages": len(detail),
                "removed_pages": entry.removed_pages,
                "pages": detail,
            }
        )

    Path(args.out).write_text(format_manifest(entries), encoding="utf-8")
    print(f"[done] manifest -> {args.out}  ({time.time() - t0:.0f}s)", file=sys.stderr)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[done] report   -> {args.report}", file=sys.stderr)
    cache.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
