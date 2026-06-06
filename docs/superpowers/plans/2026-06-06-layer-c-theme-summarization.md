# Layer C — Theme Summarization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production Layer C that summarizes each Layer B theme into a `Card` via a cheap extractive Stage 1 (map-reduce) feeding an LLM Stage 2, plus a read-only benchmark harness that runs all extractive methods over a theme for side-by-side comparison.

**Architecture:** A new `src/foodscholar/layer_c/` package mirroring `layer_b/`. Five extractive summarizers (Sumy LexRank/LSA/Luhn/TextRank + an NLTK frequency summarizer) sit behind a `BaseSummarizer` ABC, selected by a name→factory registry. `stage1` runs the chosen summarizer single-pass or map-reduce; `stage2` refines the extract with `fs.llm.generate_json` into the existing `Card` model; `builder` iterates themes → cards and persists via `graph_store.upsert_cards`. `benchmark` runs every method read-only and emits per-method JSON metrics. `sumy`/`nltk` are lazy-imported behind a new `[summarization]` extra.

**Tech Stack:** Python 3.11+, pydantic, sumy, nltk, pytest, typer (CLI). LLM access via the existing `fs.llm` (`LLMClient` protocol). In-memory stores for tests.

**Spec:** `docs/superpowers/specs/2026-06-06-layer-c-theme-summarization-design.md`

---

## File Structure

**Create:**
- `src/foodscholar/layer_c/__init__.py` — thin exports
- `src/foodscholar/layer_c/base.py` — `BaseSummarizer` ABC + `_join_sentences`/sentence helpers
- `src/foodscholar/layer_c/summarizers.py` — 5 extractive impls + `_ensure_nltk_data()`
- `src/foodscholar/layer_c/registry.py` — name→factory; `build_summarizer`, `all_summarizers`
- `src/foodscholar/layer_c/models.py` — `Stage1Output`, `MethodResult`, `LayerCReport`
- `src/foodscholar/layer_c/stage1.py` — `run_stage1` (single + map-reduce)
- `src/foodscholar/layer_c/stage2.py` — `run_stage2` (extract → `Card`)
- `src/foodscholar/layer_c/persist.py` — `persist_cards`
- `src/foodscholar/layer_c/builder.py` — `build_layer_c(fs, *, facet, dry_run)`
- `src/foodscholar/layer_c/benchmark.py` — `benchmark_theme`, `benchmark_facet`
- Tests: `tests/unit/test_layer_c_config.py`, `test_layer_c_summarizers.py`, `test_layer_c_registry.py`, `test_layer_c_models.py`, `test_layer_c_stage1.py`, `test_layer_c_stage2.py`, `test_layer_c_persist.py`, `test_layer_c_builder.py`

**Modify:**
- `pyproject.toml` — add `[summarization]` extra; add its deps to `all`
- `src/foodscholar/config.py` — extend `LayerCConfig` with new fields
- `src/foodscholar/facade.py:1120-1121` — real `build_layer_c`; add `benchmark_layer_c`
- `src/foodscholar/cli/main.py` — extend `build-layer-c` (facet/dry-run opts); add `bench-layer-c`

---

## Task 1: Dependencies — `[summarization]` extra

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the extra and extend `all`**

In `pyproject.toml`, under `[project.optional-dependencies]`, add a new line after the `bertopic` line:

```toml
summarization = ["sumy", "nltk"]
```

Then in the `all = [` list, add these two entries (anywhere in the list, e.g. after `"bertopic"` if present, else after `"umap-learn"`):

```toml
    "sumy",
    "nltk",
```

- [ ] **Step 2: Verify the file still parses**

Run: `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Install the extra into the dev environment**

Run: `pip install sumy nltk`
Expected: installs succeed (sumy pulls nltk if not already present).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build(layer-c): add [summarization] extra (sumy, nltk)"
```

---

## Task 2: Config — extend `LayerCConfig`

**Files:**
- Modify: `src/foodscholar/config.py` (the `LayerCConfig` class, currently ~lines 608-614)
- Test: `tests/unit/test_layer_c_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_layer_c_config.py`:

```python
"""Layer C config: new Stage-1 / map-reduce / benchmark fields."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from foodscholar.config import LayerCConfig


def test_layer_c_defaults() -> None:
    c = LayerCConfig()
    # existing fields preserved
    assert c.llm_model == "claude-sonnet-4-6"
    assert c.prompt_version == "v1"
    assert c.grounding_check == "strict"
    # new fields
    assert c.stage1_method == "lexrank"
    assert c.stage1_sentences == 8
    assert c.map_reduce_threshold == 400
    assert c.group_char_budget == 20_000
    assert c.max_summary_chars == 4000
    assert c.benchmark_out_dir == "data/layer_c_bench"


def test_layer_c_rejects_unknown_method() -> None:
    with pytest.raises(ValidationError):
        LayerCConfig(stage1_method="bogus")


def test_layer_c_forbids_extra() -> None:
    with pytest.raises(ValidationError):
        LayerCConfig(nonexistent_field=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_layer_c_config.py -v`
Expected: FAIL (`AttributeError`/`ValidationError` — `stage1_method` not defined).

- [ ] **Step 3: Extend `LayerCConfig`**

In `src/foodscholar/config.py`, replace the existing `LayerCConfig` class body. The current class is:

```python
class LayerCConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    llm_model: str = "claude-sonnet-4-6"
    prompt_version: str = "v1"
    sample_size: int = 12
    grounding_check: Literal["strict", "lenient", "off"] = "strict"
    safety_sensitive_facets: list[Facet] = Field(default_factory=lambda: ["allergies"])
```

Replace with (keep all existing fields, append the new block):

```python
class LayerCConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    llm_model: str = "claude-sonnet-4-6"
    prompt_version: str = "v1"
    sample_size: int = 12
    grounding_check: Literal["strict", "lenient", "off"] = "strict"
    safety_sensitive_facets: list[Facet] = Field(default_factory=lambda: ["allergies"])
    # Stage-1 extractive summarization
    stage1_method: Literal["lexrank", "lsa", "luhn", "textrank", "nltk_freq"] = "lexrank"
    stage1_sentences: int = 8
    """Sentence budget per extractive pass (top-N sentences kept)."""
    # map-reduce scaling
    map_reduce_threshold: int = 400
    """Total input sentences above which Stage 1 switches to map-reduce."""
    group_char_budget: int = 20_000
    """Max characters per map group when map-reduce is active."""
    # Stage-2 guard
    max_summary_chars: int = 4000
    """Strict-grounding length cap on the Stage-2 summary."""
    # benchmark harness
    benchmark_out_dir: str = "data/layer_c_bench"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_layer_c_config.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/config.py tests/unit/test_layer_c_config.py
git commit -m "feat(layer-c): extend LayerCConfig with stage-1/map-reduce/benchmark knobs"
```

---

## Task 3: `BaseSummarizer` ABC + sentence helpers

**Files:**
- Create: `src/foodscholar/layer_c/__init__.py`
- Create: `src/foodscholar/layer_c/base.py`
- Test: `tests/unit/test_layer_c_summarizers.py` (shared with later tasks)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_layer_c_summarizers.py`:

```python
"""Layer C extractive summarizers + the BaseSummarizer contract."""

from __future__ import annotations

import pytest

from foodscholar.layer_c.base import BaseSummarizer, split_sentences


class _Echo(BaseSummarizer):
    name = "echo"

    def summarize(self, chunks: list[str]) -> str:
        return " ".join(chunks)


def test_base_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseSummarizer()  # type: ignore[abstract]


def test_concrete_subclass_runs() -> None:
    assert _Echo().summarize(["a", "b"]) == "a b"


def test_split_sentences_counts() -> None:
    text = "Apples are sweet. Pears are juicy. Rice is a grain."
    sents = split_sentences(text)
    assert len(sents) == 3
    assert sents[0].startswith("Apples")


def test_split_sentences_empty() -> None:
    assert split_sentences("") == []
    assert split_sentences("   ") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_layer_c_summarizers.py -v`
Expected: FAIL (`ModuleNotFoundError: foodscholar.layer_c.base`).

- [ ] **Step 3: Create the package init**

Create `src/foodscholar/layer_c/__init__.py`:

```python
"""Layer C — theme summarization (extractive Stage 1 → LLM Stage 2)."""
```

- [ ] **Step 4: Create `base.py`**

Create `src/foodscholar/layer_c/base.py`:

```python
"""BaseSummarizer contract + a lightweight sentence splitter.

The splitter is regex-based and dependency-free so the ABC and helpers import
without sumy/nltk. Concrete summarizers in `summarizers.py` lazy-import their
heavy backends.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into non-empty, stripped sentences (regex, no deps)."""
    if not text or not text.strip():
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


class BaseSummarizer(ABC):
    """Common interface for every Stage-1 extractive method.

    Implementations take a list of chunk texts and return a single extractive
    summary string. The sentence budget is supplied at construction time.
    """

    name: str = "base"

    @abstractmethod
    def summarize(self, chunks: list[str]) -> str:
        """Return an extractive summary of `chunks` (joined text)."""
        ...
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/test_layer_c_summarizers.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add src/foodscholar/layer_c/__init__.py src/foodscholar/layer_c/base.py tests/unit/test_layer_c_summarizers.py
git commit -m "feat(layer-c): BaseSummarizer ABC + sentence splitter"
```

---

## Task 4: NLTK frequency summarizer (no sumy)

**Files:**
- Create: `src/foodscholar/layer_c/summarizers.py`
- Test: `tests/unit/test_layer_c_summarizers.py` (append)

The NLTK frequency summarizer is implemented first because it has the simplest backend and lets us validate the lazy-import + nltk-data pattern before adding the four sumy wrappers.

- [ ] **Step 1: Write the failing test (append to the existing file)**

Append to `tests/unit/test_layer_c_summarizers.py`:

```python
from foodscholar.layer_c.summarizers import NLTKFrequencySummarizer  # noqa: E402

_NLTK = pytest.importorskip("nltk")

_DOCS = [
    "Oats are a whole grain rich in soluble fiber called beta glucan.",
    "Beta glucan in oats can lower cholesterol and improve heart health.",
    "Rice is a staple cereal grain eaten across the world.",
    "Wheat flour is milled from wheat and used to bake bread.",
    "Barley is another cereal grain used in soups and brewing.",
]


def test_nltk_freq_respects_budget() -> None:
    s = NLTKFrequencySummarizer(n=2)
    out = s.summarize(_DOCS)
    assert out  # non-empty
    assert len(split_sentences(out)) <= 2


def test_nltk_freq_empty_input() -> None:
    assert NLTKFrequencySummarizer(n=3).summarize([]) == ""
    assert NLTKFrequencySummarizer(n=3).summarize(["", "   "]) == ""


def test_nltk_freq_fewer_than_budget_returns_all() -> None:
    s = NLTKFrequencySummarizer(n=10)
    out = s.summarize(["Only one sentence here."])
    assert "Only one sentence here." in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_layer_c_summarizers.py -v`
Expected: FAIL (`ImportError`: cannot import `NLTKFrequencySummarizer`).

- [ ] **Step 3: Create `summarizers.py` with the NLTK frequency method**

Create `src/foodscholar/layer_c/summarizers.py`:

```python
"""Five extractive Stage-1 summarizers behind BaseSummarizer.

`sumy` and `nltk` are lazy-imported inside methods (gated by the
`[summarization]` extra), so importing this module does not require them.
NLTK data (`punkt`, `punkt_tab`, `stopwords`) is fetched on first use.
"""

from __future__ import annotations

from collections import defaultdict

from foodscholar.layer_c.base import BaseSummarizer, split_sentences

_NLTK_READY = False


def _ensure_nltk_data() -> None:
    """Download the nltk resources the summarizers need, once per process."""
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk

    for pkg, path in [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)
    _NLTK_READY = True


def _concat(chunks: list[str]) -> str:
    return "\n".join(c for c in chunks if c and c.strip())


class NLTKFrequencySummarizer(BaseSummarizer):
    """Word-frequency extractive summarizer (stopword-filtered, normalized)."""

    name = "nltk_freq"

    def __init__(self, n: int = 8) -> None:
        self.n = n

    def summarize(self, chunks: list[str]) -> str:
        text = _concat(chunks)
        sentences = split_sentences(text)
        if not sentences:
            return ""
        if len(sentences) <= self.n:
            return " ".join(sentences)

        _ensure_nltk_data()
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        stop = set(stopwords.words("english"))
        freq: dict[str, float] = defaultdict(float)
        for w in word_tokenize(text.lower()):
            if w.isalpha() and w not in stop:
                freq[w] += 1.0
        if not freq:
            return " ".join(sentences[: self.n])
        peak = max(freq.values())
        for w in freq:
            freq[w] /= peak

        scored: list[tuple[int, float]] = []
        for i, sent in enumerate(sentences):
            words = [w for w in word_tokenize(sent.lower()) if w.isalpha()]
            score = sum(freq.get(w, 0.0) for w in words)
            scored.append((i, score))

        top_idx = sorted(
            (i for i, _ in sorted(scored, key=lambda t: t[1], reverse=True)[: self.n])
        )
        return " ".join(sentences[i] for i in top_idx)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_layer_c_summarizers.py -v`
Expected: PASS (the 3 new tests + the 4 from Task 3). If nltk data download is blocked, the budget test still passes for short input via the early return; the larger `_DOCS` test exercises the download path.

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/layer_c/summarizers.py tests/unit/test_layer_c_summarizers.py
git commit -m "feat(layer-c): NLTK frequency extractive summarizer + nltk-data bootstrap"
```

---

## Task 5: Sumy summarizers (LexRank, LSA, Luhn, TextRank)

**Files:**
- Modify: `src/foodscholar/layer_c/summarizers.py`
- Test: `tests/unit/test_layer_c_summarizers.py` (append)

- [ ] **Step 1: Write the failing test (append)**

Append to `tests/unit/test_layer_c_summarizers.py`:

```python
pytest.importorskip("sumy")
from foodscholar.layer_c.summarizers import (  # noqa: E402
    SumyLexRankSummarizer,
    SumyLsaSummarizer,
    SumyLuhnSummarizer,
    SumyTextRankSummarizer,
)


@pytest.mark.parametrize(
    "cls",
    [SumyLexRankSummarizer, SumyLsaSummarizer, SumyLuhnSummarizer, SumyTextRankSummarizer],
)
def test_sumy_methods_respect_budget(cls) -> None:
    out = cls(n=2).summarize(_DOCS)
    assert out
    assert len(split_sentences(out)) <= 2


@pytest.mark.parametrize(
    "cls",
    [SumyLexRankSummarizer, SumyLsaSummarizer, SumyLuhnSummarizer, SumyTextRankSummarizer],
)
def test_sumy_methods_empty_input(cls) -> None:
    assert cls(n=3).summarize([]) == ""


def test_sumy_names() -> None:
    assert SumyLexRankSummarizer(n=1).name == "lexrank"
    assert SumyLsaSummarizer(n=1).name == "lsa"
    assert SumyLuhnSummarizer(n=1).name == "luhn"
    assert SumyTextRankSummarizer(n=1).name == "textrank"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_layer_c_summarizers.py -v`
Expected: FAIL (`ImportError`: cannot import the Sumy classes).

- [ ] **Step 3: Add the sumy wrappers to `summarizers.py`**

Append to `src/foodscholar/layer_c/summarizers.py`:

```python
class _SumyBase(BaseSummarizer):
    """Shared parse/tokenize/run/join for the four sumy algorithms.

    Subclasses set `name` and implement `_algo()` returning a sumy summarizer.
    """

    def __init__(self, n: int = 8) -> None:
        self.n = n

    def _algo(self):  # noqa: ANN202 - sumy type, lazy-imported
        raise NotImplementedError

    def summarize(self, chunks: list[str]) -> str:
        text = _concat(chunks)
        sentences = split_sentences(text)
        if not sentences:
            return ""
        if len(sentences) <= self.n:
            return " ".join(sentences)

        _ensure_nltk_data()
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.parsers.plaintext import PlaintextParser

        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        picked = self._algo()(parser.document, self.n)
        out = " ".join(str(s) for s in picked)
        return out or " ".join(sentences[: self.n])


class SumyLexRankSummarizer(_SumyBase):
    name = "lexrank"

    def _algo(self):  # noqa: ANN202
        from sumy.summarizers.lex_rank import LexRankSummarizer

        return LexRankSummarizer()


class SumyLsaSummarizer(_SumyBase):
    name = "lsa"

    def _algo(self):  # noqa: ANN202
        from sumy.summarizers.lsa import LsaSummarizer

        return LsaSummarizer()


class SumyLuhnSummarizer(_SumyBase):
    name = "luhn"

    def _algo(self):  # noqa: ANN202
        from sumy.summarizers.luhn import LuhnSummarizer

        return LuhnSummarizer()


class SumyTextRankSummarizer(_SumyBase):
    name = "textrank"

    def _algo(self):  # noqa: ANN202
        from sumy.summarizers.text_rank import TextRankSummarizer

        return TextRankSummarizer()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_layer_c_summarizers.py -v`
Expected: PASS (all summarizer tests).

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/layer_c/summarizers.py tests/unit/test_layer_c_summarizers.py
git commit -m "feat(layer-c): sumy LexRank/LSA/Luhn/TextRank summarizers"
```

---

## Task 6: Registry (name → factory)

**Files:**
- Create: `src/foodscholar/layer_c/registry.py`
- Test: `tests/unit/test_layer_c_registry.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_layer_c_registry.py`:

```python
"""Layer C summarizer registry."""

from __future__ import annotations

import pytest

from foodscholar.config import LayerCConfig
from foodscholar.layer_c.registry import (
    SUMMARIZERS,
    all_summarizers,
    build_summarizer,
)


def test_registry_has_five_methods() -> None:
    assert set(SUMMARIZERS) == {"lexrank", "lsa", "luhn", "textrank", "nltk_freq"}


def test_build_summarizer_returns_named() -> None:
    cfg = LayerCConfig(stage1_sentences=5)
    s = build_summarizer("nltk_freq", cfg)
    assert s.name == "nltk_freq"
    assert s.n == 5


def test_build_summarizer_unknown_raises() -> None:
    with pytest.raises(KeyError):
        build_summarizer("bogus", LayerCConfig())


def test_all_summarizers_returns_five() -> None:
    methods = {s.name for s in all_summarizers(LayerCConfig())}
    assert methods == {"lexrank", "lsa", "luhn", "textrank", "nltk_freq"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_layer_c_registry.py -v`
Expected: FAIL (`ModuleNotFoundError: foodscholar.layer_c.registry`).

- [ ] **Step 3: Create `registry.py`**

Create `src/foodscholar/layer_c/registry.py`:

```python
"""Name → factory registry for Stage-1 summarizers.

Single source of truth: the builder selects one method via
`config.layer_c.stage1_method`; the benchmark harness iterates all of them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from foodscholar.layer_c.base import BaseSummarizer
from foodscholar.layer_c.summarizers import (
    NLTKFrequencySummarizer,
    SumyLexRankSummarizer,
    SumyLsaSummarizer,
    SumyLuhnSummarizer,
    SumyTextRankSummarizer,
)

if TYPE_CHECKING:
    from foodscholar.config import LayerCConfig

SUMMARIZERS: dict[str, Callable[["LayerCConfig"], BaseSummarizer]] = {
    "lexrank": lambda c: SumyLexRankSummarizer(n=c.stage1_sentences),
    "lsa": lambda c: SumyLsaSummarizer(n=c.stage1_sentences),
    "luhn": lambda c: SumyLuhnSummarizer(n=c.stage1_sentences),
    "textrank": lambda c: SumyTextRankSummarizer(n=c.stage1_sentences),
    "nltk_freq": lambda c: NLTKFrequencySummarizer(n=c.stage1_sentences),
}


def build_summarizer(name: str, cfg: "LayerCConfig") -> BaseSummarizer:
    """Return the BaseSummarizer for `name`, configured from `cfg`."""
    return SUMMARIZERS[name](cfg)


def all_summarizers(cfg: "LayerCConfig") -> list[BaseSummarizer]:
    """Return one instance of every registered summarizer (for the harness)."""
    return [factory(cfg) for factory in SUMMARIZERS.values()]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_layer_c_registry.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/layer_c/registry.py tests/unit/test_layer_c_registry.py
git commit -m "feat(layer-c): name->factory summarizer registry"
```

---

## Task 7: Internal models

**Files:**
- Create: `src/foodscholar/layer_c/models.py`
- Test: `tests/unit/test_layer_c_models.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_layer_c_models.py`:

```python
"""Layer C internal models."""

from __future__ import annotations

from foodscholar.layer_c.models import LayerCReport, MethodResult, Stage1Output


def test_stage1_output_fields() -> None:
    o = Stage1Output(text="x", n_input_chunks=3, n_input_chars=10,
                     strategy="single", n_groups=1)
    assert o.strategy == "single"
    assert o.n_groups == 1


def test_method_result_roundtrip() -> None:
    r = MethodResult(method="lexrank", summary="s", input_chunks=243,
                     input_chars=184532, execution_time_ms=412,
                     summary_length_chars=1840)
    d = r.model_dump()
    assert d["method"] == "lexrank"
    assert d["input_chunks"] == 243
    assert d["summary_length_chars"] == 1840


def test_layer_c_report_defaults() -> None:
    rep = LayerCReport(n_themes=5, n_cards=4, n_skipped=1, n_failed=0)
    assert rep.strategy_counts == {}
    assert "5" in str(rep)  # __str__ mentions theme count
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_layer_c_models.py -v`
Expected: FAIL (`ModuleNotFoundError: foodscholar.layer_c.models`).

- [ ] **Step 3: Create `models.py`**

Create `src/foodscholar/layer_c/models.py`:

```python
"""Internal Layer C models: Stage-1 provenance, benchmark records, run report."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Stage1Output(BaseModel):
    """The extractive summary plus the provenance the builder records."""

    model_config = ConfigDict(extra="forbid")
    text: str
    n_input_chunks: int
    n_input_chars: int
    strategy: Literal["single", "mapreduce"]
    n_groups: int = 1


class MethodResult(BaseModel):
    """One benchmark record — matches the spec's evaluation JSON exactly."""

    model_config = ConfigDict(extra="forbid")
    method: str
    summary: str
    input_chunks: int
    input_chars: int
    execution_time_ms: int
    summary_length_chars: int


class LayerCReport(BaseModel):
    """Summary of a `build_layer_c` run."""

    model_config = ConfigDict(extra="forbid")
    n_themes: int
    n_cards: int
    n_skipped: int
    n_failed: int
    strategy_counts: dict[str, int] = Field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"Layer C: {self.n_cards}/{self.n_themes} cards "
            f"(skipped {self.n_skipped}, failed {self.n_failed}); "
            f"strategies={self.strategy_counts}"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_layer_c_models.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/layer_c/models.py tests/unit/test_layer_c_models.py
git commit -m "feat(layer-c): internal models (Stage1Output, MethodResult, LayerCReport)"
```

---

## Task 8: Stage 1 — single pass + map-reduce

**Files:**
- Create: `src/foodscholar/layer_c/stage1.py`
- Test: `tests/unit/test_layer_c_stage1.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_layer_c_stage1.py`:

```python
"""Stage 1: single-pass below threshold, map-reduce above it."""

from __future__ import annotations

from foodscholar.layer_c.base import BaseSummarizer
from foodscholar.layer_c.stage1 import run_stage1


class _FirstN(BaseSummarizer):
    """Deterministic stub: keep the first n sentences of the concatenation.

    Records how many times it was called so we can assert map vs reduce passes.
    """

    name = "firstn"

    def __init__(self, n: int = 2) -> None:
        self.n = n
        self.calls = 0

    def summarize(self, chunks: list[str]) -> str:
        self.calls += 1
        joined = " ".join(c for c in chunks if c.strip())
        parts = [p.strip() for p in joined.split(".") if p.strip()]
        return ". ".join(parts[: self.n]) + ("." if parts else "")


def test_single_pass_below_threshold() -> None:
    s = _FirstN(n=2)
    chunks = ["One. Two.", "Three."]
    out = run_stage1(chunks, s, map_reduce_threshold=100, group_char_budget=10_000)
    assert out.strategy == "single"
    assert out.n_groups == 1
    assert s.calls == 1
    assert out.n_input_chunks == 2


def test_mapreduce_above_threshold() -> None:
    s = _FirstN(n=2)
    # 6 chunks, each 3 sentences = 18 sentences; threshold 5 forces map-reduce.
    chunks = [f"A{i}. B{i}. C{i}." for i in range(6)]
    out = run_stage1(chunks, s, map_reduce_threshold=5, group_char_budget=20)
    assert out.strategy == "mapreduce"
    assert out.n_groups >= 2
    # one call per group (map) + one reduce call
    assert s.calls == out.n_groups + 1
    assert out.n_input_chunks == 6


def test_empty_input() -> None:
    s = _FirstN(n=2)
    out = run_stage1([], s, map_reduce_threshold=5, group_char_budget=20)
    assert out.text == ""
    assert out.strategy == "single"
    assert out.n_input_chunks == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_layer_c_stage1.py -v`
Expected: FAIL (`ModuleNotFoundError: foodscholar.layer_c.stage1`).

- [ ] **Step 3: Create `stage1.py`**

Create `src/foodscholar/layer_c/stage1.py`:

```python
"""Stage 1 — extractive compression of a theme's chunks.

Single pass when the input is small; map-reduce when it exceeds
`map_reduce_threshold` sentences (group by char budget → summarize each group →
summarize the concatenated group summaries). Nothing is dropped.
"""

from __future__ import annotations

from foodscholar.layer_c.base import BaseSummarizer, split_sentences
from foodscholar.layer_c.models import Stage1Output


def _group_by_chars(chunks: list[str], budget: int) -> list[list[str]]:
    """Greedily pack chunks into groups whose total chars stay near `budget`."""
    groups: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for c in chunks:
        clen = len(c)
        if cur and size + clen > budget:
            groups.append(cur)
            cur, size = [], 0
        cur.append(c)
        size += clen
    if cur:
        groups.append(cur)
    return groups


def run_stage1(
    chunks: list[str],
    summarizer: BaseSummarizer,
    *,
    map_reduce_threshold: int,
    group_char_budget: int,
) -> Stage1Output:
    """Compress `chunks` into one extractive summary with provenance."""
    texts = [c for c in chunks if c and c.strip()]
    n_chunks = len(texts)
    n_chars = sum(len(c) for c in texts)

    if not texts:
        return Stage1Output(text="", n_input_chunks=0, n_input_chars=0,
                            strategy="single", n_groups=1)

    total_sentences = sum(len(split_sentences(c)) for c in texts)
    if total_sentences <= map_reduce_threshold:
        return Stage1Output(
            text=summarizer.summarize(texts),
            n_input_chunks=n_chunks, n_input_chars=n_chars,
            strategy="single", n_groups=1,
        )

    groups = _group_by_chars(texts, group_char_budget)
    group_summaries = [summarizer.summarize(g) for g in groups]
    reduced = summarizer.summarize(group_summaries)
    return Stage1Output(
        text=reduced, n_input_chunks=n_chunks, n_input_chars=n_chars,
        strategy="mapreduce", n_groups=len(groups),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_layer_c_stage1.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/layer_c/stage1.py tests/unit/test_layer_c_stage1.py
git commit -m "feat(layer-c): Stage 1 single-pass + map-reduce orchestration"
```

---

## Task 9: Stage 2 — LLM refinement into `Card`

**Files:**
- Create: `src/foodscholar/layer_c/stage2.py`
- Test: `tests/unit/test_layer_c_stage2.py`

Note on the `Card` model (from `src/foodscholar/io/graph.py:56-70`): fields are `card_id`, `target_id`, `target_type` (`"shelf"|"theme"`), `title`, `summary`, `tip`, `evidence_quality` (`"high"|"medium"|"low"|"debated"|"unclear"`), `controversy_note`, `confidence_note`, `cited_chunk_ids`, `llm_model`, `prompt_version`, `safety_flagged`, `generated_at` (auto). The `Theme` handle exposes `.theme_id`, `.label`, `.facet`, and `.model.keyword_terms`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_layer_c_stage2.py`:

```python
"""Stage 2: extract -> Card via a mock LLMClient."""

from __future__ import annotations

import pytest

from foodscholar.config import LayerCConfig
from foodscholar.layer_c.models import Stage1Output
from foodscholar.layer_c.stage2 import run_stage2


class _StubTheme:
    def __init__(self, tid: str, label: str, facet: str = "foods",
                 keyword_terms=None) -> None:
        self.theme_id = tid
        self.label = label
        self.facet = facet
        self.keyword_terms = keyword_terms or ["oat", "fiber"]


class _OKJsonLLM:
    model_id = "stub-llm"

    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.last_prompt = None

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:  # pragma: no cover
        return ""

    def generate_json(self, prompt, schema, max_tokens=1024):
        self.last_prompt = prompt
        return dict(self._payload)


_GOOD = {
    "title": "Oats and heart health",
    "summary": "Oats contain beta glucan, a soluble fiber linked to lower cholesterol.",
    "tip": "Choose whole-grain oats.",
    "evidence_quality": "high",
    "controversy_note": None,
    "confidence_note": None,
}


def _stage1() -> Stage1Output:
    return Stage1Output(text="Oats have beta glucan.", n_input_chunks=3,
                        n_input_chars=50, strategy="single", n_groups=1)


def test_stage2_builds_card() -> None:
    llm = _OKJsonLLM(_GOOD)
    card = run_stage2(llm, _stage1(), _StubTheme("t1", "Oats"),
                      ["c1", "c2", "c3"], LayerCConfig())
    assert card.target_id == "t1"
    assert card.target_type == "theme"
    assert card.title == "Oats and heart health"
    assert card.evidence_quality == "high"
    assert card.cited_chunk_ids == ["c1", "c2", "c3"]
    assert card.llm_model == LayerCConfig().llm_model
    assert card.prompt_version == "v1"
    assert card.safety_flagged is False


def test_stage2_prompt_uses_extract_not_chunks() -> None:
    llm = _OKJsonLLM(_GOOD)
    run_stage2(llm, _stage1(), _StubTheme("t1", "Oats"), ["c1"], LayerCConfig())
    assert "beta glucan" in llm.last_prompt  # the extract is in the prompt
    assert "Oats" in llm.last_prompt          # theme label too


def test_stage2_safety_flag_on_sensitive_facet() -> None:
    llm = _OKJsonLLM(_GOOD)
    cfg = LayerCConfig(safety_sensitive_facets=["allergies"])
    card = run_stage2(llm, _stage1(), _StubTheme("t2", "Peanut", facet="allergies"),
                      ["c1"], cfg)
    assert card.safety_flagged is True


def test_stage2_strict_grounding_rejects_overlong() -> None:
    over = {**_GOOD, "summary": "x" * 10}
    llm = _OKJsonLLM(over)
    cfg = LayerCConfig(grounding_check="strict", max_summary_chars=5)
    with pytest.raises(ValueError):
        run_stage2(llm, _stage1(), _StubTheme("t3", "Oats"), ["c1"], cfg)


def test_stage2_off_grounding_allows_overlong() -> None:
    over = {**_GOOD, "summary": "x" * 10}
    llm = _OKJsonLLM(over)
    cfg = LayerCConfig(grounding_check="off", max_summary_chars=5)
    card = run_stage2(llm, _stage1(), _StubTheme("t3", "Oats"), ["c1"], cfg)
    assert len(card.summary) == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_layer_c_stage2.py -v`
Expected: FAIL (`ModuleNotFoundError: foodscholar.layer_c.stage2`).

- [ ] **Step 3: Create `stage2.py`**

Create `src/foodscholar/layer_c/stage2.py`:

```python
"""Stage 2 — refine the Stage-1 extract into a Card via the LLM.

The LLM sees ONLY the compact extract (plus the theme label/keywords for
context), never the raw chunks — that is the cost win. Output is mapped onto
the existing `Card` model; `cited_chunk_ids` carry theme-level provenance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from foodscholar.io.graph import Card

if TYPE_CHECKING:
    from foodscholar.config import LayerCConfig
    from foodscholar.layer_c.models import Stage1Output

_CARD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "tip": {"type": ["string", "null"]},
        "evidence_quality": {
            "type": "string",
            "enum": ["high", "medium", "low", "debated", "unclear"],
        },
        "controversy_note": {"type": ["string", "null"]},
        "confidence_note": {"type": ["string", "null"]},
    },
    "required": ["title", "summary", "evidence_quality"],
}

_PROMPT = """You are writing a concise knowledge card about the food topic "{label}".
Related keywords: {keywords}.

Below is an extractive summary distilled from {n_chunks} source passages. Use ONLY
this material — do not invent facts.

--- EXTRACT ---
{extract}
--- END EXTRACT ---

Produce a JSON object with:
- "title": a short topic title (<= 8 words)
- "summary": a clear narrative that organizes the key messages, main claims, and
  insights from the extract into flowing prose. Remove redundancy; reorganize freely.
- "tip": one practical takeaway, or null
- "evidence_quality": one of high|medium|low|debated|unclear
- "controversy_note": note any conflicting claims, or null
- "confidence_note": caveats about coverage, or null
"""


class _ThemeLike(Protocol):
    theme_id: str
    label: str
    facet: str
    keyword_terms: list[str]


def run_stage2(
    llm: Any,
    stage1: "Stage1Output",
    theme: _ThemeLike,
    cited_chunk_ids: list[str],
    cfg: "LayerCConfig",
) -> Card:
    """Refine `stage1.text` into a Card for `theme`. Raises ValueError if the
    strict grounding guard fails."""
    prompt = _PROMPT.format(
        label=theme.label,
        keywords=", ".join(theme.keyword_terms),
        n_chunks=stage1.n_input_chunks,
        extract=stage1.text,
    )
    data = llm.generate_json(prompt, _CARD_SCHEMA, max_tokens=1024)

    summary = str(data.get("summary", "")).strip()
    if cfg.grounding_check == "strict":
        if not summary or len(summary) > cfg.max_summary_chars:
            raise ValueError(
                f"grounding(strict): summary length {len(summary)} "
                f"outside (0, {cfg.max_summary_chars}]"
            )

    safety = theme.facet in cfg.safety_sensitive_facets

    return Card(
        card_id=f"card:theme:{theme.theme_id}",
        target_id=theme.theme_id,
        target_type="theme",
        title=str(data.get("title", theme.label)).strip(),
        summary=summary,
        tip=(data.get("tip") or None),
        evidence_quality=data.get("evidence_quality", "unclear"),
        controversy_note=(data.get("controversy_note") or None),
        confidence_note=(data.get("confidence_note") or None),
        cited_chunk_ids=list(cited_chunk_ids),
        llm_model=cfg.llm_model,
        prompt_version=cfg.prompt_version,
        safety_flagged=safety,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_layer_c_stage2.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/layer_c/stage2.py tests/unit/test_layer_c_stage2.py
git commit -m "feat(layer-c): Stage 2 LLM refinement into Card"
```

---

## Task 10: Persistence

**Files:**
- Create: `src/foodscholar/layer_c/persist.py`
- Test: `tests/unit/test_layer_c_persist.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_layer_c_persist.py`:

```python
"""Layer C persistence: cards -> graph_store.upsert_cards."""

from __future__ import annotations

from foodscholar.io.graph import Card
from foodscholar.layer_c.persist import persist_cards
from foodscholar.storage.memory import InMemoryGraphStore


def _card(tid: str) -> Card:
    return Card(
        card_id=f"card:theme:{tid}", target_id=tid, target_type="theme",
        title="t", summary="s", evidence_quality="high",
        cited_chunk_ids=["c1"], llm_model="m", prompt_version="v1",
    )


def test_persist_cards_writes_to_store() -> None:
    gs = InMemoryGraphStore()
    persist_cards([_card("t1"), _card("t2")], gs)
    assert gs.get_card("t1", "theme") is not None
    assert gs.get_card("t2", "theme") is not None


def test_persist_empty_is_noop() -> None:
    gs = InMemoryGraphStore()
    persist_cards([], gs)  # must not raise
    assert gs.get_card("t1", "theme") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_layer_c_persist.py -v`
Expected: FAIL (`ModuleNotFoundError: foodscholar.layer_c.persist`).

- [ ] **Step 3: Create `persist.py`**

Create `src/foodscholar/layer_c/persist.py`:

```python
"""Persist Layer C cards. Single write — the Card model carries
`target_id`/`target_type="theme"` so the graph store routes them. Mirrors the
additive contract of `layer_b/persist.py`."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foodscholar.io.graph import Card
    from foodscholar.storage.protocols import GraphStore


def persist_cards(cards: list["Card"], graph_store: "GraphStore") -> None:
    """Upsert theme cards into the graph store. No-op on empty input."""
    if not cards:
        return
    graph_store.upsert_cards(cards)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_layer_c_persist.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/layer_c/persist.py tests/unit/test_layer_c_persist.py
git commit -m "feat(layer-c): persist cards via upsert_cards"
```

---

## Task 11: Builder — iterate themes → cards

**Files:**
- Create: `src/foodscholar/layer_c/builder.py`
- Test: `tests/unit/test_layer_c_builder.py`

The builder reads themes from `fs.graph.themes()` (returns `ThemeHandle`s with `.theme_id`, `.label`, `.facet`, and `.model.keyword_terms`), gets member chunk ids via `fs.graph_store.get_chunks_for_theme(theme_id)`, fetches text via `fs.chunk_store.get_many(ids)`, runs Stage 1 then Stage 2, and persists. A small adapter exposes the `keyword_terms` the Stage-2 `_ThemeLike` protocol expects.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_layer_c_builder.py`:

```python
"""Layer C builder: themes -> cards, skip/fail accounting, dry_run."""

from __future__ import annotations

from foodscholar.config import FoodScholarConfig, LayerCConfig
from foodscholar.io.chunk import Chunk
from foodscholar.io.graph import Shelf, Theme
from foodscholar.layer_c.builder import build_layer_c
from foodscholar.storage.memory import InMemoryChunkStore, InMemoryGraphStore


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, text=text, source_doc_id="d",
                 source_type="abstract", section_type="abstract")


def _theme(tid: str) -> Theme:
    return Theme(theme_id=tid, label="Oats", shelf_ids=["s1"], chunk_count=2,
                 discovered_by="leiden", discovery_version="v", facet="foods",
                 discovery_pass="merged", keyword_terms=["oat", "fiber"])


class _OKJsonLLM:
    model_id = "stub"

    def generate(self, prompt, max_tokens=1024):  # pragma: no cover
        return ""

    def generate_json(self, prompt, schema, max_tokens=1024):
        return {"title": "Oats", "summary": "Oats have beta glucan.",
                "tip": None, "evidence_quality": "high",
                "controversy_note": None, "confidence_note": None}


class _FailLLM(_OKJsonLLM):
    def generate_json(self, prompt, schema, max_tokens=1024):
        raise RuntimeError("llm down")


def _fs(llm):
    """Minimal stand-in for the FoodScholar facade the builder needs."""
    cs = InMemoryChunkStore()
    gs = InMemoryGraphStore()
    cs.upsert([_chunk("c1", "Oats are a whole grain. They have fiber."),
               _chunk("c2", "Beta glucan lowers cholesterol.")])
    gs.upsert_shelves([Shelf(shelf_id="s1", label="cereal", facet="foods", depth=1)])
    gs.upsert_themes([_theme("t1")])
    # signature is (chunk_id, theme_id, primary, weight)
    gs.attach_chunks_to_themes_bulk([("c1", "t1", True, 1.0), ("c2", "t1", False, 1.0)])

    class _FS:
        pass

    fs = _FS()
    fs.chunk_store = cs
    fs.graph_store = gs
    from foodscholar.graph_view import GraphView
    fs.graph = GraphView(cs, gs)
    fs.llm = llm
    fs.config = FoodScholarConfig(corpus={"chunks_path": "x"})
    fs.config.layer_c = LayerCConfig()
    return fs


def test_build_creates_card_per_theme() -> None:
    fs = _fs(_OKJsonLLM())
    rep = build_layer_c(fs)
    assert rep.n_themes == 1
    assert rep.n_cards == 1
    assert rep.n_failed == 0
    assert fs.graph_store.get_card("t1", "theme") is not None


def test_dry_run_persists_nothing() -> None:
    fs = _fs(_OKJsonLLM())
    rep = build_layer_c(fs, dry_run=True)
    assert rep.n_cards == 1
    assert fs.graph_store.get_card("t1", "theme") is None


def test_llm_failure_counted_not_raised() -> None:
    fs = _fs(_FailLLM())
    rep = build_layer_c(fs)
    assert rep.n_failed == 1
    assert rep.n_cards == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_layer_c_builder.py -v`
Expected: FAIL (`ModuleNotFoundError: foodscholar.layer_c.builder`).

- [ ] **Step 3: Create `builder.py`**

Create `src/foodscholar/layer_c/builder.py`:

```python
"""Layer C orchestrator: iterate Layer B themes, summarize each into a Card.

For each theme: gather member chunk texts → Stage 1 (map-reduce extractive) →
Stage 2 (LLM refinement) → Card. Persists via `persist_cards` unless `dry_run`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from foodscholar.layer_c.models import LayerCReport
from foodscholar.layer_c.persist import persist_cards
from foodscholar.layer_c.registry import build_summarizer
from foodscholar.layer_c.stage1 import run_stage1
from foodscholar.layer_c.stage2 import run_stage2

if TYPE_CHECKING:
    from foodscholar.facade import FoodScholar


@dataclass
class _ThemeAdapter:
    """Expose the `_ThemeLike` shape Stage 2 expects from a ThemeHandle."""

    theme_id: str
    label: str
    facet: str
    keyword_terms: list[str]


def build_layer_c(
    fs: "FoodScholar",
    *,
    facet: str = "foods",
    dry_run: bool = False,
) -> LayerCReport:
    """Build one Card per Layer B theme of `facet`."""
    cfg = fs.config.layer_c
    summarizer = build_summarizer(cfg.stage1_method, cfg)

    themes = [t for t in fs.graph.themes() if t.facet == facet]
    cards = []
    skipped = failed = 0
    strat: dict[str, int] = {}

    for th in themes:
        chunk_ids = list(fs.graph_store.get_chunks_for_theme(th.theme_id))
        texts = [c.text for c in fs.chunk_store.get_many(chunk_ids)]
        if not any(t and t.strip() for t in texts):
            skipped += 1
            continue

        s1 = run_stage1(
            texts, summarizer,
            map_reduce_threshold=cfg.map_reduce_threshold,
            group_char_budget=cfg.group_char_budget,
        )
        adapter = _ThemeAdapter(
            theme_id=th.theme_id, label=th.label, facet=th.facet,
            keyword_terms=list(th.model.keyword_terms),
        )
        try:
            card = run_stage2(fs.llm, s1, adapter, chunk_ids, cfg)
        except Exception:  # noqa: BLE001 - record + continue
            failed += 1
            continue

        cards.append(card)
        strat[s1.strategy] = strat.get(s1.strategy, 0) + 1

    if not dry_run:
        persist_cards(cards, fs.graph_store)

    return LayerCReport(
        n_themes=len(themes), n_cards=len(cards),
        n_skipped=skipped, n_failed=failed, strategy_counts=strat,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_layer_c_builder.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/layer_c/builder.py tests/unit/test_layer_c_builder.py
git commit -m "feat(layer-c): builder — themes -> cards with skip/fail/dry-run accounting"
```

---

## Task 12: Benchmark harness

**Files:**
- Create: `src/foodscholar/layer_c/benchmark.py`
- Test: `tests/unit/test_layer_c_benchmark.py`

The harness runs every registry method over one theme's chunks (single-pass — comparing raw method quality), timing each, and emits `MethodResult` records. It uses `time.perf_counter` for timing. We pass timing in via a injectable clock so the test is deterministic.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_layer_c_benchmark.py`:

```python
"""Layer C benchmark: run all methods over a theme -> MethodResult list."""

from __future__ import annotations

import pytest

from foodscholar.config import FoodScholarConfig, LayerCConfig
from foodscholar.io.chunk import Chunk
from foodscholar.io.graph import Shelf, Theme
from foodscholar.storage.memory import InMemoryChunkStore, InMemoryGraphStore

pytest.importorskip("sumy")

from foodscholar.layer_c.benchmark import benchmark_theme  # noqa: E402


def _fs():
    cs = InMemoryChunkStore()
    gs = InMemoryGraphStore()
    docs = [
        "Oats are a whole grain rich in soluble fiber called beta glucan.",
        "Beta glucan in oats can lower cholesterol and improve heart health.",
        "Rice is a staple cereal grain eaten across the world.",
        "Wheat flour is milled from wheat and used to bake bread.",
    ]
    cs.upsert([Chunk(chunk_id=f"c{i}", text=t, source_doc_id="d",
                     source_type="abstract", section_type="abstract")
               for i, t in enumerate(docs)])
    gs.upsert_shelves([Shelf(shelf_id="s1", label="cereal", facet="foods", depth=1)])
    gs.upsert_themes([Theme(theme_id="t1", label="Cereal grains", shelf_ids=["s1"],
                            chunk_count=4, discovered_by="leiden", discovery_version="v",
                            facet="foods", discovery_pass="merged",
                            keyword_terms=["oat", "rice"])])
    # signature is (chunk_id, theme_id, primary, weight)
    gs.attach_chunks_to_themes_bulk([(f"c{i}", "t1", i == 0, 1.0) for i in range(4)])

    class _FS:
        pass

    fs = _FS()
    fs.chunk_store = cs
    fs.graph_store = gs
    fs.config = FoodScholarConfig(corpus={"chunks_path": "x"})
    fs.config.layer_c = LayerCConfig(stage1_sentences=2)
    return fs


def test_benchmark_theme_returns_all_methods() -> None:
    fs = _fs()
    results = benchmark_theme(fs, "t1")
    methods = {r.method for r in results}
    assert methods == {"lexrank", "lsa", "luhn", "textrank", "nltk_freq"}
    for r in results:
        assert r.input_chunks == 4
        assert r.input_chars > 0
        assert r.summary_length_chars == len(r.summary)
        assert r.execution_time_ms >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_layer_c_benchmark.py -v`
Expected: FAIL (`ModuleNotFoundError: foodscholar.layer_c.benchmark`).

- [ ] **Step 3: Create `benchmark.py`**

Create `src/foodscholar/layer_c/benchmark.py`:

```python
"""Read-only evaluation harness: run every extractive method over a theme's
chunks and emit per-method metrics for side-by-side comparison. No LLM, no
persistence. Used to pick/tune `config.layer_c.stage1_method`."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from foodscholar.layer_c.models import MethodResult
from foodscholar.layer_c.registry import all_summarizers

if TYPE_CHECKING:
    from foodscholar.facade import FoodScholar


def benchmark_theme(fs: "FoodScholar", theme_id: str) -> list[MethodResult]:
    """Run all registered methods over `theme_id`'s chunks (single pass)."""
    cfg = fs.config.layer_c
    chunk_ids = list(fs.graph_store.get_chunks_for_theme(theme_id))
    texts = [c.text for c in fs.chunk_store.get_many(chunk_ids)]
    input_chars = sum(len(t) for t in texts)

    results: list[MethodResult] = []
    for summ in all_summarizers(cfg):
        t0 = time.perf_counter()
        summary = summ.summarize(texts)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        results.append(MethodResult(
            method=summ.name, summary=summary,
            input_chunks=len(texts), input_chars=input_chars,
            execution_time_ms=elapsed_ms, summary_length_chars=len(summary),
        ))
    return results


def benchmark_facet(
    fs: "FoodScholar",
    *,
    facet: str = "foods",
    themes: int = 5,
    out: str | None = None,
) -> dict[str, list[MethodResult]]:
    """Benchmark the `themes` largest themes of `facet`; write combined JSON."""
    cfg = fs.config.layer_c
    handles = sorted(
        (t for t in fs.graph.themes() if t.facet == facet),
        key=lambda t: t.chunk_count, reverse=True,
    )[:themes]

    by_theme = {h.theme_id: benchmark_theme(fs, h.theme_id) for h in handles}

    out_path = Path(out or cfg.benchmark_out_dir) / f"benchmark_{facet}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {tid: [r.model_dump() for r in rs] for tid, rs in by_theme.items()},
        indent=2,
    ), encoding="utf-8")
    return by_theme
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_layer_c_benchmark.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/layer_c/benchmark.py tests/unit/test_layer_c_benchmark.py
git commit -m "feat(layer-c): read-only benchmark harness (per-method JSON metrics)"
```

---

## Task 13: Facade wiring

**Files:**
- Modify: `src/foodscholar/facade.py:1120-1121` (the deferred `build_layer_c`)
- Test: `tests/unit/test_layer_c_facade.py`

The current method is `def build_layer_c(self) -> None: raise _deferred("build-layer-c")`. The existing CLI `build-layer-c` calls it with no args, and `build()` calls `self.build_layer_c()` with no args — so the new signature must keep working argument-free (facet/dry_run keyword-only with defaults).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_layer_c_facade.py`:

```python
"""Facade exposes build_layer_c / benchmark_layer_c that delegate to layer_c."""

from __future__ import annotations

import foodscholar.facade as facade_mod


def test_build_layer_c_delegates(monkeypatch) -> None:
    called = {}

    def fake_build(fs, *, facet="foods", dry_run=False):
        called["facet"] = facet
        called["dry_run"] = dry_run
        return "report"

    monkeypatch.setattr("foodscholar.layer_c.builder.build_layer_c", fake_build)

    # Build a bare facade instance without running __init__ machinery.
    fs = object.__new__(facade_mod.FoodScholar)
    out = facade_mod.FoodScholar.build_layer_c(fs, facet="foods", dry_run=True)
    assert out == "report"
    assert called == {"facet": "foods", "dry_run": True}


def test_benchmark_layer_c_delegates(monkeypatch) -> None:
    called = {}

    def fake_bench(fs, *, facet="foods", themes=5, out=None):
        called["themes"] = themes
        return {"t1": []}

    monkeypatch.setattr("foodscholar.layer_c.benchmark.benchmark_facet", fake_bench)
    fs = object.__new__(facade_mod.FoodScholar)
    out = facade_mod.FoodScholar.benchmark_layer_c(fs, themes=3)
    assert out == {"t1": []}
    assert called["themes"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_layer_c_facade.py -v`
Expected: FAIL (`build_layer_c` raises `NotImplementedError`; `benchmark_layer_c` doesn't exist).

- [ ] **Step 3: Replace the deferred method**

In `src/foodscholar/facade.py`, replace:

```python
    def build_layer_c(self) -> None:
        raise _deferred("build-layer-c")
```

with:

```python
    def build_layer_c(self, *, facet: str = "foods", dry_run: bool = False):
        """Build Layer C — one summary Card per Layer B theme of `facet`.

        Each theme's member chunks are compressed by a cheap extractive method
        (Stage 1, map-reduce when large), then refined by the LLM into a Card
        (Stage 2). `dry_run=True` runs both stages but skips persistence.
        Returns a `LayerCReport`.
        """
        from foodscholar.layer_c.builder import build_layer_c as _build_layer_c

        return _build_layer_c(self, facet=facet, dry_run=dry_run)

    def benchmark_layer_c(
        self,
        *,
        facet: str = "foods",
        themes: int = 5,
        out: str | None = None,
    ):
        """Read-only benchmark of all extractive methods over the largest
        `themes` themes of `facet`. Writes per-method JSON metrics; returns the
        results keyed by theme id. No LLM, no persistence."""
        from foodscholar.layer_c.benchmark import benchmark_facet as _bench

        return _bench(self, facet=facet, themes=themes, out=out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_layer_c_facade.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/facade.py tests/unit/test_layer_c_facade.py
git commit -m "feat(layer-c): wire build_layer_c + benchmark_layer_c into the facade"
```

---

## Task 14: CLI — facet/dry-run options + bench command

**Files:**
- Modify: `src/foodscholar/cli/main.py` (the `build-layer-c` command ~line 112)
- Test: `tests/unit/test_layer_c_cli.py`

The existing command is `_run_phase(_build(config), "build-layer-c", "build_layer_c")`, which calls `fs.build_layer_c()` with no args. We replace it with a direct call that passes options, and add a `bench-layer-c` command.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_layer_c_cli.py`:

```python
"""CLI exposes build-layer-c (with --dry-run) and bench-layer-c."""

from __future__ import annotations

from typer.testing import CliRunner

from foodscholar.cli.main import app

runner = CliRunner()


def test_bench_layer_c_in_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "bench-layer-c" in result.output


def test_build_layer_c_has_dry_run_flag() -> None:
    result = runner.invoke(app, ["build-layer-c", "--help"])
    assert result.exit_code == 0
    assert "--dry-run" in result.output
    assert "--facet" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_layer_c_cli.py -v`
Expected: FAIL (`bench-layer-c` not in help; `--dry-run` absent).

- [ ] **Step 3: Replace the `build-layer-c` command and add `bench-layer-c`**

In `src/foodscholar/cli/main.py`, replace the existing command:

```python
@app.command("build-layer-c")
def build_layer_c(config: Path = ConfigOption) -> None:
    """Build Layer C — LLM write-up cards for every shelf and theme."""
    _run_phase(_build(config), "build-layer-c", "build_layer_c")
```

with:

```python
@app.command("build-layer-c")
def build_layer_c(
    config: Path = ConfigOption,
    facet: str = typer.Option("foods", "--facet", help="Facet to summarize."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run without persisting."),
) -> None:
    """Build Layer C — one summary card per Layer B theme."""
    fs = _build(config)
    report = fs.build_layer_c(facet=facet, dry_run=dry_run)
    typer.echo(str(report))


@app.command("bench-layer-c")
def bench_layer_c(
    config: Path = ConfigOption,
    facet: str = typer.Option("foods", "--facet", help="Facet to benchmark."),
    themes: int = typer.Option(5, "--themes", help="How many largest themes."),
    out: str = typer.Option(None, "--out", help="Output dir for the JSON."),
) -> None:
    """Benchmark all extractive methods over the largest themes (read-only)."""
    fs = _build(config)
    results = fs.benchmark_layer_c(facet=facet, themes=themes, out=out)
    for tid, rows in results.items():
        typer.echo(f"\n# {tid}")
        for r in rows:
            typer.echo(
                f"  {r.method:10} {r.summary_length_chars:>6} chars "
                f"{r.execution_time_ms:>5} ms"
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_layer_c_cli.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/foodscholar/cli/main.py tests/unit/test_layer_c_cli.py
git commit -m "feat(layer-c): CLI build-layer-c (--facet/--dry-run) + bench-layer-c"
```

---

## Task 15: Package exports + full-suite gate

**Files:**
- Modify: `src/foodscholar/layer_c/__init__.py`

- [ ] **Step 1: Export the public surface**

Replace `src/foodscholar/layer_c/__init__.py` with:

```python
"""Layer C — theme summarization (extractive Stage 1 → LLM Stage 2)."""

from foodscholar.layer_c.benchmark import benchmark_facet, benchmark_theme
from foodscholar.layer_c.builder import build_layer_c
from foodscholar.layer_c.models import LayerCReport, MethodResult, Stage1Output

__all__ = [
    "build_layer_c",
    "benchmark_theme",
    "benchmark_facet",
    "LayerCReport",
    "MethodResult",
    "Stage1Output",
]
```

- [ ] **Step 2: Verify the package imports**

Run: `python -c "import foodscholar.layer_c as m; print(m.__all__)"`
Expected: prints the `__all__` list (no import error).

- [ ] **Step 3: Run the whole Layer C test set**

Run: `pytest tests/unit/ -k layer_c -v`
Expected: all Layer C tests PASS.

- [ ] **Step 4: Run the full unit suite (regression gate)**

Run: `pytest tests/unit/ -q`
Expected: no new failures; the project suite passes.

- [ ] **Step 5: Lint the new package**

Run: `ruff check src/foodscholar/layer_c/ tests/unit/test_layer_c_*.py`
Expected: clean (fix any issues, then re-run).

- [ ] **Step 6: Commit**

```bash
git add src/foodscholar/layer_c/__init__.py
git commit -m "feat(layer-c): public package exports; full-suite green"
```

---

## Self-Review notes (for the implementer)

- **Spec coverage:** BaseSummarizer (T3), 5 methods sumy×4+nltk (T4-5), registry (T6), map-reduce Stage 1 (T8), LLM Stage 2 → existing Card (T9), per-method JSON eval harness (T12), config knobs (T2), facade+CLI wiring (T13-14), persistence via upsert_cards (T10), `[summarization]` extra (T1). T5/abstractive and per-claim grounding are out of scope per the spec.
- **Type consistency:** `Stage1Output`/`MethodResult`/`LayerCReport` defined in T7 and used unchanged in T8/T11/T12. `run_stage1(chunks, summarizer, *, map_reduce_threshold, group_char_budget)` signature identical in T8 (def) and T11 (call). `run_stage2(llm, stage1, theme, cited_chunk_ids, cfg)` identical in T9 (def) and T11 (call). `build_summarizer`/`all_summarizers` from T6 used in T11/T12. `Card` constructed only in T9 with fields verified against `io/graph.py:56-70`.
- **Optional-dep guards:** sumy/nltk tests use `pytest.importorskip`; the benchmark + sumy summarizer tests skip cleanly when the extra isn't installed.
- **Backward-compat:** facade `build_layer_c` keeps an argument-free call path (keyword-only defaults), so the existing `build()` and the pre-existing CLI wiring stay valid.
