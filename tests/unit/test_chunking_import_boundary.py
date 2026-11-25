"""The chunker is usable on its own: importing it loads nothing else from the library."""

from __future__ import annotations

import subprocess
import sys

FORBIDDEN = (
    "foodscholar.facade",
    "foodscholar.storage",
    "foodscholar.llm",
    "foodscholar.ontology",
    "foodscholar.retrieval",
    "foodscholar.config",
    "foodscholar.io",
    "foodscholar.corpus",
    "foodscholar.annotate",
    "foodscholar.layer_a",
    "foodscholar.layer_b",
    "foodscholar.layer_c",
    "elasticsearch",
    "neo4j",
    "transformers",
    "torch",
)


def _loaded_after(statement: str) -> list[str]:
    code = (
        f"{statement}\nimport sys\n"
        f"print('\\n'.join(m for m in sys.modules if m.startswith({FORBIDDEN!r})))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return [line for line in out.stdout.splitlines() if line]


def test_importing_the_chunking_package_loads_no_other_part_of_the_library():
    assert _loaded_after("import foodscholar.chunking, foodscholar.chunking.evaluation") == []


def test_the_chunk_cli_module_is_light_too():
    assert _loaded_after("import foodscholar.chunking.cli") == []


def test_importing_foodscholar_itself_is_lazy_but_names_still_resolve():
    assert _loaded_after("import foodscholar") == []
    code = "from foodscholar import FoodScholar, Chunk, InMemoryChunkStore; print(FoodScholar.__name__)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "FoodScholar"
