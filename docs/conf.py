"""Sphinx configuration for the FoodScholar documentation.

Pages are authored in MyST-flavoured Markdown; the API reference (added in a
later slice) is generated from docstrings via autodoc + autosummary. Built on
Read the Docs — see ``../.readthedocs.yaml``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

# -- Project information ------------------------------------------------------

project = "FoodScholar"
author = "WiseFood"
project_copyright = "2026, WiseFood"

try:
    release = _pkg_version("foodscholar")
except PackageNotFoundError:  # package not installed in the build env (scaffold-only)
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ----------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

myst_enable_extensions = [
    "colon_fence",   # ::: fenced directives (used by sphinx-design)
    "deflist",
    "fieldlist",
    "linkify",       # bare URLs become links
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}

# Keep the published site to the curated pages. The repo's docs/ folder also holds
# internal specs/plans, talk briefs, and legacy method briefs that are not (yet)
# part of the site; exclude them so they don't trip the build. Their content is
# folded into the Concepts pages over time.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "README.md",
    "superpowers/**",
    "presentations/**",
    "deliverables/**",
    "*_brief.md",
    "methods_*.md",
    "IMPLEMENTATION_*.md",
    "DESIGN_*.md",
]

# -- Autodoc / autosummary (consumed once reference/ pages land) --------------

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": False,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Heavy optional deps that may be absent in the docs build env — mock them so
# autodoc can import modules without installing the ES/Neo4j/ML stacks.
autodoc_mock_imports = [
    "numpy", "igraph", "leidenalg", "hnswlib", "sentence_transformers",
    "gliner", "elasticsearch", "neo4j", "pronto", "rapidfuzz", "sklearn",
    "scikit_learn", "pyvis", "graphviz", "matplotlib", "hdbscan", "umap",
    "bertopic",
]

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# -- HTML output --------------------------------------------------------------

html_theme = "furo"
html_title = f"FoodScholar {version}"
