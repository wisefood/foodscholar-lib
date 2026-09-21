"""Public retrieval surface.

The library retrieves; it does not answer. `KGGenRetriever` ranks passages and
returns them with the branch scores that ranked them — prompting, citation and
answer synthesis belong to the consuming QA pipeline, which owns the model and
the editorial rules the library has no view of.

An `Answer` model (LLM-synthesized, cited) lived here while `fs.query()` was a
planned phase. That phase is not planned any more: the one consumer, the
FoodScholar API, has its own answering stack, so a second one in the library
would be a fork of it rather than a feature.
"""

from foodscholar.retrieval.kggen import KGGenRetriever, RetrievalHit, RetrievalTrace

__all__ = ["KGGenRetriever", "RetrievalHit", "RetrievalTrace"]
