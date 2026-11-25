"""Scoring chunkers — ours and anyone else's — against gold statements.

See :mod:`foodscholar.chunking.evaluation.harness` for the three stages.
"""

from foodscholar.chunking.evaluation.gold import (
    GoldStatement,
    load_fskg_statements,
    load_gold_jsonl,
    save_gold_jsonl,
)
from foodscholar.chunking.evaluation.harness import (
    BM25Retriever,
    ChunkRecord,
    EvaluationReport,
    Query,
    StatementOutcome,
    evaluate,
    evaluate_retrieval,
    hygiene,
    load_chunk_records,
    load_queries_jsonl,
    statement_queries,
)
from foodscholar.chunking.evaluation.matching import Support, content_tokens, support

__all__ = [
    "BM25Retriever",
    "ChunkRecord",
    "EvaluationReport",
    "GoldStatement",
    "Query",
    "StatementOutcome",
    "Support",
    "content_tokens",
    "evaluate",
    "evaluate_retrieval",
    "hygiene",
    "load_chunk_records",
    "load_fskg_statements",
    "load_gold_jsonl",
    "load_queries_jsonl",
    "save_gold_jsonl",
    "statement_queries",
    "support",
]
