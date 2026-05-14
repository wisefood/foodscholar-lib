"""Load chunks from a parquet/jsonl file into the in-memory chunk store."""

from pathlib import Path

from foodscholar.corpus import load_chunks
from foodscholar.storage import InMemoryChunkStore


def main(path: str) -> None:
    chunks = load_chunks(Path(path))
    store = InMemoryChunkStore()
    store.upsert(chunks)
    print(f"loaded {len(chunks)} chunks")


if __name__ == "__main__":
    import sys

    main(sys.argv[1])
