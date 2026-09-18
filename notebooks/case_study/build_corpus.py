#!/usr/bin/env python
"""Reproducible, idempotent build of the abstract-expanded case-study corpus.

Runs the full offline pipeline that turns the raw NER/NEL outputs into the two
artifacts the notebooks consume:

  data/annotated_with_abstracts.parquet   (34,359 chunks: textbook+guide+abstract)
  data/cache/bge_base_emb_13767.npy + ids  (real BGE vectors for anchor subtrees)

It is fully offline (in-memory stores, NO Elastic/Neo4j) and idempotent: a
second run reproduces byte-identical inputs and never double-appends to the
cache. Replaces the throwaway /tmp scripts the prototype was built with.

Steps:
  1. Collapse the expanded abstracts NEL (`<dochash>_<n>` sub-chunks) to a
     doc-level NEL keyed by bare `<dochash>` so it id-joins the source corpus.
  2. Filter the raw abstracts source to the NEL-covered docs only.
  3. Stage a corpus_cs/ + ner_cs/ dir (existing docs + filtered abstracts).
  4. Ingest -> annotated_with_abstracts.parquet (no source-type filter).
  5. Build Layer A, embed each anchor subtree's abstracts with real BGE, and
     write an extended, deduped, manifest-checked vector cache.

Usage (from repo root):
    conda run -n foodscholar python notebooks/case_study/build_corpus.py
    conda run -n foodscholar python notebooks/case_study/build_corpus.py --force

Determinism: SEED=42 (Leiden), fixed BGE model, sorted file iteration. The only
non-reproducible step is none — embedding is deterministic for a fixed model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(10 * 1024 * 1024)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cs_common as cs  # noqa: E402

REPO = cs.REPO_ROOT
FS = REPO / "data" / "foodscholar"
RAW_ABS = FS / "corpus" / "chunks_abstracts.csv"
EXPANDED_NEL = FS / "ner" / "nel_chunks_abstracts_combined_expanded.csv"
DOC_NEL = FS / "ner" / "nel_chunks_abstracts.csv"            # step 1 output
FILTERED_ABS = FS / "corpus_cs" / "chunks_abstracts_nelonly.csv"  # step 2 output
CORPUS_CS = FS / "corpus_cs"
NER_CS = FS / "ner_cs"
SNAPSHOT = REPO / "data" / "annotated_with_abstracts.parquet"
CACHE_DIR = REPO / "data" / "cache"
BASE_EMB = CACHE_DIR / "bge_base_emb_13252.npy"            # textbook+guide base
BASE_IDS = CACHE_DIR / "bge_base_ids_13252.json"
MANIFEST = CACHE_DIR / "MANIFEST.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def step1_doclevel_nel() -> None:
    """Collapse expanded sub-chunk NEL to doc-level keyed by bare dochash."""
    from foodscholar.corpus.nel_loader import iter_nel_rows  # noqa: F401 (validates schema)

    src_ids = set()
    with RAW_ABS.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            cid = (row.get("chunk_id") or "").strip()
            if cid:
                src_ids.add(cid)

    ner_by_doc: dict[str, list[str]] = defaultdict(list)
    uri_by_doc: dict[str, list[str]] = defaultdict(list)
    with EXPANDED_NEL.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            cid = (row.get("chunk_id") or "").strip()
            doc = cid.rsplit("_", 1)[0]
            if doc not in src_ids:
                continue
            ner = (row.get("chunk_entities_ner") or "").split(";")
            uri = (row.get("chunk_uri_nel") or "").split(";")
            for i in range(min(len(ner), len(uri))):
                s = ner[i].strip()
                if s:
                    ner_by_doc[doc].append(s)
                    uri_by_doc[doc].append(uri[i].strip())

    DOC_NEL.parent.mkdir(parents=True, exist_ok=True)
    with DOC_NEL.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "chunk_entities_ner", "chunk_uri_nel"])
        for doc in sorted(ner_by_doc):
            seen: set = set()
            ners: list[str] = []
            uris: list[str] = []
            for s, u in zip(ner_by_doc[doc], uri_by_doc[doc]):
                if (s, u) in seen:
                    continue
                seen.add((s, u))
                ners.append(s)
                uris.append(u)
            w.writerow([doc, ";".join(ners), ";".join(uris)])
    print(f"  [1] doc-level NEL: {len(ner_by_doc)} docs -> {DOC_NEL.name}")


def step2_filter_abstracts() -> None:
    from foodscholar.corpus.nel_loader import iter_nel_rows

    keep = {r[0] for r in iter_nel_rows(DOC_NEL)}
    FILTERED_ABS.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with RAW_ABS.open(newline="", encoding="utf-8", errors="replace") as f, \
            FILTERED_ABS.open("w", newline="", encoding="utf-8") as g:
        r = csv.DictReader(f)
        w = csv.DictWriter(g, fieldnames=r.fieldnames)
        w.writeheader()
        for row in r:
            if (row.get("chunk_id") or "").strip() in keep:
                w.writerow(row)
                n += 1
    print(f"  [2] filtered abstracts: {n} chunks -> {FILTERED_ABS.name}")


def step3_stage_dirs() -> None:
    def link(srcp: Path, dstp: Path) -> None:
        if dstp.is_symlink() or dstp.exists():
            dstp.unlink()
        dstp.symlink_to(srcp)

    CORPUS_CS.mkdir(parents=True, exist_ok=True)
    NER_CS.mkdir(parents=True, exist_ok=True)
    nc = 0
    for fpath in sorted((FS / "corpus").glob("chunks_*.csv")):
        if fpath.name == "chunks_abstracts.csv":
            continue
        link(fpath, CORPUS_CS / fpath.name)
        nc += 1
    nn = 0
    for fpath in sorted((FS / "ner").glob("nel_*.csv")):
        if fpath.name == "nel_chunks_abstracts_combined_expanded.csv":
            continue
        link(fpath, NER_CS / fpath.name)
        nn += 1
    print(f"  [3] staged corpus_cs ({nc} docs + filtered abstracts), ner_cs ({nn} nel)")


def step4_ingest() -> None:
    fs = cs.get_fs(with_llm=False)
    fs.ingest(str(CORPUS_CS), nel_dir=str(NER_CS), snapshot_path=str(SNAPSHOT),
              ignore_source_types=set())
    from collections import Counter
    by = Counter(c.source_type for c in fs.chunk_store.scan())
    print(f"  [4] ingested {sum(by.values())} chunks {dict(by)} -> {SNAPSHOT.name}")


def step5_embed_anchor_subtrees() -> None:
    import numpy as np

    fs = cs.get_fs(with_llm=False)
    # load the freshly-ingested snapshot directly (don't depend on the cs path)
    fs.load_chunks(str(SNAPSHOT))
    base_emb = np.load(BASE_EMB)
    base_ids = json.loads(BASE_IDS.read_text())
    # attach base vectors so Layer A/anchor resolution see real embeddings
    present = {c.chunk_id for c in fs.chunk_store.scan()}
    fs.chunk_store.update_embeddings_bulk(
        [(cid, base_emb[i].tolist(), cs.BGE_MODEL_ID)
         for i, cid in enumerate(base_ids) if cid in present]
    )
    fs.build_layer_a()
    fs.attach()
    anchors = cs.resolve_anchors(fs)

    need: list[str] = []
    for key in anchors:
        for cid in cs.subtree_chunk_ids(fs, anchors[key]["shelf"]):
            c = fs.chunk_store.get(cid)
            if c is not None and c.embedding is None:
                need.append(cid)
    need = sorted(set(need) - set(base_ids))
    print(f"  [5] anchor-subtree chunks needing embedding: {len(need)}")

    out_emb = base_emb
    out_ids = list(base_ids)
    if need:
        from foodscholar.annotate.embedder import HFEmbedder
        emb = HFEmbedder(cs.BGE_MODEL_ID)
        vecs = emb.embed([fs.chunk_store.get(c).text for c in need])
        out_emb = np.vstack([base_emb, np.asarray(vecs, dtype=np.float32)])
        out_ids = base_ids + need

    # dedup guard (idempotent): ids must be unique and aligned
    assert len(out_ids) == out_emb.shape[0], "emb/ids length mismatch"
    assert len(set(out_ids)) == len(out_ids), "duplicate ids after append"

    emb_path = CACHE_DIR / f"bge_base_emb_{len(out_ids)}.npy"
    ids_path = CACHE_DIR / f"bge_base_ids_{len(out_ids)}.json"
    np.save(emb_path, out_emb)
    ids_path.write_text(json.dumps(out_ids))
    MANIFEST.write_text(json.dumps({
        "emb_file": emb_path.name, "ids_file": ids_path.name,
        "n_vectors": len(out_ids), "dim": int(out_emb.shape[1]),
        "model": cs.BGE_MODEL_ID,
        "emb_sha16": _sha(emb_path), "ids_sha16": _sha(ids_path),
        "snapshot": SNAPSHOT.name, "snapshot_sha16": _sha(SNAPSHOT),
    }, indent=2))
    print(f"  [5] wrote cache {emb_path.name} ({out_emb.shape}) + MANIFEST.json")
    print("\nNOTE: update cs_common.BGE_EMB_NPY/BGE_IDS_JSON to "
          f"bge_base_emb_{len(out_ids)} if it changed.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the snapshot already exists")
    args = ap.parse_args()
    if SNAPSHOT.exists() and not args.force:
        print(f"snapshot {SNAPSHOT.name} exists; pass --force to rebuild. "
              "Running embed step only (idempotent).")
        step5_embed_anchor_subtrees()
        return
    print("Building abstract-expanded case-study corpus (offline, no Elastic):")
    step1_doclevel_nel()
    step2_filter_abstracts()
    step3_stage_dirs()
    step4_ingest()
    step5_embed_anchor_subtrees()
    print("done.")


if __name__ == "__main__":
    main()
