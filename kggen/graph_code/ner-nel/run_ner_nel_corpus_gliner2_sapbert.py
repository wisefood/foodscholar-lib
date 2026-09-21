"""
run_ner_nel_corpus_gliner2_sapbert.py
-------------------------------------
Corpus-wide NER + NEL pipeline with the SELECTED production defaults.

Selected on the guides + journal benchmark (see ner_nel_guides_journal.ipynb):
    NER : GLiNER2  (fastino/gliner2-large-v1)  @ confidence 0.35
          - best semantic F1 and best precision-with-usable-recall on both corpora
    NEL : HNSW + SapBERT, top-1, cosine >= 0.70, NO reranker
          - highest link rate + URI diversity, dominates BioLORD, 0.70 gate -> NIL

This is the refined successor to run_ner_nel_chunks_corpus.py, which used the
OLD defaults (GLiNER-bio v0.1 @0.4 + BioLORD). Differences:
    * GLiNER2 API (batch_extract_entities + described label set) replaces GLiNER.inference()
    * NEL goes through src.nel.HNSWNELLinker(encoder="sapbert") so the query
      encoder matches the encoder the foodon_hnsw_sapbert.bin index was built with
      (SapBERT needs CLS pooling — a plain SentenceTransformer would mismatch).
    * CORPUS_DIR -> new_experiments/corpus/big

Scans all *.csv in CORPUS_DIR (one file per source document; abstracts are one
file), runs GLiNER2 NER on each chunk, links entities to FoodOn, and writes one
output CSV per input file under OUTPUT_DIR.

Input CSV format : chunk_id, chunk_text, type, chunk_metadata
Output CSV format (prefixed nel_): chunk_id, chunk_entities_ner, chunk_uri_nel
    chunk_entities_ner — ';'-separated entity surface forms (deduped per chunk)
    chunk_uri_nel      — ';'-separated FoodOn URIs ('' for NIL), 1-to-1 with the
                         entities column

Usage:
    conda activate merlin_cuda
    cd /mnt/data/makis/merlin/LinearRAG
    python run_ner_nel_corpus_gliner2_sapbert.py

Re-running is safe: a fully-written nel_<file>.csv is skipped, so the run resumes
where it stopped. The NEL cache is shared across all files within one run.
"""

import csv
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

CORPUS_DIR      = "/mnt/data/vpitsilou/wisefood/new_experiments/corpus/big"
OUTPUT_DIR      = "/mnt/data/makis/merlin/LinearRAG/outputs/corpus_nel_gliner2_sapbert"

REPO_ROOT       = "/mnt/data/makis/merlin/LinearRAG"
HNSW_INDEX_PATH = f"{REPO_ROOT}/ontology/foodon_hnsw_sapbert.bin"
METADATA_PATH   = f"{REPO_ROOT}/ontology/foodon_metadata.json"

# NER — selected default
G2_MODEL          = "fastino/gliner2-large-v1"
GLINER_THRESHOLD  = 0.35
GLINER_BATCH_SIZE = 16

# NEL — selected default
NEL_ENCODER = "sapbert"
NEL_TOP_K   = 1
NEL_MIN_SIM = 0.70

LOG_EVERY_N = 100

# ---------------------------------------------------------------------------
# GLiNER2 label set — IDENTICAL to the benchmark that selected GLiNER2.
# (Changing the labels would invalidate the selection, so keep these verbatim.)
# ---------------------------------------------------------------------------

GLINER2_CUSTOM_LABELS = {
    "food": "Specific food items, ingredients, or beverages mentioned by name in scientific nutrition text (e.g. olive oil, red meat, green tea, whole grains, salmon, blueberries)",
    "food additive": "Substances intentionally added to food for preservation, flavouring, colouring, or texturising (e.g. sodium benzoate, aspartame, monosodium glutamate, lecithin, carrageenan)",
    "dietary pattern": "Structured overall eating patterns or dietary regimens studied as an intervention or exposure (e.g. Mediterranean diet, Western diet, DASH diet, plant-based diet, ketogenic diet, veganism)",
    "dietary supplement": "Nutritional supplements or nutraceuticals taken in addition to normal diet (e.g. fish oil capsules, multivitamins, probiotic supplements, omega-3 supplements, protein powder)",
    "nutrient": "General nutritional components not classified as a specific vitamin or mineral (e.g. dietary fibre, total protein, carbohydrate, polyphenol, antioxidant, omega-3 fatty acid, flavonoid)",
    "vitamin": "Specific vitamins identified by letter, number, or full chemical name (e.g. vitamin C, vitamin D3, retinol, alpha-tocopherol, folate, riboflavin, thiamine, niacin)",
    "mineral": "Specific dietary minerals or trace elements (e.g. calcium, magnesium, iron, zinc, selenium, potassium, sodium, phosphorus, iodine)",
    "amino acid": "Specific amino acids as building blocks of proteins or metabolic intermediates (e.g. leucine, tryptophan, glutamine, methionine, arginine, branched-chain amino acids)",
    "lipid": "Specific fats, fatty acids, cholesterol fractions, or lipid molecules (e.g. LDL cholesterol, triglycerides, arachidonic acid, DHA, EPA, saturated fatty acid)",
    "chemical": "Chemical compounds, molecules, or substances relevant to nutrition or biology that are not nutrients or drugs (e.g. resveratrol, curcumin, quercetin, ethanol, phytosterol)",
    "drug": "Pharmaceutical drugs, medications, or clinical interventions (e.g. metformin, statins, aspirin, insulin, orlistat)",
    "biomarker": "Measurable biological markers in blood, urine, or tissue used to assess metabolic or health status (e.g. HbA1c, C-reactive protein, LDL, BMI, fasting glucose, homocysteine, IL-6)",
    "enzyme": "Specific biological enzymes involved in metabolism, digestion, or biochemical pathways (e.g. lipase, amylase, COX-2, superoxide dismutase, glutathione peroxidase)",
    "hormone": "Hormones involved in metabolism, appetite regulation, or body homeostasis (e.g. insulin, leptin, ghrelin, cortisol, adiponectin, thyroid hormone)",
    "gene": "Named genes or gene symbols relevant to nutrition or metabolism research (e.g. APOE, FTO, PPARG, TCF7L2, MTHFR)",
    "genotype": "Genetic variants, single nucleotide polymorphisms, or genotypes (e.g. APOE epsilon4 allele, rs9939609, MTHFR C677T, heterozygous carriers)",
    "microbe": "Bacteria, viruses, fungi, or other microorganisms relevant to gut health or disease (e.g. Lactobacillus, Bifidobacterium, Helicobacter pylori, gut microbiota)",
    "disease": "Diagnosed medical conditions, disorders, or pathological conditions (e.g. type 2 diabetes, cardiovascular disease, obesity, metabolic syndrome, colorectal cancer)",
    "symptom": "Clinical symptoms, signs of disease, or physiological abnormalities reported in patients (e.g. hypertension, hyperglycaemia, systemic inflammation, fatigue, insulin resistance)",
    "organ or tissue": "Body organs, tissues, or anatomical structures (e.g. liver, skeletal muscle, adipose tissue, gut, colon, pancreas, small intestine)",
    "physiological process": "Biological or physiological processes and mechanisms in the body (e.g. oxidative stress, lipid oxidation, inflammation, gut microbiota composition, satiety signalling)",
    "population": "Defined human study groups or cohorts characterised by demographics, health status, or geography (e.g. postmenopausal women, elderly adults over 65, obese children, type 2 diabetic patients)",
    "life stage": "Specific stages of human life used as inclusion criteria or study context (e.g. infancy, childhood, adolescence, pregnancy, lactation, menopause, old age)",
    "exercise": "Physical exercise interventions, physical activity levels, or sport modalities (e.g. aerobic exercise, resistance training, sedentary behaviour, walking, HIIT)",
    "measurement": "Quantitative measurements, dosages, amounts, or clinical indices with units (e.g. 500 mg/day, 2 g/kg body weight, 30% energy from fat, 95th percentile BMI)",
    "time expression": "Time periods, durations, follow-up intervals, or temporal references in study design (e.g. 12 weeks, 5-year follow-up, baseline, 6 months post-intervention, 24-hour recall)",
    "country": "Countries, sovereign states, territories, or national geographic entities (e.g. Greece, United States, China, Japan, United Kingdom, South Korea).",
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Personal HF cache: GLiNER2's DeBERTa-v3-large and SapBERT both load cleanly
# here. (The shared cache has DeBERTa lock files owned by other users.)
os.environ.setdefault("HF_HOME", "/home/spapadias/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/home/spapadias/.cache/huggingface")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_gliner2(model_id: str):
    """Load GLiNER2 (quantized on GPU)."""
    logger.info(f"Loading GLiNER2: {model_id}")
    t0 = time.time()
    import torch
    from gliner2 import GLiNER2
    map_loc = "cuda" if torch.cuda.is_available() else "cpu"
    model = GLiNER2.from_pretrained(model_id, map_location=map_loc,
                                    quantize=(map_loc == "cuda"))
    logger.info(f"GLiNER2 loaded on {map_loc} in {time.time() - t0:.1f}s")
    return model


def load_nel_linker():
    """HNSW + SapBERT linker. Uses src.nel.HNSWNELLinker so the query encoder
    matches the encoder used to build foodon_hnsw_sapbert.bin."""
    logger.info(f"Loading NEL linker: HNSW + {NEL_ENCODER} "
                f"(top_k={NEL_TOP_K}, min_sim={NEL_MIN_SIM})")
    t0 = time.time()
    from src.nel import HNSWNELLinker
    linker = HNSWNELLinker(
        index_path=HNSW_INDEX_PATH,
        metadata_path=METADATA_PATH,
        encoder=NEL_ENCODER,
        top_k=NEL_TOP_K,
        min_sim=NEL_MIN_SIM,
    )
    linker.load()
    logger.info(f"NEL linker loaded in {time.time() - t0:.1f}s")
    return linker


# ---------------------------------------------------------------------------
# NER — GLiNER2 batch inference
# ---------------------------------------------------------------------------

def run_ner_batch(gliner2_model, texts: list) -> list:
    """Run GLiNER2 on a batch of texts. Returns one deduped (lowercased, first
    occurrence preserved) surface-form list per text."""
    if not texts:
        return []
    try:
        batch_results = gliner2_model.batch_extract_entities(
            texts, GLINER2_CUSTOM_LABELS,
            batch_size=len(texts), threshold=GLINER_THRESHOLD,
            include_confidence=True, include_spans=True,
        )
    except Exception as e:
        logger.warning(f"GLiNER2 batch error: {e}. Falling back to per-text.")
        batch_results = []
        for text in texts:
            try:
                batch_results.append(
                    gliner2_model.extract_entities(
                        text, GLINER2_CUSTOM_LABELS, threshold=GLINER_THRESHOLD,
                        include_confidence=True, include_spans=True)
                )
            except Exception as e2:
                logger.warning(f"GLiNER2 per-text error: {e2}")
                batch_results.append({"entities": {}})

    results = []
    for res in batch_results:
        seen, deduped = set(), []
        for _label, ents in res.get("entities", {}).items():
            for ent in ents:
                surface = " ".join(str(ent.get("text", "")).split())
                key = surface.lower()
                if surface and len(surface) >= 2 and key not in seen:
                    seen.add(key)
                    deduped.append(surface)
        results.append(deduped)
    return results


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

def read_chunks(csv_path: str) -> list:
    """Read a chunk CSV -> list of {chunk_id, chunk_text}."""
    chunks = []
    csv.field_size_limit(10 * 1024 * 1024)  # 10 MB per field
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = (row.get("chunk_id") or "").strip()
            txt = (row.get("chunk_text") or "").strip()
            if cid:
                chunks.append({"chunk_id": cid, "chunk_text": txt})
    logger.info(f"Read {len(chunks)} chunks from {Path(csv_path).name}")
    return chunks


# ---------------------------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------------------------

def process_chunks(chunks, gliner2_model, linker, nel_cache):
    """NER (batched) + NEL (per surface form, cached). Returns output rows."""
    output_rows = []
    t_start = time.time()
    total = len(chunks)

    for batch_start in range(0, total, GLINER_BATCH_SIZE):
        batch = chunks[batch_start: batch_start + GLINER_BATCH_SIZE]
        texts = [c["chunk_text"] for c in batch]
        batch_entities = run_ner_batch(gliner2_model, texts)

        for chunk, entities in zip(batch, batch_entities):
            uris = []
            for surface in entities:
                key = surface.lower()
                if key in nel_cache:
                    uri = nel_cache[key]
                else:
                    res = linker.link(surface)
                    uri = res.uri or ""        # '' for NIL (below 0.70 cosine)
                    nel_cache[key] = uri
                uris.append(uri)
            output_rows.append({
                "chunk_id":           chunk["chunk_id"],
                "chunk_entities_ner": ";".join(entities),
                "chunk_uri_nel":      ";".join(uris),
            })

        processed = min(batch_start + GLINER_BATCH_SIZE, total)
        if processed % LOG_EVERY_N == 0 or processed == total:
            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (total - processed) / rate if rate > 0 else 0
            logger.info(f"  {processed}/{total} chunks "
                        f"({rate:.1f} chunks/s, ~{remaining/60:.1f} min remaining)")
    return output_rows


def write_output(rows: list, output_path: str):
    """Write output CSV atomically (tmp + rename) so an interrupted run never
    leaves a truncated file that the skip-if-done check would treat as complete."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(output_path).with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["chunk_id", "chunk_entities_ner", "chunk_uri_nel"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(output_path)
    logger.info(f"Output written to {output_path} ({len(rows)} rows)")


def file_stats(rows: list):
    found = sum(len(r["chunk_entities_ner"].split(";")) if r["chunk_entities_ner"] else 0
                for r in rows)
    linked = sum(sum(1 for u in r["chunk_uri_nel"].split(";") if u) if r["chunk_uri_nel"] else 0
                 for r in rows)
    return len(rows), found, linked


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("Corpus-wide NER+NEL pipeline — GLiNER2 + SapBERT (selected defaults)")
    logger.info(f"NER: GLiNER2 ({G2_MODEL}) @ threshold {GLINER_THRESHOLD}")
    logger.info(f"NEL: HNSW + {NEL_ENCODER} (min_sim={NEL_MIN_SIM}, top_k={NEL_TOP_K})")
    logger.info(f"Corpus dir : {CORPUS_DIR}")
    logger.info(f"Output dir : {OUTPUT_DIR}")
    logger.info("=" * 60)

    corpus_path = Path(CORPUS_DIR)
    if not corpus_path.exists():
        logger.error(f"CORPUS_DIR does not exist: {CORPUS_DIR}")
        sys.exit(1)

    input_files = sorted(corpus_path.glob("*.csv"))
    if not input_files:
        logger.error(f"No *.csv files found in {CORPUS_DIR}")
        sys.exit(1)
    logger.info(f"Found {len(input_files)} input file(s).")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    gliner2_model = load_gliner2(G2_MODEL)
    linker = load_nel_linker()

    nel_cache = {}    # surface_form(lower) -> uri, shared across all files
    summary_rows = []

    for csv_path in input_files:
        output_path = Path(OUTPUT_DIR) / f"nel_{csv_path.name}"
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"SKIP (already done): {output_path.name}")
            continue

        logger.info("-" * 60)
        logger.info(f"Processing: {csv_path.name}")
        chunks = read_chunks(str(csv_path))
        if not chunks:
            logger.warning(f"No valid chunks in {csv_path.name}, skipping.")
            summary_rows.append((csv_path.name, 0, 0, 0))
            continue

        rows = process_chunks(chunks, gliner2_model, linker, nel_cache)
        write_output(rows, str(output_path))
        summary_rows.append((csv_path.name, *file_stats(rows)))

    # Final summary
    logger.info("=" * 60)
    logger.info("Final summary")
    logger.info("=" * 60)
    col_w = max((len(r[0]) for r in summary_rows), default=20)
    header = (f"{'filename':<{col_w}}  {'chunks':>8}  {'entities_found':>14}  "
              f"{'entities_linked':>15}  {'link_rate':>9}")
    print(header); print("-" * len(header))
    tc = tf = tl = 0
    for fname, chunks, found, linked in summary_rows:
        rate = f"{100*linked/found:.1f}%" if found > 0 else "N/A"
        print(f"{fname:<{col_w}}  {chunks:>8}  {found:>14}  {linked:>15}  {rate:>9}")
        tc += chunks; tf += found; tl += linked
    print("-" * len(header))
    trate = f"{100*tl/tf:.1f}%" if tf > 0 else "N/A"
    print(f"{'TOTAL':<{col_w}}  {tc:>8}  {tf:>14}  {tl:>15}  {trate:>9}")
    print(f"\nUnique surface forms linked this run (cache): {sum(1 for v in nel_cache.values() if v)}/{len(nel_cache)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
