#!/usr/bin/env bash
#===============================================================================
# run_all_kggen.sh
#
# Orchestrator script that:
#   1. Extracts KG triples for EACH CSV in chunks/guides/
#   2. Extracts KG triples for EACH CSV in chunks/textbooks/
#   3. Aggregates all guide graphs into one combined graph
#   4. Aggregates all textbook graphs into one combined graph
#   5. Aggregates guides + textbooks into a grand total graph
#
# Usage:
#   chmod +x run_all_kggen.sh
#   ./run_all_kggen.sh
#
#   # With a specific model:
#   ./run_all_kggen.sh --model "llama3.1:8b-instruct-fp16"
#
#   # Dry-run (print commands only):
#   ./run_all_kggen.sh --dry-run
#
#   # Process only guides:
#   ./run_all_kggen.sh --guides-only
#
#   # Process only textbooks:
#   ./run_all_kggen.sh --textbooks-only
#
# Output structure under kg_output_chunks/:
#   kg_output_chunks/
#   ├── guides/
#   │   ├── be-dietary-recommendations-for-the-belgian-population/
#   │   │   ├── chunk_graphs/
#   │   │   ├── aggregated_graph.json
#   │   │   └── ...
#   │   └── _aggregated_all_guides/
#   │       ├── aggregated_graph.json
#   │       └── ...
#   ├── textbooks/
#   │   ├── tb1_Kansas_Flexbook/
#   │   │   └── ...
#   │   └── _aggregated_all_textbooks/
#   │       └── ...
#   └── _aggregated_all/
#       ├── aggregated_graph.json
#       └── ...
#===============================================================================

set -uo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHUNKS_DIR="../data/chunks"
OUTPUT_BASE="../data/graph"

EXTRACT_SCRIPT="${SCRIPT_DIR}/extract_triplets_from_chunks.py"
AGGREGATE_SCRIPT="${SCRIPT_DIR}/aggregate_graphs.py"

DEFAULT_MODEL="mistral-small3.2:24b-instruct-2506-q8_0"
MODEL="${DEFAULT_MODEL}"

# ── Conda environment ────────────────────────────────────────────────────────
# Change to your conda env name
CONDA_ENV="${KGEN_CONDA_ENV:-python3.12_venv}"

# ── Flags ────────────────────────────────────────────────────────────────────
DRY_RUN=false
GUIDES_ONLY=false
TEXTBOOKS_ONLY=false
SKIP_EXISTING=false
VERBOSE=false
LOG_DIR=""

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|-m)
            MODEL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --guides-only)
            GUIDES_ONLY=true
            shift
            ;;
        --textbooks-only)
            TEXTBOOKS_ONLY=true
            shift
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        --conda-env)
            CONDA_ENV="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model, -m MODEL     Model to use (default: ${DEFAULT_MODEL})"
            echo "  --dry-run             Print commands without executing"
            echo "  --guides-only         Process only guides/"
            echo "  --textbooks-only      Process only textbooks/"
            echo "  --skip-existing       Skip CSVs whose output dir already has aggregated_graph.json"
            echo "  --conda-env ENV       Conda environment name (default: ${CONDA_ENV})"
            echo "  --verbose, -v         Show full Python script output in real-time"
            echo "  --log-dir DIR         Directory for per-script log files (default: none)"
            echo "  -h, --help            Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ── Helper functions ─────────────────────────────────────────────────────────

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_sep() {
    echo ""
    echo "========================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "========================================================================"
}

run_cmd() {
    local desc="$1"
    shift
    log "→ ${desc}"
    log "  CMD: $*"
    if $DRY_RUN; then
        log "  [DRY-RUN] Would execute: $*"
        return 0
    fi

    local rc=0

    # Decide how to capture output
    if [[ -n "${LOG_DIR}" ]]; then
        # Log output to a per-call log file AND optionally show on terminal
        mkdir -p "${LOG_DIR}"
        local logfile="${LOG_DIR}/$(date '+%Y%m%d_%H%M%S')_$(echo "${desc}" | tr ' /' '__').log"
        log "  Log file: ${logfile}"

        if $VERBOSE; then
            # Show output in real-time AND write to log
            "$@" 2>&1 | tee "${logfile}"
            rc=${PIPESTATUS[0]}
        else
            # Write to log only, show progress dots
            "$@" > "${logfile}" 2>&1 &
            local pid=$!
            local dots=""
            while kill -0 $pid 2>/dev/null; do
                dots="${dots}."
                printf "\r  Running%s   " "$dots"
                sleep 5
            done
            wait $pid
            rc=$?
            printf "\r  Done%s (exit=${rc})\n" "$dots"
        fi
    elif $VERBOSE; then
        # Verbose mode: just run normally so output is visible in terminal
        "$@" 2>&1
        rc=$?
    else
        # Default: run silently (output goes to terminal anyway with conda run)
        "$@"
        rc=$?
    fi

    if [[ $rc -ne 0 ]]; then
        log "  ⚠️  Command failed with exit code ${rc}, continuing..."
    else
        log "  ✓ Completed successfully"
    fi
    return $rc
}

# Derive a short name from a CSV filename
# e.g. "chunks_guide_be-dietary-recommendations-for-the-belgian-population.csv"
#   → "be-dietary-recommendations-for-the-belgian-population"
csv_short_name() {
    local csv_path="$1"
    local base
    base=$(basename "$csv_path" .csv)
    # Strip chunks_guide_ or chunks_textbook_ prefix
    base="${base#chunks_guide_}"
    base="${base#chunks_textbook_}"
    echo "$base"
}

# ── Main ─────────────────────────────────────────────────────────────────────

log_sep "KGGEN Extended — Full Chunk Processing Pipeline"
log "Model        : ${MODEL}"
log "Chunks dir   : ${CHUNKS_DIR}"
log "Output base  : ${OUTPUT_BASE}"
log "Conda env    : ${CONDA_ENV}"
log "Dry run      : ${DRY_RUN}"
log "Guides only  : ${GUIDES_ONLY}"
log "Textbooks only: ${TEXTBOOKS_ONLY}"
log "Skip existing: ${SKIP_EXISTING}"
log "Verbose      : ${VERBOSE}"
log "Log dir      : ${LOG_DIR:-'(none — output to terminal only)'}"
log ""

# ── Verify prerequisites ─────────────────────────────────────────────────────
if ! $DRY_RUN; then
    if [[ ! -f "${EXTRACT_SCRIPT}" ]]; then
        log "ERROR: Extract script not found: ${EXTRACT_SCRIPT}"
        exit 1
    fi
    if [[ ! -f "${AGGREGATE_SCRIPT}" ]]; then
        log "ERROR: Aggregate script not found: ${AGGREGATE_SCRIPT}"
        exit 1
    fi
    if [[ ! -d "${CHUNKS_DIR}" ]]; then
        log "ERROR: Chunks directory not found: ${CHUNKS_DIR}"
        exit 1
    fi
fi

# ── Build python runner prefix ───────────────────────────────────────────────
if $DRY_RUN; then
    PY_RUN="python"
else
    # Use conda run with --no-capture-output so Python output streams live to terminal
    # Without --no-capture-output, conda buffers ALL output until the process exits
    PY_RUN="conda run --no-capture-output -n ${CONDA_ENV} python"
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Extract KG for each CSV in guides/
# ══════════════════════════════════════════════════════════════════════════════

GUIDES_AGG_INPUTS=()

if ! $TEXTBOOKS_ONLY; then
    GUIDES_CSV_DIR="${CHUNKS_DIR}/guides"
    GUIDES_OUT_DIR="${OUTPUT_BASE}/guides"

    log_sep "PHASE 1 — Process each CSV in guides/"

    if [[ -d "${GUIDES_CSV_DIR}" ]]; then
        # Count total files first
        total_guides=0
        for csv_file in "${GUIDES_CSV_DIR}"/*.csv; do
            [[ -f "$csv_file" ]] && total_guides=$((total_guides + 1))
        done
        log "  Found ${total_guides} guide CSV(s)"
        guide_idx=0
        for csv_file in "${GUIDES_CSV_DIR}"/*.csv; do
            [[ -f "$csv_file" ]] || continue
            guide_idx=$((guide_idx + 1))

            short_name=$(csv_short_name "$csv_file")
            csv_out="${GUIDES_OUT_DIR}/${short_name}"

            # Skip if already done
            if $SKIP_EXISTING && [[ -f "${csv_out}/aggregated_graph.json" ]]; then
                log "  ⏭  [${guide_idx}/${total_guides}] Skipping (already exists): ${short_name}"
                GUIDES_AGG_INPUTS+=("${csv_out}/aggregated_graph.json")
                continue
            fi

            log "  📄 [${guide_idx}/${total_guides}] Processing: ${short_name}"
            run_cmd "Extract KG: ${short_name}" \
                ${PY_RUN} "${EXTRACT_SCRIPT}" \
                    --csv "${csv_file}" \
                    --out-dir "${csv_out}" \
                    --model "${MODEL}"

            if [[ -f "${csv_out}/aggregated_graph.json" ]]; then
                GUIDES_AGG_INPUTS+=("${csv_out}/aggregated_graph.json")
            fi
        done
    else
        log "  Guides CSV dir not found: ${GUIDES_CSV_DIR}"
    fi

    log "  → ${#GUIDES_AGG_INPUTS[@]} guide graph(s) ready for aggregation"
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Extract KG for each CSV in textbooks/
# ══════════════════════════════════════════════════════════════════════════════

TEXTBOOKS_AGG_INPUTS=()

if ! $GUIDES_ONLY; then
    TEXTBOOKS_CSV_DIR="${CHUNKS_DIR}/textbooks"
    TEXTBOOKS_OUT_DIR="${OUTPUT_BASE}/textbooks"

    log_sep "PHASE 2 — Process each CSV in textbooks/"

    if [[ -d "${TEXTBOOKS_CSV_DIR}" ]]; then
        # Count total files first
        total_textbooks=0
        for csv_file in "${TEXTBOOKS_CSV_DIR}"/*.csv; do
            [[ -f "$csv_file" ]] && total_textbooks=$((total_textbooks + 1))
        done
        log "  Found ${total_textbooks} textbook CSV(s)"
        textbook_idx=0
        for csv_file in "${TEXTBOOKS_CSV_DIR}"/*.csv; do
            [[ -f "$csv_file" ]] || continue
            textbook_idx=$((textbook_idx + 1))

            short_name=$(csv_short_name "$csv_file")
            csv_out="${TEXTBOOKS_OUT_DIR}/${short_name}"

            # Skip if already done
            if $SKIP_EXISTING && [[ -f "${csv_out}/aggregated_graph.json" ]]; then
                log "  ⏭  [${textbook_idx}/${total_textbooks}] Skipping (already exists): ${short_name}"
                TEXTBOOKS_AGG_INPUTS+=("${csv_out}/aggregated_graph.json")
                continue
            fi

            log "  📄 [${textbook_idx}/${total_textbooks}] Processing: ${short_name}"
            run_cmd "Extract KG: ${short_name}" \
                ${PY_RUN} "${EXTRACT_SCRIPT}" \
                    --csv "${csv_file}" \
                    --out-dir "${csv_out}" \
                    --model "${MODEL}"

            if [[ -f "${csv_out}/aggregated_graph.json" ]]; then
                TEXTBOOKS_AGG_INPUTS+=("${csv_out}/aggregated_graph.json")
            fi
        done
    else
        log "  Textbooks CSV dir not found: ${TEXTBOOKS_CSV_DIR}"
    fi

    log "  → ${#TEXTBOOKS_AGG_INPUTS[@]} textbook graph(s) ready for aggregation"
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Aggregate all guide graphs into one combined graph
# ══════════════════════════════════════════════════════════════════════════════

ALL_AGG_INPUTS=()

if ! $TEXTBOOKS_ONLY; then
    log_sep "PHASE 3 — Aggregate all guide graphs"

    GUIDES_AGG_OUT="${OUTPUT_BASE}/guides/_aggregated_all_guides"

    if [[ ${#GUIDES_AGG_INPUTS[@]} -gt 0 ]]; then
        if $SKIP_EXISTING && [[ -f "${GUIDES_AGG_OUT}/aggregated_graph.json" ]]; then
            log "  ⏭  Skipping (already exists): _aggregated_all_guides"
        else
            run_cmd "Aggregate ${#GUIDES_AGG_INPUTS[@]} guide graphs" \
                ${PY_RUN} "${AGGREGATE_SCRIPT}" \
                    --graphs "${GUIDES_AGG_INPUTS[@]}" \
                    --out-dir "${GUIDES_AGG_OUT}" \
                    --model "${MODEL}"
        fi

        if [[ -f "${GUIDES_AGG_OUT}/aggregated_graph.json" ]]; then
            ALL_AGG_INPUTS+=("${GUIDES_AGG_OUT}/aggregated_graph.json")
        fi
    else
        log "  ⚠️  No guide graphs to aggregate"
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Aggregate all textbook graphs into one combined graph
# ══════════════════════════════════════════════════════════════════════════════

if ! $GUIDES_ONLY; then
    log_sep "PHASE 4 — Aggregate all textbook graphs"

    TEXTBOOKS_AGG_OUT="${OUTPUT_BASE}/textbooks/_aggregated_all_textbooks"

    if [[ ${#TEXTBOOKS_AGG_INPUTS[@]} -gt 0 ]]; then
        if $SKIP_EXISTING && [[ -f "${TEXTBOOKS_AGG_OUT}/aggregated_graph.json" ]]; then
            log "  ⏭  Skipping (already exists): _aggregated_all_textbooks"
        else
            run_cmd "Aggregate ${#TEXTBOOKS_AGG_INPUTS[@]} textbook graphs" \
                ${PY_RUN} "${AGGREGATE_SCRIPT}" \
                    --graphs "${TEXTBOOKS_AGG_INPUTS[@]}" \
                    --out-dir "${TEXTBOOKS_AGG_OUT}" \
                    --model "${MODEL}"
        fi

        if [[ -f "${TEXTBOOKS_AGG_OUT}/aggregated_graph.json" ]]; then
            ALL_AGG_INPUTS+=("${TEXTBOOKS_AGG_OUT}/aggregated_graph.json")
        fi
    else
        log "  ⚠️  No textbook graphs to aggregate"
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: Aggregate guides + textbooks into GRAND TOTAL graph
# ══════════════════════════════════════════════════════════════════════════════

log_sep "PHASE 5 — Aggregate ALL graphs (guides + textbooks)"

GRAND_OUT="${OUTPUT_BASE}/_aggregated_all"

if [[ ${#ALL_AGG_INPUTS[@]} -gt 0 ]]; then
    if $SKIP_EXISTING && [[ -f "${GRAND_OUT}/aggregated_graph.json" ]]; then
        log "  ⏭  Skipping (already exists): _aggregated_all"
    else
        run_cmd "Aggregate ALL (${#ALL_AGG_INPUTS[@]} sub-aggregated graphs)" \
            ${PY_RUN} "${AGGREGATE_SCRIPT}" \
                --graphs "${ALL_AGG_INPUTS[@]}" \
                --out-dir "${GRAND_OUT}" \
                --model "${MODEL}"
    fi
else
    # Fallback: if no sub-aggregations were done, directly aggregate all per-CSV graphs
    ALL_DIRECT_INPUTS=("${GUIDES_AGG_INPUTS[@]}" "${TEXTBOOKS_AGG_INPUTS[@]}")
    if [[ ${#ALL_DIRECT_INPUTS[@]} -gt 0 ]]; then
        log "  (Using ${#ALL_DIRECT_INPUTS[@]} per-CSV graphs directly)"

        if $SKIP_EXISTING && [[ -f "${GRAND_OUT}/aggregated_graph.json" ]]; then
            log "  ⏭  Skipping (already exists): _aggregated_all"
        else
            run_cmd "Aggregate ALL (${#ALL_DIRECT_INPUTS[@]} per-CSV graphs)" \
                ${PY_RUN} "${AGGREGATE_SCRIPT}" \
                    --graphs "${ALL_DIRECT_INPUTS[@]}" \
                    --out-dir "${GRAND_OUT}" \
                    --model "${MODEL}"
        fi
    else
        log "  ⚠️  No graphs to aggregate for grand total"
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════

log_sep "Pipeline Complete!"
log ""
log "Output structure:"
log "  ${OUTPUT_BASE}/"
log "  ├── guides/"
log "  │   ├── <guide_name>/"
log "  │   │   ├── chunk_graphs/"
log "  │   │   ├── aggregated_graph.json"
log "  │   │   └── ..."
log "  │   └── _aggregated_all_guides/"
log "  │       ├── aggregated_graph.json"
log "  │       └── ..."
log "  ├── textbooks/"
log "  │   ├── <textbook_name>/"
log "  │   │   └── ..."
log "  │   └── _aggregated_all_textbooks/"
log "  │       └── ..."
log "  └── _aggregated_all/"
log "      ├── aggregated_graph.json"
log "      └── ..."
log ""
log "Key output files:"
log "  All guides aggregated : ${OUTPUT_BASE}/guides/_aggregated_all_guides/aggregated_graph.json"
log "  All textbooks aggregated: ${OUTPUT_BASE}/textbooks/_aggregated_all_textbooks/aggregated_graph.json"
log "  Grand total           : ${OUTPUT_BASE}/_aggregated_all/aggregated_graph.json"
