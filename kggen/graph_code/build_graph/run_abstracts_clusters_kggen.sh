#!/usr/bin/env bash
#===============================================================================
# run_abstracts_clusters_kggen.sh
#
# Orchestrator script that:
#   1. Extracts KG triples for EACH CSV in chunking/chunks/abstracts/clusters/
#   2. Optionally aggregates all cluster graphs into ONE combined graph
#      (disabled by default; enable with --aggregate-all)
#
# This script ONLY processes abstracts clusters — it does NOT touch
# guides/ or textbooks/ and does NOT create a grand-total aggregation.
#
# Usage:
#   chmod +x run_abstracts_clusters_kggen.sh
#   ./run_abstracts_clusters_kggen.sh
#
#   # With a specific model:
#   ./run_abstracts_clusters_kggen.sh --model "llama3.1:8b-instruct-fp16"
#
#   # Also combine all cluster graphs into one:
#   ./run_abstracts_clusters_kggen.sh --aggregate-all
#
#   # Dry-run (print commands only):
#   ./run_abstracts_clusters_kggen.sh --dry-run
#
# Output structure under kg_output_chunks/:
#   kg_output_chunks/
#   └── abstracts/
#       └── clusters/
#           ├── abstracts_cluster_00/
#           │   ├── chunk_graphs/
#           │   ├── aggregated_graph.json
#           │   └── ...
#           ├── abstracts_cluster_01/
#           │   └── ...
#           └── _aggregated_all_abstracts/      (only with --aggregate-all)
#               ├── aggregated_graph.json
#               └── ...
#===============================================================================

set -uo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTERS_DIR="../data/chunks/abstracts/clusters"
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
AGGREGATE_ALL=True
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
        --aggregate-all)
            AGGREGATE_ALL=true
            shift
            ;;
        --no-aggregate-all)
            AGGREGATE_ALL=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
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
            echo "  --aggregate-all       Also combine ALL cluster graphs into one (default: false)"
            echo "  --no-aggregate-all    Do NOT combine cluster graphs (default)"
            echo "  --dry-run             Print commands without executing"
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
        # Default: run silently
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
# e.g. "abstracts_cluster_00.csv" → "abstracts_cluster_00"
csv_short_name() {
    local csv_path="$1"
    basename "$csv_path" .csv
}

# ── Main ─────────────────────────────────────────────────────────────────────

log_sep "KGGEN Extended — Abstracts Clusters Processing Pipeline"
log "Model         : ${MODEL}"
log "Clusters dir  : ${CLUSTERS_DIR}"
log "Output base   : ${OUTPUT_BASE}"
log "Conda env     : ${CONDA_ENV}"
log "Dry run       : ${DRY_RUN}"
log "Aggregate all : ${AGGREGATE_ALL}"
log "Skip existing : ${SKIP_EXISTING}"
log "Verbose       : ${VERBOSE}"
log "Log dir       : ${LOG_DIR:-'(none — output to terminal only)'}"
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
    if [[ ! -d "${CLUSTERS_DIR}" ]]; then
        log "ERROR: Clusters directory not found: ${CLUSTERS_DIR}"
        exit 1
    fi
fi

# ── Build python runner prefix ───────────────────────────────────────────────
if $DRY_RUN; then
    PY_RUN="python"
else
    # Use conda run with --no-capture-output so Python output streams live to terminal
    PY_RUN="conda run --no-capture-output -n ${CONDA_ENV} python"
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Extract KG for each CSV in abstracts/clusters/
# ══════════════════════════════════════════════════════════════════════════════

CLUSTERS_AGG_INPUTS=()

CLUSTERS_OUT_DIR="${OUTPUT_BASE}/abstracts/clusters"

log_sep "PHASE 1 — Process each CSV in abstracts/clusters/"

if [[ -d "${CLUSTERS_DIR}" ]]; then
    # Count total files first
    total_clusters=0
    for csv_file in "${CLUSTERS_DIR}"/*.csv; do
        [[ -f "$csv_file" ]] && total_clusters=$((total_clusters + 1))
    done
    log "  Found ${total_clusters} abstracts cluster CSV(s)"
    cluster_idx=0
    for csv_file in "${CLUSTERS_DIR}"/*.csv; do
        [[ -f "$csv_file" ]] || continue
        cluster_idx=$((cluster_idx + 1))

        short_name=$(csv_short_name "$csv_file")
        csv_out="${CLUSTERS_OUT_DIR}/${short_name}"

        # Skip if already done
        if $SKIP_EXISTING && [[ -f "${csv_out}/aggregated_graph.json" ]]; then
            log "  ⏭  [${cluster_idx}/${total_clusters}] Skipping (already exists): ${short_name}"
            CLUSTERS_AGG_INPUTS+=("${csv_out}/aggregated_graph.json")
            continue
        fi

        log "  📄 [${cluster_idx}/${total_clusters}] Processing: ${short_name}"
        run_cmd "Extract KG: ${short_name}" \
            ${PY_RUN} "${EXTRACT_SCRIPT}" \
                --csv "${csv_file}" \
                --out-dir "${csv_out}" \
                --model "${MODEL}"

        if [[ -f "${csv_out}/aggregated_graph.json" ]]; then
            CLUSTERS_AGG_INPUTS+=("${csv_out}/aggregated_graph.json")
        fi
    done
else
    log "  ⚠️  Clusters CSV dir not found: ${CLUSTERS_DIR}"
fi

log "  → ${#CLUSTERS_AGG_INPUTS[@]} cluster graph(s) ready for aggregation"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 (OPTIONAL): Aggregate all cluster graphs into ONE combined graph
# ══════════════════════════════════════════════════════════════════════════════

if $AGGREGATE_ALL; then
    log_sep "PHASE 2 — Aggregate ALL cluster graphs into one"

    CLUSTERS_AGG_OUT="${CLUSTERS_OUT_DIR}/_aggregated_all_abstracts"

    if [[ ${#CLUSTERS_AGG_INPUTS[@]} -gt 0 ]]; then
        if $SKIP_EXISTING && [[ -f "${CLUSTERS_AGG_OUT}/aggregated_graph.json" ]]; then
            log "  ⏭  Skipping (already exists): _aggregated_all_abstracts"
        else
            run_cmd "Aggregate ${#CLUSTERS_AGG_INPUTS[@]} cluster graphs" \
                ${PY_RUN} "${AGGREGATE_SCRIPT}" \
                    --graphs "${CLUSTERS_AGG_INPUTS[@]}" \
                    --out-dir "${CLUSTERS_AGG_OUT}" \
                    --model "${MODEL}"
        fi
    else
        log "  ⚠️  No cluster graphs to aggregate"
    fi
else
    log_sep "PHASE 2 — SKIPPED (aggregate-all disabled; pass --aggregate-all to combine all cluster graphs into one)"
fi

# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════

log_sep "Pipeline Complete!"
log ""
log "Output structure:"
log "  ${OUTPUT_BASE}/"
log "  └── abstracts/"
log "      └── clusters/"
if $AGGREGATE_ALL; then
    log "          ├── abstracts_cluster_00/     (per-CSV graphs)"
    log "          ├── ..."
    log "          └── _aggregated_all_abstracts/ (combined graph)"
else
    log "          ├── abstracts_cluster_00/     (per-CSV graphs)"
    log "          ├── ..."
    log "          └── (run with --aggregate-all to also create a combined graph)"
fi
log ""
log "Key output files:"
log "  Per-cluster graphs : ${CLUSTERS_OUT_DIR}/<cluster_name>/aggregated_graph.json"
if $AGGREGATE_ALL; then
    log "  All clusters combined: ${CLUSTERS_OUT_DIR}/_aggregated_all_abstracts/aggregated_graph.json"
fi
