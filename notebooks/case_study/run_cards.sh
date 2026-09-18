#!/usr/bin/env bash
# Robust, fresh-kernel card regeneration. Reads the key from a file you control
# (never on the command line). Usage: bash run_cards.sh /path/to/groq.env
set -euo pipefail
KEYFILE="${1:?usage: run_cards.sh <keyfile with: export GROQ_API_KEY=...>}"
source "$KEYFILE"
: "${GROQ_API_KEY:?GROQ_API_KEY not set after sourcing $KEYFILE}"
cd "$(dirname "$0")"
conda run -n foodscholar python _run_nb.py NB4_cards.ipynb
