#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline.sh
# Master script — runs the full DeepLDA pipeline in order.
# Usage:
#   bash run_pipeline.sh          # all steps
#   bash run_pipeline.sh 01       # only step 01
#   bash run_pipeline.sh 02 03    # steps 02 and 03
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default: run all steps
STEPS=("01" "02" "03")
if [[ $# -gt 0 ]]; then
    STEPS=("$@")
fi

run_step() {
    local step="$1"
    local file
    case "$step" in
        01) file="${SCRIPT_DIR}/01_extract_descriptors.py" ;;
        02) file="${SCRIPT_DIR}/02_train.py" ;;
        03) file="${SCRIPT_DIR}/03_post_training_analysis.py" ;;
        04) file="${SCRIPT_DIR}/04_apply_cv.py" ;;
        *)
            echo "ERROR: Unknown step '${step}'. Valid steps: 01 02 03 04"
            exit 1
            ;;
    esac
    if [[ ! -f "$file" ]]; then
        echo "ERROR: Script not found: ${file}"
        exit 1
    fi
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  Running: $(basename "$file")"
    echo "══════════════════════════════════════════════════════════════"
    python "$file"
}

for s in "${STEPS[@]}"; do
    run_step "$s"
done

echo ""
echo "Pipeline complete."
