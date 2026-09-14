#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_getcontacts_all.sh
# Generates residue contacts for every replica of every state, then combines
# them into one contact-frequency file per state.
#
# This is a one-time preprocessing stage that runs BEFORE scripts/01. It uses
# the `getcontacts` Conda environment (vmd-python), not `deep-allo`.
#
# Usage, from the descriptors/ directory:
#   conda activate getcontacts
#   bash run_getcontacts_all.sh                   # contacts + frequencies, all states
#   bash run_getcontacts_all.sh contacts          # per-replica TSVs only
#   bash run_getcontacts_all.sh freq              # frequency files only
#
# An optional state list restricts the run to those states:
#   bash run_getcontacts_all.sh contacts Apo      # Apo replicas only
#   bash run_getcontacts_all.sh all Apo OM        # two states, both stages
# ─────────────────────────────────────────────────────────────────────────────

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STATES=(Apo Mava OM)
TRAJS_DIR="../Trajs"
OUT_DIR="getcontacts_output"
SELE="protein and resid 1 to 780"
ITYPES=(hb sb)

STAGE="${1:-all}"
case "$STAGE" in
    all|contacts|freq) ;;
    *) echo "ERROR: unknown stage '${STAGE}'. Use: all | contacts | freq"; exit 1 ;;
esac
shift || true

# Remaining arguments, if any, select state-directory names. The three bundled
# study states above are only the default; custom names are accepted.
if [[ $# -gt 0 ]]; then
    for requested in "$@"; do
        if [[ ! "$requested" =~ ^[A-Za-z0-9_.-]+$ || "$requested" == "." || "$requested" == ".." ]]; then
            echo "ERROR: invalid state-directory name '${requested}'."
            exit 1
        fi
    done
    STATES=("$@")
fi
echo "States: ${STATES[*]}   Stage: ${STAGE}"

# ── Guards ──────────────────────────────────────────────────────────────────
# GetContacts is not distributed with DeepAllo. Clone it here as described in
# README.md before running this script.
if [[ ! -f getcontacts/get_dynamic_contacts.py ]]; then
    echo "ERROR: getcontacts/ not found in $(pwd)."
    echo "       Clone it first (see README.md, section 1):"
    echo "         git clone https://github.com/getcontacts/getcontacts.git getcontacts"
    exit 1
fi

if ! python -c "import vmd" >/dev/null 2>&1; then
    echo "ERROR: 'import vmd' failed. GetContacts needs the vmd-python package,"
    echo "       which lives in the 'getcontacts' environment, not 'deep-allo':"
    echo "         conda activate getcontacts"
    exit 1
fi

# On macOS the default 'spawn' start method cannot pickle GetContacts' open
# file handles, so route through the launcher that forces 'fork'.
if [[ "$(uname -s)" == "Darwin" ]]; then
    RUNNER=(python run_getcontacts_macos.py)
else
    RUNNER=(python getcontacts/get_dynamic_contacts.py)
fi

FAILED=()

# A run that is interrupted mid-write leaves a non-empty but truncated TSV, so
# "file exists" is not a safe completion test. GetContacts records the frame
# count in the header, and frames are emitted in order, so a file is complete
# only when its last data row is frame total_frames-1. Both reads are cheap
# even on a 30 MB file. A frame with zero contacts at the very end would be
# re-run unnecessarily; that costs seconds, whereas trusting a truncated file
# silently corrupts the descriptor set.
is_complete() {
    local f="$1" total last
    [[ -s "$f" ]] || return 1
    total="$(head -1 "$f" | sed -n 's/.*total_frames:\([0-9]*\).*/\1/p')"
    [[ -n "$total" ]] || return 1
    last="$(tail -1 "$f" | cut -f1)"
    [[ "$last" =~ ^[0-9]+$ ]] || return 1
    [[ "$last" -eq $((total - 1)) ]]
}

# ── Stage 1: per-replica contacts ───────────────────────────────────────────
if [[ "$STAGE" == "all" || "$STAGE" == "contacts" ]]; then
    for state in "${STATES[@]}"; do
        topology="${TRAJS_DIR}/${state}/topology.pdb"
        if [[ ! -f "$topology" ]]; then
            echo "ERROR ${state}: no topology at ${topology}"
            FAILED+=("${state}/topology")
            continue
        fi
        mkdir -p "${OUT_DIR}/${state}"
        shopt -s nullglob
        trajectories=("${TRAJS_DIR}/${state}"/*.xtc)
        shopt -u nullglob
        if [[ ${#trajectories[@]} -eq 0 ]]; then
            echo "ERROR ${state}: no .xtc files under ${TRAJS_DIR}/${state}"
            FAILED+=("${state}/trajectories")
            continue
        fi
        for traj in "${trajectories[@]}"; do
            replica="$(basename "$traj" .xtc)"
            out="${OUT_DIR}/${state}/${replica}_HB_SB.tsv"

            if is_complete "$out"; then
                echo "SKIP  ${state}/${replica} (already done)"
                continue
            elif [[ -e "$out" ]]; then
                echo "REDO  ${state}/${replica} (incomplete output, regenerating)"
                rm -f "$out"
            fi

            echo "RUN   ${state}/${replica}"
            if ! PYTHONPATH=getcontacts "${RUNNER[@]}" \
                    --topology "$topology" \
                    --trajectory "$traj" \
                    --sele "$SELE" \
                    --itypes "${ITYPES[@]}" \
                    --output "$out"; then
                echo "FAIL  ${state}/${replica}"
                FAILED+=("${state}/${replica}")
                rm -f "$out"
            fi
        done
    done
fi

# ── Stage 2: per-state contact frequencies ──────────────────────────────────
# get_contact_frequencies.py accepts several inputs and weights each replica by
# its frame count, so the combined frequency is not a mean of per-file means.
if [[ "$STAGE" == "all" || "$STAGE" == "freq" ]]; then
    for state in "${STATES[@]}"; do
        shopt -s nullglob
        inputs=()
        for c in "${OUT_DIR}/${state}"/*_HB_SB.tsv; do
            if is_complete "$c"; then
                inputs+=("$c")
            else
                echo "WARN  excluding incomplete ${c}"
            fi
        done
        shopt -u nullglob
        if [[ ${#inputs[@]} -eq 0 ]]; then
            echo "ERROR ${state} frequencies: no complete contact files"
            FAILED+=("${state}/frequencies")
            continue
        fi
        echo "FREQ  ${state} (${#inputs[@]} replicas)"
        if ! python getcontacts/get_contact_frequencies.py \
                --input_files "${inputs[@]}" \
                --itypes all \
                --output_file "${OUT_DIR}/${state}_HB_SB_freq.tsv"; then
            echo "FAIL  ${state} frequencies"
            FAILED+=("${state}/frequencies")
        fi
    done
fi

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Completed with ${#FAILED[@]} failure(s):"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
echo "Done. Next: python preprocess_contacts.py (see README.md, section 2)."
