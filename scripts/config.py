"""
config.py  —  Apo_vs_Mava
─────────────────────────
Central configuration for the DeepLDA pipeline.
Edit only this file to adapt the pipeline to a new system.
"""

import glob as _glob
import os
import re as _re

# config.py lives in scripts/, so the project root is one level up. All data
# and output paths below are resolved relative to that root, not to scripts/.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ═════════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION — edit this block to select the two input states
# ═════════════════════════════════════════════════════════════════════════════

# Bundled strided demo trajectories:
TRAJS_DIR = os.path.join(ROOT_DIR, "examples", "Trajs")

STATE1_FOLDER = "Apo"
STATE2_FOLDER = "Mava"

# Labels drive output filenames, plot labels, and printed summaries.
STATE1_LABEL = "Apo"
STATE2_LABEL = "Mava"

CONTACTS_XLSX = os.path.join(
    ROOT_DIR, "descriptors", "Apo_vs_Mava_contacts.xlsx"
)

STRIDE_TRAIN = 1

# DeepLDA evaluates a symmetric eigendecomposition during training.
# Use "cuda" for an NVIDIA GPU. Apple MPS does not implement the required
# torch.linalg.eigh operation, so use "cpu" on Apple Silicon.
TRAIN_ACCELERATOR = "cpu"
TRAIN_DEVICES = 1

# ═════════════════════════════════════════════════════════════════════════════
# DERIVED PATHS — normally do not edit below this line
# ═════════════════════════════════════════════════════════════════════════════

STATE1_DIR = os.path.join(TRAJS_DIR, STATE1_FOLDER)
STATE2_DIR = os.path.join(TRAJS_DIR, STATE2_FOLDER)

# Each state is read with the topology stored in its own trajectory directory.
STATE1_TOPOLOGY = os.path.join(STATE1_DIR, "topology.pdb")
STATE2_TOPOLOGY = os.path.join(STATE2_DIR, "topology.pdb")


def _natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in _re.split(r"(\d+)", s)]

def _find_trajs(directory):
    files = _glob.glob(os.path.join(directory, "*.xtc"))
    if not files:
        raise FileNotFoundError(
            f"No .xtc files found in {directory!r}. "
            "Check TRAJS_DIR / STATE1_FOLDER / STATE2_FOLDER in config.py."
        )
    return sorted(files, key=_natural_sort_key)

STATE1_TRAJS = _find_trajs(STATE1_DIR)
STATE2_TRAJS = _find_trajs(STATE2_DIR)

N_REPLICAS_STATE1 = len(STATE1_TRAJS)
N_REPLICAS_STATE2 = len(STATE2_TRAJS)

# Compatibility aliases used by the existing training and analysis scripts.
APO_DIR = STATE1_DIR
MAVA_DIR = STATE2_DIR
APO_TRAJS = STATE1_TRAJS
MAVA_TRAJS = STATE2_TRAJS
N_REPLICAS_APO = N_REPLICAS_STATE1
N_REPLICAS_MAVA = N_REPLICAS_STATE2
LABEL_APO = STATE1_LABEL
LABEL_MAVA = STATE2_LABEL

OUT_DIR     = os.path.join(ROOT_DIR, "output")
MODELS_DIR  = os.path.join(ROOT_DIR, "models")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")

for _d in (OUT_DIR, MODELS_DIR, FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Trajectory / descriptor settings
# ─────────────────────────────────────────────────────────────────────────────

MIN_SEQ_SEP    = 5
BACKBONE_NAMES = {"N", "CA", "C", "O", "OXT"}

# ─────────────────────────────────────────────────────────────────────────────
# DeepLDA architecture
# ─────────────────────────────────────────────────────────────────────────────

NN_HIDDEN     = [64, 64, 32]
NN_ACTIVATION = "relu"

N_STATES = 2

# ─────────────────────────────────────────────────────────────────────────────
# Training hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_FRAC           = 0.8
VALID_FRAC           = 0.2
REPLICA_AWARE_SPLIT  = False
VALID_REPLICA_FRAC   = 0.2
BATCH_SIZE           = 0
MAX_EPOCHS           = None
EARLY_STOPPING_PATIENCE  = 50
EARLY_STOPPING_MIN_DELTA = 1e-3
SEED                 = 42

# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

FES_BINS    = 100
TEMPERATURE = 300.0
KB          = 0.008314
