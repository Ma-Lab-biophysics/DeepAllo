"""
config.py  —  Apo_vs_Mava
─────────────────────────
Central configuration for the DeepLDA pipeline.
Edit only this file to adapt the pipeline to a new system.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

# config.py lives in scripts/, so the project root is one level up. All data
# and output paths below are resolved relative to that root, not to scripts/.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAJS_DIR = os.path.join(ROOT_DIR, "Trajs")

# The topology files distributed with Apo, Mava, and Ome are identical. Use
# the Apo copy as the shared topology for descriptor extraction.
TOPOLOGY = os.path.join(TRAJS_DIR, "Apo", "topology.pdb")

APO_DIR = os.path.join(TRAJS_DIR, "Apo")       # first state
MAVA_DIR = os.path.join(TRAJS_DIR, "Mava")     # second state

import re as _re
import glob as _glob

def _natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in _re.split(r"(\d+)", s)]

def _find_trajs(directory):
    files = _glob.glob(os.path.join(directory, "*.xtc"))
    if not files:
        raise FileNotFoundError(
            f"No .xtc files found in {directory!r}. "
            "Check APO_DIR / MAVA_DIR in config.py."
        )
    return sorted(files, key=_natural_sort_key)

APO_TRAJS = _find_trajs(APO_DIR)
MAVA_TRAJS  = _find_trajs(MAVA_DIR)

N_REPLICAS_APO = len(APO_TRAJS)
N_REPLICAS_MAVA  = len(MAVA_TRAJS)

# State labels — drive ALL output filenames, plot labels, and print statements.
LABEL_APO = "Apo"
LABEL_MAVA  = "Mava"

CONTACTS_XLSX = os.path.join(ROOT_DIR, "Apo_vs_Mava_contacts.xlsx")

OUT_DIR     = os.path.join(ROOT_DIR, "output")
MODELS_DIR  = os.path.join(ROOT_DIR, "models")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")

for _d in (OUT_DIR, MODELS_DIR, FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Trajectory / descriptor settings
# ─────────────────────────────────────────────────────────────────────────────

STRIDE_TRAIN   = 1
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

# DeepLDA evaluates a symmetric eigendecomposition during training.
# torch.linalg.eigh is not implemented by PyTorch's Apple MPS backend, so use
# CPU explicitly on macOS and elsewhere for a portable, reproducible run.
TRAIN_ACCELERATOR = "cpu"
TRAIN_DEVICES = 1

# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

FES_BINS    = 100
TEMPERATURE = 300.0
KB          = 0.008314
