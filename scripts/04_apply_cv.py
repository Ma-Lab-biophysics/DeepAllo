"""
04_apply_cv.py
──────────────
Fully self-contained script. No config.py required.

Apply a trained 2-state DeepLDA collective variable to any set of
trajectories and compare them against the reference state distributions.

HOW TO USE
──────────
1. Fill in the USER CONFIGURATION block below (~20 lines).
2. python 04_apply_cv.py

REQUIRED INPUTS (all produced by the standard pipeline)
────────────────────────────────────────────────────────
  topology.pdb                      — same topology used for training
  models/deeplda_model_full.pt      — trained model (step 02 output)
  output/pairs_resids.npy           — residue pairs used as descriptors
  output/pairs_labels.npy           — pair string labels
  output/cv_values_<state1>.npy     — reference CV₁ for state 1
  output/cv_values_<state2>.npy     — reference CV₁ for state 2

OUTPUTS  (written to OUTPUT_DIR)
─────────────────────────────────
  <label>_cv_values.npy             (n_frames_total,)
  <label>_cv_values.csv             per-frame: replica, frame, cv_1
  <label>_cv_summary.csv            per-replica statistics
  <label>_distribution_metrics.csv  Wasserstein & JSD vs each reference state

  cv_vs_time.png                    CV₁ vs frame, all replicas
  cv_per_replica.png                per-replica grid of CV₁ vs time
  cv_distribution.png               histogram vs both reference states
  fes_comparison.png                FES curves: new sim + references
  cv_violin.png                     per-replica violin vs references
  probability_vs_frame.png          per-frame P(state1) and P(state2)
                                    using a sliding-window estimate
  state_assignment.png              per-frame state label (0/1/transition)
                                    stacked bar or colour-coded strip
"""

import os
import re
import sys
import glob
import time
from functools import reduce

import numpy as np
import pandas as pd
import torch
import MDAnalysis as mda
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy.ndimage import uniform_filter1d    # for sliding-window smoothing
from scipy import stats                       # Wasserstein distance
from scipy.spatial.distance import jensenshannon  # Jensen-Shannon divergence

# ═══════════════════════════════════════════════════════════════════════════════
#  USER CONFIGURATION  ← edit everything in this block
# ═══════════════════════════════════════════════════════════════════════════════

# ── Topology -----------------------------------------------------------------
TOPOLOGY = "examples/Trajs/Apo/topology.pdb"

# ── New trajectories ---------------------------------------------------------
# Folder that contains the .xtc files to analyse.
# Every .xtc inside the folder is treated as one independent replica.
SIM_DIR = "Apo_GAMD"

# Short name used in all plot titles and output filenames.
SIM_LABEL = "GAMD_test_Apo"          # e.g. "GaMD", "REST2", "Mutant_A123V"

# Use every Nth frame (set to match STRIDE_TRAIN used during training).
SIM_STRIDE = 1

# ── Trained model ------------------------------------------------------------
MODEL_PATH = "models/deeplda_model_full.pt"

# ── Pair metadata (written by step 01) --------------------------------------
PAIRS_RESIDS_PATH = "output/pairs_resids.npy"
PAIRS_LABELS_PATH = "output/pairs_labels.npy"

# ── Reference CV distributions (written by step 02) -------------------------
# State 1 — e.g. Mava
CV_STATE1_PATH  = "output/cv_values_mava.npy"
LABEL_STATE1    = "Mava"

COLOR_STATE1    =  "#d6604d"  # red

# State 2 — e.g. Ome
CV_STATE2_PATH  = "output/cv_values_apo.npy"
LABEL_STATE2    = "Apo"
COLOR_STATE2    =  "#2166ac"   # blue

# ── Output directory ---------------------------------------------------------
# All CSV files and figures are written here.
OUTPUT_DIR = "Apo_GAMD_output"

# ── Probability-vs-frame settings -------------------------------------------
# Width of the sliding window used to estimate per-frame state probability
# (in number of frames).  Larger = smoother curve, less time resolution.
# Rule of thumb: ~5–10 % of one replica length.
PROB_WINDOW = 500

# ── Plotting -----------------------------------------------------------------
SIM_COLOR   = "#984ea3"    # purple — colour for the new simulation
TEMPERATURE = 300.0        # K — for FES in kT units
KB          = 0.008314     # kJ mol⁻¹ K⁻¹
FES_BINS    = 100

# ═══════════════════════════════════════════════════════════════════════════════
#  END OF USER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.weight":       "bold",  "font.size":          22,
    "xtick.major.width": 2,       "xtick.major.size":   7,
    "xtick.major.pad":   7,       "xtick.minor.width":  1,
    "ytick.major.width": 2,       "ytick.major.size":   7,
    "ytick.major.pad":   7,       "ytick.minor.width":  1,
    "axes.linewidth":    2.5,     "xtick.direction":    "in",
    "ytick.direction":   "in",
})

CHUNK_FRAMES = 200   # frames per COM-distance computation batch


# ─────────────────────────────────────────────────────────────────────────────
# 1. Input discovery and loading
# ─────────────────────────────────────────────────────────────────────────────

def find_trajs(directory):
    def _key(s):
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r'(\d+)', s)]
    files = sorted(glob.glob(os.path.join(directory, "*.xtc")), key=_key)
    if not files:
        raise FileNotFoundError(
            f"No .xtc files found in:\n  {directory}\n"
            "Check SIM_DIR at the top of this script."
        )
    return files


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found:\n  {path}")
    model = torch.load(path, map_location="cpu", weights_only=False)
    model.eval()
    print(f"  Model        : {path}")
    return model


def load_pairs(resids_path, labels_path):
    for p in (resids_path, labels_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Pairs file not found:\n  {p}")
    pairs_resids = np.load(resids_path)
    pair_labels  = np.load(labels_path, allow_pickle=True)
    print(f"  Pairs        : {len(pair_labels)} contact pairs")
    return pairs_resids, pair_labels


def load_reference_cv(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Reference CV not found:\n  {path}\n"
            "Run 02_train.py first."
        )
    cv = np.load(path)
    if cv.ndim == 2:
        cv = cv[:, 0]    # 3-state model output — take CV1
    print(f"  {label:10s} : {len(cv):>8,} frames  "
          f"mean={cv.mean():+.3f}  std={cv.std():.3f}")
    return cv


# ─────────────────────────────────────────────────────────────────────────────
# 2. Atom selections — reconstructed from saved pair labels
# ─────────────────────────────────────────────────────────────────────────────

# Force-field protonation/disulfide variants are renamings of the same residue
# and must not be treated as a numbering error.
RESNAME_ALIASES = {
    "HIS": {"HIS", "HIE", "HID", "HIP", "HSD", "HSE", "HSP"},
    "CYS": {"CYS", "CYX", "CYM"},
    "ASP": {"ASP", "ASH"},
    "GLU": {"GLU", "GLH"},
    "LYS": {"LYS", "LYN"},
}


def _resname_matches(expected, found):
    """True if `found` is `expected` or a protonation/disulfide variant of it."""
    if expected == found:
        return True
    for canonical, variants in RESNAME_ALIASES.items():
        if expected in variants and found in variants:
            return True
    return False


def build_selections(universe, pair_labels):
    """
    Re-create the exact same whole-residue heavy-atom AtomGroups that
    were used during training, using the saved pair label strings.
    """
    def _parse(lbl):
        p1, p2  = lbl.split("-")
        rn1, id1 = p1.split(":")
        rn2, id2 = p2.split(":")
        return (rn1, int(id1)), (rn2, int(id2))

    seen, unique_res = {}, []
    for lbl in pair_labels:
        for r in _parse(lbl):
            if r[1] not in seen:
                seen[r[1]] = len(unique_res)
                unique_res.append(r)

    res_ags = []
    for resname, resid in unique_res:
        ag = universe.select_atoms(f"resid {resid} and not type H")
        if len(ag) == 0:
            raise ValueError(
                f"Residue {resname}:{resid} has no heavy atoms in topology.\n"
                "Ensure the trajectory uses the same topology as training."
            )
        residues = ag.residues
        if len(residues) != 1:
            listed = ", ".join(f"{r.resname}:{r.resid}" for r in residues)
            raise ValueError(
                f"resid {resid} matches {len(residues)} residues in "
                f"{TOPOLOGY} ({listed}).\nDescriptor residues must be "
                "uniquely identified by resid; for multi-chain systems add a "
                "segid/chainID qualifier to this selection."
            )
        found = residues[0].resname
        if not _resname_matches(resname, found):
            raise ValueError(
                f"Residue-name mismatch at resid {resid}: the saved pair "
                f"labels say {resname}, {TOPOLOGY} has {found}.\nEnsure "
                "TOPOLOGY is the same topology (and numbering) used for "
                "training."
            )
        res_ags.append(ag)

    combined      = reduce(lambda a, b: a + b, res_ags)
    heavy_indices = combined.indices

    res_slices, cursor = [], 0
    for ag in res_ags:
        n = len(ag)
        res_slices.append((cursor, cursor + n, ag.masses.astype(np.float32)))
        cursor += n

    resid_to_idx = {r[1]: i for i, r in enumerate(unique_res)}
    pair_indices = np.array(
        [(resid_to_idx[r1[1]], resid_to_idx[r2[1]])
         for r1, r2 in [_parse(lbl) for lbl in pair_labels]],
        dtype=np.int32,
    )

    print(f"  Unique residues   : {len(unique_res)}")
    print(f"  Total heavy atoms : {len(heavy_indices)}")
    return heavy_indices, res_slices, pair_indices


# ─────────────────────────────────────────────────────────────────────────────
# 3. Descriptor computation + CV projection
# ─────────────────────────────────────────────────────────────────────────────

def _coms(coords_chunk, res_slices):
    n  = coords_chunk.shape[0]
    c  = np.empty((n, len(res_slices), 3), dtype=np.float32)
    for k, (s, e, m) in enumerate(res_slices):
        c[:, k, :] = np.einsum("a,fad->fd", m, coords_chunk[:, s:e, :]) / m.sum()
    return c


def project_one_traj(traj_path, stride, heavy_indices,
                     res_slices, pair_indices, model):
    """Load one trajectory; compute descriptors in batches; return CV₁."""
    u  = mda.Universe(TOPOLOGY, traj_path)
    ag = u.atoms[heavy_indices]
    try:
        coords = u.trajectory.timeseries(
            atomgroup=ag, step=stride, order="fac"
        ).astype(np.float32)
    except TypeError:
        coords = u.trajectory.timeseries(
            asel=ag, step=stride, order="fac"
        ).astype(np.float32)

    n_frames = len(coords)
    i0, i1  = pair_indices[:, 0], pair_indices[:, 1]
    cv_vals  = np.empty(n_frames, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n_frames, CHUNK_FRAMES):
            end   = min(start + CHUNK_FRAMES, n_frames)
            coms  = _coms(coords[start:end], res_slices)
            diff  = coms[:, i0, :] - coms[:, i1, :]
            dists = torch.from_numpy(
                np.linalg.norm(diff, axis=2).astype(np.float32)
            )
            out = model(dists).squeeze().cpu().numpy()
            if out.ndim == 0: out = out.reshape(1)
            if out.ndim == 2: out = out[:, 0]
            cv_vals[start:end] = out

    del coords
    return cv_vals


def project_all(traj_list, heavy_indices, res_slices,
                pair_indices, model, stride):
    all_cv, boundaries, cursor = [], [], 0
    for rep_i, path in enumerate(traj_list):
        t0 = time.time()
        print(f"  Rep {rep_i:>2}  {os.path.basename(path)}", flush=True)
        cv = project_one_traj(path, stride, heavy_indices,
                               res_slices, pair_indices, model)
        print(f"         {len(cv):>7,} frames  "
              f"mean={cv.mean():+.3f}  std={cv.std():.3f}  "
              f"({time.time()-t0:.1f}s)")
        all_cv.append(cv)
        boundaries.append([cursor, cursor + len(cv)])
        cursor += len(cv)
    return np.concatenate(all_cv), np.array(boundaries, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# 4. State probability calculation
# ─────────────────────────────────────────────────────────────────────────────

def assign_frame_probabilities(cv_sim, cv_ref1, cv_ref2, window):
    """
    Per-frame state probability estimated two ways:

    (a) Hard assignment  — each frame is assigned to the nearer reference
        mean. Binary: 1 = state1, 0 = state2.

    (b) Soft probability — Gaussian kernel probability:
            P(state1 | s) = N(s; mu1, sigma1) / [N(s; mu1, s1) + N(s; mu2, s2)]
        smoothed with a sliding window of width `window` frames.

    Returns
    -------
    hard_assign  : int8 array  (n_frames,)  — 1=state1, 0=state2
    p_state1_raw : float32     (n_frames,)  — unsmoothed soft P(state1)
    p_state1_sm  : float32     (n_frames,)  — window-smoothed P(state1)
    """
    mu1, s1 = cv_ref1.mean(), cv_ref1.std()
    mu2, s2 = cv_ref2.mean(), cv_ref2.std()

    # Soft probability via Gaussian density ratio
    def _gauss(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / sigma

    g1 = _gauss(cv_sim, mu1, s1)
    g2 = _gauss(cv_sim, mu2, s2)
    p1_raw = g1 / (g1 + g2 + 1e-30)

    # Sliding-window smoothing
    w      = max(1, window)
    p1_sm  = uniform_filter1d(p1_raw.astype(np.float64), size=w,
                               mode="nearest").astype(np.float32)

    # Hard assignment
    hard = (np.abs(cv_sim - mu1) < np.abs(cv_sim - mu2)).astype(np.int8)

    return hard, p1_raw.astype(np.float32), p1_sm


# ─────────────────────────────────────────────────────────────────────────────
# 5. CSV output
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(cv_all, hard_assign, p1_raw, p1_sm, boundaries, traj_list):
    n = len(cv_all)
    rep   = np.full(n, -1, dtype=np.int32)
    local = np.full(n, -1, dtype=np.int32)
    fname = np.full(n, "",  dtype=object)
    for rep_i, (s, e) in enumerate(boundaries):
        rep[s:e]   = rep_i
        local[s:e] = np.arange(e - s)
        fname[s:e] = os.path.basename(traj_list[rep_i])

    df = pd.DataFrame({
        "frame_global":         np.arange(n),
        "replica":              rep,
        "trajectory":           fname,
        "frame_within_replica": local,
        "cv_1":                 cv_all,
        "hard_state":           np.where(hard_assign == 1,
                                         LABEL_STATE1, LABEL_STATE2),
        f"P_{LABEL_STATE1}":    p1_raw,
        f"P_{LABEL_STATE1}_smoothed": p1_sm,
        f"P_{LABEL_STATE2}_smoothed": 1 - p1_sm,
    })
    path = os.path.join(OUTPUT_DIR, f"{SIM_LABEL}_cv_values.csv")
    df.to_csv(path, index=False)
    print(f"  {SIM_LABEL}_cv_values.csv saved")

    # Per-replica summary
    rows = []
    for rep_i, (s, e) in enumerate(boundaries):
        v  = cv_all[s:e]
        h  = hard_assign[s:e]
        rows.append({
            "replica":            rep_i,
            "trajectory":         os.path.basename(traj_list[rep_i]),
            "n_frames":           e - s,
            "cv1_mean":           round(float(v.mean()), 4),
            "cv1_std":            round(float(v.std()),  4),
            "cv1_min":            round(float(v.min()),  4),
            "cv1_max":            round(float(v.max()),  4),
            f"pct_{LABEL_STATE1}": round(100 * h.mean(), 1),
            f"pct_{LABEL_STATE2}": round(100 * (1 - h).mean(), 1),
        })
    df_sum = pd.DataFrame(rows)
    spath  = os.path.join(OUTPUT_DIR, f"{SIM_LABEL}_cv_summary.csv")
    df_sum.to_csv(spath, index=False)
    print(f"  {SIM_LABEL}_cv_summary.csv saved")
    print()
    print(df_sum.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 6. Distribution similarity metrics
# ─────────────────────────────────────────────────────────────────────────────

def _to_pdf(data, bins=300):
    """Shared-range histogram → normalised probability vector (no zeros)."""
    all_data  = np.concatenate([data]) if data.ndim == 1 else data
    lo = min(d.min() for d in [data]) - 0.5
    hi = max(d.max() for d in [data]) + 0.5
    counts, _ = np.histogram(data, bins=bins, range=(lo, hi), density=False)
    counts     = counts.astype(np.float64) + 1e-10   # Laplace smoothing
    return counts / counts.sum()


def compute_distribution_metrics(cv_sim, cv_ref1, cv_ref2):
    """
    Compute Wasserstein distance and Jensen-Shannon divergence between
    the new simulation and each reference state.

    Wasserstein (Earth Mover's Distance)
    ─────────────────────────────────────
    • Measures the minimum "work" needed to reshape one distribution into
      another, accounting for BOTH shift in mean AND spread differences.
    • 0 = identical distributions.  Units are the same as CV₁.

    Jensen-Shannon Divergence (JSD)
    ────────────────────────────────
    • Symmetric, smoothed version of KL divergence.
    • Bounded [0, 1] (base-2 bits).  0 = identical, 1 = no overlap.
    • Penalises wrong location AND wrong width simultaneously.

    Returns
    -------
    dict with keys: w_state1, w_state2, jsd_state1, jsd_state2
    """
    # ── Wasserstein (uses raw sample arrays — no binning needed) ──────────────
    w1 = stats.wasserstein_distance(cv_sim, cv_ref1)
    w2 = stats.wasserstein_distance(cv_sim, cv_ref2)

    # ── JSD (requires probability vectors over shared bins) ───────────────────
    # Build a common bin range covering all three distributions
    lo   = min(cv_sim.min(), cv_ref1.min(), cv_ref2.min()) - 0.5
    hi   = max(cv_sim.max(), cv_ref1.max(), cv_ref2.max()) + 0.5
    bins = 300

    def _pdf(d):
        c, _ = np.histogram(d, bins=bins, range=(lo, hi), density=False)
        c    = c.astype(np.float64) + 1e-10
        return c / c.sum()

    p_sim  = _pdf(cv_sim)
    p_ref1 = _pdf(cv_ref1)
    p_ref2 = _pdf(cv_ref2)

    jsd1 = float(jensenshannon(p_sim, p_ref1, base=2))
    jsd2 = float(jensenshannon(p_sim, p_ref2, base=2))

    return dict(w_state1=w1, w_state2=w2, jsd_state1=jsd1, jsd_state2=jsd2)


def print_and_save_metrics(metrics, save_path):
    """Pretty-print metrics table and write to CSV."""
    w1, w2     = metrics["w_state1"],   metrics["w_state2"]
    jsd1, jsd2 = metrics["jsd_state1"], metrics["jsd_state2"]

    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │          Distribution Similarity Metrics            │")
    print("  ├──────────────────────────┬──────────────┬───────────┤")
    print(f"  │ Metric                   │ vs {LABEL_STATE1:<9s} │ vs {LABEL_STATE2:<6s} │")
    print("  ├──────────────────────────┼──────────────┼───────────┤")
    print(f"  │ Wasserstein distance     │ {w1:>12.4f} │ {w2:>9.4f} │")
    print(f"  │ Jensen-Shannon div (bit) │ {jsd1:>12.4f} │ {jsd2:>9.4f} │")
    print("  └──────────────────────────┴──────────────┴───────────┘")
    print("  Interpretation: lower = more similar to that reference state")
    print()

    df = pd.DataFrame([{
        "sim_label":                SIM_LABEL,
        f"wasserstein_vs_{LABEL_STATE1}": round(w1,   4),
        f"wasserstein_vs_{LABEL_STATE2}": round(w2,   4),
        f"jsd_vs_{LABEL_STATE1}":         round(jsd1, 4),
        f"jsd_vs_{LABEL_STATE2}":         round(jsd2, 4),
        "closer_to_state (Wasserstein)":  LABEL_STATE1 if w1 < w2 else LABEL_STATE2,
        "closer_to_state (JSD)":          LABEL_STATE1 if jsd1 < jsd2 else LABEL_STATE2,
    }])
    df.to_csv(save_path, index=False)
    print(f"  Saved : {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Figures
# ─────────────────────────────────────────────────────────────────────────────

def _fes(vals, n_bins=FES_BINS):
    mu, sig   = vals.mean(), vals.std()
    edges     = np.linspace(mu - 4*sig, mu + 4*sig, n_bins + 1)
    cnts, _   = np.histogram(vals, bins=edges, density=True)
    ctrs      = 0.5 * (edges[:-1] + edges[1:])
    cnts      = np.where(cnts > 0, cnts, np.nan)
    fes       = -np.log(cnts)
    fes      -= np.nanmin(fes)
    return ctrs, fes


def plot_cv_vs_time(cv_all, boundaries, save_path):
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    cmap    = matplotlib.colormaps["tab20"].resampled(len(boundaries))
    for rep_i, (s, e) in enumerate(boundaries):
        ax.plot(np.arange(e - s) + s, cv_all[s:e],
                color=cmap(rep_i), alpha=0.80, lw=1.0, label=f"Rep {rep_i}")
    ax.set_xlabel("Frame",  fontweight="bold")
    ax.set_ylabel("CV₁",    fontweight="bold")
    ax.set_title(f"{SIM_LABEL} — CV₁ vs time", fontweight="bold")
    n_cols = max(1, len(boundaries) // 8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0, frameon=True, edgecolor="black",
              fancybox=False, ncol=n_cols, handlelength=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved : {save_path}")


def plot_cv_per_replica(cv_all, boundaries, save_path):
    n_reps = len(boundaries)
    n_cols = min(4, n_reps)
    n_rows = (n_reps + n_cols - 1) // n_cols
    cmap   = matplotlib.colormaps["tab20"].resampled(n_reps)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5*n_cols, 4*n_rows), dpi=100)
    flat = np.array(axes).flatten() if n_reps > 1 else np.array([axes])
    for rep_i, (s, e) in enumerate(boundaries):
        ax = flat[rep_i]
        ax.plot(np.arange(e - s), cv_all[s:e], color=cmap(rep_i), lw=0.8)
        ax.set_title(f"replica {rep_i}", fontweight="bold", fontsize=13)
        ax.set_xlabel("frame", fontsize=11); ax.set_ylabel("CV₁", fontsize=11)
        ax.set_ylim(-2, 2)
    for ax in list(flat)[n_reps:]:
        ax.set_visible(False)
    fig.suptitle(f"{SIM_LABEL} — CV₁ per replica", fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved : {save_path}")


def plot_cv_distribution(cv_sim, cv_ref1, cv_ref2, save_path):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    for vals, label, color in [
        (cv_ref1, LABEL_STATE1, COLOR_STATE1),
        (cv_ref2, LABEL_STATE2, COLOR_STATE2),
    ]:
        mu, sig = vals.mean(), vals.std()
        ax.hist(vals, bins=np.linspace(mu-4*sig, mu+4*sig, 80),
                density=True, alpha=0.30, color=color,
                edgecolor="none", label=label)
    mu, sig = cv_sim.mean(), cv_sim.std()
    ax.hist(cv_sim, bins=np.linspace(mu-4*sig, mu+4*sig, 80),
            density=True, alpha=0.70, color=SIM_COLOR,
            edgecolor=SIM_COLOR, linewidth=1.5,
            label=SIM_LABEL, histtype="stepfilled")
    ax.set_xlabel("CV₁", fontweight="bold")
    ax.set_ylabel("Probability density", fontweight="bold")
    ax.set_title(f"{SIM_LABEL} vs reference states", fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0, frameon=True, edgecolor="black", fancybox=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved : {save_path}")


def plot_fes_comparison(cv_sim, cv_ref1, cv_ref2, save_path):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    for vals, label, color, lw, ls in [
        (cv_ref1, LABEL_STATE1, COLOR_STATE1, 2.0, "--"),
        (cv_ref2, LABEL_STATE2, COLOR_STATE2, 2.0, "--"),
        (cv_sim,  SIM_LABEL,    SIM_COLOR,    2.5, "-"),
    ]:
        ctrs, fes = _fes(vals)
        ax.plot(ctrs, fes, color=color, lw=lw, ls=ls, label=label)
    ax.set_xlabel("CV₁", fontweight="bold")
    ax.set_ylabel("F (kT)", fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.set_title(f"FES comparison  (T = {TEMPERATURE} K)", fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0, frameon=True, edgecolor="black", fancybox=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved : {save_path}")


def plot_violin_per_replica(cv_all, boundaries, cv_ref1, cv_ref2, save_path):
    ref_data   = [cv_ref1, cv_ref2]
    ref_labels = [LABEL_STATE1, LABEL_STATE2]
    ref_colors = [COLOR_STATE1, COLOR_STATE2]
    rep_data   = [cv_all[s:e] for s, e in boundaries]
    rep_labels = [f"Rep{i}" for i in range(len(boundaries))]
    all_data   = ref_data   + rep_data
    all_labels = ref_labels + rep_labels
    all_colors = ref_colors + [SIM_COLOR] * len(rep_data)

    fig, ax = plt.subplots(figsize=(max(12, len(all_data)*0.7), 7), dpi=120)
    parts = ax.violinplot(all_data, positions=range(len(all_data)),
                          showmedians=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(all_colors[i]); pc.set_alpha(0.65)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)
    ax.set_xticks(range(len(all_data)))
    ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=12)
    ax.set_ylabel("CV₁", fontweight="bold")
    ax.set_title(f"{SIM_LABEL} replicas vs reference states", fontweight="bold")
    ax.legend(handles=[
        mpatches.Patch(color=COLOR_STATE1, alpha=0.65, label=LABEL_STATE1),
        mpatches.Patch(color=COLOR_STATE2, alpha=0.65, label=LABEL_STATE2),
        mpatches.Patch(color=SIM_COLOR,    alpha=0.65, label=SIM_LABEL),
    ], loc="best", frameon=True, edgecolor="black", fancybox=False, fontsize=16)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved : {save_path}")


def plot_probability_vs_frame(p1_sm, boundaries, save_path):
    """
    Per-frame state probability for every replica.
    Top sub-panel: P(state1) and P(state2) lines — one panel per replica.
    Each replica gets its own row so the time axis is clear.

    The smoothed Gaussian-kernel probability is plotted so transitions
    appear as smooth crossings of the 0.5 line.
    """
    n_reps = len(boundaries)
    fig, axes = plt.subplots(
        n_reps, 1,
        figsize=(14, max(3, 2.5 * n_reps)),
        dpi=120,
        sharex=False,
        squeeze=False,
    )

    for rep_i, (s, e) in enumerate(boundaries):
        ax   = axes[rep_i, 0]
        x    = np.arange(e - s)
        p1   = p1_sm[s:e]
        p2   = 1.0 - p1

        ax.fill_between(x, p1, alpha=0.35, color=COLOR_STATE1, linewidth=0)
        ax.fill_between(x, p2, alpha=0.35, color=COLOR_STATE2, linewidth=0)
        ax.plot(x, p1, color=COLOR_STATE1, lw=1.5, label=LABEL_STATE1)
        ax.plot(x, p2, color=COLOR_STATE2, lw=1.5, label=LABEL_STATE2)
        ax.axhline(0.5, color="black", lw=1.0, ls="--", alpha=0.5)

        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("P(state)", fontsize=12, fontweight="bold")
        ax.set_title(f"Replica {rep_i}", fontsize=13, fontweight="bold")
        ax.tick_params(labelsize=11)

        if rep_i == 0:
            ax.legend(loc="upper right", fontsize=13,
                      frameon=True, edgecolor="black", fancybox=False)

    axes[-1, 0].set_xlabel("Frame", fontsize=14, fontweight="bold")
    fig.suptitle(
        f"{SIM_LABEL} — State probability vs frame\n"
        f"(Gaussian-kernel estimate, window = {PROB_WINDOW} frames)",
        fontweight="bold", fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved : {save_path}")


def plot_state_assignment(cv_all, hard_assign, boundaries, save_path):
    """
    Colour-coded strip plot: one horizontal strip per replica, each
    frame coloured by its hard state assignment (blue = state1,
    red = state2).  Immediately shows which replicas stay in one basin
    and which ones transition.
    """
    n_reps = len(boundaries)
    cmap   = ListedColormap([COLOR_STATE1, COLOR_STATE2])

    fig, axes = plt.subplots(
        n_reps, 1,
        figsize=(14, max(3, 1.4 * n_reps)),
        dpi=120,
        squeeze=False,
    )

    for rep_i, (s, e) in enumerate(boundaries):
        ax   = axes[rep_i, 0]
        data = hard_assign[s:e].reshape(1, -1)   # (1, n_frames)
        ax.imshow(data, aspect="auto", cmap=cmap,
                  vmin=0, vmax=1, interpolation="none",
                  extent=[0, e - s, 0, 1])
        ax.set_yticks([])
        ax.set_ylabel(f"Rep {rep_i}", fontsize=11,
                      fontweight="bold", rotation=0,
                      labelpad=35, va="center")
        if rep_i < n_reps - 1:
            ax.set_xticks([])

    axes[-1, 0].set_xlabel("Frame", fontsize=14, fontweight="bold")
    fig.suptitle(
        f"{SIM_LABEL} — per-frame state assignment\n"
        f"(blue = {LABEL_STATE1}, red = {LABEL_STATE2})",
        fontweight="bold", fontsize=16,
    )
    legend_handles = [
        mpatches.Patch(color=COLOR_STATE1, label=LABEL_STATE1),
        mpatches.Patch(color=COLOR_STATE2, label=LABEL_STATE2),
    ]
    axes[0, 0].legend(
        handles=legend_handles,
        loc="upper right", fontsize=13,
        frameon=True, edgecolor="black", fancybox=False,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved : {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print(f"  Apply Trained DeepLDA CV  [{SIM_LABEL}]")
    print("=" * 65)
    print(f"  Topology   : {TOPOLOGY}")
    print(f"  Sim dir    : {SIM_DIR}")
    print(f"  Label      : {SIM_LABEL}")
    print(f"  Stride     : every {SIM_STRIDE} frame(s)")
    print(f"  Prob window: {PROB_WINDOW} frames")
    print(f"  Output     : {OUTPUT_DIR}")

    # 1. Trajectories
    print("\n[1/8] Locating trajectory files …")
    traj_list = find_trajs(SIM_DIR)
    print(f"  Found {len(traj_list)} file(s):")
    for i, p in enumerate(traj_list):
        print(f"    [{i}] {p}")

    # 2. Model
    print("\n[2/8] Loading trained model …")
    model = load_model(MODEL_PATH)

    # 3. Pair metadata + reference CVs
    print("\n[3/8] Loading pair metadata and reference CV distributions …")
    pairs_resids, pair_labels = load_pairs(PAIRS_RESIDS_PATH, PAIRS_LABELS_PATH)
    cv_ref1 = load_reference_cv(CV_STATE1_PATH, LABEL_STATE1)
    cv_ref2 = load_reference_cv(CV_STATE2_PATH, LABEL_STATE2)

    # 4. Atom selections
    print("\n[4/8] Reconstructing atom selections …")
    ref_u = mda.Universe(TOPOLOGY)
    heavy_indices, res_slices, pair_indices = build_selections(
        ref_u, pair_labels
    )

    # 5. Project
    print(f"\n[5/8] Projecting {len(traj_list)} trajectory file(s) onto CV₁ …")
    t0 = time.time()
    cv_sim, boundaries = project_all(
        traj_list, heavy_indices, res_slices,
        pair_indices, model, SIM_STRIDE,
    )
    print(f"\n  Total frames : {len(cv_sim):,}  "
          f"({(time.time()-t0)/60:.1f} min)")
    print(f"  CV₁  mean={cv_sim.mean():+.4f}  std={cv_sim.std():.4f}  "
          f"min={cv_sim.min():.4f}  max={cv_sim.max():.4f}")

    mu1, mu2 = cv_ref1.mean(), cv_ref2.mean()
    n1 = int((np.abs(cv_sim - mu1) < np.abs(cv_sim - mu2)).sum())
    n2 = len(cv_sim) - n1
    print(f"\n  Hard classification (nearest reference mean):")
    print(f"    {LABEL_STATE1:10s} : {n1:>8,}  ({100*n1/len(cv_sim):.1f}%)")
    print(f"    {LABEL_STATE2:10s} : {n2:>8,}  ({100*n2/len(cv_sim):.1f}%)")

    # 6. Distribution similarity metrics
    print(f"\n[6/8] Computing distribution similarity metrics …")
    metrics = compute_distribution_metrics(cv_sim, cv_ref1, cv_ref2)
    print_and_save_metrics(
        metrics,
        os.path.join(OUTPUT_DIR, f"{SIM_LABEL}_distribution_metrics.csv")
    )

    # 7. Probability
    print(f"\n[7/8] Computing per-frame state probabilities …")
    hard_assign, p1_raw, p1_sm = assign_frame_probabilities(
        cv_sim, cv_ref1, cv_ref2, PROB_WINDOW
    )
    print(f"  Mean P({LABEL_STATE1}) over all frames : {p1_sm.mean():.3f}")
    print(f"  Mean P({LABEL_STATE2}) over all frames : {(1-p1_sm).mean():.3f}")

    # 8. Save + plot
    print(f"\n[8/8] Saving outputs and generating figures …")
    np.save(os.path.join(OUTPUT_DIR, f"{SIM_LABEL}_cv_values.npy"),       cv_sim)
    np.save(os.path.join(OUTPUT_DIR, f"{SIM_LABEL}_replica_boundaries.npy"), boundaries)
    np.save(os.path.join(OUTPUT_DIR, f"{SIM_LABEL}_p_state1.npy"),        p1_sm)

    save_csv(cv_sim, hard_assign, p1_raw, p1_sm, boundaries, traj_list)

    plot_cv_vs_time(
        cv_sim, boundaries,
        os.path.join(OUTPUT_DIR, "cv_vs_time.png"))
    plot_cv_per_replica(
        cv_sim, boundaries,
        os.path.join(OUTPUT_DIR, "cv_per_replica.png"))
    plot_cv_distribution(
        cv_sim, cv_ref1, cv_ref2,
        os.path.join(OUTPUT_DIR, "cv_distribution.png"))
    plot_fes_comparison(
        cv_sim, cv_ref1, cv_ref2,
        os.path.join(OUTPUT_DIR, "fes_comparison.png"))
    plot_violin_per_replica(
        cv_sim, boundaries, cv_ref1, cv_ref2,
        os.path.join(OUTPUT_DIR, "cv_violin.png"))
    plot_probability_vs_frame(
        p1_sm, boundaries,
        os.path.join(OUTPUT_DIR, "probability_vs_frame.png"))
    plot_state_assignment(
        cv_sim, hard_assign, boundaries,
        os.path.join(OUTPUT_DIR, "state_assignment.png"))

    print(f"\nDONE.  All outputs in:\n  {OUTPUT_DIR}")
    print()
    print("  Interpreting the results:")
    print(f"    cv_distribution.png     — where does {SIM_LABEL} sit relative to {LABEL_STATE1}/{LABEL_STATE2}?")
    print(f"    fes_comparison.png      — free energy landscape vs reference states")
    print(f"    probability_vs_frame.png — per-frame P({LABEL_STATE1}) and P({LABEL_STATE2}) per replica")
    print(f"    state_assignment.png    — colour-coded frame-by-frame state map")
    print(f"    cv_violin.png           — which replica sampled which basin?")


if __name__ == "__main__":
    main()
