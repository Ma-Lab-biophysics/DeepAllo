#!/usr/bin/env python3
"""Post-training analysis for an EXISTING DeepLDA model.

This script does not train or modify the model. It is designed for the
Apo/Mava analysis layout:

    config.py
    output/descriptors_apo.npy
    output/descriptors_mava.npy
    output/replica_boundaries_apo.npy
    output/replica_boundaries_mava.npy
    output/pairs_labels.npy
    models/deeplda_model_full.pt  (serialized full model), or
    deeplda_model.pt              (state dictionary, supplied with the archive)

It reconstructs the original train/validation split from config.py, computes
training-set sigma, projects the existing model, calculates sigma-scaled and
sum-normalized sensitivity, calculates descriptive Cohen's d, and generates
FES/convergence/sensitivity plots.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mlcolvar.cvs import DeepLDA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (  # noqa: E402
    OUT_DIR, MODELS_DIR, FIGURES_DIR,
    LABEL_APO, LABEL_MAVA,
    TRAIN_FRAC, REPLICA_AWARE_SPLIT, VALID_REPLICA_FRAC, SEED,
    FES_BINS, NN_HIDDEN, NN_ACTIVATION, N_STATES,
    FEATURES_FILE, CRYSTAL_NUMBERING_OFFSET,
)
from contact_features import load_contact_features  # noqa: E402

OUT_DIR = Path(OUT_DIR)
MODELS_DIR = Path(MODELS_DIR)
FIGURES_DIR = Path(FIGURES_DIR)
for directory in (OUT_DIR, MODELS_DIR, FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

TOP_N_FEATURES = 20
SENSITIVITY_BATCH_SIZE = 4096
PROJECTION_BATCH_SIZE = 8192
EPS = 1.0e-8

# Time represented by one descriptor/CV frame.
# Keep at 1.0 when trajectories were saved/strided at 1 ns per frame.
TIME_PER_FRAME_NS = 1.0

plt.rcParams.update({
    "font.weight": "bold",
    "font.size": 15,
    "axes.linewidth": 2.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
})


def _trusted_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _required(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def load_crystal_pair_labels(feature_path: Path, pair_labels, offset: int):
    """Verify DAT feature order and shift residue numbers for display."""
    feature_path = _required(Path(feature_path))
    contacts = load_contact_features(feature_path)

    def combine(pairs):
        return np.asarray([f"{left}-{right}" for left, right in pairs])

    file_sim_labels = combine(contacts.simulation_pairs)
    pair_labels = np.asarray(pair_labels, dtype=str)
    if len(file_sim_labels) != len(pair_labels):
        raise ValueError(
            "Contact count mismatch between pairs_labels.npy "
            f"({len(pair_labels)}) and {feature_path} "
            f"({len(file_sim_labels)})."
        )
    mismatches = np.flatnonzero(file_sim_labels != pair_labels)
    if len(mismatches):
        index = int(mismatches[0])
        raise ValueError(
            "Contact order mismatch between pairs_labels.npy and the feature file "
            f"at feature {index}: {pair_labels[index]!r} versus "
            f"{file_sim_labels[index]!r}."
        )

    def shift(label):
        residue_name, residue_number = label.rsplit(":", 1)
        return f"{residue_name}:{int(residue_number) + offset}"

    crystal_pairs = [
        (shift(first), shift(second))
        for first, second in contacts.simulation_pairs
    ]
    return combine(crystal_pairs)


def resolve_model_path(requested: Path | None) -> Path:
    """Resolve an explicit model or the first supported local default."""
    if requested is not None:
        return _required(requested.expanduser().resolve())

    candidates = [
        MODELS_DIR / "deeplda_model_full.pt",
        MODELS_DIR / "deeplda_model.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    choices = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "No existing DeepLDA model was found. Pass --model PATH or place one "
        f"of these files at:\n  {choices}"
    )


def load_existing_model(path: Path, n_features: int):
    """Load either a serialized DeepLDA model or a DeepLDA state dictionary."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        payload = _trusted_torch_load(path)

    if isinstance(payload, torch.nn.Module):
        model = payload
        source_type = "serialized full model"
    elif isinstance(payload, Mapping):
        model = DeepLDA(
            [n_features, *NN_HIDDEN],
            n_states=N_STATES,
            options={"nn": {"activation": NN_ACTIVATION}},
        )
        model.load_state_dict(payload, strict=True)
        source_type = "state dictionary"
    else:
        raise TypeError(
            f"Unsupported model payload in {path}: {type(payload).__name__}. "
            "Expected a torch.nn.Module or state dictionary."
        )

    model.eval().cpu()
    with torch.no_grad():
        probe = model(torch.zeros((1, n_features), dtype=torch.float32))
    if probe.numel() < 1:
        raise RuntimeError(f"Loaded model from {path} produced an empty output.")
    return model, source_type


def replica_aware_split(bounds_apo, bounds_mava, n_apo, valid_frac, rng):
    """Reproduce the split function used by the original training script."""
    def split_one(bounds, offset):
        n_reps = len(bounds)
        n_val_reps = max(1, round(n_reps * valid_frac))
        val_set = set(np.linspace(0, n_reps - 1, n_val_reps, dtype=int).tolist())
        train_parts, valid_parts = [], []
        for rep_i, (start, end) in enumerate(bounds):
            frames = np.arange(start, end, dtype=np.int64) + offset
            (valid_parts if rep_i in val_set else train_parts).append(frames)
        return np.concatenate(train_parts), np.concatenate(valid_parts)

    tr_apo, va_apo = split_one(bounds_apo, 0)
    tr_mava, va_mava = split_one(bounds_mava, n_apo)
    train_idx = np.concatenate([tr_apo, tr_mava])
    valid_idx = np.concatenate([va_apo, va_mava])
    rng.shuffle(train_idx)
    rng.shuffle(valid_idx)
    return train_idx, valid_idx


def reconstruct_split(n_apo, n_mava, bounds_apo, bounds_mava):
    rng = np.random.default_rng(SEED)
    if REPLICA_AWARE_SPLIT:
        train_idx, valid_idx = replica_aware_split(
            bounds_apo, bounds_mava, n_apo, VALID_REPLICA_FRAC, rng
        )
        strategy = "replica-aware"
    else:
        idx = np.arange(n_apo + n_mava, dtype=np.int64)
        rng.shuffle(idx)
        n_train = int(TRAIN_FRAC * len(idx))
        train_idx, valid_idx = idx[:n_train], idx[n_train:]
        strategy = "global random-frame"

    np.save(OUT_DIR / "train_indices.npy", train_idx)
    np.save(OUT_DIR / "validation_indices.npy", valid_idx)
    return train_idx, valid_idx, strategy


def gather_rows(data_apo, data_mava, global_indices):
    """Gather global indices without first stacking both complete arrays."""
    n_apo = len(data_apo)
    result = np.empty((len(global_indices), data_apo.shape[1]), dtype=np.float32)
    apo_mask = global_indices < n_apo
    result[apo_mask] = data_apo[global_indices[apo_mask]]
    result[~apo_mask] = data_mava[global_indices[~apo_mask] - n_apo]
    return result


def project_existing_model(model, data_apo, data_mava):
    outputs = []
    model.eval().cpu()
    with torch.no_grad():
        for data in (data_apo, data_mava):
            state_parts = []
            for start in range(0, len(data), PROJECTION_BATCH_SIZE):
                x = torch.from_numpy(
                    np.asarray(
                        data[start:start + PROJECTION_BATCH_SIZE], dtype=np.float32
                    ).copy()
                )
                y = model(x)
                y = y.reshape(len(x), -1)[:, 0]
                state_parts.append(y.cpu().numpy().astype(np.float32, copy=False))
            outputs.append(np.concatenate(state_parts))
    return outputs[0], outputs[1]


def compute_sensitivity(model, training_data, training_labels, training_std):
    """Training-sigma-scaled, sum-normalized gradient sensitivity.

    g_ji = |dCV_j/dx_ji| * sigma_i
    S_i  = mean_j(g_ji) / sum_k mean_j(g_jk)
    """
    sigma = torch.as_tensor(training_std, dtype=torch.float32)
    scaled_batches = []
    model.eval().cpu()

    for start in range(0, len(training_data), SENSITIVITY_BATCH_SIZE):
        x = torch.from_numpy(
            training_data[start:start + SENSITIVITY_BATCH_SIZE]
        ).clone().requires_grad_(True)
        output = model(x).reshape(len(x), -1)[:, 0]
        gradient = torch.autograd.grad(
            output, x, grad_outputs=torch.ones_like(output),
            retain_graph=False, create_graph=False,
        )[0]
        scaled_batches.append(
            (gradient.detach().abs() * sigma).cpu().numpy().astype(np.float32)
        )

    scaled = np.concatenate(scaled_batches, axis=0)
    denominator = float(scaled.mean(axis=0).sum())
    if not np.isfinite(denominator) or denominator <= 0:
        raise RuntimeError("Sensitivity normalization denominator is invalid.")

    normalized_per_frame = scaled / denominator
    scores = normalized_per_frame.mean(axis=0)
    std_scores = normalized_per_frame.std(axis=0, ddof=0)
    if not np.isclose(scores.sum(), 1.0, atol=2e-5):
        raise RuntimeError(f"Sensitivity scores sum to {scores.sum()}, not 1.")

    return scores, std_scores, normalized_per_frame, denominator


def compute_cohens_d(cv_apo, cv_mava):
    """Descriptive Cohen's d from all projected frames."""
    x = np.asarray(cv_apo, dtype=np.float64)
    y = np.asarray(cv_mava, dtype=np.float64)
    n1, n2 = len(x), len(y)
    mean1, mean2 = x.mean(), y.mean()
    sd1, sd2 = x.std(ddof=1), y.std(ddof=1)
    pooled = math.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
    d = math.nan if pooled == 0 else (mean2 - mean1) / pooled
    return {
        "population": "all projected frames (descriptive; MD frames are time-correlated)",
        "formula": "(mean_Mava - mean_Apo) / pooled_sample_standard_deviation",
        "state1": LABEL_APO,
        "state2": LABEL_MAVA,
        "n_state1": n1,
        "n_state2": n2,
        "mean_state1": mean1,
        "mean_state2": mean2,
        "sample_sd_state1": sd1,
        "sample_sd_state2": sd2,
        "pooled_sample_sd": pooled,
        "cohens_d_state2_minus_state1": d,
        "absolute_cohens_d": abs(d),
    }


def save_sensitivity(
    scores, std_scores, per_frame, labels, pair_labels,
    pair_labels_crystal, training_std, denominator,
):
    rank = np.argsort(scores)[::-1]
    ranks = np.empty(len(pair_labels), dtype=int)
    ranks[rank] = np.arange(1, len(pair_labels) + 1)
    pd.DataFrame({
        "feature_index": np.arange(len(pair_labels)),
        "pair_label": pair_labels,
        "pair_label_crystal": pair_labels_crystal,
        "training_sigma_A": training_std,
        "normalized_sensitivity": scores,
        "per_frame_sensitivity_std": std_scores,
        "rank": ranks,
    }).sort_values("rank").to_csv(OUT_DIR / "sensitivity_scores.csv", index=False)

    np.save(OUT_DIR / "sensitivity_normalized.npy", scores.astype(np.float32))
    np.save(OUT_DIR / "sensitivity_std.npy", std_scores.astype(np.float32))
    top = rank[:min(TOP_N_FEATURES, len(rank))]
    np.savez_compressed(
        OUT_DIR / "sensitivity_top_feature_gradients.npz",
        feature_indices=top,
        pair_labels=pair_labels[top],
        pair_labels_crystal=pair_labels_crystal[top],
        normalized_gradients=per_frame[:, top].astype(np.float32),
        training_labels=labels.astype(np.int8),
    )
    with open(OUT_DIR / "sensitivity_method.json", "w", encoding="utf-8") as handle:
        json.dump({
            "population": "reconstructed original training frames only",
            "formula": "mean(|dCV/dx_i| * training_sigma_i), normalized so sum_i S_i = 1",
            "normalization_denominator": denominator,
            "score_sum": float(scores.sum()),
            "n_training_frames": int(len(per_frame)),
            "n_features": int(len(pair_labels)),
            "seed": int(SEED),
            "train_fraction": float(TRAIN_FRAC),
            "replica_aware_split": bool(REPLICA_AWARE_SPLIT),
        }, handle, indent=2)
    return rank


def plot_sensitivity(scores, std_scores, per_frame, labels, pair_labels,
                     pair_labels_crystal, rank):
    """Figures are labelled with the crystallographic numbering used in the
    manuscript. pair_labels_crystal falls back to the simulation labels when
    the workbook has no '(crystal)' columns, so plots stay correct either way.
    The simulation labels are kept for the machine-readable outputs."""
    shown = pair_labels_crystal
    count = min(TOP_N_FEATURES, len(rank))
    selected = rank[:count][::-1]
    fig, ax = plt.subplots(figsize=(10.5, max(6.5, 0.36 * count)), dpi=150)
    positions = np.arange(count)
    ax.barh(positions, scores[selected], xerr=std_scores[selected], capsize=2)
    ax.set_yticks(positions)
    ax.set_yticklabels(shown[selected], fontsize=10)
    ax.set_xlabel("Normalized sensitivity", fontweight="bold")
    ax.set_title("Whole-residue COM-distance sensitivity", fontweight="bold")
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sensitivity_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    count_v = min(12, count)
    selected_v = rank[:count_v][::-1]
    fig, ax = plt.subplots(figsize=(10.5, max(6.0, 0.42 * count_v)), dpi=150)
    ax.violinplot([per_frame[:, i] for i in selected_v], positions=np.arange(count_v),
                  orientation="horizontal", showmeans=True,
                  showextrema=False, widths=0.8)
    ax.set_yticks(np.arange(count_v))
    ax.set_yticklabels(shown[selected_v], fontsize=10)
    ax.set_xlabel("Per-frame normalized sensitivity", fontweight="bold")
    ax.set_title("Training-frame sensitivity distributions", fontweight="bold")
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sensitivity_violin.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    count_s = min(10, count)
    selected_s = rank[:count_s]
    rows = []
    for feature_index in selected_s:
        for state_code, state_label in enumerate((LABEL_APO, LABEL_MAVA)):
            values = per_frame[labels == state_code, feature_index]
            rows.append({
                "pair_label_crystal": pair_labels_crystal[feature_index],
                "pair_label": pair_labels[feature_index],
                "state": state_label,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
            })
    pd.DataFrame(rows).to_csv(
        OUT_DIR / "sensitivity_by_state_top_features.csv", index=False
    )

    x = np.arange(count_s)
    width = 0.38
    fig, ax = plt.subplots(figsize=(13, 5.8), dpi=150)
    for state_code, state_label in enumerate((LABEL_APO, LABEL_MAVA)):
        means = [per_frame[labels == state_code, i].mean() for i in selected_s]
        ax.bar(x + (state_code - 0.5) * width, means, width=width, label=state_label)
    ax.set_xticks(x)
    ax.set_xticklabels(shown[selected_s], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean normalized sensitivity", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sensitivity_by_state.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _shared_edges(a, b, bins):
    lo = min(float(np.min(a)), float(np.min(b)))
    hi = max(float(np.max(a)), float(np.max(b)))
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, bins + 1)


def plot_fes(cv_apo, cv_mava):
    edges = _shared_edges(cv_apo, cv_mava, FES_BINS)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=150)
    for values, label in ((cv_apo, LABEL_APO), (cv_mava, LABEL_MAVA)):
        axes[0].hist(values, bins=edges, density=True, alpha=0.55, label=label)
        counts, _ = np.histogram(values, bins=edges, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        counts = np.where(counts > 0, counts, np.nan)
        fes = -np.log(counts)
        fes -= np.nanmin(fes)
        axes[1].plot(centers, fes, lw=2.2, label=label)
    axes[0].set_xlabel(r"DeepLDA CV$_1$", fontweight="bold")
    axes[0].set_ylabel("Probability density", fontweight="bold")
    axes[1].set_xlabel(r"DeepLDA CV$_1$", fontweight="bold")
    axes[1].set_ylabel("F (kT)", fontweight="bold")
    for ax in axes:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fes_combined.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(cv, bounds, label):
    n_reps = len(bounds)
    n_cols = min(4, n_reps)
    n_rows = (n_reps + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3.8*n_rows), dpi=120)
    flat = np.atleast_1d(axes).ravel()
    for rep_i, (start, end) in enumerate(bounds):
        time_ns = np.arange(end - start, dtype=np.float64) * TIME_PER_FRAME_NS
        flat[rep_i].plot(time_ns, cv[start:end], lw=0.8)
        flat[rep_i].set_title(f"Replica {rep_i}", fontweight="bold", fontsize=11)
        flat[rep_i].set_xlabel("Time (ns)", fontsize=10)
        flat[rep_i].set_ylabel(r"CV$_1$", fontsize=10)
        flat[rep_i].set_ylim(-2, 2)
    for ax in flat[n_reps:]:
        ax.set_visible(False)
    fig.suptitle(f"CV convergence - {label}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"convergence_{label.lower()}.png",
                dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze an existing DeepLDA model without retraining it."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "Path to a serialized full model or a state-dictionary .pt file. "
            "Defaults to models/deeplda_model_full.pt, then models/deeplda_model.pt."
        ),
    )
    args = parser.parse_args()

    print("=" * 72)
    print("POST-TRAINING ANALYSIS OF EXISTING DeepLDA MODEL — NO RETRAINING")
    print("=" * 72)

    data_apo = np.load(_required(OUT_DIR / f"descriptors_{LABEL_APO.lower()}.npy"), mmap_mode="r")
    data_mava = np.load(_required(OUT_DIR / f"descriptors_{LABEL_MAVA.lower()}.npy"), mmap_mode="r")
    bounds_apo = np.load(_required(OUT_DIR / f"replica_boundaries_{LABEL_APO.lower()}.npy"))
    bounds_mava = np.load(_required(OUT_DIR / f"replica_boundaries_{LABEL_MAVA.lower()}.npy"))
    pair_labels = np.load(_required(OUT_DIR / "pairs_labels.npy"), allow_pickle=True).astype(str)
    pair_labels_crystal = load_crystal_pair_labels(
        FEATURES_FILE, pair_labels, CRYSTAL_NUMBERING_OFFSET
    )

    model_path = resolve_model_path(args.model)
    model, model_source_type = load_existing_model(model_path, data_apo.shape[1])
    print(f"Model loaded without training ({model_source_type}): {model_path}")

    train_idx, valid_idx, strategy = reconstruct_split(
        len(data_apo), len(data_mava), bounds_apo, bounds_mava
    )
    print(f"Reconstructed split: {strategy}; train={len(train_idx)}, valid={len(valid_idx)}")

    training_data = gather_rows(data_apo, data_mava, train_idx)
    training_labels = (train_idx >= len(data_apo)).astype(np.int8)
    training_mean = training_data.mean(axis=0, dtype=np.float64).astype(np.float32)
    training_std = training_data.std(axis=0, ddof=0, dtype=np.float64).astype(np.float32)
    training_std = np.where(training_std > EPS, training_std, 1.0).astype(np.float32)
    np.save(OUT_DIR / "normalization_mean.npy", training_mean)
    np.save(OUT_DIR / "normalization_std.npy", training_std)

    print("Projecting the existing model onto all descriptor frames ...")
    cv_apo, cv_mava = project_existing_model(model, data_apo, data_mava)
    np.save(OUT_DIR / f"cv_values_{LABEL_APO.lower()}.npy", cv_apo)
    np.save(OUT_DIR / f"cv_values_{LABEL_MAVA.lower()}.npy", cv_mava)

    print("Computing corrected sensitivity ...")
    scores, std_scores, per_frame, denominator = compute_sensitivity(
        model, training_data, training_labels, training_std
    )
    rank = save_sensitivity(
        scores, std_scores, per_frame, training_labels, pair_labels,
        pair_labels_crystal, training_std, denominator,
    )
    plot_sensitivity(scores, std_scores, per_frame, training_labels, pair_labels,
                     pair_labels_crystal, rank)

    print("Computing descriptive Cohen's d ...")
    cohens = compute_cohens_d(cv_apo, cv_mava)
    pd.DataFrame([cohens]).to_csv(OUT_DIR / "cohens_d.csv", index=False)
    with open(OUT_DIR / "cohens_d.json", "w", encoding="utf-8") as handle:
        json.dump(cohens, handle, indent=2)

    plot_fes(cv_apo, cv_mava)
    plot_convergence(cv_apo, bounds_apo, LABEL_APO)
    plot_convergence(cv_mava, bounds_mava, LABEL_MAVA)

    with open(OUT_DIR / "post_training_analysis_summary.json", "w", encoding="utf-8") as handle:
        json.dump({
            "model": str(model_path),
            "model_source_type": model_source_type,
            "model_retrained": False,
            "split_strategy": strategy,
            "seed": int(SEED),
            "train_fraction": float(TRAIN_FRAC),
            "replica_aware_split": bool(REPLICA_AWARE_SPLIT),
            "n_training_frames": int(len(train_idx)),
            "sensitivity_score_sum": float(scores.sum()),
            "cohens_d_state2_minus_state1": cohens["cohens_d_state2_minus_state1"],
        }, handle, indent=2)

    print("\nTop 10 sensitivity features:")
    for position, feature_index in enumerate(rank[:10], start=1):
        print(f"  {position:>2}. {pair_labels_crystal[feature_index]:24s} "
              f"(sim {pair_labels[feature_index]:22s}) {scores[feature_index]:.6f}")
    print(f"\nCohen's d ({LABEL_MAVA} - {LABEL_APO}): "
          f"{cohens['cohens_d_state2_minus_state1']:.4f}")
    print("\nPOST-TRAINING ANALYSIS COMPLETE. The model was not retrained.\n")


if __name__ == "__main__":
    main()
