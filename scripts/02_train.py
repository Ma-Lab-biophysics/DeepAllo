"""
02_train.py — fixed version
"""

import sys
import os
import time

import numpy as np
import torch
import lightning
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Global plot style ────────────────────────────────────────────────────────
plt.rcParams['font.weight']        = 'bold'
plt.rcParams['font.size']          = 34
plt.rcParams['xtick.major.width']  = 2
plt.rcParams['xtick.major.size']   = 8
plt.rcParams['xtick.major.pad']    = 8
plt.rcParams['xtick.minor.width']  = 1
plt.rcParams['ytick.major.width']  = 2
plt.rcParams['ytick.major.size']   = 8
plt.rcParams['ytick.major.pad']    = 8
plt.rcParams['ytick.minor.width']  = 1
plt.rcParams['axes.linewidth']     = 3
plt.rcParams['xtick.direction']    = 'in'
plt.rcParams['ytick.direction']    = 'in'
# ─────────────────────────────────────────────────────────────────────────────

from lightning.pytorch import LightningDataModule
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from mlcolvar.cvs import DeepLDA
from mlcolvar.data import DictDataset, DictLoader
from mlcolvar.utils.trainer import MetricsCallback
from mlcolvar.utils.plot import plot_metrics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    OUT_DIR, MODELS_DIR, FIGURES_DIR,
    LABEL_APO, LABEL_MAVA,
    N_STATES,
    NN_HIDDEN, NN_ACTIVATION,
    TRAIN_FRAC,
    REPLICA_AWARE_SPLIT, VALID_REPLICA_FRAC,
    BATCH_SIZE, MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
    SEED, TRAIN_ACCELERATOR, TRAIN_DEVICES,
)

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_float32_matmul_precision("high")


class SplitDictModule(LightningDataModule):
    """
    Wraps two pre-built DictDatasets (train + val) into DictLoaders.

    Using a custom LightningDataModule instead of mlcolvar's DictModule
    because DictModule only accepts a single dataset and re-randomises the
    split, destroying the replica-aware boundaries we carefully computed.
    """
    def __init__(self, train_dataset, val_dataset, batch_size=0):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset   = val_dataset
        self.batch_size    = batch_size

    def train_dataloader(self):
        return DictLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DictLoader(self.val_dataset,   batch_size=self.batch_size, shuffle=False)


def load_descriptors():
    print(f"  Loading descriptors_{LABEL_APO.lower()}.npy …", end=" ", flush=True)
    data_apo = np.array(np.load(os.path.join(OUT_DIR, f"descriptors_{LABEL_APO.lower()}.npy"), mmap_mode="r"), dtype=np.float32)
    print(data_apo.shape)
    print(f"  Loading descriptors_{LABEL_MAVA.lower()}.npy  …", end=" ", flush=True)
    data_mava  = np.array(np.load(os.path.join(OUT_DIR, f"descriptors_{LABEL_MAVA.lower()}.npy"),  mmap_mode="r"), dtype=np.float32)
    print(data_mava.shape)
    bounds_apo = np.load(os.path.join(OUT_DIR, f"replica_boundaries_{LABEL_APO.lower()}.npy"))
    bounds_mava  = np.load(os.path.join(OUT_DIR, f"replica_boundaries_{LABEL_MAVA.lower()}.npy"))
    pair_labels = np.load(os.path.join(OUT_DIR, "pairs_labels.npy"), allow_pickle=True)
    return data_apo, data_mava, bounds_apo, bounds_mava, pair_labels


def replica_aware_split(bounds_apo, bounds_mava, n_apo, valid_frac, rng):
    def split_one(bounds, offset):
        n_reps     = len(bounds)
        n_val_reps = max(1, round(n_reps * valid_frac))
        val_set    = set(np.linspace(0, n_reps - 1, n_val_reps, dtype=int).tolist())
        tr, va = [], []
        for rep_i, (start, end) in enumerate(bounds):
            frames = np.arange(start, end) + offset
            (va if rep_i in val_set else tr).append(frames)
        return np.concatenate(tr), np.concatenate(va)

    tr_apo, va_apo = split_one(bounds_apo, 0)
    tr_mava, va_mava = split_one(bounds_mava,  n_apo)
    train_idx = np.concatenate([tr_apo, tr_mava])
    valid_idx = np.concatenate([va_apo, va_mava])
    rng.shuffle(train_idx)
    rng.shuffle(valid_idx)
    return train_idx, valid_idx


def build_datasets(data_apo, data_mava, bounds_apo, bounds_mava):
    """
    Build train/val DictDatasets.

    Labels are float32 (not Long/int64).
    Reason: mlcolvar's DictLoader.get_stats() calls torch.mean() on every
    key in the dataset — this raises RuntimeError for integer tensors.
    DeepLDA's FisherDiscriminantLoss casts labels to long internally.
    """
    rng    = np.random.default_rng(SEED)
    n_apo = len(data_apo)
    n_mava  = len(data_mava)

    all_data = np.vstack([data_apo, data_mava])
    # float32 labels — avoids Statistics(torch.mean) crash on Long tensors
    all_labels = np.concatenate([
        np.zeros(n_apo, dtype=np.float32),
        np.ones( n_mava,  dtype=np.float32),
    ])

    all_data_t   = torch.from_numpy(all_data)
    all_labels_t = torch.from_numpy(all_labels)

    if REPLICA_AWARE_SPLIT:
        print("  Split strategy : replica-aware")
        train_idx, valid_idx = replica_aware_split(
            bounds_apo, bounds_mava, n_apo, VALID_REPLICA_FRAC, rng
        )
    else:
        print("  Split strategy : random")
        idx = np.arange(len(all_data))
        rng.shuffle(idx)
        n_train   = int(TRAIN_FRAC * len(all_data))
        train_idx = idx[:n_train]
        valid_idx = idx[n_train:]

    lbl = all_labels
    print(f"  Train : {len(train_idx)} frames  "
          f"({LABEL_APO}: {int((lbl[train_idx]==0).sum())}, "
          f"{LABEL_MAVA}: {int((lbl[train_idx]==1).sum())})")
    print(f"  Valid : {len(valid_idx)} frames  "
          f"({LABEL_APO}: {int((lbl[valid_idx]==0).sum())}, "
          f"{LABEL_MAVA}: {int((lbl[valid_idx]==1).sum())})")

    train_ds = DictDataset({"data": all_data_t[train_idx], "labels": all_labels_t[train_idx]})
    valid_ds = DictDataset({"data": all_data_t[valid_idx], "labels": all_labels_t[valid_idx]})
    return train_ds, valid_ds, all_data_t, all_labels_t


def build_model(n_input):
    layers  = [n_input] + NN_HIDDEN
    options = {"nn": {"activation": NN_ACTIVATION}}
    model   = DeepLDA(layers, n_states=N_STATES, options=options)
    print(model)
    print(f"  Trainable parameters : {sum(p.numel() for p in model.parameters()):,}")
    return model


def plot_training_metrics(metrics_cb, save_path):
    """
    Reproduce the mlcolvar tutorial learning-curves plot.
    All 'valid' metric keys are plotted with yscale=linear.
    Legend is placed outside the plot on the right.
    """
    m = metrics_cb.metrics
    print(f"  Metric keys logged : {list(m.keys())}")
    for k, v in m.items():
        if isinstance(v, list) and v:
            print(f"    {k}: {len(v)} values  "
                  f"first={v[0]:.3f}  best={min(v):.3f}  final={v[-1]:.3f}")

    valid_keys = [k for k in m.keys() if "valid" in k and
                  isinstance(m[k], list) and len(m[k]) > 0]
    print(f"  Plotting valid keys: {valid_keys}")

    fig, ax = plt.subplots(figsize=(14, 7), dpi=120)
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(valid_keys))

    try:
        plot_metrics(m, keys=valid_keys, yscale="linear", ax=ax)
        ax.set_title("Learning curves", fontweight="bold")
        # Move the legend that plot_metrics created to outside the axes
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0,
            frameon=True,
            edgecolor="black",
            fancybox=False,
            fontsize=18,
        )
    except Exception as e:
        print(f"  plot_metrics failed ({e}), using manual plot")
        for i, key in enumerate(valid_keys):
            arr = [float(v) for v in m[key]]
            ax.plot(range(1, len(arr) + 1), arr,
                    label=key, color=cmap(i), lw=2)
        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel("Loss",  fontweight="bold")
        ax.set_title("Learning curves", fontweight="bold")
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0,
            frameon=True,
            edgecolor="black",
            fancybox=False,
            fontsize=18,
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {save_path}")


def _to_float_array(values):
    """Convert a MetricsCallback metric list to a NumPy float array."""
    arr = []
    for x in values:
        if hasattr(x, "detach"):
            arr.append(float(x.detach().cpu()))
        else:
            arr.append(float(x))
    return np.asarray(arr, dtype=float)


def plot_early_stopping_convergence(metrics_cb, early_stop, save_path):
    """
    Plot the exact convergence evidence for the early-stopping sentence.

    Early-stopping criterion used in this script:
      monitor   = valid_eigval_1_epoch
      mode      = max
      min_delta = EARLY_STOPPING_MIN_DELTA
      patience  = EARLY_STOPPING_PATIENCE

    Therefore, training stops when the validation LDA eigenvalue does not
    improve by more than delta_lambda = 1e-3 for 50 consecutive validation
    epochs, using the values defined in config.py.
    """
    m = metrics_cb.metrics
    key = "valid_eigval_1_epoch"

    if key not in m or len(m[key]) == 0:
        raise RuntimeError(
            f"Metric '{key}' was not found in MetricsCallback. "
            f"Available keys are: {list(m.keys())}"
        )

    valid_lambda = _to_float_array(m[key])
    epochs = np.arange(1, len(valid_lambda) + 1)

    # Reconstruct the early-stopping logic for mode='max'.
    # A new best is accepted only if val > best + min_delta.
    accepted_best = np.full_like(valid_lambda, np.nan, dtype=float)
    wait_count = np.zeros_like(valid_lambda, dtype=int)

    best_val = -np.inf
    best_epoch = 1
    wait = 0
    reconstructed_stop_epoch = None

    for i, val in enumerate(valid_lambda):
        epoch = i + 1
        improved = val > best_val + EARLY_STOPPING_MIN_DELTA

        if improved:
            best_val = val
            best_epoch = epoch
            wait = 0
        else:
            wait += 1

        accepted_best[i] = best_val
        wait_count[i] = wait

        if wait >= EARLY_STOPPING_PATIENCE and reconstructed_stop_epoch is None:
            reconstructed_stop_epoch = epoch

    # Lightning stores stopped_epoch only when early stopping has actually fired.
    stopped_epoch = int(getattr(early_stop, "stopped_epoch", 0))
    if stopped_epoch == 0:
        stopped_epoch = reconstructed_stop_epoch

    # Save a compact CSV for record keeping / manuscript evidence.
    csv_path = os.path.join(OUT_DIR, "early_stopping_convergence.csv")
    with open(csv_path, "w") as fh:
        fh.write("epoch,valid_lambda,accepted_best_lambda,epochs_since_last_improvement\n")
        for e, v, b, w in zip(epochs, valid_lambda, accepted_best, wait_count):
            fh.write(f"{e},{v:.10f},{b:.10f},{w}\n")

    # Save a readable summary.
    summary_path = os.path.join(OUT_DIR, "early_stopping_summary.txt")
    with open(summary_path, "w") as fh:
        fh.write("DeepLDA early-stopping convergence summary\n")
        fh.write("-----------------------------------------\n")
        fh.write(f"Monitor: {key}\n")
        fh.write("Mode: max\n")
        fh.write(f"Minimum improvement threshold, delta_lambda: {EARLY_STOPPING_MIN_DELTA}\n")
        fh.write(f"Patience: {EARLY_STOPPING_PATIENCE} validation epochs\n")
        fh.write(f"Best accepted validation lambda: {best_val:.8f}\n")
        fh.write(f"Best accepted epoch: {best_epoch}\n")
        fh.write(f"Total epochs run: {len(valid_lambda)}\n")
        fh.write(f"Stopped epoch: {stopped_epoch}\n")
        fh.write(f"Final epochs since last accepted improvement: {wait_count[-1]}\n")

    # Plot only the monitored quantity used in the sentence.
    # The title is printed to the terminal/log, not placed on the figure.
    plot_title = (
        r"DeepLDA early-stopping convergence "
        r"($\Delta\lambda = 10^{-3}$, patience = 50)"
    )
    print(f"  Plot title: {plot_title}")

    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

    ax.plot(
        epochs,
        valid_lambda,
        lw=2.2,
        marker="o",
        markersize=2.5,
        label="Validation λ",
    )

    ax.plot(
        epochs,
        accepted_best,
        lw=1.8,
        ls="--",
        label="Accepted best λ",
    )

    ax.axvline(
        best_epoch,
        lw=1.8,
        ls=":",
        label=f"Best epoch = {best_epoch}",
    )

    if stopped_epoch is not None:
        ax.axvline(
            stopped_epoch,
            lw=1.8,
            ls="-.",
            label=f"Stop epoch = {stopped_epoch}",
        )

    patience_end = min(best_epoch + EARLY_STOPPING_PATIENCE, len(valid_lambda))
    if patience_end > best_epoch:
        ax.axvspan(
            best_epoch,
            patience_end,
            alpha=0.15,
            label=f"Patience = {EARLY_STOPPING_PATIENCE}",
        )

    # Reduced and shorter axis labels for publication-style plotting.
    ax.set_xlabel("Epoch", fontweight="bold", fontsize=18)
    ax.set_ylabel("Valid λ", fontweight="bold", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14)

    ax.text(
        0.02,
        0.02,
        (
            f"No improvement > {EARLY_STOPPING_MIN_DELTA:g} "
            f"for {EARLY_STOPPING_PATIENCE} epochs"
        ),
        transform=ax.transAxes,
        fontsize=11,
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        edgecolor="black",
        fancybox=False,
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  Early-stopping convergence plot saved : {save_path}")
    print(f"  Early-stopping CSV saved              : {csv_path}")
    print(f"  Early-stopping summary saved          : {summary_path}")


def _choose_metric(metrics, preferred_keys):
    """Return the first available metric key and values from a priority list."""
    for key in preferred_keys:
        if key in metrics and isinstance(metrics[key], list) and len(metrics[key]) > 0:
            return key, _to_float_array(metrics[key])
    return None, None


def plot_loss_convergence(metrics_cb, early_stop, save_path):
    """
    Plot training and validation loss vs epoch.

    This avoids confusion with the validation-eigenvalue plot. In DeepLDA,
    the loss is usually the negative Fisher/LDA eigenvalue, so decreasing
    loss corresponds to increasing state separation. Early stopping in this
    script is still based on valid_eigval_1_epoch, not directly on loss.
    """
    m = metrics_cb.metrics

    train_key, train_loss = _choose_metric(
        m,
        ["train_loss_epoch", "train_loss"]
    )
    valid_key, valid_loss = _choose_metric(
        m,
        ["valid_loss_epoch", "valid_loss"]
    )

    if train_loss is None and valid_loss is None:
        raise RuntimeError(
            "No train/valid loss metrics were found. "
            f"Available keys are: {list(m.keys())}"
        )

    # Use the available loss curve length as epoch count.
    n_epochs = max(
        len(train_loss) if train_loss is not None else 0,
        len(valid_loss) if valid_loss is not None else 0,
    )

    # For reference, recover best and stop epochs from the validation λ curve,
    # because that is the actual early-stopping monitor in your script.
    best_epoch = None
    stopped_epoch = int(getattr(early_stop, "stopped_epoch", 0))
    valid_lambda = None
    if "valid_eigval_1_epoch" in m and len(m["valid_eigval_1_epoch"]) > 0:
        valid_lambda = _to_float_array(m["valid_eigval_1_epoch"])
        best_val = -np.inf
        wait = 0
        reconstructed_stop_epoch = None
        for i, val in enumerate(valid_lambda):
            epoch = i + 1
            if val > best_val + EARLY_STOPPING_MIN_DELTA:
                best_val = val
                best_epoch = epoch
                wait = 0
            else:
                wait += 1
            if wait >= EARLY_STOPPING_PATIENCE and reconstructed_stop_epoch is None:
                reconstructed_stop_epoch = epoch
        if stopped_epoch == 0:
            stopped_epoch = reconstructed_stop_epoch
    else:
        stopped_epoch = stopped_epoch if stopped_epoch != 0 else None

    # Save CSV for record keeping.
    csv_path = os.path.join(OUT_DIR, "loss_convergence.csv")
    with open(csv_path, "w") as fh:
        fh.write("epoch,train_loss,valid_loss\n")
        for i in range(n_epochs):
            tr = "" if train_loss is None or i >= len(train_loss) else f"{train_loss[i]:.10f}"
            va = "" if valid_loss is None or i >= len(valid_loss) else f"{valid_loss[i]:.10f}"
            fh.write(f"{i+1},{tr},{va}\n")

    plot_title = "DeepLDA loss convergence"
    print(f"  Plot title: {plot_title}")
    print(f"  Loss metrics plotted: train={train_key}, valid={valid_key}")
    print("  Note: early stopping is based on valid_eigval_1_epoch, not directly on loss.")

    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

    if train_loss is not None:
        ax.plot(
            np.arange(1, len(train_loss) + 1),
            train_loss,
            lw=2.2,
            label="Train loss",
        )

    if valid_loss is not None:
        ax.plot(
            np.arange(1, len(valid_loss) + 1),
            valid_loss,
            lw=2.2,
            ls="--",
            label="Valid loss",
        )

    if best_epoch is not None:
        ax.axvline(
            best_epoch,
            lw=1.8,
            ls=":",
            label=f"Best λ epoch = {best_epoch}",
        )

    if stopped_epoch is not None:
        ax.axvline(
            stopped_epoch,
            lw=1.8,
            ls="-.",
            label=f"Stop epoch = {stopped_epoch}",
        )

    # Short labels; no figure title.
    ax.set_xlabel("Epoch", fontweight="bold", fontsize=18)
    ax.set_ylabel("Loss", fontweight="bold", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=14)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        edgecolor="black",
        fancybox=False,
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  Loss convergence plot saved : {save_path}")
    print(f"  Loss convergence CSV saved  : {csv_path}")


def plot_cv_distribution(cv_all, labels_int, save_path):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=120)
    # Manuscript convention: red = first state (Apo), blue = second state
    # (Mava). Kept identical to COLOR_STATE1 / COLOR_STATE2 in 04_apply_cv.py.
    palette = [(LABEL_APO, "#d6604d"), (LABEL_MAVA, "#2166ac")]
    for idx, (name, color) in enumerate(palette):
        ax.hist(cv_all[labels_int == idx], bins=80, density=True,
                alpha=0.55, color=color, label=name, edgecolor="none")
    ax.set_xlabel("CV₁", fontweight="bold")
    ax.set_ylabel("Probability density", fontweight="bold")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        frameon=True,
        edgecolor="black",
        fancybox=False,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {save_path}")

def plot_cv_vs_time(cv_values, bounds, offset, state_label, save_path):
    fig, ax = plt.subplots(figsize=(14, 6), dpi=120)
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(len(bounds))
    for rep_i, (start, end) in enumerate(bounds):
        ax.plot(np.arange(end - start) + start,
                cv_values[np.arange(start, end) + offset],
                color=cmap(rep_i), alpha=0.80, lw=1.2, label=f"Rep {rep_i}")
    ax.set_xlabel("Frame (1 ns / frame)", fontweight="bold")
    ax.set_ylabel("CV₁", fontweight="bold")
    n_cols = max(1, len(bounds) // 6)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        frameon=True,
        edgecolor="black",
        fancybox=False,
        ncol=n_cols,
        handlelength=1.2,
        fontsize=22,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {save_path}")

def main():
    print("=" * 65)
    print("STEP 02 — DeepLDA Training (mlcolvar)")
    print("=" * 65)

    print("\n[1/6] Loading descriptors …")
    data_apo, data_mava, bounds_apo, bounds_mava, pair_labels = load_descriptors()
    n_pairs = data_apo.shape[1]
    print(f"  n_pairs  : {n_pairs}")

    print("\n[2/6] Building datasets …")
    train_ds, valid_ds, all_data_t, all_labels_t = build_datasets(
        data_apo, data_mava, bounds_apo, bounds_mava
    )
    datamodule = SplitDictModule(train_ds, valid_ds, batch_size=BATCH_SIZE)

    print("\n[3/6] Building model …")
    model = build_model(n_pairs)

    print("\n[4/6] Configuring trainer …")
    metrics_cb = MetricsCallback()
    # Monitor the LDA eigenvalue directly (mode=max), exactly as in the
    # official mlcolvar tutorial. This is more direct than monitoring
    # valid_loss (= -eigenvalue) with mode=min.
    early_stop = EarlyStopping(
        monitor="valid_eigval_1_epoch",
        min_delta=EARLY_STOPPING_MIN_DELTA,
        patience=EARLY_STOPPING_PATIENCE,
        mode="max",
    )
    # MAX_EPOCHS=None → -1 (Lightning ≥ 2.x "no limit"); None silently caps at 1000
    max_epochs = MAX_EPOCHS if MAX_EPOCHS is not None else -1
    trainer = lightning.Trainer(
        callbacks=[metrics_cb, early_stop],
        max_epochs=max_epochs,
        accelerator=TRAIN_ACCELERATOR,
        devices=TRAIN_DEVICES,
        logger=False,
        enable_checkpointing=False,
    )

    print("\n[5/6] Training …")
    t0 = time.time()
    trainer.fit(model, datamodule)
    print(f"\n  Done — {(time.time()-t0)/60:.1f} min")
    m = metrics_cb.metrics
    print(f"  Logged metric keys : {list(m.keys())}")
    for key, fn in [("valid_eigval_1_epoch", max), ("train_loss_epoch", min),
                    ("valid_loss", min), ("train_loss", min)]:
        if key in m and len(m[key]) > 0:
            print(f"  Best {key:30s}: {fn(m[key]):.4f}")
    epoch_key = next((k for k in ("train_loss_epoch", "train_loss") if k in m), None)
    if epoch_key:
        print(f"  Epochs run         : {len(m[epoch_key])}")

    print("\n[6/6] Saving + plotting …")
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "deeplda_model.pt"))
    torch.save(model,              os.path.join(MODELS_DIR, "deeplda_model_full.pt"))
    print(f"  Models → {MODELS_DIR}")

    plot_training_metrics(metrics_cb, os.path.join(FIGURES_DIR, "training_metrics.png"))
    plot_early_stopping_convergence(
        metrics_cb,
        early_stop,
        os.path.join(FIGURES_DIR, "early_stopping_convergence.png"),
    )
    plot_loss_convergence(
        metrics_cb,
        early_stop,
        os.path.join(FIGURES_DIR, "loss_convergence.png"),
    )

    model.eval()
    with torch.no_grad():
        cv_all = model(all_data_t).squeeze().cpu().numpy()

    labels_int = all_labels_t.numpy().astype(int)
    n_apo     = len(data_apo)

    plot_cv_distribution(cv_all, labels_int, os.path.join(FIGURES_DIR, "cv_distribution.png"))
    plot_cv_vs_time(cv_all, bounds_apo, 0,      LABEL_APO, os.path.join(FIGURES_DIR, f"cv_vs_time_{LABEL_APO.lower()}.png"))
    plot_cv_vs_time(cv_all, bounds_mava,  n_apo, LABEL_MAVA,  os.path.join(FIGURES_DIR, f"cv_vs_time_{LABEL_MAVA.lower()}.png"))

    np.save(os.path.join(OUT_DIR, f"cv_values_{LABEL_APO.lower()}.npy"), cv_all[:n_apo])
    np.save(os.path.join(OUT_DIR, f"cv_values_{LABEL_MAVA.lower()}.npy"),  cv_all[n_apo:])
    print(f"  CV arrays → {OUT_DIR}")

    print("\nSTEP 02 COMPLETE.\n")


if __name__ == "__main__":
    main()
