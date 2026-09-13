"""
01_extract_descriptors.py
─────────────────────────
Load every individual replica trajectory for both configured states, compute
whole-residue centre-of-mass (COM) distances for the contact pairs
defined in CONTACTS_XLSX, and record exact per-replica frame boundaries
for replica-aware splitting.

Descriptor definition
─────────────────────
For each contact pair (Residue 1 (sim), Residue 2 (sim)) in the xlsx:
  • Residue atoms  = ALL heavy atoms (backbone N, CA, C, O, OXT included).
  • No GLY fallback needed — every residue has backbone heavy atoms.
  • Distance = ||COM_res_1 – COM_res_2||  per frame  [Å].

Topology requirements
─────────────────────
Step 01 builds atom selections independently from the topology configured for
each state. The topology files may differ in atom count or atom ordering, but
each trajectory must match its own topology. Every residue referenced by the
"Residue 1 (sim)" and "Residue 2 (sim)" workbook columns must exist uniquely,
with consistent residue numbering and identity, in both topologies.

Outputs (in OUT_DIR) — names depend on STATE1_LABEL / STATE2_LABEL in config.py:
  descriptors_<state1>.npy          (n_frames_state1, n_pairs)  float32
  descriptors_<state2>.npy          (n_frames_state2, n_pairs)  float32
  pairs_resids.npy                  (n_pairs, 2)   int32 — (resid1, resid2)
  pairs_labels.npy                  (n_pairs,)     str   — "RES1:id1-RES2:id2"
  replica_boundaries_<state1>.npy   (N_REPLICAS_STATE1, 2)  int64
  replica_boundaries_<state2>.npy   (N_REPLICAS_STATE2, 2)  int64
"""

import sys
import os
import time
from functools import reduce

import numpy as np
import pandas as pd
import MDAnalysis as mda

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    STATE1_TOPOLOGY, STATE2_TOPOLOGY,
    STATE1_TRAJS, STATE2_TRAJS,
    N_REPLICAS_STATE1, N_REPLICAS_STATE2,
    STATE1_LABEL, STATE2_LABEL,
    STRIDE_TRAIN, OUT_DIR,
    CONTACTS_XLSX,
)

# Frames processed per vectorised chunk.
# Peak RAM per chunk ≈ CHUNK_FRAMES × n_heavy_atoms × 3 × 4 B
CHUNK_FRAMES = 100


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_residue_string(s):
    """'ASN:709' → ('ASN', 709)"""
    resname, resid = str(s).strip().split(":")
    return resname.strip(), int(resid.strip())


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


def get_residue_atomgroup(universe, resname, resid, topology_path):
    """
    Return ALL heavy atoms for a residue (backbone + side-chain).
    No GLY fallback needed — every residue has at least backbone atoms.

    Raises ValueError if the residue is absent from the topology, if the
    resid is not unique, or if the topology residue name disagrees with the
    name given in CONTACTS_XLSX. The name check guards against a silent
    numbering mismatch: the xlsx carries both simulation and crystal
    numbering (which differ by an offset), and selecting on resid alone
    would otherwise happily build descriptors from the wrong residues.
    """
    ag = universe.select_atoms(f"resid {resid} and not type H")
    if len(ag) == 0:
        raise ValueError(
            f"Residue {resname}:{resid} not found in {topology_path}. "
            f"Check that the 'Residue N (sim)' columns of {CONTACTS_XLSX} "
            f"use the same numbering as the topology."
        )

    residues = ag.residues
    if len(residues) != 1:
        listed = ", ".join(f"{r.resname}:{r.resid}" for r in residues)
        raise ValueError(
            f"resid {resid} matches {len(residues)} residues in "
            f"{topology_path} "
            f"({listed}). Descriptor residues must be uniquely identified by "
            f"resid; for multi-chain systems (e.g. motor domain + light "
            f"chain) add a segid/chainID qualifier to this selection."
        )

    found = residues[0].resname
    if not _resname_matches(resname, found):
        raise ValueError(
            f"Residue-name mismatch at resid {resid}: {CONTACTS_XLSX} says "
            f"{resname}, {topology_path} has {found}. This usually means the "
            f"contact list and the topology use different residue numbering "
            f"(the xlsx 'sim' and 'crystal' columns differ by an offset) — "
            f"verify that the 'Residue N (sim)' columns are being read."
        )
    return ag


def build_residue_selections_from_xlsx(universe, xlsx_path, topology_path):
    """
    Load contact pairs from xlsx and build whole-residue heavy-atom selections.

    Returns
    -------
    dict containing topology-specific atom selections and shared pair metadata
    """
    df = pd.read_excel(xlsx_path)
    res1_col = [c for c in df.columns if "Residue 1" in c and "sim" in c][0]
    res2_col = [c for c in df.columns if "Residue 2" in c and "sim" in c][0]

    pairs_raw = [
        (parse_residue_string(row[res1_col]),
         parse_residue_string(row[res2_col]))
        for _, row in df.iterrows()
    ]
    n_pairs = len(pairs_raw)
    print(f"    Contact pairs loaded from xlsx : {n_pairs}")

    # Unique residues (insertion-order preserved)
    seen, unique_residues = {}, []
    for r1, r2 in pairs_raw:
        for r in (r1, r2):
            key = r[1]
            if key not in seen:
                seen[key] = len(unique_residues)
                unique_residues.append(r)

    print(f"    Unique residues                : {len(unique_residues)}")

    # Per-residue all-heavy-atom AtomGroups
    res_ags = []
    atom_counts = []
    for resname, resid in unique_residues:
        ag = get_residue_atomgroup(universe, resname, resid, topology_path)
        res_ags.append(ag)
        atom_counts.append(len(ag))

    print(f"    Heavy atoms per residue — "
          f"min={min(atom_counts)}  max={max(atom_counts)}  "
          f"mean={np.mean(atom_counts):.1f}")

    # Combined AtomGroup + per-residue slice boundaries + masses
    combined           = reduce(lambda a, b: a + b, res_ags)
    heavy_atom_indices = combined.indices

    res_slices, cursor = [], 0
    for ag in res_ags:
        n = len(ag)
        res_slices.append((cursor, cursor + n, ag.masses.astype(np.float32)))
        cursor += n

    # Pair index arrays
    resid_to_idx = {r[1]: i for i, r in enumerate(unique_residues)}
    pair_indices = np.array(
        [(resid_to_idx[r1[1]], resid_to_idx[r2[1]]) for r1, r2 in pairs_raw],
        dtype=np.int32,
    )
    pairs_resids = np.array(
        [(r1[1], r2[1]) for r1, r2 in pairs_raw], dtype=np.int32
    )
    pair_labels = np.array(
        [f"{r1[0]}:{r1[1]}-{r2[0]}:{r2[1]}" for r1, r2 in pairs_raw]
    )

    print(f"    Total heavy atoms in combined AG : {len(heavy_atom_indices)}")
    return {
        "heavy_atom_indices": heavy_atom_indices,
        "res_slices": res_slices,
        "pair_indices": pair_indices,
        "pairs_resids": pairs_resids,
        "pair_labels": pair_labels,
    }


def verify_same_contact_order(state1_features, state2_features):
    """Require both states to use the same ordered workbook features."""
    for key in ("pair_indices", "pairs_resids", "pair_labels"):
        if not np.array_equal(state1_features[key], state2_features[key]):
            raise ValueError(
                f"The two states produced inconsistent {key}. Both states "
                "must use the same ordered contact pairs from CONTACTS_XLSX."
            )


def count_frames(topology, traj_path, stride):
    """Return strided frame count without loading coordinates."""
    u = mda.Universe(topology, traj_path)
    return len(u.trajectory[::stride])


def load_heavy_coords(topology, traj_path, stride, heavy_atom_indices):
    """
    Load all-heavy-atom coordinates for one replica.
    Returns float32 array (n_frames_strided, n_heavy_atoms, 3).
    """
    u      = mda.Universe(topology, traj_path)
    hvy_ag = u.atoms[heavy_atom_indices]
    try:
        coords = u.trajectory.timeseries(
            atomgroup=hvy_ag, step=stride, order="fac"
        ).astype(np.float32)
    except TypeError:
        coords = u.trajectory.timeseries(
            asel=hvy_ag, step=stride, order="fac"
        ).astype(np.float32)
    return coords   # (n_frames_strided, n_heavy_atoms, 3)


def compute_residue_coms(coords_chunk, res_slices):
    """
    Mass-weighted whole-residue COM for each residue in each frame.
    Returns float32 (n_frames, n_unique_residues, 3).
    """
    n_frames = coords_chunk.shape[0]
    coms     = np.empty((n_frames, len(res_slices), 3), dtype=np.float32)
    for k, (start, end, masses) in enumerate(res_slices):
        atoms        = coords_chunk[:, start:end, :]   # (F, n_atoms_k, 3)
        coms[:, k, :] = np.einsum("a,fad->fd", masses, atoms) / masses.sum()
    return coms


def residue_com_distances_into(coords, res_slices, pair_indices, out, row_offset):
    """
    Compute whole-residue COM distances for all frames and write into memmap.
    """
    n_frames = coords.shape[0]
    i0, i1  = pair_indices[:, 0], pair_indices[:, 1]

    for start in range(0, n_frames, CHUNK_FRAMES):
        end   = min(start + CHUNK_FRAMES, n_frames)
        chunk = coords[start:end]
        coms  = compute_residue_coms(chunk, res_slices)    # (chunk, n_res, 3)
        diff  = coms[:, i0, :] - coms[:, i1, :]           # (chunk, n_pairs, 3)
        out[row_offset + start : row_offset + end] = (
            np.linalg.norm(diff, axis=2).astype(np.float32)
        )


def process_state(traj_list, topology, features, stride, label, out_path):
    """
    Process all replicas for one state using whole-residue COM distances.
    Returns boundaries array (N_replicas, 2) int64.
    """
    heavy_atom_indices = features["heavy_atom_indices"]
    res_slices = features["res_slices"]
    pair_indices = features["pair_indices"]
    n_pairs = len(pair_indices)

    # Step 1: count frames
    print(f"  [{label}] Counting frames across {len(traj_list)} replicas …")
    frame_counts = []
    for rep_idx, tp in enumerate(traj_list):
        n = count_frames(topology, tp, stride)
        frame_counts.append(n)
        print(f"    Rep {rep_idx:>2}: {n} frames  ({os.path.basename(tp)})")
    total_frames = sum(frame_counts)
    print(f"  [{label}] Total frames     : {total_frames}")
    print(f"  [{label}] Output file size : "
          f"{total_frames * n_pairs * 4 / 1024**3:.2f} GB")

    # Step 2: pre-allocate memmap
    print(f"  [{label}] Pre-allocating {out_path} …")
    out = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float32,
        shape=(total_frames, n_pairs),
    )

    # Step 3: fill replica by replica
    boundaries, cursor = [], 0
    for rep_idx, (tp, n_frames) in enumerate(zip(traj_list, frame_counts)):
        t0 = time.time()
        print(f"\n    Rep {rep_idx:>2}  ({n_frames} frames) …")
        coords = load_heavy_coords(
            topology, tp, stride, heavy_atom_indices
        )
        print(f"      Heavy coords loaded : {time.time()-t0:.1f}s  "
              f"({coords.nbytes/1024**2:.0f} MB)")
        residue_com_distances_into(coords, res_slices, pair_indices, out, cursor)
        del coords
        print(f"      Distances done      : rows {cursor}–{cursor+n_frames-1}")
        boundaries.append([cursor, cursor + n_frames])
        cursor += n_frames

    # Step 4: flush
    out.flush()
    del out
    print(f"\n  [{label}] All replicas written → {out_path}")
    return np.array(boundaries, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("STEP 01 — Whole-Residue COM Distance Extraction (memmap)")
    print("=" * 65)
    print("  Descriptor : whole-residue COM distance [Å]")
    print("               (all heavy atoms — backbone + side-chain)")
    print(f"  Pairs      : loaded from {CONTACTS_XLSX}")
    if os.path.realpath(STATE1_TOPOLOGY) == os.path.realpath(STATE2_TOPOLOGY):
        print("\nWARNING — SHARED TOPOLOGY")
        print("  Both states are configured to use the same topology file.")
        print("  This is valid only if every trajectory has the same atom count")
        print("  and atom ordering as that topology.")

    # Replica inventory
    print(f"\n[1/5] Replica inventory:")
    print(f"  {STATE1_LABEL} : {N_REPLICAS_STATE1} replicas")
    for i, p in enumerate(STATE1_TRAJS):
        print(f"    [{i}] {p}")
    print(f"  {STATE2_LABEL} : {N_REPLICAS_STATE2} replicas")
    for i, p in enumerate(STATE2_TRAJS):
        print(f"    [{i}] {p}")

    # Build topology-specific selections from the same ordered contact list.
    print(f"\n[2/5] Building whole-residue heavy-atom selections …")
    print(f"  {STATE1_LABEL} topology: {STATE1_TOPOLOGY}")
    state1_features = build_residue_selections_from_xlsx(
        mda.Universe(STATE1_TOPOLOGY), CONTACTS_XLSX, STATE1_TOPOLOGY
    )
    print(f"  {STATE2_LABEL} topology: {STATE2_TOPOLOGY}")
    state2_features = build_residue_selections_from_xlsx(
        mda.Universe(STATE2_TOPOLOGY), CONTACTS_XLSX, STATE2_TOPOLOGY
    )
    verify_same_contact_order(state1_features, state2_features)

    pairs_resids = state1_features["pairs_resids"]
    pair_labels = state1_features["pair_labels"]
    n_pairs = len(state1_features["pair_indices"])
    print(f"\n  Contact pairs      : {n_pairs}")
    print(f"  {STATE1_LABEL} selected heavy atoms : "
          f"{len(state1_features['heavy_atom_indices'])}")
    print(f"  {STATE2_LABEL} selected heavy atoms : "
          f"{len(state2_features['heavy_atom_indices'])}")

    # Extract configured state 1
    t0 = time.time()
    state1_path = os.path.join(
        OUT_DIR, f"descriptors_{STATE1_LABEL.lower()}.npy"
    )
    print(f"\n[3/5] Extracting — {STATE1_LABEL} …")
    bounds_state1 = process_state(
        STATE1_TRAJS, STATE1_TOPOLOGY, state1_features,
        STRIDE_TRAIN, STATE1_LABEL, state1_path,
    )
    print(f"  [{STATE1_LABEL}] Wall time : {(time.time()-t0)/60:.1f} min")

    # Extract configured state 2
    t0 = time.time()
    state2_path = os.path.join(
        OUT_DIR, f"descriptors_{STATE2_LABEL.lower()}.npy"
    )
    print(f"\n[4/5] Extracting — {STATE2_LABEL} …")
    bounds_state2 = process_state(
        STATE2_TRAJS, STATE2_TOPOLOGY, state2_features,
        STRIDE_TRAIN, STATE2_LABEL, state2_path,
    )
    print(f"  [{STATE2_LABEL}] Wall time : {(time.time()-t0)/60:.1f} min")

    # Save metadata
    print(f"\n[5/5] Saving metadata …")
    np.save(os.path.join(OUT_DIR, "pairs_resids.npy"),            pairs_resids)
    np.save(os.path.join(OUT_DIR, "pairs_labels.npy"),            pair_labels)
    np.save(os.path.join(
        OUT_DIR, f"replica_boundaries_{STATE1_LABEL.lower()}.npy"
    ), bounds_state1)
    np.save(os.path.join(
        OUT_DIR, f"replica_boundaries_{STATE2_LABEL.lower()}.npy"
    ), bounds_state2)

    arr_state1 = np.load(state1_path, mmap_mode="r")
    arr_state2 = np.load(state2_path, mmap_mode="r")
    print(f"  descriptors_{STATE1_LABEL.lower()}.npy  {arr_state1.shape}  "
          f"({arr_state1.nbytes/1024**3:.2f} GB)")
    print(f"  descriptors_{STATE2_LABEL.lower()}.npy  {arr_state2.shape}  "
          f"({arr_state2.nbytes/1024**3:.2f} GB)")
    print(f"  pairs_resids.npy      {pairs_resids.shape}")
    print(f"  pairs_labels.npy      {pair_labels.shape}")
    print(f"  replica_boundaries_{STATE1_LABEL.lower()}.npy  "
          f"{bounds_state1.shape}")
    print(f"  replica_boundaries_{STATE2_LABEL.lower()}.npy  "
          f"{bounds_state2.shape}")
    del arr_state1, arr_state2
    print("\nSTEP 01 COMPLETE.\n")


if __name__ == "__main__":
    main()
