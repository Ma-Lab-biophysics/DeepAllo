# Preparing contact features for DeepAllo

`preprocess_contacts.py` converts two residue-level contact-frequency files
from GetContacts into one minimal plain-text feature file required by DeepAllo.

Each non-comment line contains two whitespace-separated simulation-numbered
residue labels, for example `GLU:773 LYS:144`. Each line is one ordered
DeepAllo input feature. Blank lines and lines beginning with `#` are ignored.
The row order must be preserved when a file is used with archived model
weights.

## 1. Generate contacts with GetContacts

GetContacts requires the VMD Python interface (`vmd-python`) to read the PDB
and XTC files and calculate contact geometries. Install it in a dedicated Conda environment
rather than in `deep-allo`: GetContacts is needed only for this preprocessing
stage, so keeping it separate leaves the pinned training environment
untouched.

```bash
conda create -n getcontacts -c conda-forge python=3.11 vmd-python=3.1.7 numpy -y
conda activate getcontacts

# Run this from the descriptors directory.
git clone https://github.com/getcontacts/getcontacts.git getcontacts
```

Run GetContacts and `preprocess_contacts.py` in the `getcontacts` environment,
then activate `deep-allo` before running the main DeepAllo pipeline.

Full trajectories are not distributed through GitHub. After cloning the
repository, obtain them from the external archive and place each state-specific
topology and its trajectories under `../Trajs/<state>/`. Each requested state
directory must contain `topology.pdb` and one or more `.xtc` files. The local
`Trajs/` directory should not be committed.

```bash
bash run_getcontacts_all.sh                 # all states
bash run_getcontacts_all.sh all Apo Mava    # selected states
bash run_getcontacts_all.sh all state_A state_B  # custom state directories
```

The run is resumable. A completed output is skipped, and one left truncated by
an interrupted run is detected from the frame count in its header and
regenerated rather than silently reused; incomplete files are also excluded
from the frequency step.

## 2. Build the DeepAllo feature files

From the `descriptors` directory, select any two state-frequency files and run:

```bash
python preprocess_contacts.py \
  --state1-frequency state_A_freq.tsv \
  --state2-frequency state_B_freq.tsv \
  --output generated_features/state_A_vs_state_B_features.dat
```

To regenerate and verify the archived Apo–Mava ordering, run:

```bash
python preprocess_contacts.py \
  --state1-frequency getcontacts_output/Apo_HB_SB_freq.tsv \
  --state2-frequency getcontacts_output/Mava_HB_SB_freq.tsv \
  --output generated_features/Apo_vs_Mava_features.dat \
  --match-archived-order
```

The output path is required, and the script refuses to write directly over
either supplied archived feature list. Set `FEATURES_FILE` to the generated
file that you want to use.

The `--match-archived-order` option verifies the selected pair sets and restores
their archived ordering when the archived feature lists are available. If a
reference DAT file is absent, the script instead warns and writes the normal
deterministic frequency-based order; that output is suitable for training a new
model but not for applying archived weights. By default, the reference is the
file with the same filename in `descriptors/`; use `--reference-features` to
specify a different reference. Omit `--match-archived-order` for a new model.
