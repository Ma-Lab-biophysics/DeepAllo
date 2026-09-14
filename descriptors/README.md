# Preparing contact features for DeepAllo

`preprocess_contacts.py` converts residue-level contact-frequency files from
GetContacts into the minimal plain-text feature files required by DeepAllo. It
creates:

- `Apo_vs_Mava_features.dat`
- `Apo_vs_OM_features.dat`

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

Before running it, place each state-specific topology and its trajectories
under `../Trajs/Apo/`, `../Trajs/Mava/`, and `../Trajs/OM/`. Each directory
must contain `topology.pdb` and one or more `.xtc` files.

```bash
bash run_getcontacts_all.sh                 # all states
bash run_getcontacts_all.sh all Apo Mava      # selected states
```

The run is resumable. A completed output is skipped, and one left truncated by
an interrupted run is detected from the frame count in its header and
regenerated rather than silently reused; incomplete files are also excluded
from the frequency step.

## 2. Build the DeepAllo feature files

From the `descriptors` directory, run:

```bash
mkdir -p generated_features

python preprocess_contacts.py \
  --apo-frequency getcontacts_output/Apo_HB_SB_freq.tsv \
  --mava-frequency getcontacts_output/Mava_HB_SB_freq.tsv \
  --om-frequency getcontacts_output/OM_HB_SB_freq.tsv \
  --output-dir generated_features \
  --match-archived-order
```

If `--output-dir` is omitted, the script writes to `generated_features/` by
default. It also refuses to write directly over either supplied archived
feature list. Set `FEATURES_FILE` to the generated file that you want to use.

The `--match-archived-order` option verifies the selected pair sets and restores
their archived ordering when the archived feature lists are available. If a
reference DAT file is absent, the script instead warns and writes the normal
deterministic frequency-based order; that output is suitable for training a new
model but not for applying archived weights. Omit `--match-archived-order` to
generate the deterministic order without trying to match an archived list.
