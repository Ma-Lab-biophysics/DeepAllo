# DeepAllo: probing protein allostery with molecular dynamics and deep learning

This tool extracts descriptors from MD trajectories and trains a deep learning model. 
The example tests a small-molecule modulator (Mava) effect on the myosin motor domain. 
Apo is state 1 and Mava is state 2.

## Directory layout

```text
DeepAllo/
├── scripts/
│   ├── 01_extract_descriptors.py
│   ├── 02_train.py
│   ├── 03_post_training_analysis.py
│   ├── 04_apply_cv.py
│   └── config.py
├── environment.yml
├── run_pipeline.sh
├── descriptors/                  # contact descriptors used in the paper
│   ├── Apo_vs_Mava_features.dat
│   └── Apo_vs_OM_features.dat
├── examples/                     # strided demo data only
│   └── Trajs/
│       ├── Apo/
│       ├── Mava/
│       └── OM/
└── weights_files/
    ├── Model_Apo_vs_Mava/
    │   ├── deeplda_model.pt
    │   └── deeplda_model_full.pt
    └── Model_Apo_vs_OM/
        ├── deeplda_model.pt
        └── deeplda_model_full.pt
```

## Install dependencies

Create the pinned `deep-allo` Conda environment from the supplied environment
file:

```bash
conda env create -f environment.yml
conda activate deep-allo
```

Activate this environment in each new terminal before using the pipeline:

```bash
conda activate deep-allo
```


## Configure the input states

All settings that normally need to be changed are grouped in the
`USER CONFIGURATION` block near the top of `scripts/config.py`. By default,
the pipeline uses the bundled trajectories under `examples/Trajs/`. Set the
state folders, labels, `FEATURES_FILE`, and `CRYSTAL_NUMBERING_OFFSET` in this
block to select a comparison.

To generate new feature lists, first use GetContacts to calculate residue-level
contact frequencies from each state. Then run `descriptors/preprocess_contacts.py`
as described in `descriptors/README.md`.

## Workflow A: train a new model from scratch

Run the complete current pipeline with:

```bash
bash run_pipeline.sh
```

The canonical steps are:

| Step | Script | Purpose |
|---|---|---|
| 01 | `scripts/01_extract_descriptors.py` | Extract Apo/Mava contact-distance descriptors |
| 02 | `scripts/02_train.py` | Train a new model and write it under `models/` |
| 03 | `scripts/03_post_training_analysis.py` | Analyze the resulting model |
| 04 | `scripts/04_apply_cv.py` | Optional application to another simulation |

NVIDIA CUDA GPUs are supported and can be enabled by setting
`TRAIN_ACCELERATOR = "cuda"` in `scripts/config.py`. On Apple Silicon, CPU execution is
recommended because the MPS backend does not support the required
eigendecomposition; set `TRAIN_ACCELERATOR = "cpu"`.

Individual steps can be run explicitly:

```bash
bash run_pipeline.sh 01
bash run_pipeline.sh 02
bash run_pipeline.sh 03
```

Step 02 writes `models/deeplda_model.pt` and
`models/deeplda_model_full.pt`. It does not alter the archived files under
`weights_files/`, but it will overwrite prior models with the same names
under `models/`. Copy any locally trained model that must be retained before
running Step 02 again.

## Workflow B: reproduce results from the archived model

This workflow never invokes Step 02 and therefore does not retrain a model.

First generate the descriptors:

```bash
python scripts/01_extract_descriptors.py
```

Then analyze the supplied state dictionary directly:

```bash
python scripts/03_post_training_analysis.py \
  --model weights_files/Model_Apo_vs_Mava/deeplda_model.pt
```

The ordered feature definitions used for the two archived DeepAllo runs are
supplied as the plain-text files `descriptors/Apo_vs_Mava_features.dat` and
`descriptors/Apo_vs_OM_features.dat`. Step 03 generates crystal-numbered labels
by adding `CRYSTAL_NUMBERING_OFFSET` to each simulation residue number. Set the
state folders, labels, `FEATURES_FILE`, and numbering offset in the
`USER CONFIGURATION` block of `scripts/config.py` to select a comparison.

Archived weights require the corresponding supplied DAT feature order. If a
feature list is regenerated from GetContacts frequencies, use
`--match-archived-order` as documented in `descriptors/README.md`. Step 03
checks this requirement automatically for the supplied archived state
dictionaries and full models using `weights_files/archived_model_manifest.json`
and stops before analysis if the configured DAT order is incompatible.

## Main outputs

Step 01 writes descriptor arrays, replica boundaries, and contact labels to
`output/`. Step 02 writes newly trained weights to `models/`. Canonical Step 03
writes CV projections, normalized sensitivities, method metadata,
and a machine-readable analysis summary to `output/`, with plots under `figures/`. 

For applying the CV to another system, edit the user-settings block at the top
of `scripts/04_apply_cv.py` before running Step 04. 

## Demo data vs. published results

`examples/Trajs/` ships a strided single replica per state (every 20th
frame; 50 frames). It exists so the pipeline can be exercised end
to end in seconds, and it is not sufficient to reproduce the published results.

To reproduce the paper, obtain the full trajectories from the external archive and create `./Trajs/<state>/` locally.
Place the full trajectories and corresponding `topology.pdb` files in each
state directory, then change `TRAJS_DIR` in `scripts/config.py` from
`examples/Trajs` to `Trajs`.
