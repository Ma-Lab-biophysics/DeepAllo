# DeepAllo: probing protein allostery with molecular dynamics and deep learning

This tool extracts residue center-of-mass distance descriptors from
MD trajectories and trains a deep learning model. 
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
│   ├── Apo_vs_Mava_contacts.xlsx
│   └── Apo_vs_Ome_contacts.xlsx
├── examples/                     # strided demo data only
│   └── Trajs/
│       ├── Apo/
│       ├── Mava/
│       └── Ome/
└── weights_files/
    ├── Model_Apo_vs_Mava/
    │   ├── deeplda_model.pt
    │   └── deeplda_model_full.pt
    └── Model_Apo_vs_OM/
        ├── deeplda_model.pt
        └── deeplda_model_full.pt
```

`scripts/config.py` uses `examples/Trajs/Apo/topology.pdb` as the shared
topology and discovers trajectories under `examples/Trajs/<state>/`. 

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

Two archived models are supplied: Model_Apo_vs_Mava, 
which uses the 46 descriptors in descriptors/Apo_vs_Mava_contacts.xlsx, 
and Model_Apo_vs_OM, which uses the 69 in descriptors/Apo_vs_Ome_contacts.xlsx. 
Set MAVA_DIR, LABEL_MAVA and CONTACTS_XLSX in scripts/config.py to select which comparison to run.

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

To reproduce the paper, one needs the full trajectories from Zenodo (28 Apo,
27 Mava and 28 OM replicas) and replace the contents of `examples/Trajs/<state>/`,
keeping the tracked `topology.pdb` in place.


