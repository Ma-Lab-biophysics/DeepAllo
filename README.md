# DeepAllo: probing protein allostery with molecular dynamics and deep-learning-based analysis

This tool extracts residue center-of-mass distance descriptors from
MD trajectories and trains a deep learning model. 
The example tests a small-molecule modulator (Mava) effect on the myosin motor domain. 
Apo is state 1 and Mava is state 2.

## Directory layout

```text
DeepAllo/
├── 01_extract_descriptors.py
├── 02_train.py
├── 03_post_training_analysis.py
├── 04_apply_cv.py
├── config.py
├── environment.yml
├── run_pipeline.sh
├── Apo_vs_Mava_contacts.xlsx
├── Apo_vs_Ome_contacts.xlsx
├── Trajs/                        # trajectories are archived on Zenodo
│   ├── Apo/topology.pdb
│   ├── Mava/topology.pdb
│   └── Ome/topology.pdb
└── weights_files/
    ├── Model_Apo_vs_Mava/
    │   ├── deeplda_model.pt
    │   └── deeplda_model_full.pt
    └── Model_Apo_vs_OM/
        ├── deeplda_model.pt
        └── deeplda_model_full.pt
```

`config.py` uses `Trajs/Apo/topology.pdb` as the shared topology and discovers
the Apo and Mava trajectories under `Trajs/Apo/` and `Trajs/Mava/`. 

Two archived models are supplied. `Model_Apo_vs_Mava` uses the 46 contact
descriptors in `Apo_vs_Mava_contacts.xlsx`; `Model_Apo_vs_OM` uses the 69
descriptors in `Apo_vs_Ome_contacts.xlsx`. Set `MAVA_DIR`, `LABEL_MAVA` and
`CONTACTS_XLSX` in `config.py` to select which comparison to run.

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
| 01 | `01_extract_descriptors.py` | Extract Apo/Mava contact-distance descriptors |
| 02 | `02_train.py` | Train a new model and write it under `models/` |
| 03 | `03_post_training_analysis.py` | Analyze the resulting model |
| 04 | `04_apply_cv.py` | Optional application to another simulation |

NVIDIA CUDA GPUs are supported and can be enabled by setting
`TRAIN_ACCELERATOR = "cuda"` in `config.py`. On Apple Silicon, CPU execution is
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

Use this workflow for manuscript/SI verification. It never invokes Step 02 and
therefore does not retrain or overwrite a model.

First generate the descriptors:

```bash
python 01_extract_descriptors.py
```

Then analyze the supplied state dictionary directly:

```bash
python 03_post_training_analysis.py \
  --model weights_files/Model_Apo_vs_Mava/deeplda_model.pt
```

## Main outputs

Step 01 writes descriptor arrays, replica boundaries, and contact labels to
`output/`. Step 02 writes newly trained weights to `models/`. Canonical Step 03
writes CV projections, normalized sensitivities, method metadata,
and a machine-readable analysis summary to `output/`, with plots under `figures/`. 

For applying the CV to another system, edit the user-settings block at the top
of `04_apply_cv.py` before running Step 04. 
