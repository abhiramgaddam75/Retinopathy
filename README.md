# DR-XAI

Explainable AI-driven deep learning model for automated diabetic retinopathy detection and severity
grading. Major project, Batch 8, 7IT2, Department of IT, KITSW.

The full specification is [docs/DESIGN.md](docs/DESIGN.md). This repository implements it milestone
by milestone (Section 15.1).

## Status

| Milestone | Content | State |
|---|---|---|
| M0 | Scaffold, config, manifest, adapters, preprocessing, splits, data-layer tests | done, tests pass on synthetic data |
| M1 | Model, loss, trainer, training run, Table 4 / Table 6 | done for the **CPU fallback model** (ConvNeXt-Tiny, 256 px). The specified model (MaxViT-Tiny, 512 px) is not trained: it needs a GPU, and the team decided on 2026-10-12 to run locally only |
| M2 | Four explainers, fidelity on IDRiD (Table 5), cross-dataset evaluation | code done and tested on synthetic data; **no numbers**: needs IDRiD, Messidor-2 and (for attention saliency) the MaxViT model |
| M3 | Ablation, efficiency, figures, notebooks | steps and six notebooks done; run on the fallback model |
| M4 | `make reproduce`, release tag | `all` step done and tested; release tag and zip not done |

Results in `results/` come from the CPU fallback model and are labelled as such in every file. Table 5 and the cross-dataset summary do not exist: they need IDRiD and Messidor-2. See [docs/HANDOFF.md](docs/HANDOFF.md) for the current state.

## Demo web app

Double-click `run_demo.bat` (or run `python -m streamlit run app/demo_app.py`) and open
<http://localhost:8501>. Upload a fundus photograph to get DR detection, the severity grade and a
Grad-CAM++ heat-map. It runs offline on the CPU with the trained fallback model. See
[docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md).

## Setup

Python 3.10 or 3.11 only.

```
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For GPU training install the CUDA build of `torch` and `torchvision` that matches the machine
before the two `pip install` lines (see pytorch.org).

## Data

See [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md) for where to download each dataset and how to place
it under `data/raw/`.

## Pipeline

```
python -m xaidr.cli prepare-data      # adapters -> data/manifests/manifest.csv -> 512 px images and masks
python -m xaidr.cli make-splits       # stratified 70/15/15 APTOS split, locked by SHA-256
python -m xaidr.cli train   --config configs/default.yaml      # M1
python -m xaidr.cli evaluate --run <run_id>                    # M1
python -m xaidr.cli xai     --run <run_id>      # Table 5 (needs IDRiD)
python -m xaidr.cli crossdataset --run <run_id> # in-domain vs external
python -m xaidr.cli train minority.mode=none                   # second arm of the ablation
python -m xaidr.cli ablation --run <oversample run_id> --against <none run_id>
python -m xaidr.cli efficiency --run <run_id>
python -m xaidr.cli figures --run <run_id> --gallery 1
python -m xaidr.cli all --run <run_id>          # every step for an existing run
python tools/make_notebooks.py --execute        # rebuild and run the six notebooks
```

Any config value can be overridden on the command line as `key.subkey=value`, for example
`python -m xaidr.cli make-splits seed=43 --resplit`. With `make` installed, the same steps are
`make prepare`, `make train`, ... (see the `Makefile`).

## Tests

```
python -m pytest tests
```

## Documents

- [docs/DESIGN.md](docs/DESIGN.md): architecture and engineering brief
- [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md): datasets, access, folder layout
- [docs/VERIFICATION.md](docs/VERIFICATION.md): outcome of every **[VERIFY]** item
- [docs/DECISIONS_LOG.md](docs/DECISIONS_LOG.md): decisions and deviations, with reasons
