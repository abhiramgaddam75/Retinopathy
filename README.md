# DR-XAI

Explainable AI-driven deep learning model for automated diabetic retinopathy detection and severity
grading. Major project, Batch 8, 7IT2, Department of IT, KITSW.

The full specification is [docs/DESIGN.md](docs/DESIGN.md). This repository implements it milestone
by milestone (Section 15.1).

## Status

| Milestone | Content | State |
|---|---|---|
| M0 | Scaffold, config, manifest, adapters, preprocessing, splits, data-layer tests | done, tests pass on synthetic data |
| M1 | Model, loss, trainer, training run, Table 4 / Table 6 | code done and tested; APTOS 2019 prepared and split. The specified model has **not** been trained (no GPU, see [docs/TRAINING_ON_GPU.md](docs/TRAINING_ON_GPU.md)); only a reduced CPU fallback run exists |
| M2 | Four explainers, fidelity on IDRiD (Table 5), cross-dataset evaluation | fidelity and agreement metrics written and tested on hand-made maps; explainers and evaluation not started: need a trained model, IDRiD, Messidor-2 |
| M3 | Ablation, efficiency, figures, notebooks | not started |
| M4 | `make reproduce`, release tag | not started |

**No results exist yet.** `results/` holds no metric until a run has produced it.

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
python -m xaidr.cli xai     --run <run_id>                     # M2
python -m xaidr.cli crossdataset --run <run_id>                # M2
python -m xaidr.cli ablation --config configs/ablation_no_oversampling.yaml   # M3
python -m xaidr.cli efficiency --run <run_id>                  # M3
python -m xaidr.cli figures --run <run_id>                     # M3
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
