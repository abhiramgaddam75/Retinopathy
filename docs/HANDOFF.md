# Handoff: where the project stands and how to continue

Updated 2026-10-12. A new Claude Code session (or a teammate) can resume from this file.

## Decision on 2026-10-12: everything runs locally

The team decided not to use Google Colab or any other GPU. The laptop cannot train the model
specified in DESIGN.md in time (measured: about 3.5 days for MaxViT-Tiny at 512 px), so **the model
of record for the submission is the CPU fallback: ConvNeXt-Tiny at 256 px**
(`configs/cpu_fallback.yaml`). Consequences:

- Every result file is labelled "Fallback baseline, NOT the specified model". The paper must
  describe ConvNeXt-Tiny at 256 px, not a "hybrid CNN/ViT" at 512 px.
- Attention-based saliency cannot be reported: it exists only for the MaxViT backbone. Three
  explainers remain (Grad-CAM++, SHAP, LIME). The abstract lists four.
- The Colab bundle, notebook and instructions stay in the repository (`docs/TRAINING_ON_GPU.md`,
  `colab/`, `tools/make_colab_bundle.py`) in case a GPU becomes available. They were never run.

## How to get back to the Engineer

- **Same conversation:** reopen the Claude desktop app, open the Code tab, and select this session in
  the sidebar (its folder is `Retinopathy`). Type your message there; the whole history is kept.
- **New conversation:** open a new session on the folder
  `C:\Users\abhir\Downloads\MJ\github\Retinopathy` and paste:

  > You are the Software Engineer for DR-XAI. Read `docs/HANDOFF.md`, then `docs/DESIGN.md` and
  > `docs/DECISIONS_LOG.md`, and continue from the "Next steps" list. Do not invent metrics.

## State

| Area | State |
|---|---|
| Data | APTOS 2019 prepared at 512 px and 256 px; 3,504 images after duplicate handling (train 2,452 / val 526 / test 526). Split locked in `data/splits/`. **IDRiD and Messidor-2 not available.** |
| Model of record | ConvNeXt-Tiny, 256 px, 20 epochs, seed 42: run `20261011-222646_convnext_tiny_oversample_seed42`. |
| Results on disk | `results/table4_detection_grading_aptos_test.csv`, `table6_comparative.csv`, `figures/`, and when produced `ablation_minority.csv`, `efficiency.csv`. |
| Not produced | Table 5 (explanation fidelity) and the cross-dataset summary: both need IDRiD; Messidor-2 as well for the latter. The code for both is written and tested on synthetic data. |
| Ablation | Done: `results/ablation_minority.csv`. Two 5-epoch arms of the fallback model (`20261012-102317_convnext_tiny_none_seed42`, `20261012-112520_convnext_tiny_oversample_seed42`), paired bootstrap on the 526 test images. Oversampling raised Proliferative sensitivity and lowered Severe sensitivity; QWK and detection AUC did not change. One run per arm. |
| Efficiency | Done on the laptop CPU: `results/efficiency.csv` (28.2 M parameters, median 98 ms per image at 256 px, 4 threads). No GPU figures. |
| Reviews | Tester: `docs/TEST_REPORT.md`, `docs/DEFECTS.md`. Researcher: `docs/SCIENTIFIC_REVIEW.md`. Issues: `docs/ISSUE_LOG.md`. Paper corrections are open and belong to the authors. |
| Git | All work is committed locally on `main`. **Not pushed**: GitHub has only the initial commit. |

## Commands (from the project folder)

```
.venv\Scripts\python.exe -m pytest tests -m "not slow"         # test suite without the slowest tests
.venv\Scripts\python.exe -m xaidr.cli all --config configs\cpu_fallback.yaml --run <run id>
.venv\Scripts\python.exe -m xaidr.cli xai --config configs\cpu_fallback.yaml --run <run id>
.venv\Scripts\python.exe tools\make_notebooks.py --execute
```

`all --run <run id>` reruns every step for an existing trained run on whatever datasets are under
`data\raw`; without `--run` it trains a new model first.

## Next steps

1. Get IDRiD (grading + segmentation parts) into `data\raw\idrid` (layout in `docs/DATA_ACCESS.md`),
   then run `all --run <run id>`: this adds Table 5 and the IDRiD rows of the cross-dataset summary.
   Expect the SHAP and LIME maps for 81 images to take a few hours on the laptop.
2. Get Messidor-2 and declare its label mapping in the config (F8); rerun `all --run <run id>`.
3. Decide on seeds 43 and 44 (ISSUE-013, about 6 hours each) and the grading-only baseline
   (ISSUE-017). Without the seeds, "explanation stability" must leave the abstract.
4. Authors: apply the paper corrections in `docs/SCIENTIFIC_REVIEW.md`.
5. Tester and Researcher re-check the fixed defects and issues; then tag `v1.0-submission` and zip.
6. Push to GitHub (GitHub Desktop: Push origin).
