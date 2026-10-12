# Handoff: where the project stands and how to continue

Written 2026-10-12. A new Claude Code session (or a teammate) can resume from this file.

## How to get back to the Engineer

- **Same conversation:** reopen the Claude desktop app, open the Code tab, and select this session in
  the sidebar (its folder is `Retinopathy`). Type your message there; the whole history is kept.
- **New conversation:** open a new session on the folder
  `C:\Users\abhir\Downloads\MJ\github\Retinopathy` and paste:

  > You are the Software Engineer for DR-XAI. Read `docs/HANDOFF.md`, then `docs/DESIGN.md` and
  > `docs/DECISIONS_LOG.md`, and continue from the "Next steps" list. Do not invent metrics.

- **After a Colab run**, say: "Colab run finished, run id `<id>`, the folder is in `checkpoints\`."
  If a Colab cell failed, paste the full error text.

## State

| Area | State |
|---|---|
| Data | APTOS 2019 prepared at 512 px and 256 px; 3,504 images after duplicate handling (train 2,452 / val 526 / test 526). Split locked in `data/splits/`. IDRiD and Messidor-2 **not available**. |
| Specified model (MaxViT-Tiny, 512 px) | **Not trained.** Needs a GPU: `docs/TRAINING_ON_GPU.md` (Colab bundle and notebook ready, untested on Colab). |
| CPU fallback model (ConvNeXt-Tiny, 256 px) | Trained, 20 epochs: run `20261011-222646_convnext_tiny_oversample_seed42`. Tables 4 and 6 in `results/` come from it and are labelled as a fallback. |
| Ablation | 5-epoch arms of the fallback model; see `results/ablation_minority.csv` when present. |
| Explainability | Fidelity and agreement metrics done. Explainers: see `docs/DECISIONS_LOG.md` for what is implemented. No fidelity numbers (no IDRiD). |
| Reviews | Tester: `docs/TEST_REPORT.md`, `docs/DEFECTS.md`. Researcher: `docs/SCIENTIFIC_REVIEW.md`. Issues: `docs/ISSUE_LOG.md`. Paper corrections are open and belong to the authors. |
| Git | All work is committed locally on `main`. **Not pushed**: GitHub has only the initial commit. |

## Commands

```
.venv\Scripts\python.exe -m pytest tests                      # full test suite
.venv\Scripts\python.exe -m xaidr.cli evaluate --run <run id>
.venv\Scripts\python.exe -m xaidr.cli ablation --run <oversample run> --against <none run>
```

A run trained elsewhere is used by copying its folder into `checkpoints\` and passing its id.

## Next steps

1. Train the specified model on Colab; copy the run folder back; `evaluate` is already run there.
2. Get IDRiD (grading + segmentation). Then: `prepare-data --datasets idrid`, the `xai` step
   (Table 5) and the cross-dataset evaluation.
3. Get Messidor-2 and declare its label mapping in the config (F8).
4. Decide on seeds 43 and 44 (ISSUE-013) and the grading-only baseline (ISSUE-017).
5. Notebooks, `all` step, release tag, zip.
6. Push to GitHub.
