# DR-XAI: project state and handoff

**Written:** 12 October 2026, end of the first working session.
**Purpose:** everything a new chat session needs to continue without the earlier conversation.
**Project folder:** `C:\Users\abhir\Downloads\MJ\github\Retinopathy`
**Deadlines:** panel demonstration on the morning of 13 October 2026; code submission 14 October 2026.

## How to continue in a new session

Open a new Claude Code session on the folder `C:\Users\abhir\Downloads\MJ\github\Retinopathy` and paste:

> You are the Software Engineer for the DR-XAI project. Read `docs/HANDOFF.md` first, then
> `docs/DECISIONS_LOG.md`. Do not invent metrics: every number must come from a file under
> `results/` or `checkpoints/`. Ask me before starting any job that runs for more than 30 minutes.

To work in another role, say "act as tester", "act as researcher" or "act as architect"; the role
briefs are `docs/VERIFICATION_AND_RESEARCH_BRIEF.md` and `docs/DESIGN.md`.

---

## 1. What the project is

An explainable deep-learning system for diabetic retinopathy (DR): one network that says whether DR
is present and grades its severity 0-4 (No DR, Mild, Moderate, Severe, Proliferative), plus saliency
maps showing where it looked. Team: K. Poojitha, K. Reethu Sri, G. Abhiram, I. Deepthi (Batch 8,
7IT2, KITSW). Supervisor: Dr. K. Praveen Kumar.

Two deliverables exist side by side:

1. **Demo web app** (the priority for the panel): upload a fundus photo, get detection, grade and a
   heat-map.
2. **Research pipeline** for the paper: data preparation, training, evaluation, explainability,
   ablation, notebooks.

## 2. Decisions the team made (do not reopen without the user)

| Date | Decision |
|---|---|
| 12 Oct | **Everything runs locally on the laptop CPU.** Google Colab and other GPUs are not used. |
| 12 Oct | **Model of record: ConvNeXt-Tiny at 256 px** (`configs/cpu_fallback.yaml`). The model in the original design (MaxViT-Tiny, 512 px) needs about 3.5 days on this laptop and is not trained. |
| 12 Oct | **The panel demo is a web app**, not notebooks. Accuracy is secondary to having a working demonstration. Plan: `docs/DEMO_APP_BRIEF.md`. |
| 12 Oct | The oversampling ablation uses **5-epoch** runs for both arms (to save time). |
| 12 Oct | IDRiD work **waits for the user's go-ahead** (see Section 7). |

## 3. Machine and environment

- Laptop: Intel i7-1165G7 (4 cores), 16 GB RAM, **no NVIDIA GPU**. Windows 11.
- Python **3.11** in `.venv` (created with `uv`). The system Python is 3.14 and must not be used.
  Run everything as `.venv\Scripts\python.exe ...` from the project folder.
- `git` is not on PATH. Use GitHub Desktop, or its bundled git at
  `C:\Users\abhir\AppData\Local\GitHubDesktop\app-3.6.7\resources\app\git\cmd\git.exe`.
- `make` is not installed; use the `python -m xaidr.cli ...` commands.
- **The laptop sleeps on battery** (Modern Standby) and this stalled one training run. For any long
  job: charger in, lid open, sleep set to "Never" when plugged in.
- PowerShell 5.1 quirk for whoever edits files by script: `Set-Content -Encoding utf8` writes a BOM.

## 4. Data on disk

| Dataset | State |
|---|---|
| **APTOS 2019** | In `data\raw\aptos2019` (3,662 labelled images; grades 0-4: 1,805 / 370 / 999 / 193 / 295). Preprocessed at 512 px (`data\processed\dr512`) and 256 px (`data\processed\dr256`). |
| **IDRiD** | **Not extracted.** The zips are in Downloads: `A. Segmentation.zip` (81 images with lesion masks: MA 81, HE 80, EX 81, SE 40, OD 81) and `B. Disease Grading.zip` (516 graded images, two label CSVs). |
| **Messidor-2** | Not available (needs a registration at adcis.net). |

Duplicates in APTOS (measured): 123 groups of byte-identical images (251 images). 96 extra copies of
same-grade images and 62 images in 30 conflicting-grade groups were dropped. **3,504 images remain:
train 2,452, validation 526, test 526.** The split is locked by SHA-256 in `data\splits\`
(`0e0ed870...`), seed 42. Loaders refuse a manifest that disagrees with the lock.

## 5. Trained models (`checkpoints\`)

| Run id | What | Result |
|---|---|---|
| `20261011-222646_convnext_tiny_oversample_seed42` | **Model of record.** ConvNeXt-Tiny, 256 px, 20 epochs, oversampling, seed 42, 5.6 h on CPU. Best epoch 17, detection threshold 0.7896. | See Section 6 |
| `20261012-102317_convnext_tiny_none_seed42` | Ablation arm, no oversampling, 5 epochs | Used only in the ablation |
| `20261012-112520_convnext_tiny_oversample_seed42` | Ablation arm, oversampling, 5 epochs | Used only in the ablation |
| `20261012-083121_convnext_tiny_none_seed42` | Abandoned 20-epoch run (2 epochs done). Unused; about 650 MB, can be deleted. | none |

## 6. Results that exist (all for the model of record unless stated)

**Table 4, APTOS test split, n = 526** (`results\table4_detection_grading_aptos_test.csv`):

| Task | Metric | Value | 95% CI |
|---|---|---|---|
| Detection | Accuracy | 0.989 | 0.979 - 0.996 |
| Detection | Sensitivity | 0.984 | 0.969 - 0.996 |
| Detection | Specificity | 0.993 | 0.982 - 1.000 |
| Detection | AUC | 0.998 | 0.995 - 1.000 |
| Grading | Accuracy | 0.867 | 0.842 - 0.892 |
| Grading | Macro sensitivity | 0.694 | 0.641 - 0.749 |
| Grading | Macro AUC | 0.957 | 0.946 - 0.969 |
| Grading | QWK | 0.910 | 0.890 - 0.929 |

Correct per grade: No DR 268/270, Mild 33/51, Moderate 126/138, Severe 16/27, **Proliferative
13/40** (20 of them graded Moderate). The model overfits: validation loss was lowest at epoch 4.

**Ablation** (`results\ablation_minority.csv`, 5 epochs per arm, paired bootstrap, n = 526):
oversampling raised Proliferative sensitivity 0.300 -> 0.625 (+0.325, CI +0.175 to +0.475) and
lowered Severe sensitivity 0.741 -> 0.519 (-0.222, CI -0.370 to -0.074). QWK (0.903 vs 0.903),
accuracy, detection AUC and the combined grade 3-4 sensitivity did not change within the interval.
One run per arm: the paper cannot claim oversampling "improves" grading.

**Efficiency** (`results\efficiency.csv`): 28.2 M parameters, 113 MB weights, median 98 ms per image
at 256 px on the CPU (IQR 92-107 ms, 4 threads). No GPU figures.

**Other files:** `results\table6_comparative.csv` (literature rows carry verification notes),
`results\figures\` (confusion matrix, training curves, class distribution, ablation plot, 12
saliency panels), `results\tables_tex\`, notebooks `00`-`05` (executed).

**Do not exist:** `table5_xai_fidelity.csv` and `crossdataset_summary.csv` (both need IDRiD).

## 7. Demo web app (built and working)

- **Start:** double-click `run_demo.bat`, wait about 30 seconds, browser opens at
  `http://localhost:8501`. Leave the black window open.
- **Files:** `app\demo_app.py` (Streamlit 1.65.0), `src\xaidr\inference.py`
  (`load_predictor`, `predict_image`, `fundus_check`, `overlay`), `tools\make_demo_samples.py`,
  samples in `data\demo_samples\` (git-ignored), presenter script `docs\DEMO_GUIDE.md`.
- **What it shows:** DR detected / not detected, grade with name, a probability bar per grade, and
  three images (upload, preprocessed, Grad-CAM++ heat-map). Sidebar: model, measured accuracy, known
  limits, "research prototype, not a medical device". It warns when a picture does not look like a
  fundus photo.
- **Checked:** renders in a browser; predictions equal the stored test predictions (difference
  below 1e-5); about 0.5 s per image.
- **Samples:** ten test-set images chosen by rule. The app grades six correctly. Good demo order:
  Sample 9 (Proliferative, correct), Sample 1 (healthy), Sample 10 (Proliferative graded Moderate:
  an honest miss).
- **Not checked:** starting through `run_demo.bat` itself, and running with Wi-Fi switched off (it
  makes no network calls by design). **The user should try both before the panel.**
- Of the ten supplied photos in `tests\fixtures\sample_fundus\`: `sample_7`, `8`, `9` come back as
  Moderate DR, the rest as No DR; `sample_3` (very dark) triggers the non-fundus warning.

## 8. Code map

```
configs\          default.yaml (specified model), cpu_fallback.yaml (model of record), ablation, convnext
src\xaidr\
  cli.py          prepare-data, make-splits, train, evaluate, ablation, xai, crossdataset,
                  efficiency, figures, all
  config.py       YAML -> frozen dataclasses, key.subkey=value overrides, "base:" inheritance
  inference.py    single-image prediction for the app
  data\           manifest, adapters (aptos2019, idrid, messidor2, eyepacs stub), preprocess
                  (crop, CLAHE, pad to square, resize), splits (dedupe, stratified, locked),
                  dataset, augment (seeded), oversample
  models\         backbone (timm), heads, dual_head
  training\       losses, metrics (bootstrap CIs), threshold (Youden), trainer, run
  xai\            gradcam_pp, shap_explainer, lime_explainer, attention (MaxViT only),
                  fidelity (IoU, Dice, pointing, recall, chance, maximum, Holm), stability
  eval\           evaluate_grading, ablation, xai_fidelity, crossdataset, efficiency
  viz\figures.py  confusion matrix, training curves, class distribution, saliency panels
app\demo_app.py   the web app
tools\            make_demo_samples, make_notebooks, make_colab_bundle, make_colab_notebook
tests\            128 tests; `-m "not slow"` runs 126
notebooks\        00-05, generated by tools\make_notebooks.py --execute
docs\             DESIGN, DECISIONS_LOG, VERIFICATION, DATA_ACCESS, TEST_REPORT, DEFECTS,
                  ISSUE_LOG, SCIENTIFIC_REVIEW, DEMO_APP_BRIEF, DEMO_GUIDE, TRAINING_ON_GPU, HANDOFF
```

Useful commands (from the project folder):

```
.venv\Scripts\python.exe -m pytest tests -m "not slow"
.venv\Scripts\python.exe -m streamlit run app\demo_app.py
.venv\Scripts\python.exe -m xaidr.cli all --config configs\cpu_fallback.yaml --run 20261011-222646_convnext_tiny_oversample_seed42
.venv\Scripts\python.exe tools\make_notebooks.py --execute
```

Timings on this laptop: one training epoch of the fallback model is 17-23 minutes; preprocessing
APTOS is 5-6 minutes; a SHAP or LIME map is 1-2 minutes; the full test suite is about 8 minutes.

## 9. Test and quality status

- Last full run (`-m "not slow"`): **126 passed, 0 failed**, after the demo app was added.
  The two slow end-to-end tests passed on the previous commit and were not rerun since.
- `ruff check src tests tools app` and `mypy src --ignore-missing-imports`: clean.
- Tester found defects DEF-001 to DEF-008; DEF-001 to DEF-006 are fixed in code. Researcher issues
  ISSUE-013 to ISSUE-016 are addressed in code. **All are still marked OPEN** in `docs\DEFECTS.md`
  and `docs\ISSUE_LOG.md`, because closing them is the Tester's and Researcher's job after re-test.
- The Tester's probe script `docs\evidence\probes\m0_probes.py` is out of date (three probes no
  longer match the fixed code).

## 10. Pending work, in suggested order

### Before the panel (13 October morning)
1. **User:** double-click `run_demo.bat`, try the samples and one upload, once with Wi-Fi off.
   Read `docs\DEMO_GUIDE.md`.

### IDRiD (waiting for the user's go-ahead; plan already agreed in outline)
2. **Fix the IDRiD adapter first.** In `B. Disease Grading.zip` the training set (413 images) and
   the testing set (103 images) both start at `IDRiD_001`, so file names collide and the current
   adapter stops with "two files share the stem". Give the two sets separate ids (for example
   `train_IDRiD_001`, `test_IDRiD_001`) and match each label CSV to its own folder. Add a test.
3. Extract both zips into `data\raw\idrid\` (layout in `docs\DATA_ACCESS.md`), then
   `prepare-data --config configs\cpu_fallback.yaml --datasets idrid`.
4. `crossdataset --run <model of record>`: external test on 516 images, about 5 minutes.
5. `xai --run <model of record>`: Table 5 on the 81 annotated images with Grad-CAM++, SHAP, LIME.
   **Estimated 3-5 hours of CPU**; maps are cached, so it can be resumed. `--limit 10` gives a
   first look in about 30 minutes. Microaneurysm rows will be weak at 256 px by construction.
6. Re-run `figures --gallery 1` and `tools\make_notebooks.py --execute`; commit.

### Still open for the submission
7. **Push to GitHub.** Every commit is local only; GitHub shows just the initial commit.
   Use GitHub Desktop -> Push origin. Note: `docs\DESIGN.md` contains the team's names and roll
   numbers and `docs\evidence\` contains local Windows paths; check whether the repository is public.
8. Decide on seeds 43 and 44 (about 6 hours each). Without them "explanation stability" must come
   out of the abstract (ISSUE-013).
9. Decide on the grading-only baseline (ISSUE-017), one extra training run, or reword the claim.
10. Messidor-2: obtain it, declare its label mapping in the config, rerun `all --run ...`.
11. Tester and Researcher re-check the fixed items; tag `v1.0-submission`; zip the repository.

### For the authors (not code)
`docs\SCIENTIFIC_REVIEW.md` lists the paper corrections. The most serious:

- Tables 4 and 6 give Akhtar et al. [49] results that belong to Shakibania et al. [27].
- Section 6.3 misreads Chilukoti et al. [48]; it does not show a cross-dataset drop.
- Van Craenendonck et al. [51]: the confidence interval is [0.46; 0.55], and ECS is not IoU or Dice.
- IDRiD has lesion masks for 81 images, not 516; its role is described two incompatible ways.
- Grad-CAM++ is attributed to Captum, which only provides the original Grad-CAM.
- The paper must now describe **ConvNeXt-Tiny at 256 px with three explainers** (Grad-CAM++, SHAP,
  LIME), not a "hybrid CNN/ViT" at 512 px with four; attention saliency exists only for MaxViT.
- "Synthetic-data augmentation" is minority oversampling; "clinically trustworthy ... deployable" is
  not supported; ref [5] has the wrong DOI; ref [20] appears to be retracted; Dinpajhouh [50] is
  unverified (paywalled).

## 11. Things that are easy to get wrong

- Result rows are labelled "Fallback baseline, NOT the specified model" on purpose. Do not relabel
  them as the proposed framework.
- Test-set metrics were computed once on the final model. Do not tune anything on the test split.
- IDRiD and Messidor-2 are test-only. Never train or select on them.
- The 5-epoch ablation runs are compared with each other only, never with the 20-epoch run.
- `data\` is git-ignored except `data\splits\`. Checkpoints are git-ignored. Losing the laptop
  loses the trained model; consider copying `checkpoints\20261011-222646_...` somewhere safe.
- The Colab files (`colab\`, `tools\make_colab_*.py`, `docs\TRAINING_ON_GPU.md`) were never run.
  `dr_xai_colab.zip` (1.2 GB) in Downloads can be deleted.

## 12. Git state

Branch `main`, all work committed locally, working tree clean at the time of writing. Latest
commits, newest first:

```
(this file)  Handoff notes
09c14bd  Demo app: fix the page-title assertion; stop tracking the local preview config
2c87921  Demo web app: upload a fundus photo, get detection, grade and a saliency map
228cd5e  Ablation and efficiency results for the CPU fallback model; notebooks re-executed
58a6e43  `all` step, six notebooks, saliency gallery; record the local-only decision
9556e85  Cross-dataset step, end-to-end test of the later steps, docs for M2/M3
ee5108f  Explainers, Table 5 pipeline, efficiency benchmark and figures
e990463  Colab training bundle, notebook and handoff notes
028e062  Add the minority-oversampling ablation step
a8ae499  First results: CPU fallback baseline on the APTOS 2019 test split
efa3d9d  M1: dual-head model, joint loss, metrics, trainer, train and evaluate steps
```

`origin/main` on GitHub (`abhiramgaddam75/Retinopathy`) is still at the initial commit `8e30639`.
