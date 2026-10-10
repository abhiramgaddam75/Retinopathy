# DR-XAI: Architecture & Engineering Brief

**Project:** Explainable AI-Driven Deep Learning Model for Automated Diabetic Retinopathy Detection and Severity Grading
**Team (Batch 8, 7IT2, KITSW Dept. of IT):** K. Poojitha (B23IT082), K. Reethu Sri (B23IT101), G. Abhiram (B23IT107), I. Deepthi (B23IT119)
**Supervisor:** Dr. K. Praveen Kumar
**Code submission deadline:** 14 October 2026
**Document produced by:** Designer/Architect role (Claude Haiku 5.5), 10 October 2026
**Intended reader:** Developer/Software Engineer role (Claude Opus or highest available model) in Claude Code
**Version:** 1.0 (design freeze candidate; decisions marked DEFAULT may be changed by the team before the Engineer starts)

---

## 0. START HERE: How to use this document

1. Create an empty folder, for example `~/xai-dr/`, and open Claude Code in it.
2. Save this file as `docs/DESIGN.md` inside that folder.
3. Paste the prompt below as your first message. The Engineer will read this document and then ask you for any missing items (dataset files, GPU, confirmation of open decisions).

> **Prompt to paste into Claude Code (Software Engineer role):**
>
> You are the **Software Engineer (Developer role)** for the DR-XAI project. Your sole job is to implement the project exactly as specified in `docs/DESIGN.md`. Read the whole file before writing any code. Do not change the architecture, file layout, metrics or paper-to-output mapping without asking me first. Work in the milestone order in Section 15. After each milestone, run the tests in Section 13 and report pass/fail. Never invent, estimate or copy metrics: every number you report must come from a run you executed, and must be saved to the `results/` path listed in Section 11. Where this document says **[VERIFY]**, check the installed library's actual API or documentation and record the outcome in `docs/VERIFICATION.md`. If a dataset file is missing or a column name differs from this document, stop and ask me; do not guess. Use Python 3.10 or 3.11 only, PyTorch and timm for models, and Jupyter notebooks for visualisation. Commit to git after every milestone with a clear message. Before you start coding, tell me which open decisions in Section 5 you accept as DEFAULT and which you want changed.

---

## 1. Project at a glance

| Item | Specification |
|---|---|
| Task | (a) Binary DR detection (DR present vs absent); (b) five-class ICDR severity grading (0 No DR, 1 Mild, 2 Moderate, 3 Severe, 4 Proliferative) |
| Model | One shared backbone, two task heads (detection, grading), jointly trained |
| Explainability | Four saliency methods: Grad-CAM++, SHAP, LIME, attention-based saliency |
| Explanation validation | Quantitative against expert lesion masks (MA, HE, EX, SE) using IoU, Dice, pointing game |
| Generalisation | Train on APTOS 2019; test in-domain on APTOS 2019; external tests on IDRiD and Messidor-2 |
| Class imbalance | Minority oversampling with augmentation for Severe (3) and Proliferative (4); ablation with and without |
| Language | Python 3.10/3.11 |
| Outputs | Results CSVs mapped to paper Tables 4, 5, 6; cross-dataset, ablation and efficiency results; figures; Jupyter notebooks; test suite; reproducibility bundle |

---

## 2. Scope for the 14 October deadline

**MUST (required for submission):**
- Data adapters for APTOS 2019, IDRiD (grading labels and lesion masks) and Messidor-2
- Preprocessing (border crop, CLAHE, resize to 512) applied identically to images and masks
- Reproducible stratified splits, saved and locked
- Dual-head model, joint loss, training loop with early stopping, checkpointing
- Grading and detection evaluation (paper Table 4 and Table 6, "Proposed framework" row)
- Four XAI methods and fidelity evaluation against IDRiD masks (paper Table 5); if attention saliency cannot be made to work with the chosen backbone, document the reason and use the ViT fallback (Section 5, D1)
- External evaluation on IDRiD and Messidor-2 (paper Section 6.3)
- Augmentation ablation, with and without minority oversampling (paper Section 6.4)
- Efficiency benchmark (paper Section 6.5)
- Jupyter notebooks, unit tests, a `make reproduce` target, and a results bundle

**SHOULD (if time allows):** bootstrap confidence intervals on all headline metrics (required for a journal, so do this before the EyePACS work); three random seeds for the primary model; a quantized/pruned variant for Section 6.5.

**OUT OF SCOPE for 14 October (keep the code stubs, disabled by config, and list as future work):**
- EyePACS pretraining and the EyePACS secondary test set (80–90 GB download; leakage risk, see Section 8.5)
- Generative synthetic images (GAN or diffusion). The "synthetic" augmentation in this build is minority oversampling with augmentation (see D2)
- Federated learning, other retinal diseases, prospective clinical validation

---

## 3. Source audit: what the project files say and what I found

I reviewed the abstract, the presentation, the two Word guides, the four supplied research papers (Alwakid et al. 2023; Bashir et al. 2023; Ishtiaq et al. 2023; Alavee et al. 2024), and the supplied sample submission and ten fundus images. Findings that change the design:

| ID | Source | Finding | Severity | Resolution in this design |
|---|---|---|---|---|
| F1 | Paper Sec. 4.1, Table 2 | Paper states IDRiD has "516 fundus images with pixel-level expert masks". **Verified against the IDRiD site:** pixel-level lesion annotations exist for **81** segmentation images only (MA on 81, EX on 81, HE on 80, SE on 40). The 516 images are the grading set without masks. | **Critical** | Fidelity evaluation uses the 81 annotated images. Paper text must be corrected. Sample sizes are reported in Table 5. |
| F2 | Paper Sec. 4.1 vs Sec. 5.3 | Sec. 4.1 says IDRiD is used "exclusively" for fidelity validation. Sec. 5.3 says IDRiD is split 70/15/15 for grading. The two contradict each other. | High | DEFAULT D3: IDRiD is an external test set (grading and fidelity) and is never used for training. Paper Sec. 5.3 must be rewritten. |
| F3 | Paper Sec. 4.4, Sec. 5.1, Presentation slide 6 | Paper says Captum implements Grad-CAM++. Captum's `LayerGradCam` is the original Grad-CAM. Grad-CAM++ is a different method. **[VERIFY]** | High | Use a library that provides Grad-CAM++ explicitly (`pytorch-grad-cam`, class `GradCAMPlusPlus`) or an own implementation with a unit test. Correct the paper's attribution. |
| F4 | Paper Sec. 4.4 vs execution guide | Paper says SHAP uses a "masked-image kernel explainer". The execution guide uses `GradientExplainer`. | Medium | Use a model-agnostic partition/kernel explainer over a coarse segment grid (Section 10.2). Paper wording is kept. |
| F5 | Paper Sec. 4.3, Table 3 | Paper says "hybrid CNN/ViT" backbone. Execution guide uses ConvNeXt-Tiny (pure CNN). Its config comment suggests ViT-B/16 at 224 px, which contradicts the 512 px input in Table 3. | High | DEFAULT D1: `maxvit_tiny_tf_512` (hybrid conv + attention, native 512 input) **[VERIFY name in installed timm]**. ConvNeXt-Tiny is the baseline and fallback. |
| F6 | Paper Sec. 4.4 | Attention-based saliency is undefined for a pure CNN. | High | Resolved by F5. Attention saliency is only computed for hybrid or ViT backbones. |
| F7 | Paper Sec. 4.2, Presentation slide 7 | Paper says "synthetic-data augmentation". Execution guide only duplicates minority-class rows and applies random augmentations. That is oversampling, not synthetic data. | High (scientific wording) | Implement oversampling-with-augmentation as the ablated method (D2). Researcher must decide whether to rename the method in the paper or add a generative method. |
| F8 | Paper Sec. 4.1, Table 2 | Messidor-2 is described as having 5-class ICDR grades. Public Messidor-2 grade labels use a coarser scale (adjudicated referable/non-referable and a 0–3 style scale depending on the release). **[VERIFY against the downloaded label file]** | High | Label-space mapping declared per dataset in config (Section 8.3). Five-class grading is reported on Messidor-2 only where the mapping is exact; detection is reported on the mapped definition, never mixed with APTOS numbers. |
| F9 | Paper Sec. 4.6 | Cross-dataset explanation stability is said to be computed "against the IDRiD-trained saliency baselines". Messidor-2 has no lesion masks, so IoU/Dice/pointing cannot be computed there. | High | Lesion-based fidelity is reported on IDRiD only. Explanation stability on Messidor-2 uses a defined agreement metric (Section 10.5), flagged for Researcher approval. |
| F10 | Paper Sec. 4.6 vs Sec. 4.1 | Plan pretrains on EyePACS and then tests on a "held-out EyePACS subset". If the pretraining set is not split, this is leakage. | High | EyePACS is out of scope for 14 October (Section 2). |
| F11 | Paper Sec. 4.5, 6.2 | IoU/Dice use a threshold on the saliency map that is not specified. A fixed threshold on min-max normalised maps is arbitrary. No chance baseline is given. | High | Top-k% thresholding (D6) with sensitivity at two other k values, and a chance baseline computed for every metric (Section 10.3). |
| F12 | Paper Sec. 6.3 vs Table 4 | Section 6.3 says the EyePACS-pretrained model reached QWK 0.854 on Messidor-2 and quotes 0.967 for the APTOS result. Table 4 gives Chilukoti et al. QWK 0.944 for Messidor. These need reconciling against the source. | Medium | Researcher to check against Chilukoti et al. [48]. |
| F13 | Paper references | Refs [4] and [5] share the DOI `10.1038/s41598-026-58292-6`. Ref [5] names Graefe's Archive but carries a Sci. Rep.-style DOI. Refs [3], [21], [37] are incomplete. | Medium | Researcher to fix all references before submission. |
| F14 | Paper Sec. 2.5, Table 1 | Claim that no reviewed study combines all three dimensions is strong and depends on a complete literature search. | Medium | Researcher to verify or soften the claim. |
| F15 | Execution guide | Contains at least 15 code defects (Section 16). | High | Engineer must not copy the guide's code; use this design's module specifications. |
| F16 | Sample submission CSV | 1,928 IDs with placeholder labels (all 0). These are the APTOS test-set IDs; the test labels are not public. | Info | Not usable for evaluation. The in-domain test set is a stratified 15% of `train.csv`. |
| F17 | 10 sample fundus photos (supplied, unlabelled) | Mixed colour and illumination, one or more with visible artefacts. No labels supplied. | Info | Use as smoke-test fixtures only (`tests/fixtures/`), subject to licence check. |

---

## 4. Architecture overview

### 4.1 Pipeline

```
 RAW DATA                  PREPARE                    TRAIN                     EXPLAIN & VALIDATE           REPORT
 ────────                  ───────                    ─────                     ───────────────────          ──────
 APTOS 2019  ─┐
 IDRiD       ─┼─► adapters ─► manifest.csv ─► crop ─► CLAHE ─► resize 512 ─┐
 Messidor-2  ─┘   (per dataset)    │                                          │
                                   └─► splits (70/15/15, stratified, locked) │
                                                                             ▼
                              Dataset ─► train-only augment (+ minority oversampling) ─► Shared backbone (hybrid, 512)
                                                                             │
                                                    ┌────────────────────────┴────────────────────────┐
                                                    ▼                                                 ▼
                                         Detection head (1 logit)                     Grading head (5 logits)
                                                    └──────────── joint loss L = 0.4·L_det + 0.6·L_grade ─┘
                                                                             │
                                  best checkpoint (selected on validation: 0.5·QWK + 0.5·det-AUC, early stopping)
                                                                             │
          ┌──────────────────────────────┬────────────────────────────┬────────┴─────────┬───────────────────────┐
          ▼                              ▼                            ▼                   ▼                       ▼
 Grading/detection metrics     Explainers (Grad-CAM++,       Cross-dataset eval     Ablation (none vs      Efficiency
 APTOS test (Table 4, 6)       SHAP, LIME, attention)        IDRiD, Messidor-2      oversampling)          (params, latency,
                               ─► fidelity vs IDRiD masks    (Sec. 6.3)             (Sec. 6.4)             memory, Sec. 6.5)
                               (Table 5)
                                          │
                                          ▼
                           results/*.csv  ─►  notebooks/*.ipynb  ─►  figures  ─►  paper tables
```

### 4.2 Design principles (non-negotiable)

1. **No leakage.** Splits are fixed once and saved. Augmentation, oversampling and any statistics (normalisation constants) are applied only to training data. IDRiD and Messidor-2 are never used for training, model selection or hyperparameter choice.
2. **One preprocessing path for images and masks.** Every geometric transform (border crop, resize) is recorded per image in the manifest and applied identically to lesion masks, with nearest-neighbour resampling for masks.
3. **Config-driven.** No hard-coded paths, hyperparameters or seeds in code. Every run writes a config snapshot.
4. **Honest metrics.** Every reported metric has a bootstrap confidence interval (or an explicit reason it was not computed) and a chance baseline where relevant.
5. **Traceability.** Every number in the paper's tables maps to one CSV file (Section 11).

---

## 5. Key design decisions

Each decision has a DEFAULT. The Engineer must confirm each before coding that module. If the team changes one, update this document first.

| ID | Decision | DEFAULT | Reason | Confirms |
|---|---|---|---|---|
| D1 | Primary backbone | `maxvit_tiny_tf_512` (hybrid conv + attention, native 512 input) **[VERIFY name in installed timm]**. Baseline and fallback: `convnext_tiny` (pure CNN). ViT-B/16 is not used because its 224 px config conflicts with Table 3. | Matches the paper's "hybrid CNN/ViT" and 512 px input, and gives attention maps for attention saliency. | Developer + Researcher |
| D2 | What "synthetic augmentation" means in this build | Minority-class oversampling (Severe and Proliferative, factor 3, training split only) with on-the-fly geometric and photometric augmentation. Ablated against no oversampling. | This is what can be built and measured by 14 October. Generative synthesis is future work. | Researcher (wording in paper) |
| D3 | IDRiD role | External test set for grading (all 516 grading images) and lesion-fidelity set (81 annotated images). Never used for training or selection. | Resolves F2. Keeps IDRiD clean for both uses. | Researcher |
| D4 | EyePACS | Out of scope for 14 October. Adapter stub present, disabled by config. | Resolves F10 and the 80–90 GB download risk. | Team |
| D5 | Detection label | DR present = grade ≥ 1 for APTOS and IDRiD. For Messidor-2, the detection label is whatever the mapped label file provides (Section 8.3); it is reported as a separate row and never pooled with APTOS. | Avoids mixing "any DR" and "referable DR" definitions. | Researcher |
| D6 | Saliency thresholding for fidelity | Top 10% of pixels set to 1 (`top_percent: 10`). Sensitivity at 5% and 20%. | Fixed thresholds on min-max maps are arbitrary (F11). | Researcher |
| D7 | Detection decision threshold | Chosen on the validation split by maximising Youden's J (sensitivity + specificity − 1), then frozen for test and external sets. | Avoids threshold tuning on test data. | Developer |
| D8 | Confidence intervals | Stratified bootstrap, 1000 resamples, 95% percentile interval, on every test and external metric. Paired bootstrap for the ablation. | Journal standard. | Developer |
| D9 | Model selection | Best validation epoch by score = 0.5 × QWK + 0.5 × detection AUC. Early stopping, patience 8. | Execution guide selected by validation loss, which mixes task weights. | Developer |
| D10 | Epochs and batch | 30 epochs maximum. Batch 16 with gradient accumulation 2 (effective 32, as in paper Table 3), AMP on. | Time budget. 40 epochs in the guide is too long for the deadline. | Developer |
| D11 | Explained class | Grading head: explain the predicted grade (argmax). Detection head: not explained in this build. | Paper explains the grading output. | Researcher |
| D12 | Seeds | One seed (42) for all headline results. Three seeds (42, 43, 44) if GPU time allows; report mean ± SD. | Time budget. | Developer |

---

## 6. Repository layout

```
dr-xai/
├── README.md                          # how to reproduce; links to this design doc
├── Makefile                           # make setup | prepare | train | evaluate | xai | crossdataset | ablation | efficiency | figures | test | reproduce
├── pyproject.toml                     # package metadata, Python >= 3.10
├── requirements.txt                   # pinned versions (see Section 14.1)
├── .gitignore                         # ignores data/, checkpoints/, results/ (except results/ summaries), *.pt
├── configs/
│   ├── default.yaml                   # primary run (Section 9.1)
│   ├── ablation_no_oversampling.yaml  # minority.mode = none
│   ├── ablation_oversampling.yaml     # minority.mode = oversample (same as default)
│   └── backbone_convnext.yaml         # baseline backbone
├── docs/
│   ├── DESIGN.md                      # this file
│   ├── VERIFICATION.md                # outcome of every [VERIFY] item
│   ├── DATA_ACCESS.md                 # where to download and how to place each dataset
│   └── DECISIONS_LOG.md               # any change to Section 5, with date and reason
├── src/xaidr/
│   ├── __init__.py
│   ├── cli.py                         # single entry point: python -m xaidr.cli <step>
│   ├── config.py                      # load + validate YAML into dataclasses
│   ├── seed.py                        # set seeds for python, numpy, torch
│   ├── logging_utils.py               # run folders, JSON metric logs, env snapshot
│   ├── data/
│   │   ├── __init__.py
│   │   ├── manifest.py                # canonical manifest schema + read/write + validation
│   │   ├── adapters/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # DatasetAdapter protocol
│   │   │   ├── aptos2019.py
│   │   │   ├── idrid.py               # grading labels + lesion masks (81 seg images)
│   │   │   ├── messidor2.py           # labels with declared mapping
│   │   │   └── eyepacs.py             # STUB, disabled (D4)
│   │   ├── preprocess.py              # FOV crop, CLAHE, resize; returns the transform record
│   │   ├── splits.py                  # stratified 70/15/15; locks split files with hashes
│   │   ├── dataset.py                 # torch Dataset classes
│   │   ├── augment.py                 # train-only transforms (albumentations)
│   │   └── oversample.py              # minority oversampling as an index-level sampler
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbone.py                # timm wrapper; exposes pooled features and feature map
│   │   ├── heads.py                   # detection head, grading head
│   │   └── dual_head.py               # DRDualHeadModel (shared backbone + both heads)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py                  # JointDRLoss (weighted BCE + class-weighted CE)
│   │   ├── metrics.py                 # all detection and grading metrics, bootstrap CIs
│   │   ├── trainer.py                 # epoch loop, AMP, accumulation, early stopping, checkpoints
│   │   └── threshold.py               # Youden threshold on validation (D7)
│   ├── xai/
│   │   ├── __init__.py
│   │   ├── base.py                    # Explainer protocol: map(img, class) -> HxW in [0, 1]
│   │   ├── gradcam_pp.py
│   │   ├── shap_explainer.py
│   │   ├── lime_explainer.py
│   │   ├── attention.py               # attention saliency (hybrid/ViT backbones only)
│   │   ├── fidelity.py                # top-k threshold, IoU, Dice, pointing game, chance baseline
│   │   └── stability.py               # cross-domain explanation agreement (Section 10.5, pending approval)
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── evaluate_grading.py        # Table 4 / Table 6
│   │   ├── xai_fidelity.py            # Table 5
│   │   ├── crossdataset.py            # Section 6.3
│   │   ├── ablation.py                # Section 6.4 (paired bootstrap)
│   │   └── efficiency.py              # Section 6.5
│   └── viz/
│       ├── __init__.py
│       └── figures.py                 # confusion matrix, saliency overlays, training curves
├── notebooks/
│   ├── 00_architecture_and_graph.ipynb
│   ├── 01_data_audit.ipynb
│   ├── 02_training_curves.ipynb
│   ├── 03_results_tables.ipynb
│   ├── 04_xai_gallery.ipynb
│   └── 05_ablation_and_efficiency.ipynb
├── tests/
│   ├── fixtures/                      # the 10 supplied sample fundus photos + tiny CSVs (subject to licence check)
│   ├── test_manifest.py
│   ├── test_adapters.py
│   ├── test_preprocess.py             # mask/image transform consistency
│   ├── test_splits.py                 # stratification, disjointness, locking
│   ├── test_no_leakage.py             # IDRiD/Messidor never in train; augmentation only in train
│   ├── test_model.py                  # output shapes, parameter groups, feature map exists
│   ├── test_loss.py
│   ├── test_metrics.py                # QWK, AUC, Youden against sklearn on random data
│   ├── test_fidelity.py               # IoU/Dice/pointing on hand-made maps with known answers
│   ├── test_explainers.py             # maps are HxW in [0,1]; shapes match input
│   └── test_cli_smoke.py              # 2-image end-to-end run in < 2 min on CPU
├── data/                              # NOT in git
│   ├── raw/                           # downloaded archives as delivered
│   ├── manifests/                     # manifest CSVs (generated)
│   ├── processed/dr512/               # preprocessed 512 px images and masks (generated)
│   └── splits/                        # locked split CSVs + hashes (generated)
├── checkpoints/                       # NOT in git; one folder per run id
└── results/                           # summary CSVs and figures (commit the CSVs and figures; not raw logs)
    ├── table4_detection_grading_aptos_test.csv
    ├── table5_xai_fidelity.csv
    ├── table6_comparative.csv
    ├── crossdataset_summary.csv
    ├── ablation_minority.csv
    ├── efficiency.csv
    ├── figures/
    └── env/                           # pip freeze, config snapshot, git commit hash, GPU info
```

---

## 7. Module specifications

Each module lists its responsibility, public API and acceptance criteria. Type hints are required on all public functions.

### 7.1 `config.py`
- Loads YAML into frozen dataclasses (`PathsCfg`, `DataCfg`, `ModelCfg`, `TrainCfg`, `MinorityCfg`, `XAICfg`, `EvalCfg`) and validates: `image_size` is an int ≥ 224; `split_ratios` sums to 1.0; `minority.target_grades ⊆ {0..4}`; `model.backbone` is a known name.
- Accepts command-line overrides in the form `key.subkey=value`.
- **Acceptance:** an invalid config fails with a readable message naming the offending key.

### 7.2 `seed.py`
- `set_seed(seed: int) -> None` sets `random`, `numpy`, `torch`, `torch.cuda`, and `PYTHONHASHSEED` guidance in the run log. Sets `torch.backends.cudnn.deterministic = True` when `train.deterministic: true`.

### 7.3 `data/manifest.py`
Canonical manifest, one row per image. Columns (exact names):

| Column | Type | Meaning |
|---|---|---|
| `dataset` | str | `aptos2019`, `idrid`, `messidor2` |
| `image_id` | str | Unique within dataset; the key used everywhere |
| `source_path` | str | Path to the original image |
| `grade` | int or empty | Five-class ICDR grade 0–4 when available; empty otherwise |
| `dr_label` | int or empty | Detection label after the dataset's mapping (D5) |
| `label_space` | str | `icdr5` or the dataset's declared scheme (e.g. `messidor_adjudicated`) |
| `patient_id` | str or empty | Empty when unknown; splitting then uses images as units |
| `has_masks` | bool | True when lesion masks exist for this image (IDRiD segmentation subset) |
| `mask_MA`, `mask_HE`, `mask_EX`, `mask_SE`, `mask_OD` | str or empty | Paths to binary masks at source resolution |
| `orig_h`, `orig_w` | int | Source resolution |
| `crop_y0`, `crop_x0`, `crop_y1`, `crop_x1` | int | Border-crop box in source pixels (required for mask alignment) |
| `processed_path` | str | Path to the 512 px image after preprocessing |
| `split` | str | `train`, `val`, `test`, `external_idrid`, `external_messidor2`, or empty |

- `read_manifest(path) -> pd.DataFrame` and `write_manifest(df, path)` validate every column and dtype. Missing required columns raise `ManifestError`.
- **Acceptance:** `test_manifest.py` rejects a manifest with a missing column and accepts a valid one.

### 7.4 `data/adapters/*`
Each adapter implements `DatasetAdapter`:
```python
class DatasetAdapter(Protocol):
    name: str
    def build_manifest(self, raw_dir: Path) -> pd.DataFrame: ...
```
- **`aptos2019.py`:** reads `train.csv` (columns `id_code`, `diagnosis`), images at `train_images/{id_code}.png`. `grade = diagnosis`; `dr_label = int(grade >= 1)`; `patient_id` empty; `split` assigned later by `splits.py`. Does not read `test.csv` labels (none are provided) and never uses `sample_submission.csv` as labels.
- **`idrid.py`:** reads the grading label CSV for the grading set (all 516 images) and maps the grading columns to `grade`. For the segmentation set (81 images), reads per-lesion binary masks (`.tif`) and sets `has_masks = True` only for those images. **[VERIFY]** exact image file names, mask file naming (`IDRiD_xx_MA.tif`-style), grading CSV column names (expected: `Image name`, `Retinopathy grade`), and whether any image appears in both the segmentation and grading sets. Overlap is logged and each overlapping image is flagged; it does not change the external-test role.
- **`messidor2.py`:** reads the label file with columns declared in config (`messidor2.label_file`, `messidor2.id_column`, `messidor2.grade_column`, `messidor2.grade_map`). The grade map is written into the manifest's `label_space` column. Images are joined by filename to labels. Rows without a label are dropped and counted in the build log. **[VERIFY]** column names and the grade scheme of the release downloaded.
- **`eyepacs.py`:** stub that raises `NotImplementedError("EyePACS is out of scope for this build (D4)")`.
- **Acceptance:** each adapter has a test on fixture CSVs that checks column names, dtype and that masks are only attached where `has_masks` is True.

### 7.5 `data/preprocess.py`
- `crop_fov(img_bgr, threshold=10) -> (cropped, (y0, x0, y1, x1))`: returns the crop box so masks can be cropped identically.
- `apply_clahe(img_bgr, clip_limit=2.0, tile_grid=(8,8))`: CLAHE on the L channel in LAB space.
- `resize_image(img, size)`: `cv2.INTER_AREA`. `resize_mask(mask, size)`: `cv2.INTER_NEAREST`, then re-binarise at 0.5.
- `preprocess_record(row, size) -> (img_512, masks_512 dict)`: applies crop, CLAHE to the image only (never to masks), resize; applies crop and resize to every mask in the record.
- `build_processed(manifest_df, out_dir, size) -> manifest_df`: writes processed images, fills `processed_path`, returns the updated manifest. Idempotent: skips files that exist with matching size unless `--force`.
- **Acceptance:** `test_preprocess.py` builds a synthetic image with a known bright square, applies the full pipeline to image and mask, and checks that the square's location in the processed mask matches the processed image within 1 pixel.

### 7.6 `data/splits.py`
- `make_splits(manifest_df, ratios=(0.70, 0.15, 0.15), seed=42, unit="patient_or_image") -> manifest_df`: stratified on `grade` for the APTOS 2019 subset only. Uses `patient_id` as the grouping unit when available; otherwise each image is a unit.
- Writes `data/splits/aptos2019_split.csv` and `data/splits/aptos2019_split.sha256`. If the hash file exists and the split differs, the function raises unless `--resplit` is given.
- IDRiD and Messidor-2 are assigned `external_idrid` and `external_messidor2` directly. They are never split.
- **Acceptance:** `test_splits.py` checks that train, val and test are disjoint, that class proportions match within 1 percentage point, and that re-running with the same seed reproduces the identical file.

### 7.7 `data/dataset.py`
- `DRDataset(manifest_df, transform, task_labels=True)`: returns `(image_tensor, dr_label, grade, image_id)`. Image read from `processed_path`, converted BGR→RGB, normalised with ImageNet mean and std.
- `LesionDataset(manifest_df, transform=None)`: for `has_masks == True` rows only. Returns `(image_tensor, masks_dict, grade, image_id)`. Mask tensors are at 512 px and binary.
- **Acceptance:** shapes, dtypes and label ranges are asserted in `test_model.py` and `test_no_leakage.py`.

### 7.8 `data/augment.py`
- `train_transform(size)`: horizontal flip (p 0.5), vertical flip (p 0.5), rotation ±30° (p 0.5), brightness and contrast jitter (p 0.3). Normalisation.
- `eval_transform(size)`: normalisation and tensor conversion only.
- Flips change laterality of the optic disc and fovea. This is standard for fundus DR training and is noted in the methods. Researcher to confirm.
- **Acceptance:** `test_no_leakage.py` asserts `train_transform` is only referenced by the training loader.

### 7.9 `data/oversample.py`
- `minority_sampler(manifest_df, target_grades=(3,4), factor=3, seed) -> torch.utils.data.Sampler`: builds an index list in which training rows with `grade ∈ target_grades` appear `factor` times, then shuffles. Duplicates are indices only; each draw gets a fresh random augmentation.
- When `minority.mode = none`, a plain shuffled sampler is returned.
- **Acceptance:** `test_no_leakage.py` confirms the sampler only draws training indices and that the grade distribution of the drawn indices matches the expected multiplier.

### 7.10 `models/backbone.py`
- `Backbone(name, pretrained=True, img_size=512)`: `timm.create_model(name, pretrained=pretrained, num_classes=0, img_size=img_size)` **[VERIFY that `img_size` is accepted for the chosen model; if not, pass the native size]**. Exposes `.num_features` and `forward_features(x)` returning the final feature map for Grad-CAM++ and attention saliency.
- **Acceptance:** `test_model.py` checks the output feature dimension for a random 2×3×512×512 input.

### 7.11 `models/heads.py` and `models/dual_head.py`
- `DetectionHead`: `Linear(F, 256) → ReLU → Dropout(0.3) → Linear(256, 1)`; output logit shape `(B,)`.
- `GradingHead`: `Linear(F, 256) → ReLU → Dropout(0.3) → Linear(256, 5)`; output `(B, 5)`.
- `DRDualHeadModel(cfg)`: `forward(x) -> dict(det_logit, grade_logits, pooled, feature_map)`.
- A `GradeOnly(model)` wrapper returns only `grade_logits`. Explainers use this wrapper so their outputs have a single tensor.
- **Acceptance:** `test_model.py` asserts shapes and that all parameters in both heads receive gradients after one backward pass.

### 7.12 `training/losses.py`
- `JointDRLoss(lambda_det=0.4, lambda_grade=0.6, grade_class_weights=None)`:
  `L = λ_det · BCEWithLogits(det_logit, dr_label) + λ_grade · CrossEntropy(grade_logits, grade, weight=class_weights)`.
- `class_weights` is inverse square-root class frequency computed on the training split only (DEFAULT). Ordinal loss is an optional future variant.
- **Acceptance:** `test_loss.py` checks the loss equals the hand-computed value on a small batch.

### 7.13 `training/trainer.py`
- `Trainer(model, loss, optimizer, scheduler, cfg, run_dir)` with `fit(train_loader, val_loader)`.
- AdamW, lr 1e-4, weight decay 0.05, cosine schedule with 2 warm-up epochs, AMP (`torch.autocast`, `GradScaler`), gradient accumulation (2 steps).
- Each epoch: train loss components; validation loss, QWK, detection AUC and grading accuracy, written to `run_dir/metrics.jsonl`.
- Selection score = 0.5 × QWK + 0.5 × detection AUC on validation (D9). Saves `best.pt` on improvement and `last.pt` each epoch. Early stopping with patience 8.
- Resumes from `last.pt` if `--resume` is given.
- **Acceptance:** a 2-image, 2-epoch smoke run completes and writes both checkpoints and `metrics.jsonl`.

### 7.14 `training/metrics.py`
- Detection: accuracy, sensitivity, specificity, AUC (`roc_auc_score`).
- Grading: accuracy, macro sensitivity (recall), one-vs-rest macro AUC, quadratic weighted kappa (`cohen_kappa_score(weights="quadratic")`). If a class is absent from a test set, the one-vs-rest AUC for that class is set to NaN, the macro is computed over present classes, and a warning is logged.
- `bootstrap_ci(metric_fn, y_true, y_pred_or_score, n=1000, seed=42, stratify=True) -> (estimate, lo, hi)`.
- `paired_bootstrap_diff(metric_fn, y_true, pred_a, pred_b, n=1000, seed=42) -> (diff, lo, hi)` for the ablation.
- **Acceptance:** `test_metrics.py` checks each metric against scikit-learn on random inputs with tolerance 1e-9.

### 7.15 `training/threshold.py`
- `youden_threshold(y_true, scores) -> float` picks the threshold maximising sensitivity + specificity − 1 on the validation split. The chosen value is written to `run_dir/threshold.json` and reused unchanged on test and external sets.

### 7.16 `xai/base.py`
```python
class Explainer(Protocol):
    name: str
    def explain(self, image_tensor: torch.Tensor, target_class: int) -> np.ndarray: ...
    # returns HxW float32 in [0,1] at the model input size (512 x 512)
```

### 7.17 `xai/gradcam_pp.py`
- Uses `GradCAMPlusPlus` from `pytorch-grad-cam` **[VERIFY class name and that it works with timm MaxViT and ConvNeXt target layers]**, or an own implementation of the Chattopadhay et al. formulation if the library cannot be used. The target layer is set in config: `xai.gradcam_layer` (for MaxViT: the last stage's final block; for ConvNeXt: `stages[-1]`).
- Target: `target_class` from the grading head (predicted grade by default).
- **Acceptance:** `test_explainers.py` checks output shape and range; a unit test checks that a synthetic model whose output depends only on one known patch produces a peak inside that patch.

### 7.18 `xai/shap_explainer.py`
- Uses `shap` with an image masker: `shap.maskers.Image("inpaint_telea", shape)` with the partition or kernel explainer, applied to a coarse grid of superpixels (default 16 × 16). **[VERIFY API for the installed shap version]**
- Background: inpainting masker (no external background dataset). `nsamples` = 512 (config).
- Output: per-superpixel attribution broadcast back to 512 × 512 and min-max normalised to [0, 1] after keeping positive contributions (`max(attr, 0)`), documented in the methods.
- **Acceptance:** runs on one image in under 60 seconds on the target GPU or CPU; output is the correct shape.

### 7.19 `xai/lime_explainer.py`
- Uses `lime.lime_image.LimeImageExplainer` with `quickshift` segmentation, `num_samples = 1000` (config), `hide_color = 0`. Maps the top-label superpixel weights to a 512 × 512 map, positive weights only, min-max normalised.
- **Acceptance:** runs on one image; output shape and range correct; deterministic given the seed (`random_state` passed).

### 7.20 `xai/attention.py`
- For hybrid or ViT backbones only. Extracts attention weights from the final attention block (timm MaxViT or ViT modules). **[VERIFY how to obtain attention weights in the installed timm version; this usually means disabling the fused attention path and registering a forward hook.]**
- Head-averaged attention from the class token (ViT) or the pooled attention for MaxViT's final stage, reshaped to the spatial grid, upsampled to 512 × 512, min-max normalised.
- If attention weights cannot be extracted within one day of starting, the Engineer stops and tells the team. Attention saliency is then reported as "not computed" for the hybrid model, and the team decides between documenting it and switching to the ViT fallback (D1).
- **Acceptance:** shape and range test; attention map of a hand-built toy tensor matches a hand-computed reference.

### 7.21 `xai/fidelity.py`
This module is the heart of the quantitative validation. Functions:

- `topk_binarise(saliency, k_percent) -> np.ndarray(bool)`: sets the top k% of pixels within the field of view to True.
- `fov_mask(img_512) -> np.ndarray(bool)`: non-black pixels of the processed image, dilated by 2 px. Pixels outside the FOV are excluded from both saliency and mask statistics.
- `iou(a, b)`, `dice(a, b)`: with empty-set handling (returns NaN when both empty).
- `pointing_hit(saliency, mask) -> bool`: argmax of saliency lies inside the mask. Ties resolved by first occurrence, documented.
- `chance_baseline(mask, fov, k_percent) -> dict`: expected IoU and pointing accuracy under a uniformly random saliency inside the FOV. Expected pointing = mask-area / FOV-area. Expected IoU is estimated by Monte Carlo (1000 random maps) with seed 42.
- `fidelity_for_image(saliency, masks, fov, k_percent) -> pd.DataFrame`: one row per (lesion, k) with IoU, Dice, pointing, and chance baselines. Skips lesions with empty mask.
- **Acceptance:** `test_fidelity.py` covers hand-made cases: perfect overlap (IoU 1, Dice 1), disjoint (IoU 0), empty mask (NaN), and a pointing-game case where the argmax is on the boundary.

### 7.22 `xai/stability.py`
- `explanation_agreement(maps_a, maps_b, fov) -> float`: Spearman rank correlation over FOV pixels between two saliency maps for the same image (for example, two seeds). Reported as the mean over Messidor-2 images with a bootstrap CI.
- **Status:** the metric is a proposal. It is implemented and tested, but **not reported as a result until the Researcher approves it** (F9).

### 7.23 `eval/*`
- `evaluate_grading.py`: loads `best.pt` and `threshold.json`; computes Table 4 metrics on the APTOS 2019 test split with bootstrap CIs; writes `results/table4_detection_grading_aptos_test.csv` and the "Proposed framework (this study)" row of `results/table6_comparative.csv`.
- `xai_fidelity.py`: runs all configured explainers on the 81 IDRiD images with masks (at 512 px after the stored transform), computes fidelity at k = 10 (primary), 5 and 20; writes `results/table5_xai_fidelity.csv` with the columns `method, lesion, k_percent, n_images, mean_iou, iou_ci_lo, iou_ci_hi, mean_dice, dice_ci_lo, dice_ci_hi, pointing_acc, pointing_ci_lo, pointing_ci_hi, chance_iou, chance_pointing`.
- `crossdataset.py`: APTOS test (in-domain), IDRiD grading (external), Messidor-2 (external). Writes `results/crossdataset_summary.csv` with metrics per dataset, the QWK difference from in-domain with a bootstrap CI, and the detection label definition used per dataset.
- `ablation.py`: trains `minority.mode = none` and `minority.mode = oversample` with the same seed and split; compares macro sensitivity on grades 3 and 4, QWK, detection AUC and mean IDRiD IoU (k = 10); writes `results/ablation_minority.csv` with paired bootstrap differences.
- `efficiency.py`: parameter counts (total and trainable), model file size, latency at batch 1 (fp32 and fp16 on GPU; fp32 on CPU), peak GPU memory. 20 warm-up runs, 200 timed runs, median and IQR. Writes `results/efficiency.csv`.

### 7.24 `viz/figures.py`
- `confusion_matrix_fig(y_true, y_pred, labels)` with the five ICDR labels.
- `saliency_panel(image, saliency, mask, title)`: three panels (image, saliency overlay, lesion mask overlay).
- `training_curves_fig(metrics_jsonl)`.
- `class_distribution_fig(manifest_df)`: reproduces the paper's Fig. 2 from the manifest, log scale.
- All figures saved at 220 dpi to `results/figures/`.

### 7.25 `cli.py`
Single entry point. Subcommands run in order via `all`:
```
python -m xaidr.cli prepare-data     # adapters -> manifests -> preprocessing
python -m xaidr.cli make-splits
python -m xaidr.cli train   --config configs/default.yaml
python -m xaidr.cli evaluate --run <run_id>
python -m xaidr.cli xai     --run <run_id>
python -m xaidr.cli crossdataset --run <run_id>
python -m xaidr.cli ablation --config configs/ablation_no_oversampling.yaml
python -m xaidr.cli efficiency --run <run_id>
python -m xaidr.cli figures --run <run_id>
python -m xaidr.cli all --config configs/default.yaml
```
`make reproduce` runs `all` followed by `make test`.

---

## 8. Data

### 8.1 Sources and access

| Dataset | Use in this build | Where to get it | Access steps | Notes |
|---|---|---|---|---|
| APTOS 2019 Blindness Detection | Train, validation, in-domain test | kaggle.com/c/aptos2019-blindness-detection | Free Kaggle account; accept rules; download via Kaggle API | 3,662 labelled training images, 5-class grades. Test labels are not public. |
| IDRiD (Indian Diabetic Retinopathy Image Dataset) | External grading test (516 images); lesion fidelity (81 segmentation images) | ieee-dataport.org (search "Indian Diabetic Retinopathy Image Dataset") or idrid.grand-challenge.org | Free registration on the challenge site or IEEE DataPort | Masks: MA, HE, EX, SE, OD, plus fovea centres. Verify the segmentation / grading split in the download. |
| Messidor-2 | External test | Images: adcis.net (registration form). Labels: search Kaggle for `messidor2-dr-grades` | Registration can take days. **Start this on day 1.** | 1,748 images from 874 examinations. Labels distributed separately; join by filename. |
| EyePACS (Kaggle DR Detection) | OUT OF SCOPE (D4) | kaggle.com/c/diabetic-retinopathy-detection | Kaggle account | About 80–90 GB. Do not download for this deadline. |

**Models:** ImageNet-pretrained weights from timm (downloaded automatically on first use). Primary: `maxvit_tiny_tf_512` **[VERIFY]**. Baseline: `convnext_tiny`. No external medical pretrained weights are used in this build.

**Access the Engineer cannot perform alone:** the Kaggle API key (`~/.kaggle/kaggle.json`), IDRiD and Messidor-2 registrations, and the GPU. The Engineer must ask the team for these and must not attempt to bypass any access control.

### 8.2 Expected folder layout (raw)

```
data/raw/
  aptos2019/
    train.csv                  # id_code, diagnosis
    train_images/              # {id_code}.png
    test_images/               # unlabelled; not used for metrics
    sample_submission.csv      # placeholder labels; not used for metrics
  idrid/
    grading/images/            # .jpg, grading set (516)
    grading/labels.csv         # [VERIFY columns]
    segmentation/images/       # .jpg, 81 annotated images
    segmentation/masks/MA/ HE/ EX/ SE/ OD/   # binary .tif per lesion [VERIFY naming]
  messidor2/
    images/
    labels.csv                 # [VERIFY columns and grade scheme]
```

### 8.3 Label mapping per dataset

| Dataset | Native label | Mapped `grade` (ICDR 0–4) | `dr_label` (D5) | Notes |
|---|---|---|---|---|
| APTOS 2019 | `diagnosis` 0–4 | identity | `grade ≥ 1` | Primary |
| IDRiD | Retinopathy grade 0–4 | identity | `grade ≥ 1` | External |
| Messidor-2 | Release-specific adjudicated grade, declared in config | Only where an exact mapping to ICDR 0–4 exists; otherwise empty | Mapped definition from config, e.g. referable DR | Report under its own label definition. Never pool with APTOS. Researcher to confirm (F8). |

### 8.4 Preprocessing transform record
Every processed image has its crop box and original size in the manifest. Masks are cropped with the same box, then resized to 512 × 512 with nearest-neighbour interpolation. The test in Section 7.5 verifies this alignment.

### 8.5 Leakage rules (checked by `test_no_leakage.py`)
1. Rows with `split ∈ {external_idrid, external_messidor2}` never appear in the training sampler.
2. Augmentation transforms are attached only to training loaders.
3. Oversampling duplicates training indices only.
4. Normalisation constants are the ImageNet constants (no dataset statistics computed from test data).
5. Splits are stratified and locked by SHA-256; the lock file is committed.
6. If an IDRiD image appears in both the grading and segmentation sets, it is logged and it remains external.

---

## 9. Model and training configuration

### 9.1 `configs/default.yaml`
```yaml
seed: 42
deterministic: true

paths:
  raw: data/raw
  manifests: data/manifests
  processed: data/processed/dr512
  splits: data/splits
  checkpoints: checkpoints
  results: results

data:
  image_size: 512
  crop_threshold: 10
  clahe: {clip_limit: 2.0, tile_grid: [8, 8]}
  split_ratios: [0.70, 0.15, 0.15]
  detection_positive_min_grade: 1      # D5 for APTOS and IDRiD
  messidor2:
    label_file: labels.csv             # [VERIFY]
    id_column: image_id                # [VERIFY]
    grade_column: adjudicated_dr_grade # [VERIFY]
    grade_map: {}                      # filled after verification (F8)
    detection_definition: referable    # [VERIFY with Researcher]

model:
  backbone: maxvit_tiny_tf_512         # D1; fallback convnext_tiny
  pretrained: true
  head_hidden: 256
  dropout: 0.3

train:
  epochs: 30                           # D10
  batch_size: 16                       # x grad_accum 2 = effective 32
  grad_accum: 2
  optimizer: adamw
  lr: 1.0e-4
  weight_decay: 0.05
  scheduler: cosine
  warmup_epochs: 2
  amp: true
  deterministic: true
  num_workers: 4
  early_stopping: {monitor: selection_score, patience: 8}   # D9
  loss:
    lambda_det: 0.4
    lambda_grade: 0.6
    grade_class_weights: inverse_sqrt_freq

minority:
  mode: oversample                     # none | oversample (D2)
  target_grades: [3, 4]
  factor: 3

xai:
  methods: [gradcam_pp, shap, lime, attention]
  gradcam_layer: auto                  # resolved per backbone in xai/gradcam_pp.py
  fidelity_k_percent: [10, 5, 20]      # D6; first value is primary
  shap: {segments: [16, 16], nsamples: 512}
  lime: {num_samples: 1000, hide_color: 0}

eval:
  bootstrap_iterations: 1000           # D8
  detection_threshold: youden          # D7
  chance_monte_carlo: 1000
```

### 9.2 Optimisation details
- Input 512 × 512, RGB, CLAHE-enhanced, ImageNet normalisation.
- Loss: `0.4 · BCE + 0.6 · weighted CE` (paper Sec. 4.3 and Table 3).
- Mixed precision on. Gradient accumulation 2. Gradient clipping at norm 1.0 (added for stability with AMP; document it).
- Memory fallback: if out of memory at batch 16, drop to batch 8 with accumulation 4 and record the change in the run's config snapshot. Do not change the effective batch size silently.

---

## 10. Explainability specification

### 10.1 Common rules
- All maps are produced on the 512 × 512 processed image.
- All maps are float32, in [0, 1], and aligned with the processed masks (Section 8.4).
- Explanations are generated for the grading head's predicted class, unless `xai.target` is set otherwise (D11).
- Every method is run on the same 81 IDRiD images, with the same seed, and the runtime per image is logged.

### 10.2 Methods
| Method | Implementation | Applies to | Notes |
|---|---|---|---|
| Grad-CAM++ | `pytorch-grad-cam` `GradCAMPlusPlus` **[VERIFY]**, or own implementation | Any backbone | Target layer set per backbone |
| SHAP | `shap` image masker with a coarse superpixel grid, inpainting background | Any backbone | Kernel-style estimator, as stated in paper. Clipped at zero, then normalised. |
| LIME | `lime_image.LimeImageExplainer`, quickshift | Any backbone | Positive weights only, then normalised |
| Attention | Attention weights from the final attention block | MaxViT / ViT only | See Section 7.20 |

### 10.3 Fidelity protocol (Table 5)
For each method and each image with masks:
1. Generate the map at 512 × 512.
2. Restrict to the FOV.
3. Binarise the top k% of FOV pixels (k = 10 primary; 5 and 20 sensitivity).
4. Compute IoU and Dice against each lesion mask (MA, HE, EX, SE) and against the union of all lesions ("any lesion").
5. Compute pointing-game hit against each lesion mask.
6. Compute the chance baseline for IoU and pointing for the same mask and FOV.
7. Aggregate per method and lesion over images containing that lesion. Report the mean with a bootstrap CI over images, and the chance baseline alongside.

Reporting the chance baseline is required. Without it, the reader cannot tell whether IoU of 0.05 is good or bad for a lesion that covers 2% of the field.

### 10.4 Sample sizes
Lesion-level sample sizes will be small (for example, SE on 40 images). Table 5 must show `n_images` for every row. Results with fewer than 20 images are reported as exploratory.

### 10.5 Cross-domain explanation agreement (pending approval, F9)
Messidor-2 has no lesion masks. The Researcher must approve one of these operational definitions before it is reported:
- **Proposal A (preferred):** Spearman correlation between saliency maps of the same image produced by models trained with three seeds (requires seeds, see D12).
- **Proposal B:** agreement between saliency maps computed on APTOS-trained and IDRiD-trained models on the same external images.

Until approved, the code computes Proposal A and the result is marked "pending" in `crossdataset_summary.csv`.

---

## 11. Evaluation outputs mapped to the paper

| Paper location | File | Content | Produced by |
|---|---|---|---|
| Table 4 | `results/table4_detection_grading_aptos_test.csv` | Detection: accuracy, sensitivity, specificity, AUC. Grading: accuracy, macro sensitivity, macro AUC, QWK. Each with 95% CI. | `evaluate_grading.py` |
| Table 5 | `results/table5_xai_fidelity.csv` | IoU, Dice, pointing per method and lesion, with n, CI and chance | `xai_fidelity.py` |
| Table 6 | `results/table6_comparative.csv` | Published rows (copied verbatim from paper, cited) and the proposed row from Table 4 | `evaluate_grading.py` |
| Sec. 6.3 | `results/crossdataset_summary.csv` | In-domain vs external metrics, QWK delta with CI, label definition per dataset, Proposal A status | `crossdataset.py` |
| Sec. 6.4 | `results/ablation_minority.csv` | Minority sensitivity, QWK, det AUC, IoU for none vs oversample, paired diff with CI | `ablation.py` |
| Sec. 6.5 | `results/efficiency.csv` | Parameters, size, latency (median, IQR), peak memory | `efficiency.py` |
| Figs | `results/figures/*.png` | Confusion matrix, saliency panels, training curves, class distribution | `figures.py` |

**Rule:** published numbers from the cited literature are placed in Table 6 and never mixed with this build's results in the same column without a row label.

---

## 12. Jupyter notebooks (visualisation requirement)

All notebooks read from `results/` and `data/manifests/`. None trains a model.

| Notebook | Purpose |
|---|---|
| `00_architecture_and_graph.ipynb` | Model summary (`torchinfo`), module tree, dataflow diagram, shapes at each stage |
| `01_data_audit.ipynb` | Per-dataset class counts, image sizes, before/after CLAHE examples, mask coverage per lesion, reproduces the paper's Fig. 2 from the manifest |
| `02_training_curves.ipynb` | Loss components, QWK, detection AUC per epoch; marks the selected epoch |
| `03_results_tables.ipynb` | Renders Tables 4, 5, 6 and the cross-dataset summary with CIs; exports LaTeX snippets to `results/tables_tex/` |
| `04_xai_gallery.ipynb` | Image | saliency | mask panels for one good, one average and one poor case per method; confusion matrix |
| `05_ablation_and_efficiency.ipynb` | Paired differences with CIs; efficiency bar chart with IQR |

Each notebook starts with a "Run all" cell that asserts the required result files exist and prints a clear message if they do not.

---

## 13. Testing and acceptance criteria

The Tester role (Sonnet) uses this section. The Engineer must pass all of the following before any result is reported:

| Test file | What it must prove |
|---|---|
| `test_manifest.py` | Required columns enforced; invalid dtypes rejected |
| `test_adapters.py` | Each adapter builds a correct manifest from fixture CSVs; masks attached only where `has_masks` |
| `test_preprocess.py` | Mask and image stay aligned after crop and resize (within 1 px) |
| `test_splits.py` | Disjointness; stratification within 1 percentage point; reproducible; lock file respected |
| `test_no_leakage.py` | External sets never in training; augmentation only in training; oversampling draws training indices only |
| `test_model.py` | Output shapes; both heads receive gradients |
| `test_loss.py` | Loss equals hand-computed value |
| `test_metrics.py` | Metrics equal scikit-learn within 1e-9 on random data |
| `test_fidelity.py` | Known answers for IoU, Dice, pointing, FOV, empty masks |
| `test_explainers.py` | Shapes and range; synthetic known-patch test for Grad-CAM++ |
| `test_cli_smoke.py` | End-to-end on 2 images, CPU, under 2 minutes |

**Scientific acceptance checks (must be reported, not hidden):**
- No reported metric comes from a run without a saved config snapshot and git commit hash.
- Test-set metrics are computed once per final model. No tuning on test data.
- Every table row states n, CI and (for fidelity) chance baseline.
- The number of IDRiD segmentation images with each lesion is printed in the run log and equals the values in F1 (81 / 81 / 80 / 40), or the difference is explained.

**Tester instruction for errors:** any failing test or logical error found is reported back to the Engineer with the file, line and expected behaviour. The Tester does not edit the implementation.

---

## 14. Reproducibility and deliverables

### 14.1 Environment
- Python 3.10 or 3.11. Pinned in `requirements.txt` (the Engineer chooses exact versions after install and records them in `results/env/pip_freeze.txt`).
- Required packages: `torch`, `torchvision`, `timm`, `pytorch-grad-cam`, `captum` (only if used), `shap`, `lime`, `albumentations`, `opencv-python`, `scikit-learn`, `scikit-image`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `tqdm`, `pyyaml`, `kaggle`, `torchinfo`, `pytest`, `nbformat`, `nbconvert`.
- Hardware: NVIDIA GPU with ≥ 16 GB VRAM recommended. Kaggle Notebooks or Colab Pro are acceptable. CPU runs are for tests only.

### 14.2 Run record (written automatically per run)
`checkpoints/<run_id>/run.json` and `results/env/`: config snapshot, git commit, seed, package versions, GPU name, wall-clock time per stage, dataset manifest hash, split hash, threshold value.

### 14.3 Final deliverables for 14 October
1. Git repository with tagged release `v1.0-submission`.
2. `README.md` with exact reproduction steps.
3. `results/` summary CSVs, figures and `results/tables_tex/`.
4. Notebooks, executed, with outputs saved.
5. `docs/VERIFICATION.md`, `docs/DATA_ACCESS.md`, `docs/DECISIONS_LOG.md` completed.
6. A zipped copy of the repository without data or checkpoints.

---

## 15. Schedule and risk register

### 15.1 Schedule (today is Saturday 10 October 2026; code is due Wednesday 14 October)

**This schedule is tight. It assumes a GPU is available from Monday and that Messidor-2 access is approved early. If either fails, follow the fallback in the risk register.**

| Day | Milestone | Exit criterion |
|---|---|---|
| Sat 10 – Sun 11 Oct | M0: repo scaffold, config, manifest, adapters for APTOS and IDRiD, preprocessing, splits, all unit tests for those modules. Start Messidor-2 access request. | `pytest tests/` passes for those modules |
| Mon 12 Oct | M1: model, loss, trainer; full training run of the primary model; `evaluate_grading.py` produces Table 4 and Table 6 row | Table 4 CSV exists with CIs; training curves saved |
| Mon 12 – Tue 13 Oct | M2: four explainers; fidelity on 81 IDRiD images (Table 5); Messidor-2 adapter and cross-dataset eval if data available | Table 5 CSV exists with chance baselines |
| Tue 13 Oct | M3: ablation (two runs; the no-oversampling run can start in the background on Monday night); efficiency benchmark; figures; notebooks 00–05 | Ablation and efficiency CSVs exist; notebooks run end to end |
| Wed 14 Oct (deadline) | M4: full `make reproduce` on a clean clone; Tester final pass; tag `v1.0-submission`; hand results to Researcher and Editor | `make reproduce` passes; tag exists |

### 15.2 Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| No GPU for the team | Medium | Critical | Kaggle Notebooks or Colab Pro from day 1. Reduce to `convnext_tiny`, batch 8, 20 epochs if needed. Record every change. |
| Messidor-2 registration not approved in time | High | High | Report cross-dataset on IDRiD only; state Messidor-2 as pending in `crossdataset_summary.csv`. Do not fabricate. |
| Attention weights hard to extract from MaxViT | Medium | Medium | Time-box to one day (Section 7.20). Fall back to ViT (D1) or document "not computed". |
| SHAP or LIME too slow for 81 images | Medium | Medium | Reduce `nsamples` and grid, run overnight, cache per-image maps to disk. |
| Captum or pytorch-grad-cam API differs from this document | High | Low | Check installed API, record in `docs/VERIFICATION.md`, adapt without changing outputs. |
| IDRiD file naming or label columns differ | High | Medium | Adapter reads from config; Engineer asks the team to confirm after first inspection. |
| Out-of-memory at 512 px | Medium | Medium | Fallback in Section 9.2; record change. |
| Claude Code session runs out of context or usage limit mid-task | Medium | Medium | Commit after each module. Start a fresh session from `docs/DESIGN.md` and the git log if interrupted. Keep each session to one milestone. |
| Paper claims (F1–F14) not corrected before submission | High | High | Researcher corrects paper using the list in Section 17. |

---

## 16. Defects in the Experimentation Execution Guide (do not copy)

The guide is a useful starting point, but the Engineer must not paste its code. These defects are known from reading it:

| # | Location in guide | Defect | Correct behaviour in this design |
|---|---|---|---|
| G1 | 3.10 `xai_fidelity.py` | `LayerGradCam` called with `target=None` on a model returning a tuple | Grade-only wrapper, explicit target class (Section 7.11) |
| G2 | 3.10 | Iterates the dataset directly; saliency is 512 px but masks are at source resolution; masks not cropped or resized with the image | Stored transform record; masks aligned (Sections 7.5, 8.4) |
| G3 | 3.10 | `iou_dice` uses a fixed 0.5 threshold on a min-max map | Top-k% with sensitivity (Section 7.21) |
| G4 | 3.10 | SHAP and LIME are described but not implemented | Implemented (Sections 7.18–7.19) |
| G5 | 3.10 | `target_layer = model.backbone.stages[-1]` works only for ConvNeXt | Backbone-specific layer resolution (Section 7.17) |
| G6 | 3.2, 3.10 | IDRiD mask path `mask_dir/LESION/image_id` and column names `image_id`, `retinopathy_grade` are not verified | Adapter with config and verification (Section 7.4) |
| G7 | 3.5 `oversample_minpriority_classes` | Uses `df.append` (removed in pandas 2.x) and is never called by training | Oversampling sampler (Section 7.9), wired to config |
| G8 | 3.12 `ablation_augmentation.py` | `use_synthetic_oversampling` is never read; both runs use the same code path; checkpoint path is fixed so the second run overwrites the first | Config-driven runs with separate run IDs |
| G9 | 3.11 `evaluate_crossdataset.py` | Calls `evaluate()` on Messidor-2 without passing its label column; the default `diagnosis` does not exist there | Label column from manifest (Section 7.23) |
| G10 | 3.4 `DRGradingDataset` | Computes `detect = grade > 0` for every dataset, including Messidor-2 where the label is referable/adjudicated | Per-dataset detection definition (D5, Section 8.3) |
| G11 | 3.8 `train.py` | Ignores the YAML file; hard-coded paths; no seeds; no AMP; selects best model on validation loss | Config-driven, seeded, selection score (D9) |
| G12 | 3.13 `benchmark_efficiency.py` | CUDA only | CUDA and CPU (Section 7.23) |
| G13 | 3.1 config | `backbone: convnext_tiny` with comment about `vit_base_patch16_224` at 224 px | Section 9.1 |
| G14 | 3.7 `losses.py` | Paper says "class-weighted" but the code passes no weights | Inverse square-root class weights (Section 7.12) |
| G15 | 3.3 `make_splits.py` | Splits APTOS only and writes to fixed paths without a lock | Locked, hashed splits (Section 7.6) |

---

## 17. Open issues for the Researcher and team

These do not block coding, but they must be resolved before the paper is submitted. The Researcher role (Sonnet or Opus) should own them.

1. **F1:** Correct the IDRiD mask statement (81 annotated images, not 516) in Section 4.1 and Table 2.
2. **F2 / D3:** Rewrite Section 5.3 so IDRiD is an external test set and not split.
3. **F3:** Verify whether Captum implements Grad-CAM++. Correct Sections 4.4 and 5.1 accordingly, and cite the implementation actually used.
4. **F5 / F6 / D1:** Replace "hybrid CNN/ViT" in Table 3 with the backbone actually used.
5. **F7 / D2:** Decide whether the paper keeps the term "synthetic-data augmentation" (it must then include generative synthesis) or uses "minority oversampling with augmentation".
6. **F8 / D5:** Confirm the Messidor-2 label release, the grade scheme and the detection definition. Revise Sections 4.1 and 6.3 and Table 2.
7. **F9:** Approve Proposal A or B (Section 10.5), or propose an alternative for cross-domain explanation stability.
8. **F10:** Confirm EyePACS is future work in Sections 4.1 and 4.6, and remove the held-out EyePACS claim.
9. **F11 / D6:** Replace the unspecified IoU threshold in Sections 4.5 and 6.2 with the top-k% protocol and chance baseline.
10. **F12:** Reconcile the Chilukoti et al. QWK values quoted in Section 6.3 against Table 4 and the source paper.
11. **F13:** Fix all references: duplicate DOI for [4] and [5], journal for [5], incomplete entries [3], [21], [37].
12. **F14:** Verify or soften the claim in Section 2.5 and Table 1 that no reviewed study combines all three dimensions.
13. **Sample size:** Report IDRiD lesion sample sizes (81 / 81 / 80 / 40) in Table 5, and describe the effect of small n on the conclusions in Section 8 (Limitations).
14. **Literature review:** Section 2 cites 38 works. Check whether each citation supports the sentence it is attached to. The supplied research papers (Alwakid et al. 2023, Bashir et al. 2023, Ishtiaq et al. 2023, Alavee et al. 2024) show that several studies report accuracies from augmented data before the split. The paper should state how leakage was avoided in its own design and why it matters.

---

## 18. Instructions for the Engineer (summary)

1. Read this entire file. Then reply with: (a) what you understood the deliverable to be; (b) which Section 5 decisions you accept; (c) which datasets and access you already have and which you need; (d) the GPU you have or need.
2. Create the scaffold from Section 6 and commit it.
3. Implement milestones M0 to M4 (Section 15.1) in order. Run the tests after each milestone and report results.
4. Resolve every **[VERIFY]** item and record it in `docs/VERIFICATION.md`.
5. Never produce a number that is not from an executed run. Never copy numbers from the paper into this build's results.
6. If anything in this document conflicts with the Experimentation Execution Guide, this document wins.
7. When finished, hand back: the repository, the results CSVs, the list of **[VERIFY]** outcomes, the list of deviations from this document (with reasons), and any open questions.

---

## 19. Next steps for the team (before the Engineer starts)

1. **Confirm the Section 5 decisions**, especially D1 (backbone), D2 (wording of synthetic augmentation), D3 (IDRiD role) and D4 (EyePACS out of scope).
2. **Start access requests today:** Messidor-2 registration (adcis.net); IDRiD registration; Kaggle API key for APTOS 2019.
3. **Secure a GPU** for Monday 12 October (Kaggle, Colab Pro, or lab hardware).
4. **Share the sample submission and sample images** (already supplied) as smoke-test fixtures only.
5. **Choose the Researcher and Editor** timings: Researcher checks Section 17 during the Monday and Tuesday training runs.

---

*Designer/Architect role output. This document describes the design only. It contains no results. All metric claims in the paper are to be replaced by values produced by this build.*
