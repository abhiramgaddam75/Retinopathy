# Decisions log

Changes to DESIGN.md Section 5, and implementation choices the design left open. Newest last.

## Section 5 decisions

| Date | Decision | Status |
|---|---|---|
| 2026-10-10 | D1-D12 | Engineer proceeds on every DEFAULT. Team confirmation still requested for D1, D2, D3, D4 (Section 19). None of them affects milestone M0. |

## Implementation choices not fixed by the design

| Date | Module | Choice | Reason |
|---|---|---|---|
| 2026-10-10 | Environment | Python 3.11.15 in `.venv`, created with `uv`. The machine only had Python 3.14, which the design excludes. | Section 14.1 |
| 2026-10-10 | Environment | `opencv-python-headless` instead of `opencv-python`. | Same API without GUI dependencies; nothing in the pipeline opens a window. |
| 2026-10-10 | Environment | `captum` is not installed. | Section 14.1 lists it "only if used"; Grad-CAM++ comes from `pytorch-grad-cam` (F3). |
| 2026-10-10 | `config.py` | A config may name a parent with `base: default.yaml`; the child is deep-merged over it. | Keeps the ablation and baseline configs to the keys that differ (G8). |
| 2026-10-10 | `config.py` | Added `data.datasets`, `data.idrid.*`, `data.messidor2.images`, `data.messidor2.detection_map`, `train.grad_clip_norm`. | Section 7.4 needs IDRiD names read from config; D5 needs an explicit Messidor-2 detection mapping; Section 9.2 specifies clipping at 1.0. |
| 2026-10-10 | `idrid.py` | Segmentation images get the id `seg_<name>`; one combined manifest `data/manifests/manifest.csv` holds all datasets. | Segmentation and grading images are numbered separately; ids must be unique within a dataset (Section 7.3). |
| 2026-10-10 | `idrid.py` | Overlap between the segmentation and grading sets is detected by SHA-256 of the image files and written to `idrid_overlap.csv`; a byte-identical twin passes its grade to the segmentation row. | Section 7.4 requires overlap to be logged and flagged; the manifest schema has no flag column. Re-encoded copies would not be detected: check when the data arrives. |
| 2026-10-10 | `preprocess.py` | **Superseded on 2026-10-11 (ISSUE-015): the crop is now padded to a square.** Originally: the cropped field of view is resized straight to 512 x 512 without padding to a square. | Section 7.5 specifies crop, CLAHE, resize. Images whose retina is cut off at the top and bottom are therefore stretched vertically; masks get the identical transform, so alignment holds. Padding to a square first would avoid the stretch; raise with the Designer if wanted. |
| 2026-10-10 | `dataset.py` | `LesionDataset(manifest_df, size, transform=None)` takes the size and rebuilds masks from the source masks with the stored crop box. `make_train_loader` / `make_eval_loader` live here. | One code path for mask geometry (Section 4.2 rule 2); gives the leakage tests a single place where `train_transform` is used. |
| 2026-10-10 | `dataset.py` | A missing label is returned as `-1`. | Messidor-2 rows may have no five-class grade (Section 8.3). |
| 2026-10-10 | `cli.py` | `prepare-data` fails when a configured dataset folder is missing; `--datasets` selects a subset. | "If a dataset file is missing, stop and ask" (Section 0). |

## Milestones M2 and M3 code, 2026-10-12

Written and tested on synthetic data and untrained or fallback models. **No fidelity or
cross-dataset number exists**: IDRiD and Messidor-2 are not available.

| Module | Choice | Reason / departure |
|---|---|---|
| `xai/base.py` | Explainers take the normalised image tensor and a target: `"disease"` (log-odds of any DR, the default) or a grade. SHAP and LIME perturb the de-normalised uint8 image and read P(target). | ISSUE-014; replaces D11. |
| `xai/gradcam_pp.py` | Own implementation of the first-order closed form on the backbone's final feature map; a test checks it against `pytorch_grad_cam.GradCAMPlusPlus` (correlation > 0.99 on ConvNeXt). | DESIGN 7.17 allows either. No hooks or reshape transform are needed because the model exposes its feature map. Resolves V4. |
| `xai/shap_explainer.py` | `shap.PartitionExplainer` with `shap.maskers.Image("inpaint_telea")`, 512 evaluations. The partition is the explainer's own hierarchy, about 16 x 16 regions at that budget, not a fixed grid. | DESIGN 7.18 allows partition or kernel. The paper must say "partition explainer with an inpainting masker", not "kernel explainer" (F4). Resolves V6: runs with opencv-python-headless 5.0. |
| `xai/lime_explainer.py` | Quickshift (kernel 4, max distance 200, ratio 0.2), 1,000 samples, hidden colour 0, two-class output (not DR, DR). | DESIGN 7.19; the settings must be reported in the paper (review B3 item 3). |
| `xai/attention.py` | Attention each position receives in MaxViT's last grid-attention block, averaged over 16 heads and all queries; the fused path is switched off for one pass and restored. At 512 px that block is one global window over the 16 x 16 map. It ignores the target. | Resolves V8 within the one-day box. Unavailable on ConvNeXt: reported as "not computed", never silently skipped. |
| `eval/xai_fidelity.py` | Saliency maps cached as float32 in the run folder. Table 5 adds lesion recall, maximum attainable IoU and Dice, a one-sided Wilcoxon test of IoU against chance and Holm-adjusted p-values over the primary-k rows. `fidelity_summary.json` records images per lesion, images predicted grade 0 and seconds per image. | DESIGN 7.23 plus ISSUE-014. |
| `eval/crossdataset.py` | One long table; datasets missing from the manifest are written as "not available". QWK difference uses independent bootstrap resampling of the two test sets. Explanation stability is written as "not computed". IDRiD segmentation images are excluded from grading metrics (they repeat grading images). | DESIGN 7.23; ISSUE-013. |
| `eval/efficiency.py` | CPU fp32 always; GPU fp32 and fp16 when a GPU exists. No quantised or pruned variant. | DESIGN 7.23; the compressed variant is a SHOULD item not done, so the paper must not promise it. |
| `viz/figures.py` | Class-distribution bars use a linear axis from zero with counts labelled, not a log axis. | DESIGN 7.24 asked for log scale; bar length on a log axis is not meaningful. |

## Milestone M1 and the first real data, 2026-10-11

| Topic | Decision or fact | Reason |
|---|---|---|
| APTOS 2019 on disk | 3,662 labelled images; grades 0-4: 1,805 / 370 / 999 / 193 / 295. `train.csv` SHA-256 `B2A34796...DD79A5`. | Matches the published dataset (brief Section 2). |
| Duplicates measured | 123 groups of byte-identical images (251 images). 96 extra same-grade copies and 62 images in 30 conflicting-grade groups were dropped; **3,504 images remain**: train 2,452, validation 526, test 526. | DEF-002 / ISSUE-016 on real data. The paper's "one image per patient" sentence is wrong for APTOS. |
| Learning-rate schedule | Stepped once per epoch: linear warm-up over `warmup_epochs`, then cosine decay; the first cosine epoch is at the peak rate. | DESIGN 7.13 does not say per step or per epoch. |
| Validation loss | Computed on CPU from the stored probabilities with the same loss weights as training. | One prediction pass per epoch serves loss, QWK and AUC. |
| Pooled features | The backbone's own pooling and normalisation (`forward_head(pre_logits=True)`); 512 features for MaxViT-Tiny, 768 for ConvNeXt-Tiny. | DESIGN 7.10. |
| Pretrained weights | timm's default tag for each name. For `convnext_tiny` that is `in12k_ft_in1k` (ImageNet-12k, fine-tuned on ImageNet-1k); for `maxvit_tiny_tf_512` it is `in1k`. | DESIGN says "ImageNet-pretrained weights from timm"; the paper must name the tag. |
| Table 6 literature rows | Written with a `note` column carrying each row's verification status. The Akhtar et al. row from the draft is replaced by Shakibania et al. [27]; Dinpajhouh [50] is marked NOT VERIFIED. | DESIGN 7.23 says "copied verbatim from paper", but the Researcher found those rows wrong (ISSUE-009). |
| **No GPU: CPU fallback run** | The specified model needs about 3.5 days on the laptop (measured 3.31 s per image). `configs/cpu_fallback.yaml` trains `convnext_tiny` at 256 px, at most 20 epochs, same split and seed. **Departs from D1 and the 512 px input.** Its results are a fallback baseline, not the proposed framework's result. The specified run is described in `docs/TRAINING_ON_GPU.md`. | Risk register, "No GPU for the team": reduce the model and record every change. |
| Ablation command | `ablation --run <oversample run> --against <none run>` compares two finished runs; each arm is trained with the ordinary `train` step (`minority.mode=none` for the second). It refuses runs that differ in anything but `minority.mode`, or in seed, split or manifest hash. An arm's test predictions are produced without overwriting Table 4. The IDRiD IoU row is written as "not computed" until the explainers and IDRiD exist. | DESIGN 7.23 has `ablation` train both arms from one command; separate training keeps runs resumable on the laptop and makes run ids explicit (G8). |
| OOM fallback | Not automatic. Rerun with `train.batch_size=8 train.grad_accum=4`; the override is saved in the run's config snapshot. | DESIGN 9.2 asks that the change be recorded, not silent. |

## Changes for Researcher issues (docs/ISSUE_LOG.md), 2026-10-11

For the Tester to re-test and the Researcher to close. Several change DESIGN.md; the Designer has
not signed these off (open question 6 in docs/SCIENTIFIC_REVIEW.md).

| Issue | Change | Departs from the design? |
|---|---|---|
| ISSUE-015 (DEF-007) | The field-of-view crop is centred on a black square (`pad_to_square`) before resizing, for images and masks alike, so the aspect ratio is kept. The padding is a function of the crop size, so the crop box in the manifest still describes the full geometry and no column was added. `preprocess.json` records `pad_to_square`, so data built the old way is refused. | Yes: Section 7.5 (crop, CLAHE, resize) |
| ISSUE-016 (DEF-002) | `make-splits` keeps one image (the first by id) from each group of byte-identical same-grade copies and drops the rest. Groups with conflicting grades are dropped from every split; the Researcher allowed "training only, or drop", and dropping avoids training on one photograph with two labels. Dropped images have an empty `split` and are listed with an `action` in `aptos2019_duplicates.csv`; they are part of the locked table. Counts are logged. | Extends Section 7.6 |
| ISSUE-014 | `xai/fidelity.py` implemented (it is an M2 module; done now because it needs no data). Each (lesion, k) row carries IoU, Dice, pointing, **lesion recall at k**, their chance values, the **maximum attainable IoU and Dice**, and the lesion's share of the field of view. Chance IoU and Dice sample the hypergeometric overlap directly instead of drawing 1,000 random maps; the distribution is the same. `holm_adjust` is provided for the 20 primary comparisons. New config key `xai.target`, default `disease` (grades 1-4 together), replacing "explain the predicted grade". | Yes: D11; Section 7.21 columns; Table 5 gains columns |
| ISSUE-013 | `xai/stability.py` implemented (Proposal A, Spearman over field-of-view pixels). New config key `eval.stability_seeds: [42, 43, 44]`. **Not done:** the three training runs and the per-dataset report, which need data and a GPU. | Yes: D12 becomes three seeds if stability is to be claimed |
| ISSUE-017 | **Not done.** The grading-only baseline needs the model code (M1). Planned as one extra config, `baseline_grading_only.yaml`, if the team wants the comparison. | Adds a run |
| ISSUE-012 | No code change: the code already follows the facts (IDRiD external, EyePACS disabled, `pytorch-grad-cam`). The paper is what changes. | No |

## Fixes for Tester defects (docs/DEFECTS.md), 2026-10-10

Each fix is for the Tester to re-test and close. Where a fix departs from DESIGN.md it is marked.

| Defect | Change | Departs from the design? |
|---|---|---|
| DEF-001 | `train_transform(size, seed=None)` seeds the albumentations pipeline (from `seed`, or from the seeded python generator). `make_train_loader` passes `cfg.seed`, gives the `DataLoader` a seeded generator, and reseeds the pipeline in each worker from the worker seed, so worker runs are reproducible and each epoch still draws new augmentations. | No |
| DEF-002 | **New manifest column `content_hash`** (SHA-256 of the source file), filled by `prepare-data`. `make-splits` treats byte-identical APTOS images as one unit, refuses to run without hashes, writes `aptos2019_duplicates.csv`, and warns about groups with conflicting grades and about APTOS images identical to an external-test image. Nothing is removed: what to do with conflicting grades is the Researcher's decision (ISSUE-002). Only byte-identical copies are found, not re-encoded ones. | Yes: Section 7.3 lists exact columns |
| DEF-003 | `make_train_loader` and `make_eval_loader` call `verify_split_lock` and refuse a manifest whose APTOS split column differs from the lock, or when no lock exists. `.gitignore` now ignores `data/*` except `data/splits/`, so the split CSV and its hash can be committed. | Yes: Section 6 lists all of `data/` as not in git; Section 8.5 rule 5 requires the lock to be committed |
| DEF-004 | `prepare-data` stores the preprocessing parameters in `data/processed/dr512/<dataset>/preprocess.json` and stops when the config differs, unless `--force` rebuilds. | Extends Section 7.5 |
| DEF-005 | Unreadable images and impossible splits end with a one-line error and exit code 1. `make-splits` also refuses a split in which a grade would be missing from train, val or test (found while fixing: a grade with two images was silently absent from val and test). New `SplitError`; `SplitLockError` is a subclass. | No |
| DEF-006 | `train.deterministic` removed; the top-level `deterministic` is the only key. | Yes: Section 9.1 lists both |
| Tester file | `tests/test_sanity.py` is Tester-owned. The Engineer removed its two DEF-001 `xfail` markers and pointed its config at the test's split folder, because the suite would otherwise fail after the fixes. The Tester should review that edit. | n/a |
