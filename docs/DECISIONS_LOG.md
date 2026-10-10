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
| 2026-10-10 | `preprocess.py` | The cropped field of view is resized straight to 512 x 512 without padding to a square. | Section 7.5 specifies crop, CLAHE, resize. Images whose retina is cut off at the top and bottom are therefore stretched vertically; masks get the identical transform, so alignment holds. Padding to a square first would avoid the stretch; raise with the Designer if wanted. |
| 2026-10-10 | `dataset.py` | `LesionDataset(manifest_df, size, transform=None)` takes the size and rebuilds masks from the source masks with the stored crop box. `make_train_loader` / `make_eval_loader` live here. | One code path for mask geometry (Section 4.2 rule 2); gives the leakage tests a single place where `train_transform` is used. |
| 2026-10-10 | `dataset.py` | A missing label is returned as `-1`. | Messidor-2 rows may have no five-class grade (Section 8.3). |
| 2026-10-10 | `cli.py` | `prepare-data` fails when a configured dataset folder is missing; `--datasets` selects a subset. | "If a dataset file is missing, stop and ask" (Section 0). |
