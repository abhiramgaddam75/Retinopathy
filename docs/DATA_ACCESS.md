# Data access and placement

Nothing under `data/` is committed. Download each dataset yourself (the registrations and the
Kaggle key belong to the team) and place the files exactly as below, then run

```
python -m xaidr.cli prepare-data
python -m xaidr.cli make-splits
```

`prepare-data` stops with an error naming any dataset folder that is missing. To prepare only the
datasets you already have, pass for example `--datasets aptos2019,idrid`; datasets prepared earlier
are kept in the manifest.

## Status on 10 October 2026

| Dataset | On disk | Needed for |
|---|---|---|
| APTOS 2019 | **no** | training, validation, in-domain test (M1) |
| IDRiD | **no** | external grading test, lesion fidelity (M2) |
| Messidor-2 | **no** (registration can take days) | external test (M2) |

Only `sample_submission.csv` (placeholder labels, not usable, F16) and ten unlabelled sample photos
were supplied. The photos are in `tests/fixtures/sample_fundus/` as smoke-test fixtures and are
git-ignored until their licence is checked (F17).

## APTOS 2019 Blindness Detection

Source: <https://www.kaggle.com/c/aptos2019-blindness-detection> (Kaggle account, accept the rules).

```
data/raw/aptos2019/
  train.csv            # columns: id_code, diagnosis
  train_images/        # {id_code}.png, 3,662 images
  test_images/         # optional, unlabelled, never used for metrics
  sample_submission.csv  # optional, placeholder labels, never used
```

With a Kaggle API key in `~/.kaggle/kaggle.json`:

```
pip install kaggle
kaggle competitions download -c aptos2019-blindness-detection -p data/raw/aptos2019
```

then unzip in place.

## IDRiD

Source: IEEE DataPort, "Indian Diabetic Retinopathy Image Dataset (IDRiD)", or
<https://idrid.grand-challenge.org>. Two parts of the download are used: *Disease Grading* and
*Segmentation*. Copy the training **and** testing parts of each into one tree:

```
data/raw/idrid/
  grading/
    images/            # all 516 grading images (IDRiD_001.jpg ...), sub-folders allowed
    labels*.csv        # one or more label CSVs; columns "Image name", "Retinopathy grade"
  segmentation/
    images/            # the 81 annotated images (IDRiD_01.jpg ...), sub-folders allowed
    masks/             # every mask file, any sub-folder layout, named {image}_{MA|HE|EX|SE|OD}.tif
```

The adapter searches these folders recursively and matches files by name, so the original
sub-folders (`a. Training Set`, `1. Microaneurysms`, ...) can be copied as they are. Both label CSVs
(train and test) can be dropped in as `labels_train.csv` and `labels_test.csv`. The two sets reuse
the names `IDRiD_001` ..., so keep their images in the original `a. Training Set` and
`b. Testing Set` folders: a CSV named for a set is matched to that set's folder and its ids become
`train_IDRiD_001` and `test_IDRiD_001`. Folder and column
names are configurable under `data.idrid` in the config. If a column or file name differs,
`prepare-data` stops and says which; do not rename things by guesswork, tell the Engineer.

Segmentation images are numbered separately from grading images, so they get the manifest id
`seg_IDRiD_xx`. If a segmentation image is byte-identical to a grading image it is listed in
`data/manifests/idrid_overlap.csv` and inherits that image's grade. All IDRiD rows are
`external_idrid`.

## Messidor-2

Images: registration form at <https://www.adcis.net/en/third-party/messidor2/>. Labels are
distributed separately (for example the Kaggle dataset `messidor2-dr-grades`).

```
data/raw/messidor2/
  images/              # 1,748 images, sub-folders allowed
  labels.csv           # label file as downloaded
```

Before `prepare-data` will accept Messidor-2, open the label file and fill these keys in
`configs/default.yaml` (F8); the adapter refuses to run with an empty `detection_map`:

```yaml
data:
  messidor2:
    label_file: labels.csv
    id_column: ...            # column holding the image file name
    grade_column: ...         # column holding the native grade
    grade_map: {}             # native grade -> ICDR 0-4, only where the mapping is exact
    detection_map: {}         # native grade -> 0/1 under detection_definition
    detection_definition: referable
```

Rows without a grade and rows without an image file are dropped and counted in the log.

## EyePACS

Out of scope for this build (D4). The adapter is a stub that raises `NotImplementedError`.
