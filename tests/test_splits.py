from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xaidr.data.manifest import normalise
from xaidr.data.splits import SplitError, SplitLockError, make_splits, verify_split_lock

# APTOS 2019 class counts (3,662 labelled images)
APTOS_COUNTS = {0: 1805, 1: 370, 2: 999, 3: 193, 4: 295}


def manifest(seed: int = 0) -> pd.DataFrame:
    grades = np.repeat(list(APTOS_COUNTS), list(APTOS_COUNTS.values()))
    np.random.default_rng(seed).shuffle(grades)
    aptos = pd.DataFrame(
        {"dataset": "aptos2019", "image_id": [f"a{i:04d}" for i in range(len(grades))],
         "source_path": "x.png", "grade": grades, "dr_label": (grades >= 1).astype(int)}
    )
    external = pd.DataFrame(
        {"dataset": ["idrid", "idrid", "messidor2"], "image_id": ["i1", "i2", "m1"],
         "source_path": "x.jpg", "grade": [0, 3, 2], "dr_label": [0, 1, 1]}
    )
    df = normalise(pd.concat([aptos, external], ignore_index=True))
    return df.assign(content_hash="h-" + df["image_id"])


def test_disjoint_complete_and_sized(tmp_path: Path) -> None:
    df = make_splits(manifest(), splits_dir=tmp_path)
    aptos = df[df["dataset"] == "aptos2019"]
    assert set(aptos["split"]) == {"train", "val", "test"}
    assert aptos["image_id"].is_unique and len(aptos) == 3662
    sets = {name: set(group["image_id"]) for name, group in aptos.groupby("split")}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
    fractions = aptos["split"].value_counts(normalize=True)
    assert abs(fractions["train"] - 0.70) < 0.01
    assert abs(fractions["val"] - 0.15) < 0.01 and abs(fractions["test"] - 0.15) < 0.01


def test_stratified_within_one_percentage_point(tmp_path: Path) -> None:
    df = make_splits(manifest(), splits_dir=tmp_path)
    aptos = df[df["dataset"] == "aptos2019"]
    overall = aptos["grade"].value_counts(normalize=True)
    for _, group in aptos.groupby("split"):
        assert (group["grade"].value_counts(normalize=True) - overall).abs().max() < 0.01


def test_external_sets_are_never_split(tmp_path: Path) -> None:
    df = make_splits(manifest(), splits_dir=tmp_path)
    assert set(df.loc[df["dataset"] == "idrid", "split"]) == {"external_idrid"}
    assert set(df.loc[df["dataset"] == "messidor2", "split"]) == {"external_messidor2"}
    written = pd.read_csv(tmp_path / "aptos2019_split.csv")
    assert set(written["dataset"]) == {"aptos2019"}


def test_reproducible_and_locked(tmp_path: Path) -> None:
    make_splits(manifest(), seed=42, splits_dir=tmp_path)
    csv, lock = tmp_path / "aptos2019_split.csv", tmp_path / "aptos2019_split.sha256"
    first_csv, first_lock = csv.read_bytes(), lock.read_text()
    # same seed, even with the rows in another order, reproduces the identical file
    make_splits(manifest().sample(frac=1.0, random_state=1), seed=42, splits_dir=tmp_path)
    assert csv.read_bytes() == first_csv and lock.read_text() == first_lock

    with pytest.raises(SplitLockError, match="resplit"):
        make_splits(manifest(), seed=43, splits_dir=tmp_path)
    assert csv.read_bytes() == first_csv  # a refused split leaves the locked files untouched
    make_splits(manifest(), seed=43, splits_dir=tmp_path, resplit=True)
    assert csv.read_bytes() != first_csv and lock.read_text() != first_lock


def test_patients_are_kept_together(tmp_path: Path) -> None:
    df = manifest()
    aptos = df["dataset"] == "aptos2019"
    df.loc[aptos, "patient_id"] = [f"p{i // 2}" for i in range(aptos.sum())]
    out = make_splits(df, splits_dir=tmp_path)
    per_patient = out[aptos].groupby("patient_id")["split"].nunique()
    assert (per_patient == 1).all()


def test_identical_images_stay_in_one_split(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    df = manifest()
    aptos = df.index[df["dataset"] == "aptos2019"]
    # 300 pairs of byte-identical files under different ids, 40 of them with conflicting grades
    grade0, grade2 = aptos[df.loc[aptos, "grade"] == 0], aptos[df.loc[aptos, "grade"] == 2]
    df.loc[grade0[300:560], "content_hash"] = df.loc[grade0[:260], "content_hash"].to_numpy()
    df.loc[grade2[:40], "content_hash"] = df.loc[grade0[560:600], "content_hash"].to_numpy()
    df.loc[df["image_id"] == "i1", "content_hash"] = df.loc[aptos[0], "content_hash"]
    with caplog.at_level("WARNING"):
        out = make_splits(df, splits_dir=tmp_path)
    aptos_out = out[out["dataset"] == "aptos2019"]
    used = aptos_out[aptos_out["split"] != ""]
    # one copy of each same-grade pair is kept; conflicting pairs are dropped from every split
    assert len(used) == 3662 - 260 - 80 and used["content_hash"].is_unique
    assert set(used["split"]) == {"train", "val", "test"}
    listed = pd.read_csv(tmp_path / "aptos2019_duplicates.csv", keep_default_na=False)
    assert len(listed) == 600 and listed["content_hash"].nunique() == 300
    assert listed["action"].value_counts().to_dict() == {
        "kept": 260, "dropped_duplicate_copy": 260, "dropped_conflicting_grades": 80,
    }
    assert (listed.loc[listed["action"] != "kept", "split"] == "").all()
    conflicting = listed[listed["action"] == "dropped_conflicting_grades"]
    assert (conflicting.groupby("content_hash")["grade"].nunique() == 2).all()
    assert "300 groups of byte-identical images (600 images). Dropped 260 extra copies" in caplog.text
    assert "80 images in 40 groups with conflicting grades; 3322 images remain" in caplog.text
    assert "byte-identical to an external-test image" in caplog.text
    fractions = used["split"].value_counts(normalize=True)
    assert abs(fractions["train"] - 0.70) < 0.01
    verify_split_lock(out, tmp_path)  # dropped images are part of the locked table
    revived = out.copy()
    revived.loc[aptos_out.index[aptos_out["split"] == ""][0], "split"] = "train"
    with pytest.raises(SplitLockError):
        verify_split_lock(revived, tmp_path)


def test_split_needs_content_hashes(tmp_path: Path) -> None:
    with pytest.raises(SplitError, match="prepare-data"):
        make_splits(manifest().assign(content_hash=""), splits_dir=tmp_path)


def test_verify_split_lock(tmp_path: Path) -> None:
    with pytest.raises(SplitLockError, match="make-splits"):
        verify_split_lock(manifest(), tmp_path)
    df = make_splits(manifest(), splits_dir=tmp_path)
    verify_split_lock(df, tmp_path)
    verify_split_lock(df.sample(frac=1.0, random_state=3), tmp_path)  # row order does not matter
    moved = df.copy()
    moved.loc[moved.index[moved["split"] == "test"][0], "split"] = "train"
    with pytest.raises(SplitLockError, match="does not match"):
        verify_split_lock(moved, tmp_path)


def test_too_small_a_class_is_a_readable_error(tmp_path: Path) -> None:
    df = manifest()
    df = pd.concat([df[df["grade"] != 4], df[df["grade"] == 4].head(2)], ignore_index=True)
    with pytest.raises(SplitError, match="grade counts"):
        make_splits(df, splits_dir=tmp_path)
    assert not (tmp_path / "aptos2019_split.sha256").exists()
