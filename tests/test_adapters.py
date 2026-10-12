from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from xaidr.config import Messidor2Cfg
from xaidr.data.adapters import (
    Aptos2019Adapter,
    DataMissingError,
    EyepacsAdapter,
    IdridAdapter,
    Messidor2Adapter,
)
from xaidr.data.manifest import MASK_COLUMNS, normalise

MESSIDOR_CFG = Messidor2Cfg(
    grade_map={0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
    detection_map={0: 0, 1: 0, 2: 1, 3: 1, 4: 1},
    detection_definition="referable",
)


def test_aptos_manifest(aptos_raw: Path) -> None:
    df = normalise(Aptos2019Adapter().build_manifest(aptos_raw))
    assert len(df) == 100 and set(df["dataset"]) == {"aptos2019"}
    assert df["grade"].value_counts().to_dict() == {0: 40, 1: 20, 2: 20, 3: 10, 4: 10}
    assert (df["dr_label"] == (df["grade"] >= 1).astype(int)).all()
    assert not df["has_masks"].any() and (df["split"] == "").all()
    assert all(Path(p).is_file() for p in df["source_path"])
    # the placeholder ids in sample_submission.csv never become labelled rows (F16)
    assert not df["image_id"].isin(["t1", "t2"]).any()


def test_aptos_reports_missing_files_and_columns(aptos_raw: Path) -> None:
    (aptos_raw / "train_images" / "img000.png").unlink()
    with pytest.raises(DataMissingError, match="img000"):
        Aptos2019Adapter().build_manifest(aptos_raw)
    pd.DataFrame({"id": ["x"], "label": [0]}).to_csv(aptos_raw / "train.csv", index=False)
    with pytest.raises(DataMissingError, match="diagnosis"):
        Aptos2019Adapter().build_manifest(aptos_raw)


def test_idrid_manifest(idrid_raw: Path) -> None:
    adapter = IdridAdapter()
    df = normalise(adapter.build_manifest(idrid_raw))
    grading = df[~df["has_masks"]]
    seg = df[df["has_masks"]].set_index("image_id")
    assert grading.set_index("image_id")["grade"].to_dict() == {
        "IDRiD_001": 0, "IDRiD_002": 2, "IDRiD_003": 4, "IDRiD_004": 1,
    }
    assert (grading["dr_label"] == (grading["grade"] >= 1).astype(int)).all()
    # masks are attached only to segmentation images, and only for lesions that exist
    assert (grading[list(MASK_COLUMNS)] == "").all().all()
    assert list(seg.index) == ["seg_IDRiD_01", "seg_IDRiD_02"]
    assert seg.loc["seg_IDRiD_01", "mask_MA"].endswith("IDRiD_01_MA.tif")
    assert seg.loc["seg_IDRiD_01", "mask_HE"] and seg.loc["seg_IDRiD_01", "mask_EX"] == ""
    assert seg.loc["seg_IDRiD_02", "mask_EX"] and seg.loc["seg_IDRiD_02", "mask_HE"] == ""
    assert (seg["mask_SE"] == "").all()
    # the byte-identical twin is logged and inherits its grade; the other image stays ungraded
    assert adapter.overlap.to_dict("records") == [
        {"seg_image_id": "seg_IDRiD_01", "grading_image_id": "IDRiD_003"}
    ]
    assert seg.loc["seg_IDRiD_01", "grade"] == 4 and pd.isna(seg.loc["seg_IDRiD_02", "grade"])


def test_idrid_reports_unexpected_columns(idrid_raw: Path) -> None:
    labels = idrid_raw / "grading" / "labels.csv"
    pd.read_csv(labels).rename(columns={"Retinopathy grade": "retinopathy_grade"}).to_csv(labels, index=False)
    with pytest.raises(DataMissingError, match="Retinopathy grade"):
        IdridAdapter().build_manifest(idrid_raw)


def test_idrid_train_and_test_sets_share_file_names(tmp_path: Path) -> None:
    """The real download numbers the training and testing sets from IDRiD_001 each."""
    from conftest import fundus, write_image

    raw = tmp_path / "idrid"
    sets = {"a. Training Set": ("labels_train.csv", {"IDRiD_001": 3, "IDRiD_002": 0}),
            "b. Testing Set": ("labels_test.csv", {"IDRiD_001": 4, "IDRiD_002": 1})}
    for n, (folder, (csv, grades)) in enumerate(sets.items()):
        for i, name in enumerate(grades):
            write_image(raw / "grading" / "images" / folder / f"{name}.jpg", fundus(10 * n + i))
        pd.DataFrame({"Image name": list(grades), "Retinopathy grade": list(grades.values())}).to_csv(
            raw / "grading" / csv, index=False)
    seg = raw / "segmentation"
    write_image(seg / "images" / "IDRiD_01.jpg", fundus(99))
    write_image(seg / "masks" / "MA" / "IDRiD_01_MA.tif", fundus(98)[:, :, 0])

    df = IdridAdapter().build_manifest(raw)
    grading = df[~df["has_masks"]].set_index("image_id")
    assert grading["grade"].to_dict() == {
        "train_IDRiD_001": 3, "train_IDRiD_002": 0, "test_IDRiD_001": 4, "test_IDRiD_002": 1,
    }
    # each id points at the image of its own set
    assert all("Training" in grading.loc[i, "source_path"] for i in grading.index if i.startswith("train_"))
    assert all("Testing" in grading.loc[i, "source_path"] for i in grading.index if i.startswith("test_"))


def test_messidor_manifest(messidor_raw: Path) -> None:
    adapter = Messidor2Adapter(MESSIDOR_CFG)
    df = normalise(adapter.build_manifest(messidor_raw))
    assert df.set_index("image_id")["grade"].to_dict() == {"a": 0, "b": 3}
    assert df.set_index("image_id")["dr_label"].to_dict() == {"a": 0, "b": 1}
    assert (adapter.dropped_unlabelled, adapter.dropped_no_image) == (1, 1)
    assert df["label_space"].str.startswith("messidor2:").all()
    assert '"detection_definition": "referable"' in df["label_space"].iloc[0]
    assert not df["has_masks"].any()


def test_messidor_without_grade_map_leaves_grade_empty(messidor_raw: Path) -> None:
    cfg = dataclasses.replace(MESSIDOR_CFG, grade_map={})
    df = normalise(Messidor2Adapter(cfg).build_manifest(messidor_raw))
    assert df["grade"].isna().all() and df["dr_label"].notna().all()


def test_messidor_requires_a_declared_mapping(messidor_raw: Path) -> None:
    with pytest.raises(DataMissingError, match="detection_map"):
        Messidor2Adapter(Messidor2Cfg()).build_manifest(messidor_raw)
    partial = dataclasses.replace(MESSIDOR_CFG, detection_map={0: 0})
    with pytest.raises(DataMissingError, match="not in"):
        Messidor2Adapter(partial).build_manifest(messidor_raw)


def test_eyepacs_is_disabled(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="D4"):
        EyepacsAdapter().build_manifest(tmp_path)
