from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from xaidr.data.manifest import COLUMNS, ManifestError, normalise, read_manifest, write_manifest


def valid() -> pd.DataFrame:
    return normalise(
        pd.DataFrame(
            {
                "dataset": ["aptos2019", "idrid"],
                "image_id": ["a", "seg_b"],
                "source_path": ["a.png", "b.jpg"],
                "grade": [2, pd.NA],
                "dr_label": [1, pd.NA],
                "label_space": "icdr5",
                "has_masks": [False, True],
                "mask_MA": ["", "b_MA.tif"],
            }
        )
    )


def test_valid_manifest_round_trips(tmp_path: Path) -> None:
    df = valid()
    assert tuple(df.columns) == COLUMNS
    path = tmp_path / "m" / "manifest.csv"
    write_manifest(df, path)
    back = read_manifest(path)
    pd.testing.assert_frame_equal(back, df)
    assert str(back["grade"].dtype) == "Int64" and back["has_masks"].dtype == bool
    assert pd.isna(back.loc[1, "grade"]) and back.loc[1, "has_masks"]


def test_missing_column_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    valid().drop(columns=["crop_y0"]).to_csv(path, index=False)
    with pytest.raises(ManifestError, match="crop_y0"):
        read_manifest(path)
    with pytest.raises(ManifestError, match="source_path"):
        normalise(pd.DataFrame({"dataset": ["idrid"], "image_id": ["x"]}))


@pytest.mark.parametrize(
    "column, value, match",
    [
        ("grade", "severe", "grade"),
        ("grade", 2.5, "grade"),
        ("grade", 7, "grade"),
        ("dr_label", 2, "dr_label"),
        ("has_masks", "maybe", "has_masks"),
        ("dataset", "eyepacs", "dataset"),
        ("split", "holdout", "split"),
    ],
)
def test_invalid_values_are_rejected(column: str, value: object, match: str) -> None:
    df = valid().astype({column: object})
    df.loc[0, column] = value
    with pytest.raises(ManifestError, match=match):
        normalise(df)


def test_duplicate_ids_and_orphan_masks_are_rejected() -> None:
    df = valid()
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ManifestError, match="unique"):
        normalise(dup)
    orphan = valid()
    orphan.loc[0, "mask_HE"] = "x_HE.tif"
    with pytest.raises(ManifestError, match="has_masks"):
        normalise(orphan)
