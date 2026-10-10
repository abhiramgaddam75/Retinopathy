"""Canonical manifest: one row per image (DESIGN.md Section 7.3)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATASETS = ("aptos2019", "idrid", "messidor2")
SPLITS = ("", "train", "val", "test", "external_idrid", "external_messidor2")
LESIONS = ("MA", "HE", "EX", "SE", "OD")
MASK_COLUMNS = tuple(f"mask_{name}" for name in LESIONS)

STR_COLUMNS = (
    "dataset", "image_id", "source_path", "label_space", "patient_id",
    *MASK_COLUMNS, "processed_path", "split",
)
INT_COLUMNS = ("grade", "dr_label", "orig_h", "orig_w", "crop_y0", "crop_x0", "crop_y1", "crop_x1")
BOOL_COLUMNS = ("has_masks",)
CROP_COLUMNS = ("crop_y0", "crop_x0", "crop_y1", "crop_x1")

COLUMNS = (
    "dataset", "image_id", "source_path", "grade", "dr_label", "label_space", "patient_id",
    "has_masks", *MASK_COLUMNS, "orig_h", "orig_w", *CROP_COLUMNS, "processed_path", "split",
)


class ManifestError(ValueError):
    """Raised when a manifest does not follow the canonical schema."""


def _to_bool(value: object, column: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    raise ManifestError(f"column '{column}' must be boolean, got {value!r}")


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Add optional columns that are still empty, coerce dtypes and validate."""
    out = df.copy()
    required = ("dataset", "image_id", "source_path")
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ManifestError(f"manifest is missing required column(s): {missing}")
    for col in COLUMNS:
        if col not in out.columns:
            out[col] = False if col in BOOL_COLUMNS else (pd.NA if col in INT_COLUMNS else "")
    return _coerce(out[list(COLUMNS)])


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ManifestError(f"manifest is missing required column(s): {missing}")
    out = df.copy()
    for col in STR_COLUMNS:
        out[col] = out[col].fillna("").astype(str)
    for col in INT_COLUMNS:
        series = out[col].replace("", pd.NA)
        try:
            numeric = pd.to_numeric(series, errors="raise")
        except (ValueError, TypeError) as exc:
            raise ManifestError(f"column '{col}' must hold integers or be empty: {exc}") from exc
        if (numeric.dropna() % 1 != 0).any():
            raise ManifestError(f"column '{col}' must hold integers, found a non-integer value")
        out[col] = numeric.astype("Int64")
    for col in BOOL_COLUMNS:
        out[col] = out[col].map(lambda v, c=col: _to_bool(v, c)).astype(bool)
    validate_manifest(out)
    return out


def validate_manifest(df: pd.DataFrame) -> None:
    bad = sorted(set(df["dataset"]) - set(DATASETS))
    if bad:
        raise ManifestError(f"column 'dataset' has unknown value(s): {bad}")
    if (df["image_id"] == "").any():
        raise ManifestError("column 'image_id' must not be empty")
    dup = df.duplicated(subset=["dataset", "image_id"])
    if dup.any():
        raise ManifestError(
            f"image_id must be unique within a dataset; duplicate: {df.loc[dup, 'image_id'].iloc[0]}"
        )
    grades = df["grade"].dropna()
    if not grades.isin(range(5)).all():
        raise ManifestError("column 'grade' must be in 0..4 or empty")
    labels = df["dr_label"].dropna()
    if not labels.isin((0, 1)).all():
        raise ManifestError("column 'dr_label' must be 0, 1 or empty")
    bad = sorted(set(df["split"]) - set(SPLITS))
    if bad:
        raise ManifestError(f"column 'split' has unknown value(s): {bad}")
    any_mask = (df[list(MASK_COLUMNS)] != "").any(axis=1)
    if (any_mask & ~df["has_masks"]).any():
        raise ManifestError("mask paths are set on a row where has_masks is False")
    if (df["has_masks"] & ~any_mask).any():
        raise ManifestError("has_masks is True on a row without any mask path")


def read_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"manifest not found: {path}")
    return _coerce(pd.read_csv(path, dtype=str, keep_default_na=False))


def write_manifest(df: pd.DataFrame, path: str | Path) -> None:
    out = _coerce(df)[list(COLUMNS)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, lineterminator="\n")
