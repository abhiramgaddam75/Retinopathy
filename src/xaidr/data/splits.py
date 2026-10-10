"""Stratified 70/15/15 split of APTOS 2019, locked by SHA-256 (DESIGN.md Section 7.6)."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from xaidr.data.manifest import normalise

log = logging.getLogger(__name__)

SPLIT_DATASET = "aptos2019"
EXTERNAL_SPLITS = {"idrid": "external_idrid", "messidor2": "external_messidor2"}


class SplitLockError(RuntimeError):
    """The computed split differs from the locked one."""


def _assign(units: pd.DataFrame, ratios: tuple[float, float, float], seed: int) -> pd.Series:
    """Two stratified cuts: train vs rest, then rest into val and test."""
    train, rest = train_test_split(
        units, train_size=ratios[0], stratify=units["label"], random_state=seed
    )
    val, test = train_test_split(
        rest, train_size=ratios[1] / (ratios[1] + ratios[2]), stratify=rest["label"], random_state=seed
    )
    split = pd.Series("", index=units.index, dtype=object)
    split[train.index], split[val.index], split[test.index] = "train", "val", "test"
    return split


def make_splits(
    manifest_df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    unit: str = "patient_or_image",
    splits_dir: str | Path = "data/splits",
    resplit: bool = False,
) -> pd.DataFrame:
    if unit != "patient_or_image":
        raise ValueError(f"unsupported split unit '{unit}'")
    df = normalise(manifest_df)
    for dataset, split in EXTERNAL_SPLITS.items():
        df.loc[df["dataset"] == dataset, "split"] = split

    aptos = df[df["dataset"] == SPLIT_DATASET]
    if aptos.empty:
        raise ValueError(f"the manifest has no '{SPLIT_DATASET}' rows to split")
    if aptos["grade"].isna().any():
        raise ValueError("every APTOS 2019 row needs a grade before splitting")
    # A patient is one unit when known; otherwise each image is its own unit.
    unit_key = aptos["patient_id"].where(aptos["patient_id"] != "", "img:" + aptos["image_id"])
    units = aptos.assign(unit=unit_key).groupby("unit", sort=True)["grade"].max().rename("label").to_frame()
    unit_split = _assign(units, ratios, seed)
    df.loc[aptos.index, "split"] = unit_key.map(unit_split).to_numpy()

    table = (
        df.loc[aptos.index, ["dataset", "image_id", "split"]]
        .sort_values("image_id")
        .to_csv(index=False, lineterminator="\n")
        .encode("utf-8")
    )
    digest = hashlib.sha256(table).hexdigest()
    splits_dir = Path(splits_dir)
    csv_path = splits_dir / f"{SPLIT_DATASET}_split.csv"
    lock_path = splits_dir / f"{SPLIT_DATASET}_split.sha256"
    if lock_path.is_file() and not resplit:
        locked = lock_path.read_text(encoding="utf-8").split()[0]
        if locked != digest:
            raise SplitLockError(
                f"the computed split ({digest[:12]}) differs from the locked split ({locked[:12]}) in "
                f"{lock_path}. The data, seed or ratios changed; pass --resplit only if this is intended."
            )
    splits_dir.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(table)
    lock_path.write_text(f"{digest}  {csv_path.name}\n", encoding="utf-8")
    counts = df.loc[aptos.index].groupby(["split", "grade"]).size().unstack(fill_value=0)
    log.info("APTOS 2019 split (seed %d), images per grade:\n%s", seed, counts)
    return df
