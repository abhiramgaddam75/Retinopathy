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
KEPT, DROPPED_COPY, DROPPED_CONFLICT = "kept", "dropped_duplicate_copy", "dropped_conflicting_grades"


class SplitError(RuntimeError):
    """The split cannot be made from this manifest."""


class SplitLockError(SplitError):
    """The split in use differs from the locked one, or no lock exists."""


def _assign(units: pd.DataFrame, ratios: tuple[float, float, float], seed: int) -> pd.Series:
    """Two stratified cuts: train vs rest, then rest into val and test."""
    try:
        train, rest = train_test_split(
            units, train_size=ratios[0], stratify=units["label"], random_state=seed
        )
        val, test = train_test_split(
            rest, train_size=ratios[1] / (ratios[1] + ratios[2]), stratify=rest["label"],
            random_state=seed,
        )
    except ValueError as exc:
        counts = units["label"].value_counts().sort_index().to_dict()
        raise SplitError(
            f"cannot make a stratified {ratios} split from these grade counts {counts}: {exc}"
        ) from exc
    labels = set(units["label"])
    for name, part in (("train", train), ("val", val), ("test", test)):
        absent = sorted(labels - set(part["label"]))
        if absent:
            counts = units["label"].value_counts().sort_index().to_dict()
            raise SplitError(
                f"grade(s) {absent} would be absent from the {name} split; grade counts {counts} "
                f"are too small for a stratified {ratios} split"
            )
    split = pd.Series("", index=units.index, dtype=object)
    split[train.index], split[val.index], split[test.index] = "train", "val", "test"
    return split


def _split_table(df: pd.DataFrame) -> bytes:
    """Canonical bytes of the locked split: independent of row order and platform."""
    rows = df.loc[df["dataset"] == SPLIT_DATASET, ["dataset", "image_id", "split"]]
    return rows.sort_values("image_id").to_csv(index=False, lineterminator="\n").encode("utf-8")


def _lock_paths(splits_dir: str | Path) -> tuple[Path, Path]:
    splits_dir = Path(splits_dir)
    return splits_dir / f"{SPLIT_DATASET}_split.csv", splits_dir / f"{SPLIT_DATASET}_split.sha256"


def verify_split_lock(manifest_df: pd.DataFrame, splits_dir: str | Path) -> None:
    """Refuse a manifest whose APTOS split column is not exactly the locked split."""
    _, lock_path = _lock_paths(splits_dir)
    if not lock_path.is_file():
        raise SplitLockError(f"no split lock at {lock_path}; run `xaidr.cli make-splits` first")
    locked = lock_path.read_text(encoding="utf-8").split()[0]
    digest = hashlib.sha256(_split_table(manifest_df)).hexdigest()
    if digest != locked:
        raise SplitLockError(
            f"the manifest's split column ({digest[:12]}) does not match the locked split "
            f"({locked[:12]}) in {lock_path}; the manifest was edited or belongs to another split"
        )


def _unit_keys(aptos: pd.DataFrame) -> pd.Series:
    """Splitting unit per image: the patient when known, else the image's content hash.
    Callers pass de-duplicated rows, so a hash names one image; keying on it still guarantees
    that copies of one photograph could never sit in two splits."""
    key = "dup:" + aptos["content_hash"]
    return key.where(aptos["patient_id"] == "", "pat:" + aptos["patient_id"])


def _duplicate_actions(aptos: pd.DataFrame) -> pd.Series:
    """What happens to each APTOS image because of byte-identical copies (Researcher rule, ISSUE-016).

    A group of copies with one grade keeps its first image and drops the rest, so no photograph
    is weighted or counted twice. A group with conflicting grades has no defined reference
    label and is dropped from every split.
    """
    action = pd.Series(KEPT, index=aptos.index, dtype=object)
    for _, group in aptos.sort_values("image_id").groupby("content_hash", sort=False):
        if len(group) == 1:
            continue
        if group["grade"].nunique() > 1:
            action[group.index] = DROPPED_CONFLICT
        else:
            action[group.index[1:]] = DROPPED_COPY
    return action


def _report_duplicates(df: pd.DataFrame, action: pd.Series, splits_dir: Path) -> None:
    hashed = df[df["content_hash"] != ""]
    aptos = df.loc[action.index].assign(action=action)
    sizes = aptos.groupby("content_hash")["image_id"].transform("size")
    listing = aptos[sizes > 1].sort_values(["content_hash", "image_id"])
    listing[["content_hash", "image_id", "grade", "action", "split"]].to_csv(
        splits_dir / f"{SPLIT_DATASET}_duplicates.csv", index=False, lineterminator="\n"
    )
    if len(listing):
        conflict = listing[listing["action"] == DROPPED_CONFLICT]
        log.warning(
            "APTOS 2019: %d groups of byte-identical images (%d images). Dropped %d extra copies of "
            "same-grade images and %d images in %d groups with conflicting grades; %d images remain. "
            "See %s_duplicates.csv",
            listing["content_hash"].nunique(), len(listing), int((listing["action"] == DROPPED_COPY).sum()),
            len(conflict), conflict["content_hash"].nunique(), int((action == KEPT).sum()), SPLIT_DATASET,
        )
    shared = set(aptos["content_hash"]) & set(hashed.loc[hashed["dataset"] != SPLIT_DATASET, "content_hash"])
    if shared:
        log.warning(
            "%d APTOS 2019 image(s) are byte-identical to an external-test image: the external "
            "sets are not fully independent", len(shared),
        )


def make_splits(
    manifest_df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    unit: str = "patient_or_image",
    splits_dir: str | Path = "data/splits",
    resplit: bool = False,
) -> pd.DataFrame:
    if unit != "patient_or_image":
        raise SplitError(f"unsupported split unit '{unit}'")
    df = normalise(manifest_df)
    for dataset, split in EXTERNAL_SPLITS.items():
        df.loc[df["dataset"] == dataset, "split"] = split

    aptos = df[df["dataset"] == SPLIT_DATASET]
    if aptos.empty:
        raise SplitError(f"the manifest has no '{SPLIT_DATASET}' rows to split")
    if aptos["grade"].isna().any():
        raise SplitError("every APTOS 2019 row needs a grade before splitting")
    if (aptos["content_hash"] == "").any():
        # without the hashes, copies of one photograph could land in two splits
        raise SplitError(
            "APTOS 2019 rows have no content hash; run `xaidr.cli prepare-data` before make-splits"
        )
    action = _duplicate_actions(aptos)
    kept = aptos[action == KEPT]
    unit_key = _unit_keys(kept)
    units = kept.assign(unit=unit_key).groupby("unit", sort=True)["grade"].max().rename("label").to_frame()
    unit_split = _assign(units, ratios, seed)
    df.loc[aptos.index, "split"] = ""  # dropped images belong to no split
    df.loc[kept.index, "split"] = unit_key.map(unit_split).to_numpy()

    table = _split_table(df)
    digest = hashlib.sha256(table).hexdigest()
    splits_dir = Path(splits_dir)
    csv_path, lock_path = _lock_paths(splits_dir)
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
    _report_duplicates(df, action, splits_dir)
    counts = df.loc[kept.index].groupby(["split", "grade"]).size().unstack(fill_value=0)
    log.info("APTOS 2019 split (seed %d), images per grade:\n%s", seed, counts)
    return df
