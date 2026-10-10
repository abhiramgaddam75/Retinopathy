"""Leakage rules from DESIGN.md Section 8.5."""
from __future__ import annotations

import dataclasses
import re
from collections import Counter
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import pytest
import torch

from conftest import REPO_ROOT
from xaidr.config import load_config
from xaidr.data import augment
from xaidr.data.adapters import Aptos2019Adapter, IdridAdapter
from xaidr.data.dataset import DRDataset, LesionDataset, make_eval_loader, make_train_loader
from xaidr.data.manifest import normalise
from xaidr.data.oversample import minority_sampler
from xaidr.data.preprocess import build_processed
from xaidr.data.splits import SplitLockError, make_splits

SIZE = 224
RANDOM_TRANSFORMS = (A.HorizontalFlip, A.VerticalFlip, A.Rotate, A.RandomBrightnessContrast)


@pytest.fixture
def prepared(aptos_raw: Path, idrid_raw: Path, tmp_path: Path) -> pd.DataFrame:
    manifest = pd.concat(
        [normalise(Aptos2019Adapter().build_manifest(aptos_raw)),
         normalise(IdridAdapter().build_manifest(idrid_raw))], ignore_index=True,
    )
    manifest = build_processed(manifest, tmp_path / "processed", SIZE, num_workers=2)
    return make_splits(manifest, splits_dir=tmp_path / "splits")


@pytest.fixture
def cfg(tmp_path: Path):
    return load_config(
        REPO_ROOT / "configs" / "default.yaml",
        [f"data.image_size={SIZE}", "train.batch_size=4", "train.num_workers=0",
         f"paths.splits={(tmp_path / 'splits').as_posix()}"],
    )


def test_train_loader_sees_training_rows_only(prepared: pd.DataFrame, cfg) -> None:
    loader = make_train_loader(prepared, cfg)
    train_ids = set(prepared.loc[prepared["split"] == "train", "image_id"])
    assert set(loader.dataset.df["image_id"]) == train_ids
    assert set(loader.dataset.df["dataset"]) == {"aptos2019"}
    drawn = {loader.dataset.df.loc[i, "image_id"] for i in loader.sampler}
    assert drawn <= train_ids
    external = set(prepared.loc[prepared["split"].str.startswith("external"), "image_id"])
    assert external and not (drawn & external)


def test_augmentation_only_in_training(prepared: pd.DataFrame, cfg) -> None:
    train_types = {type(t) for t in make_train_loader(prepared, cfg).dataset.transform.transforms}
    assert set(RANDOM_TRANSFORMS) <= train_types
    for split in ("val", "test", "external_idrid"):
        loader = make_eval_loader(prepared, split, cfg)
        assert set(loader.dataset.df["split"]) == {split}
        types = {type(t) for t in loader.dataset.transform.transforms}
        assert not (types & set(RANDOM_TRANSFORMS))
        first = loader.dataset[0][0]
        assert torch.equal(first, loader.dataset[0][0])  # evaluation inputs are deterministic
    with pytest.raises(ValueError, match="make_train_loader"):
        make_eval_loader(prepared, "train", cfg)


def test_train_transform_is_referenced_only_by_the_training_loader() -> None:
    uses = []
    for path in (REPO_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        uses += [(path.name, m.start(), text) for m in re.finditer(r"train_transform\(", text)]
    assert sorted({name for name, _, _ in uses}) == ["augment.py", "dataset.py"]
    for name, start, text in uses:
        if name == "dataset.py":
            enclosing = re.findall(r"^def (\w+)", text[:start], flags=re.M)[-1]
            assert enclosing == "make_train_loader"


def test_normalisation_uses_imagenet_constants() -> None:
    assert augment.IMAGENET_MEAN == (0.485, 0.456, 0.406)
    assert augment.IMAGENET_STD == (0.229, 0.224, 0.225)
    for build in (augment.train_transform, augment.eval_transform):
        norm = [t for t in build(SIZE).transforms if isinstance(t, A.Normalize)]
        assert len(norm) == 1
        assert tuple(norm[0].mean) == augment.IMAGENET_MEAN and tuple(norm[0].std) == augment.IMAGENET_STD


def test_sampler_matches_the_expected_multiplier(prepared: pd.DataFrame) -> None:
    train = prepared[prepared["split"] == "train"].reset_index(drop=True)
    base = Counter(train["grade"].astype(int))
    for mode, factor in (("oversample", 3), ("none", 1)):
        sampler = minority_sampler(train, (3, 4), 3, seed=42, mode=mode)
        epoch = list(sampler)
        drawn = Counter(int(train.loc[i, "grade"]) for i in epoch)
        expected = {g: n * (factor if g in (3, 4) else 1) for g, n in base.items()}
        assert drawn == expected and len(sampler) == sum(expected.values())
        assert list(sampler) != epoch  # reshuffled every epoch
        assert list(minority_sampler(train, (3, 4), 3, seed=42, mode=mode)) == epoch  # seeded


def test_sampler_rejects_non_training_rows(prepared: pd.DataFrame) -> None:
    for split in ("val", "test", "external_idrid"):
        with pytest.raises(ValueError, match="training rows"):
            minority_sampler(prepared[prepared["split"].isin(["train", split])])


def test_oversampled_duplicates_are_augmented_independently(prepared: pd.DataFrame, cfg) -> None:
    dataset = make_train_loader(prepared, cfg).dataset
    draws = [dataset[0][0] for _ in range(8)]
    assert any(not torch.equal(draws[0], d) for d in draws[1:])


def test_dataset_outputs(prepared: pd.DataFrame, cfg) -> None:
    image, dr_label, grade, image_id = make_eval_loader(prepared, "val", cfg).dataset[0]
    assert image.shape == (3, SIZE, SIZE) and image.dtype == torch.float32
    assert dr_label in (0, 1) and 0 <= grade <= 4 and dr_label == int(grade >= 1)
    assert isinstance(image_id, str)
    batch = next(iter(make_train_loader(prepared, cfg)))
    assert batch[0].shape == (4, 3, SIZE, SIZE) and batch[2].dtype == torch.int64

    lesions = LesionDataset(prepared, SIZE)
    assert len(lesions) == 2 and set(lesions.df["split"]) == {"external_idrid"}
    image, masks, grade, image_id = lesions[0]
    assert image.shape == (3, SIZE, SIZE) and image_id == "seg_IDRiD_01"
    assert set(masks) == {"MA", "HE"}
    for mask in masks.values():
        assert mask.shape == (SIZE, SIZE) and set(np.unique(mask.numpy())) <= {0, 1}
    with pytest.raises(ValueError, match="processed_path"):
        DRDataset(prepared.assign(processed_path=""), augment.eval_transform(SIZE))


def test_split_lock_is_written(prepared: pd.DataFrame, tmp_path: Path) -> None:
    lock = tmp_path / "splits" / "aptos2019_split.sha256"
    assert lock.is_file() and len(lock.read_text().split()[0]) == 64


def test_loaders_refuse_a_manifest_that_differs_from_the_lock(
    prepared: pd.DataFrame, cfg, tmp_path: Path
) -> None:
    edited = prepared.copy()
    edited.loc[edited["split"] == "test", "split"] = "train"
    with pytest.raises(SplitLockError, match="does not match the locked split"):
        make_train_loader(edited, cfg)
    with pytest.raises(SplitLockError, match="does not match the locked split"):
        make_eval_loader(edited, "val", cfg)
    (tmp_path / "splits" / "aptos2019_split.sha256").unlink()
    with pytest.raises(SplitLockError, match="make-splits"):
        make_train_loader(prepared, cfg)


def first_batches(prepared: pd.DataFrame, cfg, epochs: int = 2) -> list[torch.Tensor]:
    loader = make_train_loader(prepared, cfg)
    return [next(iter(loader))[0] for _ in range(epochs)]


def test_training_batches_follow_the_seed(prepared: pd.DataFrame, cfg) -> None:
    first, second = first_batches(prepared, cfg), first_batches(prepared, cfg)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert not torch.equal(first[0], first[1])  # a new epoch draws new augmentations
    other = first_batches(prepared, dataclasses.replace(cfg, seed=7))
    assert not torch.equal(first[0], other[0])


@pytest.mark.slow
def test_training_batches_follow_the_seed_with_workers(prepared: pd.DataFrame, cfg) -> None:
    workers = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, num_workers=2))
    first, second = first_batches(prepared, workers), first_batches(prepared, workers)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert not torch.equal(first[0], first[1])
