"""Torch datasets and loaders over the canonical manifest."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from xaidr.config import Config
from xaidr.data.augment import eval_transform, train_transform
from xaidr.data.oversample import minority_sampler
from xaidr.data.preprocess import preprocess_record, read_image
from xaidr.data.splits import verify_split_lock

MISSING_LABEL = -1  # used where a dataset has no five-class grade (e.g. Messidor-2 without a map)


def _load_rgb(path: str) -> np.ndarray:
    return cv2.cvtColor(read_image(path), cv2.COLOR_BGR2RGB)


def _label(value: Any) -> int:
    return MISSING_LABEL if pd.isna(value) else int(value)


class DRDataset(Dataset):
    """Returns ``(image_tensor, dr_label, grade, image_id)`` from the processed images."""

    def __init__(self, manifest_df: pd.DataFrame, transform: Callable, task_labels: bool = True) -> None:
        if (manifest_df["processed_path"] == "").any():
            raise ValueError("every row needs a processed_path; run `xaidr.cli prepare-data` first")
        self.df = manifest_df.reset_index(drop=True)
        self.transform = transform
        self.task_labels = task_labels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int, str]:
        row = self.df.iloc[index]
        image = self.transform(image=_load_rgb(row["processed_path"]))["image"]
        if not self.task_labels:
            return image, MISSING_LABEL, MISSING_LABEL, row["image_id"]
        return image, _label(row["dr_label"]), _label(row["grade"]), row["image_id"]


class LesionDataset(Dataset):
    """Rows with lesion masks only. Returns ``(image_tensor, masks_dict, grade, image_id)``.

    Masks are binary uint8 tensors at the model input size, produced from the source masks
    with the crop box stored in the manifest (the same path as the image).
    """

    def __init__(self, manifest_df: pd.DataFrame, size: int, transform: Callable | None = None) -> None:
        self.df = manifest_df[manifest_df["has_masks"]].reset_index(drop=True)
        if self.df[["crop_y0", "crop_x0", "crop_y1", "crop_x1"]].isna().any().any():
            raise ValueError("lesion rows need a stored crop box; run `xaidr.cli prepare-data` first")
        self.size = size
        self.transform = transform or eval_transform(size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], int, str]:
        row = self.df.iloc[index].to_dict()
        _, masks = preprocess_record(row, self.size)
        image = self.transform(image=_load_rgb(row["processed_path"]))["image"]
        mask_tensors = {k: torch.from_numpy(v) for k, v in masks.items()}
        return image, mask_tensors, _label(row["grade"]), row["image_id"]


def _seed_worker_augmentation(worker_id: int) -> None:
    """Give each worker its own augmentation stream, derived from the loader's seeded generator."""
    info = get_worker_info()
    if info is not None:
        cast(Any, info.dataset).transform.set_random_seed(info.seed % 2**32)


def make_train_loader(manifest_df: pd.DataFrame, cfg: Config) -> DataLoader:
    """The only loader that augments and oversamples; it only ever sees ``split == 'train'``."""
    verify_split_lock(manifest_df, cfg.paths.splits)
    train_df = manifest_df[manifest_df["split"] == "train"].reset_index(drop=True)
    dataset = DRDataset(train_df, train_transform(cfg.data.image_size, cfg.seed))
    sampler = minority_sampler(
        train_df, cfg.minority.target_grades, cfg.minority.factor, cfg.seed, cfg.minority.mode
    )
    return DataLoader(
        dataset, batch_size=cfg.train.batch_size, sampler=sampler,
        num_workers=cfg.train.num_workers, drop_last=True, pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker_augmentation, generator=torch.Generator().manual_seed(cfg.seed),
    )


def make_eval_loader(manifest_df: pd.DataFrame, split: str, cfg: Config) -> DataLoader:
    """Deterministic, un-augmented loader for validation, test and external splits."""
    if split == "train":
        raise ValueError("use make_train_loader for the training split")
    verify_split_lock(manifest_df, cfg.paths.splits)
    subset = manifest_df[manifest_df["split"] == split].reset_index(drop=True)
    dataset = DRDataset(subset, eval_transform(cfg.data.image_size))
    return DataLoader(
        dataset, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, pin_memory=torch.cuda.is_available(),
    )
