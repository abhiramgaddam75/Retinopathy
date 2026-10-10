"""Albumentations pipelines. ``train_transform`` may only be attached to the training loader."""
from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

# Fixed constants, never dataset statistics (leakage rule 4, DESIGN.md Section 8.5).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _check_size(size: int) -> None:
    if size < 1:
        raise ValueError(f"size must be positive, got {size}")


def train_transform(size: int) -> A.Compose:
    """Flips, rotation +-30 deg and brightness/contrast jitter, then normalisation.

    Flips change the laterality of the optic disc and fovea; this is standard for DR training.
    """
    _check_size(size)
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def eval_transform(size: int) -> A.Compose:
    """Normalisation and tensor conversion only."""
    _check_size(size)
    return A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])
