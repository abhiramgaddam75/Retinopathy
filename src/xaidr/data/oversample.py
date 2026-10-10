"""Minority-class oversampling as an index-level sampler (D2).

Duplicates are indices only: every draw of a duplicated image passes through the training
augmentation again, so no augmented copy is ever stored or shared with another split.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence

import pandas as pd
import torch
from torch.utils.data import Sampler


class ShuffledIndexSampler(Sampler[int]):
    """Yields a fixed multiset of indices in a new seeded order every epoch."""

    def __init__(self, indices: Sequence[int], seed: int) -> None:
        self.indices = list(indices)
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self) -> Iterator[int]:
        order = torch.randperm(len(self.indices), generator=self.generator).tolist()
        return iter(self.indices[i] for i in order)

    def __len__(self) -> int:
        return len(self.indices)


def minority_sampler(
    manifest_df: pd.DataFrame,
    target_grades: Sequence[int] = (3, 4),
    factor: int = 3,
    seed: int = 42,
    mode: str = "oversample",
) -> ShuffledIndexSampler:
    """Sampler over the positions of ``manifest_df``, which must hold training rows only."""
    if mode not in ("none", "oversample"):
        raise ValueError(f"minority mode must be 'none' or 'oversample', got '{mode}'")
    not_train = manifest_df["split"] != "train"
    if not_train.any():
        raise ValueError(
            f"the sampler only accepts training rows; found split(s) "
            f"{sorted(set(manifest_df.loc[not_train, 'split']))}"
        )
    indices: list[int] = []
    for position, grade in enumerate(manifest_df["grade"]):
        repeat = factor if mode == "oversample" and not pd.isna(grade) and int(grade) in target_grades else 1
        indices.extend([position] * repeat)
    return ShuffledIndexSampler(indices, seed)
