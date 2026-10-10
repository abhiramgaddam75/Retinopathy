"""Tester sanity tests (VERIFICATION_AND_RESEARCH_BRIEF.md, A3) and Tester-added M0 checks.

S1-S8 and S10 need the metrics, threshold, loss and fidelity modules (milestones M1/M2) and are
added when those exist. Tests marked xfail document an open defect in docs/DEFECTS.md; they are
strict, so they start failing the suite as soon as the defect is fixed and the marker must go.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import albumentations as A
import pandas as pd
import pytest
import torch

from conftest import REPO_ROOT
from xaidr.config import load_config
from xaidr.data.adapters import Aptos2019Adapter, IdridAdapter
from xaidr.data.augment import train_transform
from xaidr.data.dataset import make_eval_loader, make_train_loader
from xaidr.data.manifest import normalise
from xaidr.data.preprocess import build_processed
from xaidr.data.splits import make_splits
from xaidr.seed import set_seed

SIZE = 224
OVERRIDES = [f"data.image_size={SIZE}", "train.batch_size=4", "train.num_workers=0"]


@pytest.fixture
def prepared(aptos_raw: Path, idrid_raw: Path, tmp_path: Path) -> pd.DataFrame:
    manifest = pd.concat(
        [normalise(Aptos2019Adapter().build_manifest(aptos_raw)),
         normalise(IdridAdapter().build_manifest(idrid_raw))], ignore_index=True,
    )
    manifest = build_processed(manifest, tmp_path / "processed", SIZE, num_workers=2)
    return make_splits(manifest, splits_dir=tmp_path / "splits")


def config(*extra: str):
    return load_config(REPO_ROOT / "configs" / "default.yaml", [*OVERRIDES, *extra])


def test_s9_eval_loaders_are_deterministic(prepared: pd.DataFrame) -> None:
    """S9: no random transform on evaluation loaders; two passes give identical outputs."""
    cfg = config()
    for split in ("val", "test", "external_idrid"):
        loader = make_eval_loader(prepared, split, cfg)
        randomised = [t for t in loader.dataset.transform.transforms
                      if not isinstance(t, (A.Normalize, A.pytorch.ToTensorV2))]
        assert randomised == [], f"{split}: unexpected transform(s) {randomised}"
        first = [(images, list(ids)) for images, _, _, ids in loader]
        second = [(images, list(ids)) for images, _, _, ids in loader]
        assert len(first) == len(second) > 0
        for (images_a, ids_a), (images_b, ids_b) in zip(first, second, strict=True):
            assert ids_a == ids_b and torch.equal(images_a, images_b)


@pytest.mark.parametrize("mode, factor", [("oversample", 3), ("none", 1)])
def test_training_loader_applies_the_configured_oversampling(
    prepared: pd.DataFrame, mode: str, factor: int
) -> None:
    """R5: the loader, not only the sampler function, must honour ``minority.mode``."""
    loader = make_train_loader(prepared, config(f"minority.mode={mode}"))
    frame = loader.dataset.df
    base = Counter(frame["grade"].astype(int))
    drawn = Counter(int(frame.loc[i, "grade"]) for i in loader.sampler)
    assert drawn == {g: n * (factor if g in (3, 4) else 1) for g, n in base.items()}


def test_ablation_configs_differ_only_in_minority_mode() -> None:
    """J3: the two ablation arms share every setting except ``minority.mode``."""
    off = load_config(REPO_ROOT / "configs" / "ablation_no_oversampling.yaml").to_dict()
    on = load_config(REPO_ROOT / "configs" / "ablation_oversampling.yaml").to_dict()
    assert (off["minority"].pop("mode"), on["minority"].pop("mode")) == ("none", "oversample")
    assert off == on
    assert on | {"minority": {**on["minority"], "mode": "oversample"}} == load_config(
        REPO_ROOT / "configs" / "default.yaml"
    ).to_dict()


@pytest.mark.xfail(strict=True, reason="DEF-001: training augmentation ignores the seed")
def test_training_augmentation_is_reproducible_under_the_seed() -> None:
    """R6: the same seed must give the same augmented training inputs."""
    image = torch.randint(0, 255, (SIZE, SIZE, 3), generator=torch.Generator().manual_seed(0))
    image = image.numpy().astype("uint8")
    runs = []
    for _ in range(2):
        set_seed(42)
        transform = train_transform(SIZE)
        runs.append(torch.stack([transform(image=image)["image"] for _ in range(6)]))
    assert torch.equal(runs[0], runs[1])


@pytest.mark.xfail(strict=True, reason="DEF-001: training augmentation ignores the seed")
def test_training_batches_are_reproducible_under_the_seed(prepared: pd.DataFrame) -> None:
    """R6: two fresh training loaders with the same seed must yield the same first batch."""
    cfg = config()
    batches = []
    for _ in range(2):
        set_seed(cfg.seed)
        images, _, _, ids = next(iter(make_train_loader(prepared, cfg)))
        batches.append((images, list(ids)))
    assert batches[0][1] == batches[1][1]
    assert torch.equal(batches[0][0], batches[1][0])
