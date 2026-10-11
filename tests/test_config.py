from __future__ import annotations

from pathlib import Path

import pytest

from conftest import REPO_ROOT
from xaidr.config import ConfigError, load_config

CONFIGS = REPO_ROOT / "configs"


def test_default_config_matches_design() -> None:
    cfg = load_config(CONFIGS / "default.yaml")
    assert cfg.seed == 42
    assert cfg.data.image_size == 512
    assert cfg.data.split_ratios == (0.70, 0.15, 0.15)
    assert cfg.model.backbone == "maxvit_tiny_tf_512"
    assert (cfg.train.batch_size, cfg.train.grad_accum, cfg.train.epochs) == (16, 2, 30)
    assert (cfg.train.loss.lambda_det, cfg.train.loss.lambda_grade) == (0.4, 0.6)
    assert (cfg.minority.mode, cfg.minority.target_grades, cfg.minority.factor) == ("oversample", (3, 4), 3)
    assert cfg.xai.fidelity_k_percent[0] == 10
    assert cfg.xai.target == "disease" and cfg.eval.stability_seeds == (42, 43, 44)
    assert isinstance(cfg.paths.raw, Path)


@pytest.mark.parametrize("name", ["ablation_no_oversampling", "ablation_oversampling", "backbone_convnext"])
def test_variant_configs_inherit_default(name: str) -> None:
    cfg = load_config(CONFIGS / f"{name}.yaml")
    assert cfg.train.epochs == 30
    if name == "ablation_no_oversampling":
        assert cfg.minority.mode == "none"
    if name == "backbone_convnext":
        assert cfg.model.backbone == "convnext_tiny"
        assert "attention" not in cfg.xai.methods


def test_overrides() -> None:
    cfg = load_config(CONFIGS / "default.yaml", ["train.epochs=3", "minority.mode=none", "paths.raw=x/y"])
    assert cfg.train.epochs == 3
    assert cfg.minority.mode == "none"
    assert cfg.paths.raw == Path("x/y")


@pytest.mark.parametrize(
    "override, key",
    [
        ("data.image_size=128", "data.image_size"),
        ("data.image_size=big", "data.image_size"),
        ("data.split_ratios=[0.7,0.2,0.2]", "data.split_ratios"),
        ("minority.target_grades=[3,7]", "minority.target_grades"),
        ("model.backbone=resnet18", "model.backbone"),
        ("minority.mode=smote", "minority.mode"),
        ("xai.target=grade3", "xai.target"),
        ("eval.stability_seeds=[42,42]", "eval.stability_seeds"),
        ("train.epoch=3", "train.epoch"),
    ],
)
def test_invalid_config_names_the_key(override: str, key: str) -> None:
    with pytest.raises(ConfigError, match=key.replace(".", r"\.")):
        load_config(CONFIGS / "default.yaml", [override])
