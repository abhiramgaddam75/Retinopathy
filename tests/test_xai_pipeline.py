"""Table 5 pipeline, efficiency benchmark and figures on synthetic data and an untrained model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import REPO_ROOT
from xaidr.config import Config, load_config
from xaidr.data.adapters import DataMissingError, IdridAdapter
from xaidr.data.manifest import normalise, write_manifest
from xaidr.data.preprocess import build_processed
from xaidr.eval.efficiency import benchmark_run, parameter_counts, time_forward, weights_megabytes
from xaidr.eval.xai_fidelity import aggregate, run_fidelity
from xaidr.models.dual_head import DRDualHeadModel
from xaidr.viz.figures import (
    class_distribution_fig,
    confusion_matrix_fig,
    saliency_panel,
    save,
    training_curves_fig,
)

SIZE = 224


def make_cfg(tmp_path: Path, *extra: str) -> Config:
    names = ("raw", "manifests", "processed", "splits", "checkpoints", "results")
    return load_config(
        REPO_ROOT / "configs" / "default.yaml",
        [*[f"paths.{n}={(tmp_path / n).as_posix()}" for n in names], f"data.image_size={SIZE}",
         "model.backbone=convnext_tiny", "model.pretrained=false", "eval.bootstrap_iterations=40",
         "eval.chance_monte_carlo=100", *extra],
    )


@pytest.fixture
def untrained_run(tmp_path: Path) -> tuple[Config, Path]:
    cfg = make_cfg(tmp_path, "xai.methods=[gradcam_pp,lime,attention]", "xai.lime={num_samples: 16}")
    run_dir = cfg.paths.checkpoints / "run"
    run_dir.mkdir(parents=True)
    torch.manual_seed(0)
    model = DRDualHeadModel(cfg.model, SIZE)
    torch.save({"model": model.state_dict()}, run_dir / "best.pt")
    return cfg, run_dir


def test_table5_pipeline(untrained_run: tuple[Config, Path], idrid_raw: Path, tmp_path: Path) -> None:
    cfg, run_dir = untrained_run
    manifest = normalise(IdridAdapter().build_manifest(idrid_raw)).assign(split="external_idrid")
    manifest = build_processed(manifest, cfg.paths.processed, SIZE, num_workers=2)
    manifest_file = cfg.paths.manifests / "manifest.csv"
    write_manifest(manifest, manifest_file)

    table = run_fidelity(cfg, run_dir, manifest_file)
    written = pd.read_csv(cfg.paths.results / "table5_xai_fidelity.csv")
    assert len(written) == len(table)
    for column in ("method", "lesion", "k_percent", "n_images", "mean_iou", "iou_ci_lo", "iou_ci_hi",
                   "mean_dice", "dice_ci_lo", "dice_ci_hi", "pointing_acc", "pointing_ci_lo",
                   "pointing_ci_hi", "chance_iou", "chance_pointing", "max_iou", "mean_recall",
                   "chance_recall", "p_holm"):
        assert column in written.columns, column
    assert set(written["method"]) == {"gradcam_pp", "lime"}  # attention: not available on ConvNeXt
    assert set(written["lesion"]) == {"MA", "HE", "EX", "any"}
    assert set(written["k_percent"]) == {10, 5, 20}
    lime_primary = written[(written["method"] == "lime") & (written["k_percent"] == 10)]
    counts = lime_primary.set_index("lesion")["n_images"]
    assert counts.to_dict() == {"MA": 2, "HE": 1, "EX": 1, "any": 2}  # as in the fixture
    assert written["exploratory"].all() and (written["target"] == "disease").all()
    assert written.loc[written["primary"], "p_holm"].notna().all()
    assert written.loc[~written["primary"], "p_holm"].isna().all()
    assert (written["mean_iou"] <= written["max_iou"] + 1e-9).all()

    summary = json.loads((run_dir / "fidelity_summary.json").read_text())
    assert summary["n_images"] == 2 and "attention" in summary["methods_not_computed"]
    assert summary["images_per_lesion"] == {"EX": 1, "HE": 1, "MA": 2, "any": 2}
    cached = sorted(p.name for p in (run_dir / "saliency" / "lime_disease").iterdir())
    assert cached == ["seg_IDRiD_01.npy", "seg_IDRiD_02.npy"]
    # a second call reuses the cached maps and reproduces the table
    again = run_fidelity(cfg, run_dir, manifest_file)
    pd.testing.assert_frame_equal(again, table)

    without_masks = cfg.paths.manifests / "none.csv"
    write_manifest(manifest[~manifest["has_masks"]], without_masks)
    with pytest.raises(DataMissingError, match="lesion masks"):
        run_fidelity(cfg, run_dir, without_masks)


def test_aggregate_statistics() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for image in range(30):
        for method, lift in (("good", 0.2), ("random", -0.004)):
            rows.append({
                "method": method, "image_id": f"i{image}", "lesion": "HE", "k_percent": 10,
                "iou": 0.05 + lift + rng.normal(scale=0.02), "dice": 0.1, "pointing": float(image % 2),
                "recall": 0.3, "chance_iou": 0.05, "chance_dice": 0.09, "chance_pointing": 0.02,
                "chance_recall": 0.1, "max_iou": 0.4, "max_dice": 0.5, "lesion_fraction": 0.04,
            })
    table = aggregate(pd.DataFrame(rows), primary_k=10, n_boot=100).set_index("method")
    assert table.loc["good", "n_images"] == 30 and not table.loc["good", "exploratory"]
    assert table.loc["good", "p_holm"] < 0.001 < 0.05 < table.loc["random", "p_holm"]
    assert table.loc["good", "iou_ci_lo"] < table.loc["good", "mean_iou"] < table.loc["good", "iou_ci_hi"]
    assert table.loc["good", "pointing_acc"] == 0.5


def test_efficiency(untrained_run: tuple[Config, Path]) -> None:
    cfg, run_dir = untrained_run
    model = DRDualHeadModel(cfg.model, SIZE)
    counts = parameter_counts(model)
    assert counts["params_total"] == sum(p.numel() for p in model.parameters()) == counts["params_trainable"]
    assert weights_megabytes(model) == pytest.approx(counts["params_total"] * 4 / 1e6, rel=0.01)
    timing = time_forward(model.eval(), torch.randn(1, 3, SIZE, SIZE), warmup=1, timed=5)
    assert timing["latency_q25_ms"] <= timing["latency_median_ms"] <= timing["latency_q75_ms"]
    assert timing["latency_iqr_ms"] >= 0 and timing["n_timed"] == 5

    table = benchmark_run(cfg, run_dir, warmup=1, timed=3)
    written = pd.read_csv(cfg.paths.results / "efficiency.csv")
    assert len(written) == len(table) >= 1 and written.loc[0, "device_type"] == "cpu"
    assert written.loc[0, "params_total"] == counts["params_total"] and written.loc[0, "precision"] == "fp32"
    assert written.loc[0, "model"].endswith("convnext_tiny, 224 px")
    assert written.loc[0, "latency_median_ms"] > 0


def test_figures(tmp_path: Path) -> None:
    matrix = np.array(
        [[268, 2, 0, 0, 0], [1, 33, 17, 0, 0], [1, 7, 126, 3, 1], [0, 0, 5, 16, 6], [0, 1, 20, 6, 13]]
    )
    files = [save(confusion_matrix_fig(matrix, title="test"), tmp_path / "confusion.png")]
    log = tmp_path / "metrics.jsonl"
    log.write_text("\n".join(json.dumps({
        "epoch": e, "train_loss": 1 / (e + 1), "val_loss": 0.6 + 0.05 * e, "val_qwk": 0.8 + 0.02 * e,
        "val_det_auc": 0.99, "selection_score": 0.9 + 0.01 * e}) for e in range(4)))
    files.append(save(training_curves_fig(log, "run"), tmp_path / "curves.png"))
    manifest = pd.DataFrame({
        "dataset": ["aptos2019"] * 6 + ["idrid"] * 3,
        "image_id": [f"a{i}" for i in range(6)] + ["i1", "i2", "seg_x"],
        "grade": [0, 0, 0, 1, 2, 4, 0, 3, 3],
    })
    files.append(save(class_distribution_fig(manifest), tmp_path / "classes.png"))
    rgb = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    saliency = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    mask = np.zeros((64, 64), np.uint8)
    mask[20:30, 20:30] = 1
    files.append(save(saliency_panel(rgb, saliency, mask, "panel"), tmp_path / "panel.png"))
    files.append(save(saliency_panel(rgb, saliency, None), tmp_path / "panel_nomask.png"))
    for path in files:
        assert path.is_file() and path.stat().st_size > 5000, path
