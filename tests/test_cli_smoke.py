"""End to end on a tiny synthetic dataset, on CPU: prepare, split, train two epochs, evaluate."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from conftest import REPO_ROOT
from xaidr.cli import main


def run(tmp_path: Path, *args: str) -> int:
    names = ("raw", "manifests", "processed", "splits", "checkpoints", "results")
    overrides = [f"paths.{name}={(tmp_path / name).as_posix()}" for name in names]
    settings = [
        "data.image_size=224", "model.backbone=convnext_tiny", "model.pretrained=false",
        "train.epochs=2", "train.batch_size=8", "train.grad_accum=2", "train.num_workers=0",
        "train.warmup_epochs=2", "eval.bootstrap_iterations=50",
    ]
    step, *flags = args
    config = str(REPO_ROOT / "configs" / "default.yaml")
    return main([step, "--config", config, *flags, *overrides, *settings])


def test_prepare_split_train_evaluate(aptos_raw: Path, tmp_path: Path) -> None:
    assert run(tmp_path, "prepare-data", "--datasets", "aptos2019") == 0
    assert run(tmp_path, "make-splits") == 0
    assert run(tmp_path, "evaluate", "--run", "nothing") == 1  # no such run: readable error
    assert run(tmp_path, "train") == 0

    (run_dir,) = (tmp_path / "checkpoints").iterdir()
    assert "convnext_tiny_oversample_seed42" in run_dir.name
    for name in ("best.pt", "last.pt", "metrics.jsonl", "config.yaml", "threshold.json", "run.json",
                 "predictions_val.csv"):
        assert (run_dir / name).is_file(), name
    epochs = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [e["epoch"] for e in epochs] == [0, 1]
    for key in ("train_loss", "train_loss_det", "train_loss_grade", "val_loss", "val_qwk", "val_det_auc",
                "val_grade_acc", "selection_score", "lr"):
        assert key in epochs[0], key
    assert epochs[0]["lr"] != epochs[1]["lr"]
    record = json.loads((run_dir / "run.json").read_text())
    lock = (tmp_path / "splits" / "aptos2019_split.sha256").read_text().split()[0]
    assert record["split_sha256"] == lock and record["seed"] == 42 and len(record["class_weights"]) == 5
    assert len(record["manifest_sha256"]) == 64 and record["packages"] and record["best_epoch"] in (0, 1)
    threshold = json.loads((run_dir / "threshold.json").read_text())["threshold"]
    assert 0.0 < threshold < 1.0 and threshold == record["threshold"]

    # resuming a finished run trains no further epochs and keeps the record
    assert run(tmp_path, "train", "--resume", run_dir.name) == 0
    assert len((run_dir / "metrics.jsonl").read_text().splitlines()) == 2

    assert run(tmp_path, "evaluate", "--run", run_dir.name) == 0
    table = pd.read_csv(tmp_path / "results" / "table4_detection_grading_aptos_test.csv")
    assert list(table["metric"]) == ["accuracy", "sensitivity", "specificity", "auc",
                                     "accuracy", "macro_sensitivity", "macro_auc", "qwk"]
    predictions = pd.read_csv(run_dir / "predictions_test.csv")
    assert (table["n"] == len(predictions)).all() and len(predictions) == 15
    assert (table["detection_threshold"] == threshold).all()
    assert ((table["ci_lo"] <= table["value"] + 1e-9) & (table["value"] <= table["ci_hi"] + 1e-9)).all()
    assert (predictions[[f"p{g}" for g in range(5)]].sum(axis=1) - 1).abs().max() < 1e-4
    table6 = pd.read_csv(tmp_path / "results" / "table6_comparative.csv")
    proposed = table6[table6["method"].str.startswith("Proposed")].iloc[0]
    qwk = table[table["metric"] == "qwk"].iloc[0]["value"]
    assert proposed["qwk"] == qwk and (table6["source"].iloc[:-1] == "published").all()
    assert (run_dir / "confusion_matrix_test.csv").is_file()
