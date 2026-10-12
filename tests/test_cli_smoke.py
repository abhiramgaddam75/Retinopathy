"""End to end on a tiny synthetic dataset, on CPU: prepare, split, train two epochs, evaluate."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

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
    proposed = table6.iloc[-1]
    assert proposed["method"].startswith("Fallback baseline, NOT the specified model")
    assert proposed["method"].endswith("convnext_tiny, 224 px")
    assert (table["model"] == proposed["method"]).all()
    qwk = table[table["metric"] == "qwk"].iloc[0]["value"]
    assert proposed["qwk"] == qwk and (table6["source"].iloc[:-1] == "published").all()
    assert (run_dir / "confusion_matrix_test.csv").is_file()


def test_crossdataset_efficiency_figures(aptos_raw: Path, idrid_raw: Path, tmp_path: Path) -> None:
    """The later steps on a one-epoch run: APTOS in-domain, IDRiD external, Messidor-2 absent."""
    assert run(tmp_path, "prepare-data", "--datasets", "aptos2019,idrid") == 0
    assert run(tmp_path, "make-splits") == 0
    assert run(tmp_path, "train", "train.epochs=1") == 0
    (run_dir,) = (tmp_path / "checkpoints").iterdir()
    assert run(tmp_path, "evaluate", "--run", run_dir.name) == 0

    assert run(tmp_path, "crossdataset", "--run", run_dir.name) == 0
    summary = pd.read_csv(tmp_path / "results" / "crossdataset_summary.csv")
    status = summary.groupby("dataset")["status"].first().to_dict()
    assert status == {"aptos2019": "computed", "idrid": "computed", "messidor2": "not available"}
    idrid = summary[summary["dataset"] == "idrid"]
    assert (idrid["n"] == 4).all() and (idrid["split"] == "external_idrid").all()  # grading images only
    assert set(idrid["metric"]) >= {"auc", "qwk"}
    assert idrid["detection_definition"].iloc[0] == "any DR (grade >= 1)"
    assert idrid.loc[idrid["metric"] == "qwk", "qwk_diff_ci_lo"].notna().all()
    aptos = summary[summary["dataset"] == "aptos2019"]
    table4 = pd.read_csv(tmp_path / "results" / "table4_detection_grading_aptos_test.csv")
    assert aptos["value"].tolist() == pytest.approx(table4["value"].tolist(), nan_ok=True)
    assert summary["explanation_stability"].str.startswith("not computed").all()
    assert len(pd.read_csv(run_dir / "predictions_external_idrid.csv")) == 4

    assert run(tmp_path, "figures", "--run", run_dir.name) == 0
    names = sorted(p.name for p in (tmp_path / "results" / "figures").iterdir())
    assert names == ["class_distribution.png", f"confusion_matrix_{run_dir.name}.png",
                     f"training_curves_{run_dir.name}.png"]
    assert run(tmp_path, "xai", "--run", run_dir.name, "--limit", "1",
               "xai.methods=[gradcam_pp]") == 0
    table5 = pd.read_csv(tmp_path / "results" / "table5_xai_fidelity.csv")
    assert set(table5["method"]) == {"gradcam_pp"} and (table5["n_images"] == 1).all()


@pytest.mark.slow
def test_all_step(aptos_raw: Path, idrid_raw: Path, tmp_path: Path) -> None:
    """`all` runs every step on the datasets that are present (Messidor-2 is not)."""
    assert run(tmp_path, "all", "train.epochs=1", "xai.methods=[gradcam_pp]") == 0
    results = tmp_path / "results"
    for name in ("table4_detection_grading_aptos_test.csv", "table5_xai_fidelity.csv",
                 "table6_comparative.csv", "crossdataset_summary.csv", "efficiency.csv"):
        assert (results / name).is_file(), name
    assert len(list((results / "figures").glob("*.png"))) == 3
    (run_dir,) = (p for p in (tmp_path / "checkpoints").iterdir() if p.is_dir())
    # a second call reuses the trained run instead of training again
    assert run(tmp_path, "all", "--run", run_dir.name, "xai.methods=[gradcam_pp]") == 0
    assert len([p for p in (tmp_path / "checkpoints").iterdir() if p.is_dir()]) == 1
