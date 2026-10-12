"""Minority-oversampling ablation: paired comparison of two runs (paper Section 6.4).

Both arms must share seed, split and every config value except ``minority.mode``. Differences are
``oversample - none`` with a paired bootstrap interval over the same test images. With one seed
per arm the interval reflects test-set sampling only, not run-to-run training variability.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from xaidr.training import metrics as M
from xaidr.training.trainer import PROB_COLUMNS

log = logging.getLogger(__name__)

OUTPUT = "ablation_minority.csv"
MINORITY_GRADES = (3, 4)


class AblationError(RuntimeError):
    """The two runs are not a valid pair of ablation arms."""


def grade_recall(grade: int) -> M.MetricFn:
    def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        members = y_true == grade
        return float("nan") if not members.any() else float((y_pred[members] == grade).mean())

    return recall


def minority_sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean recall over Severe (3) and Proliferative (4)."""
    return float(np.mean([grade_recall(g)(y_true, y_pred) for g in MINORITY_GRADES]))


def check_arms(run_none: Path, run_oversample: Path) -> dict[str, object]:
    """Confirm the runs differ only in ``minority.mode``; returns the shared seed and split hash."""
    configs, records = [], []
    for run_dir in (run_none, run_oversample):
        configs.append(yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")))
        records.append(json.loads((run_dir / "run.json").read_text(encoding="utf-8")))
    modes = [cfg["minority"].pop("mode") for cfg in configs]
    if modes != ["none", "oversample"]:
        raise AblationError(f"expected minority.mode none then oversample, got {modes}")
    if configs[0] != configs[1]:
        differing = sorted(k for k in configs[0] if configs[0][k] != configs[1].get(k))
        raise AblationError(f"the runs differ in more than minority.mode: section(s) {differing}")
    for key in ("seed", "split_sha256", "manifest_sha256"):
        if records[0][key] != records[1][key]:
            raise AblationError(f"the runs differ in {key}: {records[0][key]} vs {records[1][key]}")
    return {"seed": records[0]["seed"], "split_sha256": records[0]["split_sha256"]}


def ablation_table(
    none: pd.DataFrame, oversample: pd.DataFrame, threshold_none: float, threshold_oversample: float,
    n_boot: int = 1000,
) -> pd.DataFrame:
    """One row per metric: value in each arm, paired difference and its 95% interval."""
    none = none.sort_values("image_id").reset_index(drop=True)
    oversample = oversample.sort_values("image_id").reset_index(drop=True)
    if list(none["image_id"]) != list(oversample["image_id"]):
        raise AblationError("the two runs were not evaluated on the same test images")
    grade, dr = none["grade"].to_numpy(), none["dr_label"].to_numpy()
    grade_pred = [frame[PROB_COLUMNS].to_numpy().argmax(axis=1) for frame in (none, oversample)]
    det_score = [frame["det_score"].to_numpy() for frame in (none, oversample)]
    # each arm keeps the threshold chosen on its own validation predictions
    thresholds = (threshold_none, threshold_oversample)
    det_pred = [(score >= t).astype(int) for score, t in zip(det_score, thresholds, strict=True)]
    comparisons: list[tuple[str, M.MetricFn, np.ndarray, list[np.ndarray]]] = [
        ("minority_sensitivity_grades_3_4", minority_sensitivity, grade, grade_pred),
        ("sensitivity_grade_3", grade_recall(3), grade, grade_pred),
        ("sensitivity_grade_4", grade_recall(4), grade, grade_pred),
        ("grading_macro_sensitivity", M.macro_sensitivity, grade, grade_pred),
        ("grading_accuracy", lambda y, p: float((y == p).mean()), grade, grade_pred),
        ("grading_qwk", M.quadratic_weighted_kappa, grade, grade_pred),
        ("detection_auc", M.auc, dr, det_score),
        ("detection_sensitivity", M.sensitivity, dr, det_pred),
        ("detection_specificity", M.specificity, dr, det_pred),
    ]
    rows = []
    for name, fn, y_true, (pred_none, pred_over) in comparisons:
        diff, lo, hi = M.paired_bootstrap_diff(fn, y_true, pred_none, pred_over, n=n_boot)
        rows.append(
            {
                "metric": name, "none": fn(y_true, pred_none), "oversample": fn(y_true, pred_over),
                "diff_oversample_minus_none": diff, "ci_lo": lo, "ci_hi": hi,
                "ci_excludes_zero": bool(lo > 0 or hi < 0), "n": len(y_true),
            }
        )
    return pd.DataFrame(rows)


def write_ablation(
    run_none: Path, run_oversample: Path, results_dir: Path, n_boot: int = 1000
) -> pd.DataFrame:
    """Compare two finished, evaluated runs and write ``results/ablation_minority.csv``."""
    shared = check_arms(run_none, run_oversample)
    frames, thresholds = [], []
    for run_dir in (run_none, run_oversample):
        frames.append(pd.read_csv(run_dir / "predictions_test.csv", dtype={"image_id": str}))
        thresholds.append(json.loads((run_dir / "threshold.json").read_text(encoding="utf-8"))["threshold"])
    table = ablation_table(frames[0], frames[1], thresholds[0], thresholds[1], n_boot)
    fidelity = {
        "metric": "mean_idrid_iou_k10", "none": np.nan, "oversample": np.nan,
        "diff_oversample_minus_none": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
        "ci_excludes_zero": False, "n": 0,
    }
    table = pd.concat([table, pd.DataFrame([fidelity])], ignore_index=True)
    table["note"] = ""
    table.loc[table["metric"] == "mean_idrid_iou_k10", "note"] = (
        "not computed: needs IDRiD lesion masks and the explainers"
    )
    table["run_none"], table["run_oversample"] = run_none.name, run_oversample.name
    table["seed"] = str(shared["seed"])
    table["split_sha256"] = str(shared["split_sha256"])
    table["interval"] = "paired bootstrap over test images, one training run per arm"
    results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_dir / OUTPUT, index=False)
    shown = table[["metric", "none", "oversample", "diff_oversample_minus_none", "ci_lo", "ci_hi"]]
    log.info("wrote %s\n%s", results_dir / OUTPUT, shown.round(4).to_string(index=False))
    return table
