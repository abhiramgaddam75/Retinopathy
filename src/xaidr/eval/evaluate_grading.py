"""In-domain evaluation on the APTOS 2019 test split: paper Table 4 and the proposed row of Table 6."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix

from xaidr.config import Config
from xaidr.data.dataset import make_eval_loader
from xaidr.data.manifest import read_manifest
from xaidr.training import metrics as M
from xaidr.training.run import load_best_model
from xaidr.training.trainer import PROB_COLUMNS, predict

log = logging.getLogger(__name__)

TABLE4 = "table4_detection_grading_aptos_test.csv"
TABLE6 = "table6_comparative.csv"

# Published rows for Table 6. Each carries its verification status from docs/SCIENTIFIC_REVIEW.md;
# nothing here is a result of this build.
LITERATURE_ROWS = [
    {
        "method": "Chilukoti et al. [48], 2024", "dataset": "APTOS 2019", "grading_accuracy": np.nan,
        "qwk": 0.967, "source": "published",
        "note": "Checked against the source. EyePACS-pretrained, fine-tuned, checkpoint ensemble; "
                "APTOS split not reported, so not comparable with a held-out test split.",
    },
    {
        "method": "Shakibania et al. [27], 2024", "dataset": "APTOS 2019", "grading_accuracy": 0.8960,
        "qwk": 0.93, "source": "published",
        "note": "Seen only as quoted by a second paper; the draft attributed these values to [49]. "
                "Verify against [27] before use.",
    },
    {
        "method": "Dinpajhouh & Seyyedsalehi [50], 2023", "dataset": "APTOS 2019",
        "grading_accuracy": 0.8369, "qwk": 0.8926, "source": "published",
        "note": "NOT VERIFIED: values copied from the draft; the article could not be opened.",
    },
]


SPECIFIED_MODEL = ("maxvit_tiny_tf_512", 512)  # D1 and the 512 px input of DESIGN.md


def method_label(cfg: Config) -> str:
    """Row label for this build; a run that is not the specified model says so in its name."""
    model = f"{cfg.model.backbone}, {cfg.data.image_size} px"
    if (cfg.model.backbone, cfg.data.image_size) == SPECIFIED_MODEL:
        return f"Proposed framework (this study): {model}"
    return f"Fallback baseline, NOT the specified model (this study): {model}"


def _row(task: str, name: str, fn: M.MetricFn, y: np.ndarray, values: np.ndarray, n_boot: int) -> dict:
    estimate, lo, hi = M.bootstrap_ci(fn, y, values, n=n_boot)
    return {"task": task, "metric": name, "value": estimate, "ci_lo": lo, "ci_hi": hi, "n": len(y)}


def table4_rows(predictions: pd.DataFrame, threshold: float, n_boot: int) -> pd.DataFrame:
    dr, score = predictions["dr_label"].to_numpy(), predictions["det_score"].to_numpy()
    grade, probs = predictions["grade"].to_numpy(), predictions[PROB_COLUMNS].to_numpy()
    y_pred_det = (score >= threshold).astype(int)
    y_pred_grade = probs.argmax(axis=1)
    rows = [
        _row("detection", "accuracy", accuracy_score, dr, y_pred_det, n_boot),
        _row("detection", "sensitivity", M.sensitivity, dr, y_pred_det, n_boot),
        _row("detection", "specificity", M.specificity, dr, y_pred_det, n_boot),
        _row("detection", "auc", M.auc, dr, score, n_boot),
        _row("grading", "accuracy", accuracy_score, grade, y_pred_grade, n_boot),
        _row("grading", "macro_sensitivity", M.macro_sensitivity, grade, y_pred_grade, n_boot),
        _row("grading", "macro_auc", M.macro_ovr_auc, grade, probs, n_boot),
        _row("grading", "qwk", M.quadratic_weighted_kappa, grade, y_pred_grade, n_boot),
    ]
    return pd.DataFrame(rows)


def evaluate_run(cfg: Config, run_dir: Path, manifest_path: Path) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = read_manifest(manifest_path)
    loader = make_eval_loader(manifest, "test", cfg)
    model = load_best_model(run_dir, cfg, device)
    predictions = predict(model, loader, device, cfg.train.amp)
    predictions.to_csv(run_dir / "predictions_test.csv", index=False)
    threshold = json.loads((run_dir / "threshold.json").read_text(encoding="utf-8"))["threshold"]

    table = table4_rows(predictions, threshold, cfg.eval.bootstrap_iterations)
    table["detection_threshold"] = threshold
    table["dataset"] = "aptos2019_test"
    table["model"] = method_label(cfg)
    table["run_id"] = run_dir.name
    results = cfg.paths.results
    results.mkdir(parents=True, exist_ok=True)
    table.to_csv(results / TABLE4, index=False)

    grade = predictions["grade"].to_numpy()
    predicted = predictions[PROB_COLUMNS].to_numpy().argmax(axis=1)
    matrix = pd.DataFrame(
        confusion_matrix(grade, predicted, labels=list(M.GRADES)),
        index=[f"true_{g}" for g in M.GRADES], columns=[f"pred_{g}" for g in M.GRADES],
    )
    matrix.to_csv(run_dir / "confusion_matrix_test.csv")

    by_metric = table[table["task"] == "grading"].set_index("metric")
    proposed = {
        "method": method_label(cfg), "dataset": "APTOS 2019 (held-out 15% test split)",
        "grading_accuracy": by_metric.loc["accuracy", "value"], "qwk": by_metric.loc["qwk", "value"],
        "qwk_ci_lo": by_metric.loc["qwk", "ci_lo"], "qwk_ci_hi": by_metric.loc["qwk", "ci_hi"],
        "source": f"this build, run {run_dir.name}", "note": f"n = {len(predictions)}; one seed",
    }
    pd.DataFrame([*LITERATURE_ROWS, proposed]).to_csv(results / TABLE6, index=False)
    log.info("wrote %s and %s\n%s", results / TABLE4, results / TABLE6,
             table[["task", "metric", "value", "ci_lo", "ci_hi", "n"]].round(4).to_string(index=False))
    return table
