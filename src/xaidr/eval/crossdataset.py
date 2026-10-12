"""In-domain versus external performance (paper Section 6.3, DESIGN.md Section 7.23).

The model is applied without any fine-tuning to the external sets. The detection threshold is
the one frozen on APTOS validation data. Each dataset keeps its own detection definition and is
reported in its own rows; nothing is pooled. A dataset that is not in the manifest is written
as "not available" rather than left out.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

from xaidr.config import Config
from xaidr.data.dataset import make_eval_loader
from xaidr.data.manifest import read_manifest
from xaidr.eval.evaluate_grading import method_label, table4_rows
from xaidr.training import metrics as M
from xaidr.training.run import load_best_model
from xaidr.training.trainer import PROB_COLUMNS, predict

log = logging.getLogger(__name__)

OUTPUT = "crossdataset_summary.csv"
SPLITS = {"aptos2019": "test", "idrid": "external_idrid", "messidor2": "external_messidor2"}
STABILITY_STATUS = "not computed: needs three seeds on every dataset (ISSUE-013)"


def detection_definition(rows: pd.DataFrame, min_grade: int) -> str:
    space = rows["label_space"].iloc[0]
    return f"any DR (grade >= {min_grade})" if space == "icdr5" else space


def qwk_difference(
    in_domain: pd.DataFrame, external: pd.DataFrame, n_boot: int, seed: int = 42
) -> tuple[float, float, float]:
    """External QWK minus in-domain QWK; the two test sets are resampled independently."""
    rng = np.random.default_rng(seed)
    pairs = []
    for frame in (in_domain, external):
        grade = frame["grade"].to_numpy()
        pairs.append((grade, frame[PROB_COLUMNS].to_numpy().argmax(axis=1)))
    draws = []
    for _ in range(n_boot):
        values = []
        for grade, predicted in pairs:
            idx = rng.integers(0, len(grade), len(grade))
            values.append(M.quadratic_weighted_kappa(grade[idx], predicted[idx]))
        draws.append(values[1] - values[0])
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    point = M.quadratic_weighted_kappa(*pairs[1]) - M.quadratic_weighted_kappa(*pairs[0])
    return float(point), float(lo), float(hi)


def dataset_rows(predictions: pd.DataFrame, threshold: float, n_boot: int) -> pd.DataFrame:
    """Detection rows always; grading rows only when every image has a five-class grade."""
    if (predictions["grade"] >= 0).all():
        return table4_rows(predictions, threshold, n_boot)
    dr, score = predictions["dr_label"].to_numpy(), predictions["det_score"].to_numpy()
    y_pred = (score >= threshold).astype(int)
    rows = []
    for name, fn, values in (("accuracy", accuracy_score, y_pred), ("sensitivity", M.sensitivity, y_pred),
                             ("specificity", M.specificity, y_pred), ("auc", M.auc, score)):
        estimate, lo, hi = M.bootstrap_ci(fn, dr, values, n=n_boot)
        row = {"task": "detection", "metric": name, "value": estimate, "ci_lo": lo, "ci_hi": hi}
        rows.append({**row, "n": len(dr)})
    return pd.DataFrame(rows)


def cross_dataset(cfg: Config, run_dir: Path, manifest_path: Path) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = read_manifest(manifest_path)
    # segmentation images repeat grading images under another id; they are scored in the xai step
    manifest = manifest[~manifest["image_id"].str.startswith("seg_")]
    threshold = json.loads((run_dir / "threshold.json").read_text(encoding="utf-8"))["threshold"]
    model = load_best_model(run_dir, cfg, device)
    n_boot = cfg.eval.bootstrap_iterations

    frames, predictions = [], {}
    for dataset, split in SPLITS.items():
        rows = manifest[(manifest["dataset"] == dataset) & (manifest["split"] == split)]
        rows = rows[rows["dr_label"].notna()]
        if rows.empty:
            missing = {"dataset": dataset, "split": split, "n": 0, "status": "not available"}
            frames.append(pd.DataFrame([missing]))
            continue
        frame = predict(model, make_eval_loader(manifest, split, cfg), device, cfg.train.amp)
        frame = frame[frame["image_id"].isin(set(rows["image_id"]))].reset_index(drop=True)
        frame.to_csv(run_dir / f"predictions_{split}.csv", index=False)
        predictions[dataset] = frame
        table = dataset_rows(frame, threshold, n_boot)
        table.insert(0, "split", split)
        table.insert(0, "dataset", dataset)
        table["status"] = "computed"
        table["detection_definition"] = detection_definition(rows, cfg.data.detection_positive_min_grade)
        table["label_space"] = rows["label_space"].iloc[0]
        frames.append(table)
    summary = pd.concat(frames, ignore_index=True)

    for column in ("qwk_diff_vs_in_domain", "qwk_diff_ci_lo", "qwk_diff_ci_hi"):
        summary[column] = np.nan
    for dataset, frame in predictions.items():
        if dataset == "aptos2019" or "aptos2019" not in predictions or not (frame["grade"] >= 0).all():
            continue
        diff = qwk_difference(predictions["aptos2019"], frame, n_boot, cfg.seed)
        row = (summary["dataset"] == dataset) & (summary["metric"] == "qwk")
        summary.loc[row, ["qwk_diff_vs_in_domain", "qwk_diff_ci_lo", "qwk_diff_ci_hi"]] = diff
    summary["detection_threshold"] = threshold
    summary["explanation_stability"] = STABILITY_STATUS
    summary["model"] = method_label(cfg)
    summary["run_id"] = run_dir.name
    cfg.paths.results.mkdir(parents=True, exist_ok=True)
    summary.to_csv(cfg.paths.results / OUTPUT, index=False)
    log.info("wrote %s\n%s", cfg.paths.results / OUTPUT,
             summary[["dataset", "n", "status"]].drop_duplicates().to_string(index=False))
    return summary
