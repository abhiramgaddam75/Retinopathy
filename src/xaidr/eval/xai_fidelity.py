"""Explanation fidelity against lesion masks: paper Table 5 (DESIGN.md Sections 7.23 and 10.3).

Every configured explainer is run on every image that has lesion masks (the IDRiD segmentation
set). Maps are cached in the run folder. Per-image scores are aggregated per method, lesion and
k, with a bootstrap interval over images, the chance and maximum-attainable reference values,
and a one-sided Wilcoxon signed-rank test of IoU against chance. Holm correction is applied
across the primary-k rows only (ISSUE-014); rows for other k are sensitivity analyses.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from tqdm import tqdm

from xaidr.config import Config
from xaidr.data.adapters import DataMissingError
from xaidr.data.dataset import LesionDataset
from xaidr.data.manifest import read_manifest
from xaidr.training.metrics import bootstrap_ci
from xaidr.training.run import load_best_model
from xaidr.xai import build_explainer
from xaidr.xai.attention import AttentionUnavailableError
from xaidr.xai.base import Explainer, Target, denormalise
from xaidr.xai.fidelity import fidelity_for_image, fov_mask, holm_adjust

log = logging.getLogger(__name__)

TABLE5 = "table5_xai_fidelity.csv"
EXPLORATORY_BELOW = 20  # DESIGN 10.4


def _mean_ci(values: np.ndarray, n_boot: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    labels = np.zeros(len(values))  # unused: images are resampled without strata
    return bootstrap_ci(lambda _y, v: float(np.mean(v)), labels, values, n=n_boot, stratify=False)


def _p_above_chance(score: np.ndarray, chance: np.ndarray) -> float:
    """One-sided Wilcoxon signed-rank p-value that the score exceeds its chance value."""
    diff = np.asarray(score, dtype=float) - np.asarray(chance, dtype=float)
    if len(diff) < 2 or not np.any(diff != 0):
        return 1.0
    return float(wilcoxon(diff, alternative="greater", zero_method="zsplit").pvalue)


def aggregate(per_image: pd.DataFrame, primary_k: float, n_boot: int) -> pd.DataFrame:
    """Per-image fidelity rows -> one row per (method, lesion, k)."""
    rows = []
    for (method, lesion, k), group in per_image.groupby(["method", "lesion", "k_percent"], sort=False):
        iou_mean, iou_lo, iou_hi = _mean_ci(group["iou"].to_numpy(), n_boot)
        dice_mean, dice_lo, dice_hi = _mean_ci(group["dice"].to_numpy(), n_boot)
        point, point_lo, point_hi = _mean_ci(group["pointing"].to_numpy(), n_boot)
        recall, recall_lo, recall_hi = _mean_ci(group["recall"].to_numpy(), n_boot)
        rows.append(
            {
                "method": method, "lesion": lesion, "k_percent": k, "n_images": len(group),
                "mean_iou": iou_mean, "iou_ci_lo": iou_lo, "iou_ci_hi": iou_hi,
                "mean_dice": dice_mean, "dice_ci_lo": dice_lo, "dice_ci_hi": dice_hi,
                "pointing_acc": point, "pointing_ci_lo": point_lo, "pointing_ci_hi": point_hi,
                "mean_recall": recall, "recall_ci_lo": recall_lo, "recall_ci_hi": recall_hi,
                "chance_iou": group["chance_iou"].mean(), "chance_dice": group["chance_dice"].mean(),
                "chance_pointing": group["chance_pointing"].mean(),
                "chance_recall": group["chance_recall"].mean(),
                "max_iou": group["max_iou"].mean(), "max_dice": group["max_dice"].mean(),
                "mean_lesion_fraction": group["lesion_fraction"].mean(),
                "p_iou_above_chance": _p_above_chance(
                    group["iou"].to_numpy(), group["chance_iou"].to_numpy()
                ),
                "primary": bool(k == primary_k), "exploratory": bool(len(group) < EXPLORATORY_BELOW),
            }
        )
    table = pd.DataFrame(rows)
    table["p_holm"] = np.nan
    primary = table["primary"].to_numpy()
    if primary.any():
        table.loc[primary, "p_holm"] = holm_adjust(table.loc[primary, "p_iou_above_chance"].to_numpy())
    return table


def _saliency(
    explainer: Explainer, image: torch.Tensor, target: Target, cache: Path
) -> tuple[np.ndarray, float]:
    if cache.is_file():
        return np.load(cache), float("nan")
    started = time.time()
    saliency = explainer.explain(image, target)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, saliency.astype(np.float32))
    return saliency, time.time() - started


def run_fidelity(cfg: Config, run_dir: Path, manifest_path: Path, limit: int | None = None) -> pd.DataFrame:
    manifest = read_manifest(manifest_path)
    lesion_rows = manifest[manifest["has_masks"]]
    if lesion_rows.empty:
        raise DataMissingError(
            "the manifest has no images with lesion masks; prepare IDRiD first "
            "(docs/DATA_ACCESS.md, then `prepare-data --datasets idrid`)"
        )
    if (lesion_rows["split"] == "train").any():
        raise DataMissingError("lesion-annotated images must never be training images")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_best_model(run_dir, cfg, device)
    dataset = LesionDataset(lesion_rows.head(limit) if limit else lesion_rows, cfg.data.image_size)

    explainers: dict[str, Explainer] = {}
    not_computed: dict[str, str] = {}
    for name in cfg.xai.methods:
        try:
            explainers[name] = build_explainer(name, model, cfg.xai, cfg.seed)
        except AttentionUnavailableError as exc:
            not_computed[name] = str(exc)
            log.warning("%s saliency not computed: %s", name, exc)

    target: Target = cfg.xai.target if cfg.xai.target == "disease" else -1
    records = []
    seconds: dict[str, list[float]] = {name: [] for name in explainers}
    predicted_no_dr = 0
    for index in tqdm(range(len(dataset)), desc="fidelity"):
        image, masks, _grade, image_id = dataset[index]
        with torch.no_grad():
            predicted = int(model(image[None].to(device))["grade_logits"].argmax(dim=1))
        predicted_no_dr += predicted == 0
        image_target: Target = predicted if target == -1 else target
        fov = fov_mask(denormalise(image))
        mask_arrays = {name: mask.numpy() for name, mask in masks.items()}
        for name, explainer in explainers.items():
            cache = run_dir / "saliency" / f"{name}_{cfg.xai.target}" / f"{image_id}.npy"
            saliency, elapsed = _saliency(explainer, image, image_target, cache)
            if not np.isnan(elapsed):
                seconds[name].append(elapsed)
            frame = fidelity_for_image(
                saliency.astype(np.float32), mask_arrays, fov, cfg.xai.fidelity_k_percent,
                cfg.eval.chance_monte_carlo, cfg.seed,
            )
            frame.insert(0, "predicted_grade", predicted)
            frame.insert(0, "image_id", image_id)
            frame.insert(0, "method", name)
            records.append(frame)
    per_image = pd.concat(records, ignore_index=True)
    per_image.to_csv(run_dir / "fidelity_per_image.csv", index=False)

    table = aggregate(per_image, cfg.xai.fidelity_k_percent[0], cfg.eval.bootstrap_iterations)
    table["target"] = cfg.xai.target
    table["run_id"] = run_dir.name
    cfg.paths.results.mkdir(parents=True, exist_ok=True)
    table.to_csv(cfg.paths.results / TABLE5, index=False)
    summary = {
        "n_images": len(dataset), "images_predicted_grade_0": int(predicted_no_dr),
        "target": cfg.xai.target, "methods_not_computed": not_computed,
        "median_seconds_per_image": {k: float(np.median(v)) for k, v in seconds.items() if v},
        "images_per_lesion": per_image[per_image["k_percent"] == cfg.xai.fidelity_k_percent[0]]
        .groupby("lesion")["image_id"].nunique().to_dict(),
    }
    (run_dir / "fidelity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("wrote %s (%d rows); summary: %s", cfg.paths.results / TABLE5, len(table), summary)
    return table
