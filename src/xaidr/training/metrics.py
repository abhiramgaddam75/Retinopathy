"""Detection and grading metrics with bootstrap confidence intervals (DESIGN.md Section 7.14)."""
from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, recall_score, roc_auc_score

log = logging.getLogger(__name__)

GRADES = (0, 1, 2, 3, 4)
MetricFn = Callable[[np.ndarray, np.ndarray], float]


def auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC; NaN when only one class is present."""
    return float("nan") if len(np.unique(y_true)) < 2 else float(roc_auc_score(y_true, scores))


def sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    positives = np.asarray(y_true) == 1
    return float("nan") if not positives.any() else float((np.asarray(y_pred)[positives] == 1).mean())


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    negatives = np.asarray(y_true) == 0
    return float("nan") if not negatives.any() else float((np.asarray(y_pred)[negatives] == 0).mean())


def detection_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": sensitivity(y_true, y_pred),
        "specificity": specificity(y_true, y_pred),
        "auc": auc(y_true, scores),
    }


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=list(GRADES)))


def macro_sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean recall over the grades present in ``y_true``."""
    return float(recall_score(y_true, y_pred, labels=np.unique(y_true), average="macro", zero_division=0))


def macro_ovr_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    """One-vs-rest AUC averaged over grades; an absent grade is NaN and left out, with a warning."""
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    per_class = [auc((y_true == g).astype(int), probs[:, g]) for g in GRADES]
    absent = [g for g, value in zip(GRADES, per_class, strict=True) if np.isnan(value)]
    if absent:
        log.warning("one-vs-rest AUC is undefined for grade(s) %s; macro AUC uses the others", absent)
    return float("nan") if len(absent) == len(GRADES) else float(np.nanmean(per_class))


def grading_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    y_pred = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_sensitivity": macro_sensitivity(y_true, y_pred),
        "macro_auc": macro_ovr_auc(y_true, probs),
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
    }


def _resample(y_true: np.ndarray, rng: np.random.Generator, stratify: bool) -> np.ndarray:
    if not stratify:
        return rng.integers(0, len(y_true), len(y_true))
    groups = [np.flatnonzero(y_true == value) for value in np.unique(y_true)]
    return np.concatenate([rng.choice(group, len(group)) for group in groups])


def bootstrap_ci(
    metric_fn: MetricFn,
    y_true: np.ndarray,
    y_pred_or_score: np.ndarray,
    n: int = 1000,
    seed: int = 42,
    stratify: bool = True,
) -> tuple[float, float, float]:
    """Point estimate and 95% percentile interval; resampling keeps each class's size when stratified."""
    y_true, values = np.asarray(y_true), np.asarray(y_pred_or_score)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        idx = _resample(y_true, rng, stratify)
        draws.append(metric_fn(y_true[idx], values[idx]))
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(metric_fn(y_true, values)), float(lo), float(hi)


def paired_bootstrap_diff(
    metric_fn: MetricFn,
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """``metric(b) - metric(a)`` with a 95% interval; both arms use the same resampled images."""
    y_true, pred_a, pred_b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        idx = _resample(y_true, rng, True)
        draws.append(metric_fn(y_true[idx], pred_b[idx]) - metric_fn(y_true[idx], pred_a[idx]))
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(metric_fn(y_true, pred_b) - metric_fn(y_true, pred_a)), float(lo), float(hi)
