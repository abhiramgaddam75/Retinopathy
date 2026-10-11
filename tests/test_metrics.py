from __future__ import annotations

import logging

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)

from xaidr.training import metrics as M
from xaidr.training.threshold import youden_threshold
from xaidr.training.trainer import selection_score, warmup_cosine

TOL = 1e-9


@pytest.fixture
def data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 400
    grade = rng.integers(0, 5, n)
    logits = rng.normal(size=(n, 5)) + 1.5 * np.eye(5)[grade]
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    dr = (grade >= 1).astype(int)
    score = np.clip(0.5 + 0.25 * (dr - 0.5) + rng.normal(scale=0.2, size=n), 0, 1)
    return {"grade": grade, "probs": probs, "dr": dr, "score": score}


def test_detection_metrics_match_sklearn(data: dict[str, np.ndarray]) -> None:
    got = M.detection_metrics(data["dr"], data["score"], 0.55)
    pred = (data["score"] >= 0.55).astype(int)
    tn, fp, fn, tp = confusion_matrix(data["dr"], pred).ravel()
    assert got["accuracy"] == pytest.approx(accuracy_score(data["dr"], pred), abs=TOL)
    assert got["sensitivity"] == pytest.approx(tp / (tp + fn), abs=TOL)
    assert got["specificity"] == pytest.approx(tn / (tn + fp), abs=TOL)
    assert got["auc"] == pytest.approx(roc_auc_score(data["dr"], data["score"]), abs=TOL)


def test_grading_metrics_match_sklearn(data: dict[str, np.ndarray]) -> None:
    got = M.grading_metrics(data["grade"], data["probs"])
    pred = data["probs"].argmax(axis=1)
    assert got["accuracy"] == pytest.approx(accuracy_score(data["grade"], pred), abs=TOL)
    macro_recall = recall_score(data["grade"], pred, average="macro")
    assert got["macro_sensitivity"] == pytest.approx(macro_recall, abs=TOL)
    assert got["qwk"] == pytest.approx(cohen_kappa_score(data["grade"], pred, weights="quadratic"), abs=TOL)
    expected = roc_auc_score(data["grade"], data["probs"], multi_class="ovr", average="macro")
    assert got["macro_auc"] == pytest.approx(expected, abs=TOL)


def test_absent_class_gives_nan_auc_with_a_warning(
    data: dict[str, np.ndarray], caplog: pytest.LogCaptureFixture
) -> None:
    keep = data["grade"] != 3
    grade, probs = data["grade"][keep], data["probs"][keep]
    with caplog.at_level(logging.WARNING):
        value = M.macro_ovr_auc(grade, probs)
    assert "grade(s) [3]" in caplog.text
    per_class = [roc_auc_score((grade == g).astype(int), probs[:, g]) for g in (0, 1, 2, 4)]
    assert value == pytest.approx(np.mean(per_class), abs=TOL)
    assert np.isnan(M.auc(np.ones(5), np.linspace(0, 1, 5)))
    # recall is averaged over the grades that exist, even if grade 3 is predicted
    assert M.macro_sensitivity(grade, np.full(len(grade), 3)) == 0.0
    assert M.quadratic_weighted_kappa(grade, grade) == 1.0


def test_youden_threshold(data: dict[str, np.ndarray]) -> None:
    threshold = youden_threshold(data["dr"], data["score"])

    def youden(t: float) -> float:
        m = M.detection_metrics(data["dr"], data["score"], t)
        return m["sensitivity"] + m["specificity"] - 1

    best = max(youden(t) for t in np.unique(data["score"]))
    assert youden(threshold) == pytest.approx(best, abs=TOL)
    assert youden(threshold) >= youden(0.5) and np.isfinite(threshold)
    assert youden_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(0.8)
    with pytest.raises(ValueError, match="both classes"):
        youden_threshold([1, 1], [0.2, 0.9])


def test_bootstrap_ci(data: dict[str, np.ndarray]) -> None:
    estimate, lo, hi = M.bootstrap_ci(M.auc, data["dr"], data["score"], n=300)
    assert estimate == pytest.approx(roc_auc_score(data["dr"], data["score"]), abs=TOL)
    assert lo < estimate < hi and hi - lo < 0.15
    assert (estimate, lo, hi) == M.bootstrap_ci(M.auc, data["dr"], data["score"], n=300)  # seeded
    assert M.bootstrap_ci(M.auc, data["dr"], data["score"], n=300, seed=1)[1] != lo
    # works for 2-D scores and for the unstratified variant
    qwk = M.bootstrap_ci(M.macro_ovr_auc, data["grade"], data["probs"], n=50, stratify=False)
    assert qwk[1] < qwk[0] < qwk[2]


def test_bootstrap_interval_covers_a_known_rate() -> None:
    rng = np.random.default_rng(3)
    covered = 0
    for repeat in range(200):
        correct = (rng.random(150) < 0.8).astype(int)
        _, lo, hi = M.bootstrap_ci(
            lambda y, p: float(p.mean()), np.zeros(150, int), correct, n=300, seed=repeat
        )
        covered += lo <= 0.8 <= hi
    assert 0.90 <= covered / 200 <= 0.99


def test_paired_bootstrap_diff(data: dict[str, np.ndarray]) -> None:
    grade = data["grade"]
    good = data["probs"].argmax(axis=1)
    rng = np.random.default_rng(5)
    worse = np.where(rng.random(len(grade)) < 0.3, rng.integers(0, 5, len(grade)), good)
    diff, lo, hi = M.paired_bootstrap_diff(M.quadratic_weighted_kappa, grade, worse, good, n=300)
    expected = M.quadratic_weighted_kappa(grade, good) - M.quadratic_weighted_kappa(grade, worse)
    assert diff == pytest.approx(expected, abs=TOL) and lo > 0 and lo < diff < hi
    same = M.paired_bootstrap_diff(M.quadratic_weighted_kappa, grade, good, good, n=100)
    assert same == (0.0, 0.0, 0.0)


def test_schedule_and_selection_score() -> None:
    factors = [warmup_cosine(e, 2, 30) for e in range(30)]
    assert factors[0] == 0.5 and factors[1] == factors[2] == 1.0 == max(factors)  # warm-up, then the peak
    assert all(a > b for a, b in zip(factors[2:], factors[3:], strict=False))  # cosine decay
    assert 0 < factors[-1] < 0.01
    assert selection_score(0.8, 0.9) == pytest.approx(0.85)
