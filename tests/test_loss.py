from __future__ import annotations

import math

import pytest
import torch

from xaidr.training.losses import JointDRLoss, inverse_sqrt_class_weights

DET_LOGIT = torch.tensor([2.0, -1.0, 0.5, -3.0])
DR_LABEL = torch.tensor([1, 0, 1, 0])
GRADE_LOGITS = torch.tensor(
    [[0.1, 2.0, 0.3, -1.0, 0.0], [3.0, 0.2, -0.5, 0.1, 0.0],
     [0.0, 0.5, 0.2, 1.5, -0.2], [1.0, 1.0, 1.0, 1.0, 1.0]]
)
GRADE = torch.tensor([1, 0, 3, 4])


def by_hand(weights: list[float] | None) -> tuple[float, float]:
    bce = [
        -math.log(1 / (1 + math.exp(-z))) if y == 1 else -math.log(1 - 1 / (1 + math.exp(-z)))
        for z, y in zip(DET_LOGIT.tolist(), DR_LABEL.tolist(), strict=True)
    ]
    ce, norm = 0.0, 0.0
    for row, target in zip(GRADE_LOGITS.tolist(), GRADE.tolist(), strict=True):
        log_p = row[target] - math.log(sum(math.exp(v) for v in row))
        w = 1.0 if weights is None else weights[target]
        ce += -w * log_p
        norm += w
    return sum(bce) / 4, ce / norm


@pytest.mark.parametrize("weights", [None, [0.5, 1.0, 0.8, 1.6, 1.3]])
def test_loss_equals_hand_computed_value(weights: list[float] | None) -> None:
    det, grading = by_hand(weights)
    tensor = None if weights is None else torch.tensor(weights)
    total, parts = JointDRLoss(0.4, 0.6, tensor)(DET_LOGIT, GRADE_LOGITS, DR_LABEL, GRADE)
    assert float(total) == pytest.approx(0.4 * det + 0.6 * grading, abs=1e-6)
    assert parts["loss_det"] == pytest.approx(det, abs=1e-6)
    assert parts["loss_grade"] == pytest.approx(grading, abs=1e-6)
    assert parts["loss"] == pytest.approx(float(total), abs=1e-9)


def test_loss_weights_are_applied() -> None:
    det, grading = by_hand(None)
    only_det, _ = JointDRLoss(1.0, 0.0)(DET_LOGIT, GRADE_LOGITS, DR_LABEL, GRADE)
    only_grade, _ = JointDRLoss(0.0, 1.0)(DET_LOGIT, GRADE_LOGITS, DR_LABEL, GRADE)
    assert float(only_det) == pytest.approx(det, abs=1e-6)
    assert float(only_grade) == pytest.approx(grading, abs=1e-6)


def test_inverse_sqrt_class_weights() -> None:
    grades = [0] * 100 + [1] * 25 + [2] * 49 + [3] * 4 + [4] * 16
    weights = inverse_sqrt_class_weights(grades)
    raw = torch.tensor([1 / 10, 1 / 5, 1 / 7, 1 / 2, 1 / 4])
    assert torch.allclose(weights, raw * 5 / raw.sum(), atol=1e-6)
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-6)
    assert weights[3] > weights[4] > weights[1] > weights[2] > weights[0]  # rarer grade, larger weight
    with pytest.raises(ValueError, match="every grade"):
        inverse_sqrt_class_weights([0, 1, 2, 3])
