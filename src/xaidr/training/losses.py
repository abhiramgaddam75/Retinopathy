"""Joint loss: weighted binary cross-entropy (detection) + class-weighted cross-entropy (grading)."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from xaidr.models.heads import NUM_GRADES


def inverse_sqrt_class_weights(train_grades: Sequence[int]) -> torch.Tensor:
    """Weight 1/sqrt(n_c) per grade, scaled to mean 1. Pass grades of the training split only."""
    counts = np.bincount(np.asarray(train_grades, dtype=int), minlength=NUM_GRADES).astype(float)
    if (counts == 0).any():
        raise ValueError(f"every grade needs at least one training image; counts are {counts.tolist()}")
    weights = 1.0 / np.sqrt(counts)
    return torch.tensor(weights * NUM_GRADES / weights.sum(), dtype=torch.float32)


class JointDRLoss(nn.Module):
    """``L = lambda_det * BCE(det_logit, dr_label) + lambda_grade * CE(grade_logits, grade)``."""

    def __init__(
        self,
        lambda_det: float = 0.4,
        lambda_grade: float = 0.6,
        grade_class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.lambda_det = lambda_det
        self.lambda_grade = lambda_grade
        self.grade_class_weights: torch.Tensor | None
        self.register_buffer("grade_class_weights", grade_class_weights)

    def forward(
        self, det_logit: torch.Tensor, grade_logits: torch.Tensor, dr_label: torch.Tensor, grade: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        det = F.binary_cross_entropy_with_logits(det_logit.float(), dr_label.float())
        grading = F.cross_entropy(grade_logits.float(), grade.long(), weight=self.grade_class_weights)
        total = self.lambda_det * det + self.lambda_grade * grading
        parts = {"loss": total.item(), "loss_det": det.item(), "loss_grade": grading.item()}
        return total, parts
