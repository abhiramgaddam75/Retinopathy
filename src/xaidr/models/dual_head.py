"""Shared backbone with a detection head and a grading head."""
from __future__ import annotations

import torch
from torch import nn

from xaidr.config import ModelCfg
from xaidr.models.backbone import Backbone
from xaidr.models.heads import DetectionHead, GradingHead


class DRDualHeadModel(nn.Module):
    def __init__(self, cfg: ModelCfg, img_size: int = 512) -> None:
        super().__init__()
        self.backbone = Backbone(cfg.backbone, cfg.pretrained, img_size)
        features = self.backbone.num_features
        self.detection_head = DetectionHead(features, cfg.head_hidden, cfg.dropout)
        self.grading_head = GradingHead(features, cfg.head_hidden, cfg.dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_map = self.backbone.forward_features(x)
        pooled = self.backbone.pool(feature_map)
        return {
            "det_logit": self.detection_head(pooled),
            "grade_logits": self.grading_head(pooled),
            "pooled": pooled,
            "feature_map": feature_map,
        }


class GradeOnly(nn.Module):
    """Returns only the grade logits, so explainers see a model with a single tensor output."""

    def __init__(self, model: DRDualHeadModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["grade_logits"]
