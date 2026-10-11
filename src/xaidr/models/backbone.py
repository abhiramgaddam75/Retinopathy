"""timm backbone wrapper exposing the final feature map and the pooled features."""
from __future__ import annotations

from typing import Any

import timm
import torch
from torch import nn


class Backbone(nn.Module):
    def __init__(self, name: str, pretrained: bool = True, img_size: int = 512) -> None:
        super().__init__()
        # MaxViT builds its attention windows for a fixed input size; ConvNeXt is size-agnostic
        # and rejects the argument (docs/VERIFICATION.md, V2).
        kwargs: dict[str, Any] = {"img_size": img_size} if name.startswith("maxvit") else {}
        self.name = name
        self.model: Any = timm.create_model(name, pretrained=pretrained, num_classes=0, **kwargs)
        width: Any = getattr(self.model, "head_hidden_size", None) or self.model.num_features
        self.num_features = int(width)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Final feature map ``(B, C, H, W)``, the target of Grad-CAM++ and attention saliency."""
        return self.model.forward_features(x)

    def pool(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Pooled features ``(B, num_features)`` through the backbone's own pooling and norm."""
        return self.model.forward_head(feature_map, pre_logits=True)
