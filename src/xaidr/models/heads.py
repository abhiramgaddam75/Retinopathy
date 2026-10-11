"""Task heads on the shared pooled features."""
from __future__ import annotations

import torch
from torch import nn

NUM_GRADES = 5


def _mlp(in_features: int, hidden: int, dropout: float, out_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, out_features)
    )


class DetectionHead(nn.Module):
    """One logit per image: DR present vs absent. Output shape ``(B,)``."""

    def __init__(self, in_features: int, hidden: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = _mlp(in_features, hidden, dropout, 1)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled).squeeze(-1)


class GradingHead(nn.Module):
    """Five ICDR grade logits. Output shape ``(B, 5)``."""

    def __init__(self, in_features: int, hidden: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = _mlp(in_features, hidden, dropout, NUM_GRADES)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)
