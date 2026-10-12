"""Explainer protocol and the pieces every explainer shares (DESIGN.md Sections 7.16 and 10.1).

All explainers work on one processed image at the model input size and return a float32 map in
[0, 1] of the same height and width. The explained quantity is set by ``xai.target``:

* ``"disease"`` (default, ISSUE-014): the log-odds of any DR, ``logsumexp(grades 1-4) - logit(0)``.
  It stays meaningful when the model predicts grade 0 for an image that has lesions.
* an integer grade: that grade's logit (use the predicted grade to reproduce D11).
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
from torch import nn

from xaidr.data.augment import IMAGENET_MEAN, IMAGENET_STD
from xaidr.models.dual_head import DRDualHeadModel

Target = str | int
DISEASE = "disease"


class Explainer(Protocol):
    name: str

    def explain(self, image_tensor: torch.Tensor, target: Target = DISEASE) -> np.ndarray: ...


def check_target(target: Target) -> None:
    if target != DISEASE and not (isinstance(target, int) and 0 <= target <= 4):
        raise ValueError(f"target must be '{DISEASE}' or a grade 0-4, got {target!r}")


def target_score(grade_logits: torch.Tensor, target: Target) -> torch.Tensor:
    """Scalar score per image, shape ``(B,)``, for the explained target."""
    check_target(target)
    if target == DISEASE:
        return torch.logsumexp(grade_logits[:, 1:], dim=1) - grade_logits[:, 0]
    return grade_logits[:, int(target)]


def target_probability(grade_logits: torch.Tensor, target: Target) -> torch.Tensor:
    """Probability per image, shape ``(B,)``: P(any DR) or P(grade)."""
    check_target(target)
    probs = torch.softmax(grade_logits.float(), dim=1)
    return 1.0 - probs[:, 0] if target == DISEASE else probs[:, int(target)]


def minmax(saliency: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]; a constant map becomes all zeros."""
    saliency = np.asarray(saliency, dtype=np.float32)
    low, high = float(saliency.min()), float(saliency.max())
    if high - low < 1e-12:
        return np.zeros_like(saliency, dtype=np.float32)
    return ((saliency - low) / (high - low)).astype(np.float32)


def denormalise(image_tensor: torch.Tensor) -> np.ndarray:
    """Normalised ``(3, H, W)`` tensor back to the uint8 RGB image it was made from."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = (image_tensor.detach().cpu().float() * std + mean) * 255.0
    return image.round().clamp(0, 255).byte().permute(1, 2, 0).numpy()


def normalise_batch(images_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """uint8 or float RGB batch ``(N, H, W, 3)`` in 0-255 to a normalised ``(N, 3, H, W)`` tensor."""
    batch = torch.as_tensor(np.asarray(images_rgb), dtype=torch.float32, device=device) / 255.0
    mean = torch.tensor(IMAGENET_MEAN, device=device)
    std = torch.tensor(IMAGENET_STD, device=device)
    return ((batch - mean) / std).permute(0, 3, 1, 2).contiguous()


class ProbabilityFn:
    """``f(images_rgb) -> P(target)`` for perturbation-based explainers (SHAP, LIME)."""

    def __init__(self, model: DRDualHeadModel, target: Target, batch_size: int = 16) -> None:
        check_target(target)
        self.model = model.eval()
        self.target = target
        self.batch_size = batch_size
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def __call__(self, images_rgb: np.ndarray) -> np.ndarray:
        images_rgb = np.asarray(images_rgb)
        out = []
        for start in range(0, len(images_rgb), self.batch_size):
            batch = normalise_batch(images_rgb[start:start + self.batch_size], self.device)
            out.append(target_probability(self.model(batch)["grade_logits"], self.target).cpu().numpy())
        return np.concatenate(out).astype(np.float64)


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device
