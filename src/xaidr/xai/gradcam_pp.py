"""Grad-CAM++ on the backbone's final feature map.

Own implementation of the closed form of Chattopadhay et al. (WACV 2018) that uses first-order
gradients only, the same form as ``pytorch_grad_cam.GradCAMPlusPlus``; ``tests/test_explainers.py``
checks the two against each other. Writing it here avoids hooks: the model already exposes its
final feature map, which is the target layer for both ConvNeXt and MaxViT.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from xaidr.models.dual_head import DRDualHeadModel
from xaidr.xai.base import DISEASE, Target, minmax, model_device, target_score


def gradcam_pp_weights(activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
    """Channel weights ``(B, C)`` from activations and gradients of shape ``(B, C, H, W)``."""
    grads_2, grads_3 = gradients.pow(2), gradients.pow(3)
    sum_activations = activations.sum(dim=(2, 3), keepdim=True)
    denominator = 2 * grads_2 + sum_activations * grads_3
    alpha = grads_2 / torch.where(denominator != 0, denominator, torch.ones_like(denominator))
    return (alpha * torch.relu(gradients)).sum(dim=(2, 3))


class GradCAMPlusPlus:
    name = "gradcam_pp"

    def __init__(self, model: DRDualHeadModel) -> None:
        self.model = model.eval()

    def explain(self, image_tensor: torch.Tensor, target: Target = DISEASE) -> np.ndarray:
        x = image_tensor.unsqueeze(0).to(model_device(self.model))
        with torch.enable_grad():
            feature_map = self.model.backbone.forward_features(x)
            if not feature_map.requires_grad:  # a backbone without trainable layers before the map
                feature_map.requires_grad_(True)
            feature_map.retain_grad()
            logits = self.model.grading_head(self.model.backbone.pool(feature_map))
            score = target_score(logits, target).sum()
            self.model.zero_grad(set_to_none=True)
            score.backward()
        if feature_map.grad is None:
            raise RuntimeError("the target score does not depend on the feature map")
        activations, gradients = feature_map.detach().float(), feature_map.grad.detach().float()
        self.model.zero_grad(set_to_none=True)
        weights = gradcam_pp_weights(activations, gradients)
        cam = torch.relu((weights[:, :, None, None] * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        return minmax(cam[0, 0].cpu().numpy())
