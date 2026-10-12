"""Attention saliency for the hybrid backbone (DESIGN.md Section 7.20).

The map is the attention each position of the final feature map *receives* in the last
attention block of MaxViT, averaged over heads and over all querying positions. It is a
model-internal signal: it does not depend on the explained target and is not, by itself,
evidence of what drove the prediction (docs/SCIENTIFIC_REVIEW.md, F-012).

timm computes attention through a fused kernel that never materialises the weights
(docs/VERIFICATION.md, V8). For one forward pass the block is switched to the explicit path
and the softmax output is captured at ``attn_drop``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from xaidr.models.dual_head import DRDualHeadModel
from xaidr.xai.base import DISEASE, Target, minmax, model_device


class AttentionUnavailableError(RuntimeError):
    """The backbone has no attention block this method can read."""


def attention_received(attn: torch.Tensor) -> torch.Tensor:
    """Mean attention each key position receives: ``(B, heads, N, N) -> (B, N)``.

    ``attn[b, h, i, j]`` is how much query ``i`` attends to key ``j``; rows sum to one.
    """
    return attn.mean(dim=1).mean(dim=1)


class AttentionSaliency:
    name = "attention"

    def __init__(self, model: DRDualHeadModel) -> None:
        self.model = model.eval()
        try:
            self.block = model.backbone.model.stages[-1].blocks[-1].attn_grid.attn
        except AttributeError as exc:
            raise AttentionUnavailableError(
                f"attention saliency needs a MaxViT backbone; '{model.backbone.name}' has no attention block"
            ) from exc

    def explain(self, image_tensor: torch.Tensor, target: Target = DISEASE) -> np.ndarray:
        captured: list[torch.Tensor] = []
        hook = self.block.attn_drop.register_forward_hook(lambda _m, args, _o: captured.append(args[0]))
        fused = self.block.fused_attn
        self.block.fused_attn = False
        try:
            with torch.no_grad():
                feature_map = self.model.backbone.forward_features(
                    image_tensor.unsqueeze(0).to(model_device(self.model))
                )
        finally:
            self.block.fused_attn = fused
            hook.remove()
        attn = captured[-1].float()
        grid_h, grid_w = feature_map.shape[-2:]
        if attn.shape[0] != 1 or attn.shape[-1] != grid_h * grid_w:
            raise AttentionUnavailableError(
                f"the last attention block covers {attn.shape[0]} partition(s) of {attn.shape[-1]} tokens, "
                f"not one partition spanning the {grid_h} x {grid_w} feature map"
            )
        received = attention_received(attn).reshape(1, 1, grid_h, grid_w)
        saliency = F.interpolate(received, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        return minmax(saliency[0, 0].cpu().numpy())
