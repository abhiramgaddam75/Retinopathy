"""SHAP with an image masker (DESIGN.md Section 7.18).

``shap.PartitionExplainer`` over ``shap.maskers.Image``: hidden regions are replaced by the
masker's background (Telea inpainting by default, so no external background set is needed), and
the image is split hierarchically until the evaluation budget ``nsamples`` is used. With the
default 512 evaluations the finest regions are roughly a 16 x 16 grid at 512 px. Positive
attributions are kept, summed over colour channels and min-max normalised.
"""
from __future__ import annotations

import numpy as np
import shap
import torch

from xaidr.models.dual_head import DRDualHeadModel
from xaidr.xai.base import DISEASE, ProbabilityFn, Target, denormalise, minmax


class ShapExplainer:
    name = "shap"

    def __init__(
        self, model: DRDualHeadModel, nsamples: int = 512, masker: str = "inpaint_telea",
        batch_size: int = 16, seed: int = 42,
    ) -> None:
        self.model = model.eval()
        self.nsamples = nsamples
        self.masker = masker
        self.batch_size = batch_size
        self.seed = seed

    def explain(self, image_tensor: torch.Tensor, target: Target = DISEASE) -> np.ndarray:
        image = denormalise(image_tensor)
        predict = ProbabilityFn(self.model, target, self.batch_size)
        masker = shap.maskers.Image(self.masker, image.shape)
        explainer = shap.PartitionExplainer(predict, masker, seed=self.seed)
        explanation = explainer(
            image[None], max_evals=self.nsamples, batch_size=self.batch_size, silent=True
        )
        values = np.asarray(explanation.values)[0]  # (H, W, 3) for the single scalar output
        values = values.reshape(*image.shape[:2], -1).sum(axis=2)
        return minmax(np.maximum(values, 0.0))
