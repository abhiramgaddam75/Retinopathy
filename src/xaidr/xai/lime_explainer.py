"""LIME over quickshift superpixels (DESIGN.md Section 7.19).

Superpixels are switched to ``hide_color`` at random, a weighted linear model is fitted to the
model's response, and each superpixel's positive weight is painted back on the image.
"""
from __future__ import annotations

import numpy as np
import torch
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

from xaidr.models.dual_head import DRDualHeadModel
from xaidr.xai.base import DISEASE, ProbabilityFn, Target, denormalise, minmax

QUICKSHIFT = {"kernel_size": 4, "max_dist": 200, "ratio": 0.2}


class LimeExplainer:
    name = "lime"

    def __init__(
        self, model: DRDualHeadModel, num_samples: int = 1000, hide_color: int = 0,
        batch_size: int = 16, seed: int = 42,
    ) -> None:
        self.model = model.eval()
        self.num_samples = num_samples
        self.hide_color = hide_color
        self.batch_size = batch_size
        self.seed = seed

    def explain(self, image_tensor: torch.Tensor, target: Target = DISEASE) -> np.ndarray:
        image = denormalise(image_tensor)
        probability = ProbabilityFn(self.model, target, self.batch_size)

        def two_class(images: np.ndarray) -> np.ndarray:
            p = probability(images)
            return np.stack([1.0 - p, p], axis=1)

        explanation = lime_image.LimeImageExplainer(random_state=self.seed).explain_instance(
            image, two_class, labels=(1,), top_labels=None, hide_color=self.hide_color,
            num_samples=self.num_samples, batch_size=self.batch_size, random_seed=self.seed,
            segmentation_fn=SegmentationAlgorithm("quickshift", random_seed=self.seed, **QUICKSHIFT),
        )
        weights = dict(explanation.local_exp[1])
        saliency = np.zeros(explanation.segments.shape, dtype=np.float32)
        for segment, weight in weights.items():
            if weight > 0:
                saliency[explanation.segments == segment] = weight
        return minmax(saliency)
