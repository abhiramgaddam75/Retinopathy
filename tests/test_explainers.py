from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch import nn

from xaidr.config import ModelCfg, XAICfg
from xaidr.data.augment import eval_transform
from xaidr.models.dual_head import DRDualHeadModel, GradeOnly
from xaidr.xai import build_explainer
from xaidr.xai.attention import AttentionSaliency, AttentionUnavailableError, attention_received
from xaidr.xai.base import (
    ProbabilityFn,
    denormalise,
    minmax,
    target_probability,
    target_score,
)
from xaidr.xai.gradcam_pp import GradCAMPlusPlus, gradcam_pp_weights

SIZE = 224


@pytest.fixture(scope="module")
def convnext() -> DRDualHeadModel:
    torch.manual_seed(0)
    return DRDualHeadModel(ModelCfg(backbone="convnext_tiny", pretrained=False), SIZE).eval()


@pytest.fixture(scope="module")
def image() -> torch.Tensor:
    rng = np.random.default_rng(0)
    rgb = rng.integers(30, 220, size=(SIZE, SIZE, 3), dtype=np.uint8)
    rgb[:20] = 0  # a black band, like the border of a fundus photo
    return eval_transform(SIZE)(image=rgb)["image"]


def assert_saliency(saliency: np.ndarray) -> None:
    assert saliency.shape == (SIZE, SIZE) and saliency.dtype == np.float32
    assert saliency.min() >= 0.0 and saliency.max() <= 1.0 and np.isfinite(saliency).all()


def test_targets() -> None:
    logits = torch.tensor([[2.0, 0.0, 1.0, -1.0, 0.5], [0.0, 3.0, 0.0, 0.0, 0.0]])
    probs = torch.softmax(logits, dim=1)
    assert torch.allclose(target_probability(logits, "disease"), 1 - probs[:, 0])
    assert torch.allclose(target_probability(logits, 2), probs[:, 2])
    # the disease score is the log-odds of any DR
    odds = (1 - probs[:, 0]) / probs[:, 0]
    assert torch.allclose(target_score(logits, "disease"), odds.log(), atol=1e-6)
    assert torch.equal(target_score(logits, 3), logits[:, 3])
    for bad in ("grade", 5, -1, 1.5):
        with pytest.raises(ValueError, match="target"):
            target_score(logits, bad)  # type: ignore[arg-type]


def test_denormalise_and_minmax(image: torch.Tensor) -> None:
    rgb = denormalise(image)
    assert rgb.shape == (SIZE, SIZE, 3) and rgb.dtype == np.uint8 and not rgb[:20].any()
    assert torch.allclose(eval_transform(SIZE)(image=rgb)["image"], image, atol=1e-6)
    assert minmax(np.array([[2.0, 4.0], [3.0, 6.0]])).tolist() == [[0.0, 0.5], [0.25, 1.0]]
    assert not minmax(np.full((3, 3), 7.0)).any()


def test_probability_fn(convnext: DRDualHeadModel, image: torch.Tensor) -> None:
    rgb = denormalise(image)
    batch = np.stack([rgb, rgb[::-1], rgb])
    p = ProbabilityFn(convnext, "disease", batch_size=2)(batch)
    with torch.no_grad():
        direct = 1 - torch.softmax(GradeOnly(convnext)(image[None]), dim=1)[0, 0]
    assert p.shape == (3,) and p[0] == pytest.approx(float(direct), abs=1e-5) and p[0] == pytest.approx(p[2])


def test_gradcam_pp_weights_by_hand() -> None:
    activations = torch.tensor([[[[1.0, 2.0], [0.0, 1.0]]]])  # sum 4
    gradients = torch.tensor([[[[1.0, -1.0], [0.0, 2.0]]]])
    # alpha = g^2 / (2 g^2 + 4 g^3): 1/6, 1/(2-4) = -1/2, 0 -> guarded, 4/(8+32) = 1/10
    # weight = sum(alpha * relu(g)) = 1/6 * 1 + 0 + 0 + 1/10 * 2
    assert gradcam_pp_weights(activations, gradients).item() == pytest.approx(1 / 6 + 0.2)


class PatchBackbone(nn.Module):
    """Feature map = 16 px average pooling of the image; pooled features = its spatial mean."""

    name = "toy"

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.avg_pool2d(x, 16)

    def pool(self, feature_map: torch.Tensor) -> torch.Tensor:
        return feature_map.mean(dim=(2, 3))


class PatchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = PatchBackbone()
        self.grading_head = nn.Linear(3, 5)
        with torch.no_grad():
            self.grading_head.weight.copy_(torch.tensor([[-1.0] * 3] + [[1.0] * 3] * 4))
            self.grading_head.bias.zero_()


def test_gradcam_pp_peaks_inside_the_only_patch_that_drives_the_output() -> None:
    """The image is empty except for one patch, so the model's output depends on that patch alone."""
    cell_row, cell_col = 3, 9
    image = torch.zeros(3, SIZE, SIZE)
    image[:, cell_row * 16:(cell_row + 1) * 16, cell_col * 16:(cell_col + 1) * 16] = 1.0
    saliency = GradCAMPlusPlus(PatchModel()).explain(image)  # type: ignore[arg-type]
    assert_saliency(saliency)
    row, col = np.unravel_index(int(saliency.argmax()), saliency.shape)
    assert cell_row * 16 <= row < (cell_row + 1) * 16 and cell_col * 16 <= col < (cell_col + 1) * 16
    far = np.ones((SIZE, SIZE), bool)
    far[(cell_row - 1) * 16:(cell_row + 2) * 16, (cell_col - 1) * 16:(cell_col + 2) * 16] = False
    assert saliency[far].max() == 0.0 and saliency.max() == 1.0
    # explaining "grade 0" for the same image gives no positive evidence anywhere
    assert not GradCAMPlusPlus(PatchModel()).explain(image, target=0).any()  # type: ignore[arg-type]


def test_gradcam_pp_matches_pytorch_grad_cam(convnext: DRDualHeadModel, image: torch.Tensor) -> None:
    from pytorch_grad_cam import GradCAMPlusPlus as Reference
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    # The library adds 1e-7 to the Grad-CAM++ denominator. An untrained head gives gradients so
    # small that this constant dominates, so the comparison uses a head with a realistic scale.
    model = copy.deepcopy(convnext)
    with torch.no_grad():
        model.grading_head.net[-1].weight.mul_(1e3)
    ours = GradCAMPlusPlus(model).explain(image, target=2)
    assert_saliency(ours)
    reference = Reference(model=GradeOnly(model), target_layers=[model.backbone.model.stages[-1]])
    theirs = reference(input_tensor=image[None], targets=[ClassifierOutputTarget(2)])[0]
    assert np.corrcoef(ours.ravel(), theirs.ravel())[0, 1] > 0.99
    # the default target is the disease score, a different quantity from one grade's logit
    assert_saliency(GradCAMPlusPlus(convnext).explain(image))
    assert all(p.grad is None for p in convnext.parameters())  # no gradients left behind


def test_shap_and_lime(convnext: DRDualHeadModel, image: torch.Tensor) -> None:
    cfg = XAICfg(shap={"nsamples": 48}, lime={"num_samples": 24, "hide_color": 0})
    shap_map = build_explainer("shap", convnext, cfg).explain(image)
    assert_saliency(shap_map)
    lime_explainer = build_explainer("lime", convnext, cfg)
    lime_map = lime_explainer.explain(image)
    assert_saliency(lime_map)
    assert np.array_equal(lime_map, lime_explainer.explain(image))  # deterministic given the seed
    assert len(np.unique(lime_map)) > 1 and len(np.unique(shap_map)) > 1
    assert build_explainer("gradcam_pp", convnext, cfg).name == "gradcam_pp"
    with pytest.raises(KeyError):
        build_explainer("saliency", convnext, cfg)


def test_attention_received_by_hand() -> None:
    attn = torch.tensor([[[[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.2, 0.3, 0.5]],
                          [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.5, 0.5]]]])  # (1, 2 heads, 3, 3)
    expected = torch.tensor([[(0.5 + 0.0 + 0.2 + 1.0 + 0.0 + 0.0) / 6,
                              (0.5 + 1.0 + 0.3 + 0.0 + 0.0 + 0.5) / 6,
                              (0.0 + 0.0 + 0.5 + 0.0 + 1.0 + 0.5) / 6]])
    assert torch.allclose(attention_received(attn), expected)
    assert attention_received(attn).sum().item() == pytest.approx(1.0)


def test_attention_needs_a_hybrid_backbone(convnext: DRDualHeadModel) -> None:
    with pytest.raises(AttentionUnavailableError, match="MaxViT"):
        AttentionSaliency(convnext)


def test_attention_saliency_on_maxvit() -> None:
    torch.manual_seed(0)
    model = DRDualHeadModel(ModelCfg(backbone="maxvit_tiny_tf_512", pretrained=False), 512).eval()
    block = model.backbone.model.stages[-1].blocks[-1].attn_grid.attn
    x = torch.randn(3, 512, 512)
    with torch.no_grad():
        before = model(x[None])["grade_logits"]
    saliency = AttentionSaliency(model).explain(x)
    assert saliency.shape == (512, 512) and saliency.dtype == np.float32
    assert saliency.min() == 0.0 and saliency.max() == 1.0
    # the block is restored and the model's output is unchanged
    assert block.fused_attn is True and len(block.attn_drop._forward_hooks) == 0
    with torch.no_grad():
        assert torch.allclose(model(x[None])["grade_logits"], before, atol=1e-5)
    assert GradCAMPlusPlus(model).explain(x).shape == (512, 512)
