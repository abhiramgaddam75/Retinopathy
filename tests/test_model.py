from __future__ import annotations

import pytest
import torch

from xaidr.config import ModelCfg
from xaidr.models.dual_head import DRDualHeadModel, GradeOnly


def model_for(backbone: str, size: int) -> DRDualHeadModel:
    torch.manual_seed(0)
    return DRDualHeadModel(ModelCfg(backbone=backbone, pretrained=False), size)


@pytest.mark.parametrize(
    "backbone, size, features, grid",
    [("convnext_tiny", 224, 768, 7), ("maxvit_tiny_tf_512", 512, 512, 16)],
)
def test_output_shapes(backbone: str, size: int, features: int, grid: int) -> None:
    model = model_for(backbone, size).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, size, size))
    assert out["det_logit"].shape == (2,)
    assert out["grade_logits"].shape == (2, 5)
    assert out["pooled"].shape == (2, features) == (2, model.backbone.num_features)
    assert out["feature_map"].shape[0] == 2 and out["feature_map"].shape[-2:] == (grid, grid)
    assert GradeOnly(model)(torch.randn(1, 3, size, size)).shape == (1, 5)


def test_one_backbone_feeds_both_heads_and_all_get_gradients() -> None:
    model = model_for("convnext_tiny", 224).train()
    assert sum(1 for name, _ in model.named_children() if name == "backbone") == 1
    head_inputs = []
    hooks = [head.register_forward_hook(lambda _m, args, _o: head_inputs.append(args[0]))
             for head in (model.detection_head, model.grading_head)]
    out = model(torch.randn(2, 3, 224, 224))
    assert head_inputs[0] is head_inputs[1] is out["pooled"]  # the same pooled tensor
    for hook in hooks:
        hook.remove()

    (out["det_logit"].sum() + out["grade_logits"].sum()).backward()
    for head in (model.detection_head, model.grading_head):
        for name, parameter in head.named_parameters():
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
    backbone_grads = [p.grad for p in model.backbone.parameters() if p.requires_grad]
    assert all(g is not None for g in backbone_grads)

    # each head alone also reaches the shared backbone
    for key in ("det_logit", "grade_logits"):
        model.zero_grad()
        model(torch.randn(2, 3, 224, 224))[key].sum().backward()
        first = next(model.backbone.parameters())
        assert first.grad is not None and first.grad.abs().sum() > 0
