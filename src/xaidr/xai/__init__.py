"""Explainers and their validation metrics."""
from __future__ import annotations

from xaidr.config import XAICfg
from xaidr.models.dual_head import DRDualHeadModel
from xaidr.xai.base import Explainer


def build_explainer(name: str, model: DRDualHeadModel, cfg: XAICfg, seed: int = 42) -> Explainer:
    """Explainer by config name; imports are local so that unused libraries are not loaded."""
    if name == "gradcam_pp":
        from xaidr.xai.gradcam_pp import GradCAMPlusPlus

        return GradCAMPlusPlus(model)
    if name == "shap":
        from xaidr.xai.shap_explainer import ShapExplainer

        return ShapExplainer(model, nsamples=int(cfg.shap.get("nsamples", 512)), seed=seed)
    if name == "lime":
        from xaidr.xai.lime_explainer import LimeExplainer

        return LimeExplainer(
            model, num_samples=int(cfg.lime.get("num_samples", 1000)),
            hide_color=int(cfg.lime.get("hide_color", 0)), seed=seed,
        )
    if name == "attention":
        from xaidr.xai.attention import AttentionSaliency

        return AttentionSaliency(model)
    raise KeyError(f"unknown explainer '{name}'")
