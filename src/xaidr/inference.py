"""Single-image prediction for the demo app (docs/DEMO_APP_BRIEF.md).

One uploaded photo goes through exactly the code path used in training and evaluation: border
crop, CLAHE, square padding, resize, ImageNet normalisation, the trained dual-head model, the
detection threshold frozen on validation data, and Grad-CAM++ for the "any DR" score.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from xaidr.config import Config
from xaidr.data.augment import eval_transform
from xaidr.data.preprocess import preprocess_arrays
from xaidr.eval.evaluate_grading import method_label
from xaidr.models.dual_head import DRDualHeadModel
from xaidr.training.run import load_best_model, run_config
from xaidr.xai.gradcam_pp import GradCAMPlusPlus

GRADE_NAMES = ("No DR", "Mild", "Moderate", "Severe", "Proliferative")


@dataclass(frozen=True)
class Predictor:
    model: DRDualHeadModel
    cfg: Config
    threshold: float
    run_id: str
    label: str


@dataclass(frozen=True)
class Prediction:
    dr_present: bool
    detection_score: float
    threshold: float
    grade: int
    grade_name: str
    probabilities: tuple[float, ...]
    processed_rgb: np.ndarray
    saliency: np.ndarray | None
    looks_like_fundus: bool
    fundus_note: str


def load_predictor(run_dir: str | Path) -> Predictor:
    """Load a trained run once; the checkpoint supplies every weight, so nothing is downloaded."""
    run_dir = Path(run_dir)
    if not (run_dir / "best.pt").is_file():
        raise FileNotFoundError(f"no trained model at {run_dir / 'best.pt'}")
    cfg = run_config(run_dir)
    model = load_best_model(run_dir, cfg, torch.device("cpu"))
    threshold = json.loads((run_dir / "threshold.json").read_text(encoding="utf-8"))["threshold"]
    return Predictor(model, cfg, float(threshold), run_dir.name, method_label(cfg))


def fundus_check(image_rgb: np.ndarray) -> tuple[bool, str]:
    """Rough test that a picture is a colour fundus photo: red-dominant retina on a dark frame.

    The model returns a grade for any picture, so the app warns when this test fails. It is a
    plausibility check, not a guarantee.
    """
    small = cv2.resize(image_rgb, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
    gray = small.mean(axis=2)
    field = gray > 25
    if field.mean() < 0.15:
        return False, "the picture is almost entirely dark"
    red, green, blue = (float(small[..., c][field].mean()) for c in range(3))
    if not (red > green > blue and red > 1.25 * blue):
        return False, "the colours are not those of a fundus photo (red-orange retina expected)"
    corners = [gray[:10, :10], gray[:10, -10:], gray[-10:, :10], gray[-10:, -10:]]
    if sum(float(c.mean()) < 40 for c in corners) < 2:
        return False, "no dark frame around a round field of view was found"
    return True, ""


@torch.no_grad()
def _forward(predictor: Predictor, tensor: torch.Tensor) -> tuple[float, np.ndarray]:
    out = predictor.model(tensor[None])
    score = float(torch.sigmoid(out["det_logit"].float())[0])
    return score, torch.softmax(out["grade_logits"].float(), dim=1)[0].numpy()


def predict_image(predictor: Predictor, image_rgb: np.ndarray, explain: bool = True) -> Prediction:
    """Diagnose one RGB uint8 image of any size."""
    image_rgb = np.ascontiguousarray(image_rgb[..., :3]).astype(np.uint8)
    cfg = predictor.cfg.data
    processed_bgr, _, _ = preprocess_arrays(
        image_rgb[..., ::-1], {}, cfg.image_size, None, cfg.crop_threshold,
        cfg.clahe.clip_limit, cfg.clahe.tile_grid,
    )
    processed_rgb = np.ascontiguousarray(processed_bgr[..., ::-1])
    tensor = eval_transform(cfg.image_size)(image=processed_rgb)["image"]
    score, probs = _forward(predictor, tensor)
    grade = int(probs.argmax())
    saliency = GradCAMPlusPlus(predictor.model).explain(tensor) if explain else None
    ok, note = fundus_check(image_rgb)
    return Prediction(
        dr_present=score >= predictor.threshold, detection_score=score, threshold=predictor.threshold,
        grade=grade, grade_name=GRADE_NAMES[grade], probabilities=tuple(float(p) for p in probs),
        processed_rgb=processed_rgb, saliency=saliency, looks_like_fundus=ok, fundus_note=note,
    )


def overlay(processed_rgb: np.ndarray, saliency: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """Heat-map of the saliency blended over the image; transparent where saliency is low."""
    heat = cv2.applyColorMap((np.clip(saliency, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_rgb = heat[..., ::-1].astype(np.float32)
    alpha = (np.clip(saliency, 0, 1) * strength)[..., None]
    return (processed_rgb.astype(np.float32) * (1 - alpha) + heat_rgb * alpha).round().astype(np.uint8)
