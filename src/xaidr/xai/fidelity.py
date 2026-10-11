"""Agreement between a saliency map and expert lesion masks (DESIGN.md Sections 7.21 and 10.3).

Every score is reported with three reference values, because a raw overlap number cannot be
read on its own: the value expected from a uniformly random map (chance), the largest value
any map could reach for that lesion at that k (maximum), and the lesion's share of the field
of view. Lesion recall at k is included as a measure that does not shrink with lesion size.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np
import pandas as pd

FIDELITY_LESIONS = ("MA", "HE", "EX", "SE")  # the optic disc mask is anatomy, not a lesion
ANY_LESION = "any"


def fov_mask(img_512: np.ndarray, threshold: int = 10, dilate_px: int = 2) -> np.ndarray:
    """Field of view of a processed image: non-black pixels, dilated by ``dilate_px``."""
    gray = img_512 if img_512.ndim == 2 else img_512.max(axis=2)
    inside = (gray > threshold).astype(np.uint8)
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        inside = cv2.dilate(inside, kernel)
    return inside.astype(bool)


def n_selected(fov: np.ndarray, k_percent: float) -> int:
    return int(round(k_percent / 100.0 * int(fov.sum())))


def topk_binarise(saliency: np.ndarray, k_percent: float, fov: np.ndarray | None = None) -> np.ndarray:
    """True for the top k% of field-of-view pixels; ties are broken by first occurrence."""
    fov = np.ones(saliency.shape, bool) if fov is None else fov.astype(bool)
    out = np.zeros(saliency.shape, bool)
    count = n_selected(fov, k_percent)
    if count == 0:
        return out
    values = np.where(fov, saliency, -np.inf).ravel()
    out.ravel()[np.argsort(-values, kind="stable")[:count]] = True
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    union = int((a | b).sum())
    return float("nan") if union == 0 else int((a & b).sum()) / union


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    total = int(a.sum()) + int(b.sum())
    return float("nan") if total == 0 else 2 * int((a & b).sum()) / total


def lesion_recall(binary: np.ndarray, mask: np.ndarray) -> float:
    """Share of lesion pixels inside the selected region; NaN for an empty mask."""
    mask = mask.astype(bool)
    area = int(mask.sum())
    return float("nan") if area == 0 else int((binary.astype(bool) & mask).sum()) / area


def pointing_hit(saliency: np.ndarray, mask: np.ndarray, fov: np.ndarray | None = None) -> bool:
    """Whether the most salient field-of-view pixel lies in the mask (first occurrence on ties)."""
    values = saliency if fov is None else np.where(fov.astype(bool), saliency, -np.inf)
    return bool(mask.astype(bool).ravel()[int(np.argmax(values))])


def max_attainable(mask_area: int, selected: int) -> dict[str, float]:
    """Best IoU and Dice any map could reach when ``selected`` pixels are chosen."""
    if mask_area == 0 or selected == 0:
        return {"max_iou": float("nan"), "max_dice": float("nan")}
    small, large = min(mask_area, selected), max(mask_area, selected)
    return {"max_iou": small / large, "max_dice": 2 * small / (mask_area + selected)}


def chance_baseline(
    mask: np.ndarray, fov: np.ndarray, k_percent: float, n_mc: int = 1000, seed: int = 42
) -> dict[str, float]:
    """Expected scores of a uniformly random saliency map inside the field of view.

    Pointing and recall have closed forms. IoU and Dice are Monte Carlo means: the overlap of a
    random k% selection with the mask is hypergeometric, which is sampled directly rather than
    by drawing whole random maps (identical in distribution, far cheaper).
    """
    fov = fov.astype(bool)
    area = int((mask.astype(bool) & fov).sum())
    total = int(fov.sum())
    selected = n_selected(fov, k_percent)
    if area == 0 or total == 0 or selected == 0:
        nan = float("nan")
        return {"chance_iou": nan, "chance_dice": nan, "chance_pointing": nan, "chance_recall": nan}
    overlap = np.random.default_rng(seed).hypergeometric(area, total - area, selected, size=n_mc)
    return {
        "chance_iou": float(np.mean(overlap / (area + selected - overlap))),
        "chance_dice": float(np.mean(2 * overlap / (area + selected))),
        "chance_pointing": area / total,
        "chance_recall": selected / total,
    }


def fidelity_for_image(
    saliency: np.ndarray,
    masks: Mapping[str, np.ndarray],
    fov: np.ndarray,
    k_percents: Sequence[float] = (10, 5, 20),
    n_mc: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """One row per (lesion, k) for one image; lesions with an empty mask are skipped."""
    fov = fov.astype(bool)
    targets = {name: masks[name].astype(bool) & fov for name in FIDELITY_LESIONS if name in masks}
    if targets:
        targets[ANY_LESION] = np.logical_or.reduce(list(targets.values()))
    rows = []
    for k in k_percents:
        binary = topk_binarise(saliency, k, fov)
        selected = int(binary.sum())
        for lesion, mask in targets.items():
            area = int(mask.sum())
            if area == 0:
                continue
            rows.append(
                {
                    "lesion": lesion, "k_percent": k, "lesion_fraction": area / int(fov.sum()),
                    "iou": iou(binary, mask), "dice": dice(binary, mask),
                    "recall": lesion_recall(binary, mask),
                    "pointing": float(pointing_hit(saliency, mask, fov)),
                    **chance_baseline(mask, fov, k, n_mc, seed), **max_attainable(area, selected),
                }
            )
    return pd.DataFrame(rows)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values, in the order given."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p, kind="stable")
    scaled = (len(p) - np.arange(len(p))) * p[order]
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(np.maximum.accumulate(scaled), 1.0)
    return adjusted
