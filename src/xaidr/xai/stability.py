"""Between-seed explanation agreement (DESIGN.md Section 10.5, Proposal A).

Approved by the Researcher (ISSUE-013) on two conditions: three seeds are trained, and the
agreement is reported for the in-domain test set and both external sets alike. It measures how
much two models agree on where they look; it says nothing about whether either is right.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def explanation_agreement(map_a: np.ndarray, map_b: np.ndarray, fov: np.ndarray) -> float:
    """Spearman rank correlation of two saliency maps over field-of-view pixels.

    NaN when either map is constant inside the field of view.
    """
    if map_a.shape != map_b.shape or map_a.shape != fov.shape:
        raise ValueError(f"shapes differ: {map_a.shape}, {map_b.shape}, {fov.shape}")
    inside = fov.astype(bool)
    a, b = map_a[inside], map_b[inside]
    if a.size < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)
