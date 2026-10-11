from __future__ import annotations

import numpy as np
import pytest

from xaidr.xai.fidelity import (
    chance_baseline,
    dice,
    fidelity_for_image,
    fov_mask,
    holm_adjust,
    iou,
    lesion_recall,
    max_attainable,
    pointing_hit,
    topk_binarise,
)
from xaidr.xai.stability import explanation_agreement

SIZE = 100


def square(y0: int, x0: int, side: int) -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), bool)
    mask[y0:y0 + side, x0:x0 + side] = True
    return mask


def test_iou_and_dice_known_answers() -> None:
    a, b = square(10, 10, 20), square(10, 20, 20)  # half of each overlaps
    assert iou(a, a) == 1.0 and dice(a, a) == 1.0
    assert iou(a, square(60, 60, 20)) == 0.0 and dice(a, square(60, 60, 20)) == 0.0
    assert iou(a, b) == pytest.approx(200 / 600) and dice(a, b) == pytest.approx(0.5)
    empty = np.zeros((SIZE, SIZE), bool)
    assert np.isnan(iou(empty, empty)) and np.isnan(dice(empty, empty))
    assert iou(a, empty) == 0.0 and np.isnan(lesion_recall(a, empty))
    assert lesion_recall(a, b) == pytest.approx(0.5)


@pytest.mark.parametrize("k", [5, 10, 20, 0.3])
def test_topk_selects_k_percent_of_the_field_of_view(k: float) -> None:
    rng = np.random.default_rng(0)
    saliency = rng.random((SIZE, SIZE)).astype(np.float32)
    fov = np.zeros((SIZE, SIZE), bool)
    fov[:, :60] = True
    saliency[:, 60:] = 5.0  # the largest values are outside the field of view
    binary = topk_binarise(saliency, k, fov)
    assert abs(int(binary.sum()) - k / 100 * 6000) <= 1
    assert not binary[:, 60:].any()
    assert saliency[binary].min() >= saliency[fov & ~binary].max()


def test_topk_ties_and_whole_image() -> None:
    flat = np.zeros((SIZE, SIZE), np.float32)
    binary = topk_binarise(flat, 10)
    assert binary.sum() == 1000 and binary.ravel()[:1000].all()  # first occurrence wins
    assert topk_binarise(flat, 10, np.zeros((SIZE, SIZE), bool)).sum() == 0


def test_pointing_game() -> None:
    mask = square(40, 40, 10)
    saliency = np.zeros((SIZE, SIZE), np.float32)
    saliency[49, 49] = 1.0  # last pixel inside the mask
    assert pointing_hit(saliency, mask)
    saliency[:] = 0
    saliency[50, 49] = 1.0  # first pixel below the mask
    assert not pointing_hit(saliency, mask)
    saliency[45, 45] = 1.0  # tie: the first occurrence (inside) decides
    assert pointing_hit(saliency, mask)
    # a maximum outside the field of view is ignored
    saliency[:] = 0
    saliency[0, 0], saliency[45, 45] = 9.0, 1.0
    fov = np.ones((SIZE, SIZE), bool)
    fov[:5] = False
    assert not pointing_hit(saliency, mask) and pointing_hit(saliency, mask, fov)


def test_fov_mask_excludes_black_border() -> None:
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    img[20:80, 20:80] = 120
    fov = fov_mask(img, dilate_px=2)
    assert fov[20:80, 20:80].all() and fov[18, 50] and not fov[17, 50] and not fov[0, 0]
    assert fov_mask(img, dilate_px=0).sum() == 3600


def test_max_attainable() -> None:
    assert max_attainable(200, 1000) == {"max_iou": 0.2, "max_dice": pytest.approx(400 / 1200)}
    assert max_attainable(1000, 1000) == {"max_iou": 1.0, "max_dice": 1.0}
    assert max_attainable(2000, 1000)["max_iou"] == 0.5
    assert np.isnan(max_attainable(0, 1000)["max_iou"])


def test_chance_baseline_matches_random_maps() -> None:
    fov = np.ones((SIZE, SIZE), bool)
    fov[:, 80:] = False
    mask = square(30, 30, 20)
    chance = chance_baseline(mask, fov, 10, n_mc=4000)
    assert chance["chance_pointing"] == pytest.approx(400 / 8000)
    assert chance["chance_recall"] == pytest.approx(0.10)
    rng = np.random.default_rng(1)
    simulated = [topk_binarise(rng.random((SIZE, SIZE)), 10, fov) for _ in range(400)]
    assert chance["chance_iou"] == pytest.approx(np.mean([iou(s, mask) for s in simulated]), abs=0.003)
    assert chance["chance_dice"] == pytest.approx(np.mean([dice(s, mask) for s in simulated]), abs=0.005)
    assert chance == chance_baseline(mask, fov, 10, n_mc=4000)  # seeded
    assert np.isnan(chance_baseline(np.zeros_like(mask), fov, 10)["chance_iou"])


def test_fidelity_for_image() -> None:
    fov = np.ones((SIZE, SIZE), bool)
    fov[:, 90:] = False
    masks = {
        "MA": square(10, 10, 5), "HE": square(40, 40, 20), "EX": np.zeros((SIZE, SIZE), bool),
        "OD": square(60, 5, 20),
    }
    saliency = np.zeros((SIZE, SIZE), np.float32)
    saliency[40:60, 40:60] = 1.0  # exactly the haemorrhage
    table = fidelity_for_image(saliency, masks, fov, k_percents=(10, 5, 20))
    assert set(table["lesion"]) == {"MA", "HE", "any"}  # empty EX skipped, OD is not a lesion
    assert len(table) == 9 and list(table["k_percent"].unique()) == [10, 5, 20]
    he = table[(table["lesion"] == "HE") & (table["k_percent"] == 10)].iloc[0]
    assert he["pointing"] == 1.0 and he["recall"] == 1.0
    assert he["iou"] == pytest.approx(400 / 900) == he["max_iou"]  # 900 pixels selected, 400 in the lesion
    assert he["lesion_fraction"] == pytest.approx(400 / 9000) and he["chance_recall"] == pytest.approx(0.1)
    ma = table[(table["lesion"] == "MA") & (table["k_percent"] == 10)].iloc[0]
    assert ma["iou"] == 0.0 and ma["pointing"] == 0.0 and ma["max_iou"] == pytest.approx(25 / 900)
    assert (table["iou"] <= table["max_iou"] + 1e-12).all()
    assert fidelity_for_image(saliency, {"EX": masks["EX"]}, fov).empty


def test_shifted_mask_lowers_the_score() -> None:
    fov = np.ones((SIZE, SIZE), bool)
    saliency = np.zeros((SIZE, SIZE), np.float32)
    saliency[40:60, 40:60] = 1.0
    good = fidelity_for_image(saliency, {"HE": square(40, 40, 20)}, fov, (5,))
    shifted = fidelity_for_image(saliency, {"HE": square(40, 70, 20)}, fov, (5,))
    assert good["iou"].iloc[0] > 0.7 and shifted["iou"].iloc[0] == 0.0


def test_holm_adjust() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert holm_adjust([]).size == 0


def test_explanation_agreement() -> None:
    rng = np.random.default_rng(0)
    a = rng.random((SIZE, SIZE))
    fov = np.ones((SIZE, SIZE), bool)
    fov[:10] = False
    assert explanation_agreement(a, a, fov) == pytest.approx(1.0)
    assert explanation_agreement(a, a ** 3 + 2, fov) == pytest.approx(1.0)  # rank-based
    assert explanation_agreement(a, -a, fov) == pytest.approx(-1.0)
    assert abs(explanation_agreement(a, rng.random((SIZE, SIZE)), fov)) < 0.05
    b = a.copy()
    b[:10] = 99.0  # differences outside the field of view do not count
    assert explanation_agreement(a, b, fov) == pytest.approx(1.0)
    assert np.isnan(explanation_agreement(a, np.ones_like(a), fov))
    with pytest.raises(ValueError, match="shapes"):
        explanation_agreement(a, a[:50], fov)
