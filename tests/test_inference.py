from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from conftest import REPO_ROOT, fundus
from xaidr.config import load_config
from xaidr.data.preprocess import read_image
from xaidr.inference import (
    GRADE_NAMES,
    Predictor,
    fundus_check,
    load_predictor,
    overlay,
    predict_image,
)
from xaidr.models.dual_head import DRDualHeadModel

SIZE = 224
REAL_RUN = REPO_ROOT / "checkpoints" / "20261011-222646_convnext_tiny_oversample_seed42"


def fundus_like(seed: int = 0, h: int = 300, w: int = 400) -> np.ndarray:
    """RGB test picture with a red-orange disc on a black frame."""
    bgr = fundus(seed, h, w).astype(np.float32)
    disc = bgr.sum(axis=2, keepdims=True) > 0
    tint = np.array([0.25, 0.5, 1.0], np.float32)  # B, G, R
    return np.where(disc, bgr * tint, 0).astype(np.uint8)[..., ::-1].copy()


@pytest.fixture(scope="module")
def predictor(tmp_path_factory: pytest.TempPathFactory) -> Predictor:
    run_dir = tmp_path_factory.mktemp("ckpt") / "run"
    run_dir.mkdir()
    cfg = load_config(
        REPO_ROOT / "configs" / "default.yaml",
        [f"data.image_size={SIZE}", "model.backbone=convnext_tiny", "model.pretrained=false"],
    )
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict()))
    (run_dir / "threshold.json").write_text(json.dumps({"threshold": 0.5}))
    torch.manual_seed(0)
    torch.save({"model": DRDualHeadModel(cfg.model, SIZE).state_dict()}, run_dir / "best.pt")
    return load_predictor(run_dir)


def test_prediction_fields(predictor: Predictor) -> None:
    result = predict_image(predictor, fundus_like())
    assert isinstance(result.dr_present, bool) and 0.0 <= result.detection_score <= 1.0
    assert result.dr_present == (result.detection_score >= result.threshold == 0.5)
    assert result.grade in range(5) and result.grade_name == GRADE_NAMES[result.grade]
    assert len(result.probabilities) == 5 and sum(result.probabilities) == pytest.approx(1.0, abs=1e-5)
    assert result.grade == int(np.argmax(result.probabilities))
    assert result.processed_rgb.shape == (SIZE, SIZE, 3) and result.processed_rgb.dtype == np.uint8
    assert result.saliency is not None and result.saliency.shape == (SIZE, SIZE)
    assert 0.0 <= result.saliency.min() and result.saliency.max() <= 1.0
    assert result.looks_like_fundus and result.fundus_note == ""
    blended = overlay(result.processed_rgb, result.saliency)
    assert blended.shape == (SIZE, SIZE, 3) and blended.dtype == np.uint8


def test_prediction_is_repeatable_and_accepts_other_shapes(predictor: Predictor) -> None:
    image = fundus_like(1)
    first, second = predict_image(predictor, image), predict_image(predictor, image)
    assert first.probabilities == second.probabilities and first.detection_score == second.detection_score
    assert np.array_equal(first.saliency, second.saliency)
    no_map = predict_image(predictor, image, explain=False)
    assert no_map.saliency is None and no_map.probabilities == first.probabilities
    rgba = np.dstack([image, np.full(image.shape[:2], 255, np.uint8)])  # PNG with an alpha channel
    assert predict_image(predictor, rgba, explain=False).probabilities == first.probabilities
    tall = predict_image(predictor, fundus_like(2, 500, 320), explain=False)
    assert tall.processed_rgb.shape == (SIZE, SIZE, 3)


@pytest.mark.parametrize(
    "picture, reason",
    [
        (np.zeros((200, 300, 3), np.uint8), "dark"),
        (np.full((200, 300, 3), 240, np.uint8), "colours"),
        (np.random.default_rng(0).integers(0, 255, (200, 300, 3), dtype=np.uint8), "colours"),
        (np.full((200, 300, 3), (205, 150, 125), np.uint8), "dark frame"),
    ],
)
def test_non_fundus_pictures_are_flagged(predictor: Predictor, picture: np.ndarray, reason: str) -> None:
    ok, note = fundus_check(picture)
    assert not ok and reason in note
    result = predict_image(predictor, picture, explain=False)
    assert not result.looks_like_fundus and result.grade in range(5)  # still answers, but warns


def test_missing_run_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="best.pt"):
        load_predictor(tmp_path / "nothing")


needs_trained_run = pytest.mark.skipif(
    not (REAL_RUN / "predictions_test.csv").is_file(), reason="trained run not on this machine"
)


@needs_trained_run
def test_demo_app_renders_a_sample_without_errors() -> None:
    """The Streamlit page itself: start, pick a sample, get a result."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(REPO_ROOT / "app" / "demo_app.py"), default_timeout=120).run()
    assert not app.exception and "Diabetic retinopathy" in app.title[0].value
    assert any("Upload a fundus photograph or pick a sample" in box.value for box in app.info)
    app.selectbox[0].select("Sample 5").run()
    assert not app.exception
    text = " ".join(m.value for m in app.markdown)
    assert "Severity grade" in text and "Reference grade from the dataset: **2 · Moderate**" in text
    assert len(app.error) + len(app.success) == 1  # exactly one detection verdict


@needs_trained_run
def test_app_path_reproduces_the_evaluated_predictions() -> None:
    """A raw APTOS test image through the app's function equals the stored test prediction."""
    real = load_predictor(REAL_RUN)
    stored = pd.read_csv(REAL_RUN / "predictions_test.csv", dtype={"image_id": str}).head(4)
    raw = REPO_ROOT / "data" / "raw" / "aptos2019" / "train_images"
    for _, row in stored.iterrows():
        result = predict_image(real, read_image(raw / f"{row['image_id']}.png")[..., ::-1], explain=False)
        assert result.detection_score == pytest.approx(row["det_score"], abs=1e-5)
        expected = row[[f"p{g}" for g in range(5)]].to_numpy(dtype=float)
        assert np.allclose(result.probabilities, expected, atol=1e-5)
        assert result.looks_like_fundus
