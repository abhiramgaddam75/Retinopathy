"""Synthetic raw-data trees that follow the layout in DESIGN.md Section 8.2."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def fundus(seed: int, h: int = 120, w: int = 160) -> np.ndarray:
    """Black frame with a textured disc, like a fundus photo with its border."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    disc = np.zeros((h, w), np.uint8)
    cv2.circle(disc, (w // 2, h // 2), min(h, w) // 2 - 6, 255, -1)
    texture = rng.integers(60, 200, size=(h, w, 3), dtype=np.uint8)
    img[disc > 0] = texture[disc > 0]
    return img


def write_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), img)


@pytest.fixture
def aptos_raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw" / "aptos2019"
    grades = [0] * 40 + [1] * 20 + [2] * 20 + [3] * 10 + [4] * 10
    ids = [f"img{i:03d}" for i in range(len(grades))]
    raw.mkdir(parents=True)
    pd.DataFrame({"id_code": ids, "diagnosis": grades}).to_csv(raw / "train.csv", index=False)
    for i, code in enumerate(ids):
        write_image(raw / "train_images" / f"{code}.png", fundus(i, 48, 64))
    # placeholder labels that must never be read as ground truth (F16)
    pd.DataFrame({"id_code": ["t1", "t2"], "diagnosis": [0, 0]}).to_csv(
        raw / "sample_submission.csv", index=False
    )
    return raw


@pytest.fixture
def idrid_raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw" / "idrid"
    grading = {"IDRiD_001": 0, "IDRiD_002": 2, "IDRiD_003": 4, "IDRiD_004": 1}
    for i, name in enumerate(grading):
        write_image(raw / "grading" / "images" / f"{name}.jpg", fundus(100 + i))
    # header spacing mirrors the published CSV, which pads some column names
    pd.DataFrame(
        {"Image name": list(grading), "Retinopathy grade": list(grading.values()),
         "Risk of macular edema ": [0, 1, 2, 0]}
    ).to_csv(raw / "grading" / "labels.csv", index=False)

    seg_images = raw / "segmentation" / "images"
    # IDRiD_01 is a byte-identical copy of grading image IDRiD_003
    seg_images.mkdir(parents=True)
    (seg_images / "IDRiD_01.jpg").write_bytes((raw / "grading" / "images" / "IDRiD_003.jpg").read_bytes())
    write_image(seg_images / "IDRiD_02.jpg", fundus(201))
    mask = np.zeros((120, 160), np.uint8)
    mask[40:60, 50:80] = 255
    masks = raw / "segmentation" / "masks"
    for lesion, stems in {"MA": ["IDRiD_01", "IDRiD_02"], "HE": ["IDRiD_01"], "EX": ["IDRiD_02"]}.items():
        for stem in stems:
            write_image(masks / lesion / f"{stem}_{lesion}.tif", mask)
    return raw


@pytest.fixture
def messidor_raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw" / "messidor2"
    names = ["a.png", "b.png", "c.png", "d.png"]
    for i, name in enumerate(names[:3]):  # d.png has a label but no image
        write_image(raw / "images" / name, fundus(300 + i))
    pd.DataFrame(
        {"image_id": names, "adjudicated_dr_grade": ["0", "3", "", "1"]}  # c.png is ungradable
    ).to_csv(raw / "labels.csv", index=False)
    return raw
