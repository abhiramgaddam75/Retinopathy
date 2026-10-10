from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import REPO_ROOT, write_image
from xaidr.data.adapters import IdridAdapter
from xaidr.data.manifest import normalise
from xaidr.data.preprocess import (
    PreprocessError,
    apply_clahe,
    build_processed,
    crop_fov,
    preprocess_arrays,
    preprocess_record,
    read_image,
    resize_mask,
)

SIZE = 256


def bbox(binary: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.flatnonzero(binary.any(axis=1)), np.flatnonzero(binary.any(axis=0))
    return rows[0], cols[0], rows[-1], cols[-1]


def bordered_image() -> tuple[np.ndarray, np.ndarray]:
    """A dim disc off-centre in a black frame, with one bright square and its mask."""
    img = np.zeros((300, 420, 3), np.uint8)
    cv2.circle(img, (230, 160), 120, (40, 40, 40), -1)
    mask = np.zeros((300, 420), np.uint8)
    mask[120:160, 250:300] = 255
    img[mask > 0] = 255
    return img, mask


def test_crop_fov_returns_the_box() -> None:
    img, _ = bordered_image()
    cropped, box = crop_fov(img, threshold=10)
    assert box == (40, 110, 281, 351)
    assert cropped.shape[:2] == (241, 241)
    black = np.zeros((10, 12, 3), np.uint8)
    assert crop_fov(black)[1] == (0, 0, 10, 12)


def test_mask_and_image_stay_aligned() -> None:
    img, mask = bordered_image()
    out_img, out_masks, _ = preprocess_arrays(img, {"EX": mask}, SIZE)
    assert out_img.shape == (SIZE, SIZE, 3) and out_masks["EX"].shape == (SIZE, SIZE)
    assert set(np.unique(out_masks["EX"])) == {0, 1}
    gray = cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY).astype(int)
    square = gray > (gray.max() + np.median(gray[gray > 0])) / 2
    assert np.abs(np.subtract(bbox(square), bbox(out_masks["EX"]))).max() <= 1


def test_clahe_is_never_applied_to_masks() -> None:
    img, mask = bordered_image()
    plain = resize_mask(mask[40:281, 110:351], SIZE)
    _, out_masks, _ = preprocess_arrays(img, {"EX": mask}, SIZE)
    assert np.array_equal(out_masks["EX"], plain)
    assert not np.array_equal(apply_clahe(img), img)


def test_mask_size_mismatch_is_an_error() -> None:
    img, mask = bordered_image()
    with pytest.raises(PreprocessError, match="EX"):
        preprocess_arrays(img, {"EX": mask[:100]}, SIZE)


def test_build_processed_records_the_transform(idrid_raw: Path, tmp_path: Path) -> None:
    manifest = normalise(IdridAdapter().build_manifest(idrid_raw))
    out = build_processed(manifest, tmp_path / "processed", SIZE, num_workers=2)
    assert (out[["orig_h", "orig_w"]] == [120, 160]).all().all()
    assert out[["crop_y0", "crop_x0", "crop_y1", "crop_x1"]].notna().all().all()
    for path in out["processed_path"]:
        assert read_image(path).shape == (SIZE, SIZE, 3)
    seg = out[out["image_id"] == "seg_IDRiD_01"].iloc[0]
    assert (tmp_path / "processed" / "idrid" / "masks" / "MA" / "seg_IDRiD_01.png").is_file()
    # the stored box reproduces the same masks later (the path LesionDataset uses)
    _, masks = preprocess_record(seg.to_dict(), SIZE)
    assert set(masks) == {"MA", "HE"} and masks["MA"].sum() > 0

    # idempotent: an existing file is not rewritten unless forced
    target = Path(seg["processed_path"])
    write_image(target, np.full((SIZE, SIZE, 3), 7, np.uint8))
    again = build_processed(out, tmp_path / "processed", SIZE, num_workers=2)
    assert (read_image(target) == 7).all() and again.equals(out)
    build_processed(out, tmp_path / "processed", SIZE, force=True, num_workers=2)
    assert not (read_image(target) == 7).all()


def test_build_processed_records_hashes_and_parameters(idrid_raw: Path, tmp_path: Path) -> None:
    manifest = normalise(IdridAdapter().build_manifest(idrid_raw))
    out = build_processed(manifest, tmp_path / "processed", SIZE, num_workers=2)
    by_id = out.set_index("image_id")["content_hash"]
    assert by_id.str.len().eq(64).all()
    assert by_id["seg_IDRiD_01"] == by_id["IDRiD_003"] and by_id.nunique() == len(by_id) - 1
    assert by_id["IDRiD_001"] == hashlib.sha256(
        (idrid_raw / "grading" / "images" / "IDRiD_001.jpg").read_bytes()
    ).hexdigest()

    # the same data with other preprocessing parameters is refused unless forced
    target = Path(out["processed_path"].iloc[0])
    before = target.read_bytes()
    for change in ({"clip_limit": 8.0}, {"crop_threshold": 30}, {"tile_grid": (4, 4)}):
        with pytest.raises(PreprocessError, match="--force"):
            build_processed(out, tmp_path / "processed", SIZE, num_workers=2, **change)
    with pytest.raises(PreprocessError, match="--force"):
        build_processed(out, tmp_path / "processed", SIZE + 32, num_workers=2)
    assert target.read_bytes() == before
    build_processed(out, tmp_path / "processed", SIZE, num_workers=2, clip_limit=8.0, force=True)
    assert target.read_bytes() != before
    build_processed(out, tmp_path / "processed", SIZE, num_workers=2, clip_limit=8.0)  # now the record


def test_supplied_sample_photos_preprocess() -> None:
    photos = sorted((REPO_ROOT / "tests" / "fixtures" / "sample_fundus").glob("*.jpg"))
    if not photos:
        pytest.skip("sample photos are not committed (licence check pending, F17)")
    for path in photos:
        source = read_image(path)
        out, _, (y0, x0, y1, x1) = preprocess_arrays(source, {}, 512)
        assert out.shape == (512, 512, 3) and out.dtype == np.uint8
        # every photo has a black border, so the crop must remove columns and keep the retina
        assert (x1 - x0) < source.shape[1] and (y1 - y0) > source.shape[0] * 0.5
