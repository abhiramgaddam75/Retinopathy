"""Border crop, CLAHE and resize, applied identically to images and lesion masks.

The crop box found on the image is stored in the manifest and reused for every mask of
that image, so saliency maps and masks stay aligned at the model input size.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from xaidr.data.manifest import CROP_COLUMNS, LESIONS, normalise

log = logging.getLogger(__name__)

Box = tuple[int, int, int, int]  # y0, x0, y1, x1 in source pixels, end-exclusive
PARAMS_FILE = "preprocess.json"


class PreprocessError(RuntimeError):
    pass


def read_image(path: str | Path) -> np.ndarray:
    """Read a colour image as BGR uint8 (np.fromfile keeps non-ASCII Windows paths working)."""
    return _read_image_and_hash(path)[0]


def _read_image_and_hash(path: str | Path) -> tuple[np.ndarray, str]:
    """Decoded image and the SHA-256 of the file bytes (used to find duplicate images)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise PreprocessError(f"cannot read image: {path} ({exc})") from exc
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise PreprocessError(f"cannot read image: {path}")
    return img, hashlib.sha256(data.tobytes()).hexdigest()


def read_mask(path: str | Path) -> np.ndarray:
    """Read a lesion mask as binary uint8 {0, 1}; any non-zero pixel in any channel is lesion."""
    raw = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None:
        from PIL import Image

        raw = np.asarray(Image.open(path).convert("L"))
    if raw.ndim == 3:
        raw = raw[..., :3].max(axis=2)
    return (raw > 0).astype(np.uint8)


def crop_fov(img_bgr: np.ndarray, threshold: int = 10) -> tuple[np.ndarray, Box]:
    """Crop the black border around the field of view; returns the crop and its box."""
    gray = img_bgr if img_bgr.ndim == 2 else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    foreground = gray > threshold
    h, w = gray.shape
    if not foreground.any():
        return img_bgr, (0, 0, h, w)
    rows = np.flatnonzero(foreground.any(axis=1))
    cols = np.flatnonzero(foreground.any(axis=0))
    box = (int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1)
    return apply_crop(img_bgr, box), box


def apply_crop(arr: np.ndarray, box: Box) -> np.ndarray:
    y0, x0, y1, x1 = box
    return arr[y0:y1, x0:x1]


def apply_clahe(
    img_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """CLAHE on the L channel in LAB space."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tuple(tile_grid))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def resize_image(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    resized = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_NEAREST)
    return (resized > 0.5).astype(np.uint8)


def preprocess_arrays(
    img_bgr: np.ndarray,
    masks: Mapping[str, np.ndarray],
    size: int,
    box: Box | None = None,
    crop_threshold: int = 10,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> tuple[np.ndarray, dict[str, np.ndarray], Box]:
    """Crop, CLAHE (image only) and resize; masks get the same crop box and nearest resize."""
    if box is None:
        cropped, box = crop_fov(img_bgr, crop_threshold)
    else:
        cropped = apply_crop(img_bgr, box)
    out_img = resize_image(apply_clahe(cropped, clip_limit, tile_grid), size)
    out_masks = {}
    for name, mask in masks.items():
        if mask.shape[:2] != img_bgr.shape[:2]:
            raise PreprocessError(
                f"mask '{name}' is {mask.shape[:2]} but its image is {img_bgr.shape[:2]}"
            )
        out_masks[name] = resize_mask(apply_crop(mask, box), size)
    return out_img, out_masks, box


def _row_box(row: Mapping) -> Box | None:
    y0, x0, y1, x1 = (row.get(c) for c in CROP_COLUMNS)
    if y0 is None or x0 is None or y1 is None or x1 is None or pd.isna([y0, x0, y1, x1]).any():
        return None
    return int(y0), int(x0), int(y1), int(x1)


def _row_masks(row: Mapping) -> dict[str, np.ndarray]:
    return {name: read_mask(row[f"mask_{name}"]) for name in LESIONS if row.get(f"mask_{name}")}


def preprocess_record(
    row: Mapping,
    size: int,
    crop_threshold: int = 10,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Processed image and masks for one manifest row, using the stored crop box when present."""
    img, masks, _ = preprocess_arrays(
        read_image(row["source_path"]), _row_masks(row), size, _row_box(row),
        crop_threshold, clip_limit, tile_grid,
    )
    return img, masks


def build_processed(
    manifest_df: pd.DataFrame,
    out_dir: Path,
    size: int,
    crop_threshold: int = 10,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
    force: bool = False,
    num_workers: int = 4,
) -> pd.DataFrame:
    """Write processed images (and masks) and fill ``processed_path`` and the transform record.

    Idempotent: a row whose processed file already exists at the right size and whose crop
    box and content hash are already recorded is skipped unless ``force`` is set. The
    parameters used are stored beside the images; a rerun with different parameters stops
    unless ``force`` is set, so processed data can never silently disagree with the config.
    """
    df = normalise(manifest_df)
    out_dir = Path(out_dir)
    params = {
        "image_size": size, "crop_threshold": crop_threshold,
        "clip_limit": float(clip_limit), "tile_grid": [int(v) for v in tile_grid],
    }
    for dataset in sorted(set(df["dataset"])):
        record = out_dir / dataset / PARAMS_FILE
        if record.is_file() and not force:
            stored = json.loads(record.read_text(encoding="utf-8"))
            if stored != params:
                raise PreprocessError(
                    f"{out_dir / dataset} was built with {stored} but the config now asks for {params}. "
                    "Rerun prepare-data with --force to rebuild it."
                )

    def work(idx: int) -> dict:
        row = df.loc[idx].to_dict()
        target = out_dir / row["dataset"] / "images" / f"{row['image_id']}.png"
        update: dict[str, object] = {"processed_path": target.as_posix()}
        if not force and target.is_file() and _row_box(row) is not None and row["content_hash"]:
            existing = cv2.imdecode(np.fromfile(str(target), dtype=np.uint8), cv2.IMREAD_COLOR)
            if existing is not None and existing.shape[:2] == (size, size):
                return update
        source, update["content_hash"] = _read_image_and_hash(row["source_path"])
        img, masks, box = preprocess_arrays(
            source, _row_masks(row), size, None, crop_threshold, clip_limit, tile_grid
        )
        _write_png(target, img)
        for name, mask in masks.items():
            _write_png(out_dir / row["dataset"] / "masks" / name / f"{row['image_id']}.png", mask * 255)
        update.update(
            orig_h=source.shape[0], orig_w=source.shape[1], **dict(zip(CROP_COLUMNS, box, strict=True))
        )
        return update

    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        updates = list(tqdm(pool.map(work, df.index), total=len(df), desc="preprocess"))
    for idx, update in zip(df.index, updates, strict=True):
        for key, value in update.items():
            df.at[idx, key] = value
    for dataset in sorted(set(df["dataset"])):
        (out_dir / dataset).mkdir(parents=True, exist_ok=True)
        (out_dir / dataset / PARAMS_FILE).write_text(json.dumps(params, indent=2), encoding="utf-8")
    return normalise(df)


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise PreprocessError(f"cannot encode image for {path}")
    buf.tofile(str(path))
