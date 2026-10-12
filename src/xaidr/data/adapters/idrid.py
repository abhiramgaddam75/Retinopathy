"""IDRiD: grading set (labels, no masks) and segmentation set (81 images with lesion masks)."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from xaidr.config import IdridCfg
from xaidr.data.adapters.base import (
    IMAGE_EXTENSIONS,
    DataMissingError,
    index_files,
    read_label_csv,
    require_columns,
    require_dir,
)
from xaidr.data.manifest import LESIONS
from xaidr.logging_utils import sha256_file

log = logging.getLogger(__name__)

# Segmentation images are numbered separately from the grading images (IDRiD_01 vs IDRiD_001),
# so their manifest ids carry this prefix to stay unique within the dataset.
SEG_PREFIX = "seg_"


def _set_tag(parts: tuple[str, ...]) -> str | None:
    """'train' or 'test' when a folder or file name says which IDRiD set it belongs to."""
    for part in parts:
        low = part.lower()
        if "train" in low:
            return "train"
        if "test" in low:
            return "test"
    return None


class IdridAdapter:
    name = "idrid"

    def __init__(self, cfg: IdridCfg | None = None, positive_min_grade: int = 1) -> None:
        self.cfg = cfg or IdridCfg()
        self.positive_min_grade = positive_min_grade
        self.overlap: pd.DataFrame = pd.DataFrame(columns=["seg_image_id", "grading_image_id"])

    def _grading_rows(self, raw_dir: Path) -> pd.DataFrame:
        """Grading rows. The training and testing sets both start at IDRiD_001, so a label CSV
        named for a set ("train"/"test") is matched only to the images in that set's folder and
        its ids get the set as a prefix (``train_IDRiD_001``)."""
        cfg = self.cfg
        image_dir = require_dir(raw_dir / cfg.grading_images, "IDRiD grading image folder")
        label_files = sorted(raw_dir.glob(cfg.grading_labels))
        if not label_files:
            raise DataMissingError(
                f"no IDRiD grading label CSV matches {raw_dir / cfg.grading_labels} (see docs/DATA_ACCESS.md)"
            )
        by_set: dict[str, dict[str, Path]] = {}
        for path in sorted(image_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                tag = _set_tag(path.relative_to(image_dir).parent.parts) or ""
                files = by_set.setdefault(tag, {})
                if path.stem.lower() in files:
                    raise DataMissingError(
                        f"two files share the stem '{path.stem}': {files[path.stem.lower()]} and {path}"
                    )
                files[path.stem.lower()] = path
        frames = []
        for path in label_files:
            frame = read_label_csv(path)
            require_columns(frame, [cfg.id_column, cfg.grade_column], path)
            frame = frame[[cfg.id_column, cfg.grade_column]]
            frame = frame[frame[cfg.id_column].str.strip() != ""].copy()
            frame["set"] = _set_tag((path.stem,)) or ""
            frames.append(frame)
        labels = pd.concat(frames, ignore_index=True)
        stems = labels[cfg.id_column].str.strip().map(lambda name: Path(name).stem)
        sources: list[str] = []
        missing: list[str] = []
        for tag, stem in zip(labels["set"], stems, strict=True):
            found = by_set.get(tag, {}).get(stem.lower())
            if found is None and tag:  # images not in a set folder: fall back to the flat folder
                found = by_set.get("", {}).get(stem.lower())
            if found is None:
                missing.append(f"{tag}/{stem}" if tag else stem)
            else:
                sources.append(found.as_posix())
        if missing:
            raise DataMissingError(
                f"{len(missing)} IDRiD grading image(s) named in the labels are missing, e.g. {missing[:3]}"
            )
        ids = [f"{tag}_{stem}" if tag else stem for tag, stem in zip(labels["set"], stems, strict=True)]
        if len(set(ids)) != len(ids):
            raise DataMissingError("IDRiD grading labels repeat an image id; check the label CSVs")
        grade = labels[cfg.grade_column].astype(int)
        return pd.DataFrame(
            {
                "dataset": self.name,
                "image_id": ids,
                "source_path": sources,
                "grade": grade.to_numpy(),
                "dr_label": (grade >= self.positive_min_grade).astype(int).to_numpy(),
                "label_space": "icdr5",
                "has_masks": False,
            }
        )

    def _segmentation_rows(self, raw_dir: Path) -> pd.DataFrame:
        cfg = self.cfg
        image_dir = require_dir(raw_dir / cfg.seg_images, "IDRiD segmentation image folder")
        mask_dir = require_dir(raw_dir / cfg.seg_masks, "IDRiD segmentation mask folder")
        masks = index_files(mask_dir)
        rows = []
        for stem, path in index_files(image_dir).items():
            row: dict[str, object] = {
                "dataset": self.name,
                "image_id": SEG_PREFIX + path.stem,
                "source_path": path.as_posix(),
                "label_space": "icdr5",
            }
            for lesion in LESIONS:
                mask = masks.get(f"{stem}_{lesion.lower()}")
                row[f"mask_{lesion}"] = mask.as_posix() if mask else ""
            row["has_masks"] = any(row[f"mask_{lesion}"] for lesion in LESIONS)
            rows.append(row)
        seg = pd.DataFrame(rows)
        counts = {lesion: int((seg[f"mask_{lesion}"] != "").sum()) for lesion in LESIONS}
        # Scientific acceptance check (Section 13): expected 81 / 80 / 81 / 40 for MA / HE / EX / SE.
        log.info("IDRiD segmentation images: %d; images with each lesion mask: %s", len(seg), counts)
        return seg

    def _attach_overlap(self, grading: pd.DataFrame, seg: pd.DataFrame) -> pd.DataFrame:
        """Byte-identical grading twins give a segmentation image its grade; every overlap is logged."""
        by_hash = {sha256_file(Path(p)): (i, g) for p, i, g in
                   zip(grading["source_path"], grading["image_id"], grading["grade"], strict=True)}
        seg = seg.copy()
        seg["grade"] = pd.array([pd.NA] * len(seg), dtype="Int64")
        seg["dr_label"] = pd.array([pd.NA] * len(seg), dtype="Int64")
        pairs = []
        for idx, path in zip(seg.index, seg["source_path"], strict=True):
            twin = by_hash.get(sha256_file(Path(path)))
            if twin is not None:
                pairs.append((seg.at[idx, "image_id"], twin[0]))
                seg.at[idx, "grade"] = int(twin[1])
                seg.at[idx, "dr_label"] = int(twin[1] >= self.positive_min_grade)
        self.overlap = pd.DataFrame(pairs, columns=["seg_image_id", "grading_image_id"])
        log.info(
            "IDRiD: %d of %d segmentation images are byte-identical to a grading image; "
            "all remain external (never used for training)", len(pairs), len(seg),
        )
        return seg

    def build_manifest(self, raw_dir: Path) -> pd.DataFrame:
        grading = self._grading_rows(raw_dir)
        seg = self._attach_overlap(grading, self._segmentation_rows(raw_dir))
        return pd.concat([grading, seg], ignore_index=True)
