"""Messidor-2: images joined by filename to a label file whose scheme is declared in config (F8)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from xaidr.config import Messidor2Cfg
from xaidr.data.adapters.base import (
    DataMissingError,
    index_files,
    read_label_csv,
    require_columns,
    require_dir,
)

log = logging.getLogger(__name__)


def _int_keys(mapping: dict, key: str) -> dict[int, int | None]:
    try:
        return {int(k): (None if v is None else int(v)) for k, v in mapping.items()}
    except (TypeError, ValueError) as exc:
        raise DataMissingError(f"'data.messidor2.{key}' must map integer grades to integers: {exc}") from exc


class Messidor2Adapter:
    name = "messidor2"

    def __init__(self, cfg: Messidor2Cfg) -> None:
        self.cfg = cfg
        self.dropped_unlabelled = 0
        self.dropped_no_image = 0

    def build_manifest(self, raw_dir: Path) -> pd.DataFrame:
        cfg = self.cfg
        detection_map = _int_keys(cfg.detection_map, "detection_map")
        grade_map = _int_keys(cfg.grade_map, "grade_map")
        if not detection_map:
            raise DataMissingError(
                "'data.messidor2.detection_map' is empty. Verify the downloaded label file (F8), "
                "then declare the native-grade -> 0/1 mapping in the config; do not guess."
            )
        label_path = raw_dir / cfg.label_file
        if not label_path.is_file():
            raise DataMissingError(f"Messidor-2 label file not found: {label_path} (see docs/DATA_ACCESS.md)")
        images = index_files(require_dir(raw_dir / cfg.images, "Messidor-2 image folder"))
        labels = read_label_csv(label_path)
        require_columns(labels, [cfg.id_column, cfg.grade_column], label_path)

        native = pd.to_numeric(labels[cfg.grade_column].replace("", pd.NA), errors="coerce")
        self.dropped_unlabelled = int(native.isna().sum())
        labels = labels[native.notna()].assign(_native=native.dropna().astype(int))
        unmapped = sorted(set(labels["_native"]) - set(detection_map))
        if unmapped:
            raise DataMissingError(
                f"Messidor-2 grade value(s) {unmapped} are not in 'data.messidor2.detection_map'"
            )
        stems = labels[cfg.id_column].str.strip().map(lambda name: Path(name).stem)
        has_image = stems.str.lower().isin(images)
        self.dropped_no_image = int((~has_image).sum())
        labels, stems = labels[has_image], stems[has_image]
        log.info(
            "Messidor-2: kept %d rows; dropped %d without a label and %d without an image file",
            len(labels), self.dropped_unlabelled, self.dropped_no_image,
        )
        grade = labels["_native"].map(lambda g: grade_map.get(g)).astype("Int64")
        label_space = "messidor2:" + json.dumps(
            {
                "grade_column": cfg.grade_column,
                "grade_map": grade_map,
                "detection_map": detection_map,
                "detection_definition": cfg.detection_definition,
            },
            sort_keys=True,
        )
        return pd.DataFrame(
            {
                "dataset": self.name,
                "image_id": stems.to_numpy(),
                "source_path": [images[s.lower()].as_posix() for s in stems],
                "grade": grade.to_numpy(),
                "dr_label": labels["_native"].map(detection_map).astype(int).to_numpy(),
                "label_space": label_space,
                "has_masks": False,
            }
        )
