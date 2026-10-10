"""APTOS 2019 Blindness Detection: train.csv + train_images/{id_code}.png."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from xaidr.data.adapters.base import DataMissingError, read_label_csv, require_columns, require_dir


class Aptos2019Adapter:
    name = "aptos2019"

    def __init__(self, positive_min_grade: int = 1) -> None:
        self.positive_min_grade = positive_min_grade

    def build_manifest(self, raw_dir: Path) -> pd.DataFrame:
        # Only train.csv carries labels; sample_submission.csv is placeholders (F16).
        csv_path = raw_dir / "train.csv"
        if not csv_path.is_file():
            raise DataMissingError(f"APTOS 2019 label file not found: {csv_path} (see docs/DATA_ACCESS.md)")
        image_dir = require_dir(raw_dir / "train_images", "APTOS 2019 image folder")
        labels = read_label_csv(csv_path)
        require_columns(labels, ["id_code", "diagnosis"], csv_path)
        grade = labels["diagnosis"].astype(int)
        paths = [image_dir / f"{code}.png" for code in labels["id_code"]]
        missing = [p.name for p in paths if not p.is_file()]
        if missing:
            raise DataMissingError(
                f"{len(missing)} APTOS 2019 image(s) listed in train.csv are missing, e.g. {missing[:3]}"
            )
        return pd.DataFrame(
            {
                "dataset": self.name,
                "image_id": labels["id_code"],
                "source_path": [p.as_posix() for p in paths],
                "grade": grade,
                "dr_label": (grade >= self.positive_min_grade).astype(int),
                "label_space": "icdr5",
                "has_masks": False,
            }
        )
