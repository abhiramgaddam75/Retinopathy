"""Adapter protocol and helpers shared by the dataset adapters."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


class DataMissingError(FileNotFoundError):
    """A raw dataset file or column is missing or differs from the documented layout."""


class DatasetAdapter(Protocol):
    name: str

    def build_manifest(self, raw_dir: Path) -> pd.DataFrame: ...


def require_dir(path: Path, what: str) -> Path:
    if not path.is_dir():
        raise DataMissingError(f"{what} not found: {path} (see docs/DATA_ACCESS.md)")
    return path


def require_columns(df: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataMissingError(
            f"{source} is missing column(s) {missing}; found {list(df.columns)}. "
            "Update the config or ask the team; do not guess."
        )


def index_files(root: Path, extensions: tuple[str, ...] = IMAGE_EXTENSIONS) -> dict[str, Path]:
    """Map lower-cased file stem -> path for every image below ``root``."""
    index: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            key = path.stem.lower()
            if key in index:
                raise DataMissingError(f"two files share the stem '{path.stem}': {index[key]} and {path}")
            index[key] = path
    return index


def read_label_csv(path: Path) -> pd.DataFrame:
    """Read a label CSV, stripping header whitespace and dropping empty columns."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]
