"""Pack the code, the 512 px preprocessed APTOS images, the manifest and the locked split into
one zip for training on a GPU machine (see docs/TRAINING_ON_GPU.md).

Run from the repository root:  python tools/make_colab_bundle.py <output.zip>

The training loaders read only the processed images, so the raw download is not needed there.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", ".idea", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__",
             "data", "checkpoints", "xaidr.egg-info"}
DATA = ("data/manifests/manifest.csv", "data/splits", "data/processed/dr512")


def code_files() -> list[Path]:
    files = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if path.is_file() and not (set(relative.parts) & SKIP_DIRS) and path.suffix != ".zip":
            files.append(path)
    return files


def data_files() -> list[Path]:
    files: list[Path] = []
    for entry in DATA:
        path = ROOT / entry
        if not path.exists():
            raise SystemExit(f"missing {entry}: run prepare-data and make-splits first")
        files.extend([path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file()))
    return files


def main(output: str) -> None:
    code, data = code_files(), data_files()
    # PNG files are already compressed; storing them keeps packing and unpacking fast
    sys.path.insert(0, str(ROOT / "src"))
    from xaidr.logging_utils import git_commit

    with zipfile.ZipFile(output, "w", zipfile.ZIP_STORED) as bundle:
        bundle.writestr("COMMIT", git_commit(ROOT) + "\n")
        for path in code:
            bundle.write(path, path.relative_to(ROOT).as_posix(), zipfile.ZIP_DEFLATED)
        for path in data:
            bundle.write(path, path.relative_to(ROOT).as_posix())
    size = Path(output).stat().st_size / 1e6
    print(f"{output}: {len(code)} code files, {len(data)} data files, {size:.0f} MB")


if __name__ == "__main__":
    main(sys.argv[1])
