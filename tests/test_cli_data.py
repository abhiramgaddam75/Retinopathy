"""prepare-data and make-splits end to end on synthetic raw data (the M0 part of the CLI)."""
from __future__ import annotations

from pathlib import Path

from conftest import REPO_ROOT
from xaidr.cli import main
from xaidr.data.manifest import read_manifest


def run(tmp_path: Path, *args: str) -> int:
    paths = {
        "raw": tmp_path / "raw", "manifests": tmp_path / "manifests", "processed": tmp_path / "processed",
        "splits": tmp_path / "splits", "checkpoints": tmp_path / "ckpt", "results": tmp_path / "results",
    }
    overrides = [f"paths.{k}={v.as_posix()}" for k, v in paths.items()]
    step, *flags = args
    return main([step, "--config", str(REPO_ROOT / "configs" / "default.yaml"), *flags,
                 *overrides, "data.image_size=224", "train.num_workers=2"])


def test_prepare_and_split(aptos_raw: Path, idrid_raw: Path, tmp_path: Path) -> None:
    # Messidor-2 is not on disk: the default dataset list must stop with an error, not skip it
    assert run(tmp_path, "prepare-data") == 1
    assert not (tmp_path / "manifests" / "manifest.csv").exists()

    assert run(tmp_path, "prepare-data", "--datasets", "aptos2019") == 0
    assert run(tmp_path, "make-splits") == 0
    assert run(tmp_path, "prepare-data", "--datasets", "idrid") == 0
    manifest = read_manifest(tmp_path / "manifests" / "manifest.csv")
    assert manifest["dataset"].value_counts().to_dict() == {"aptos2019": 100, "idrid": 6}
    # adding a dataset later keeps the split that was already assigned
    assert set(manifest.loc[manifest["dataset"] == "aptos2019", "split"]) == {"train", "val", "test"}
    assert (manifest["processed_path"] != "").all()

    assert run(tmp_path, "make-splits") == 0  # same data, same seed: the lock is satisfied
    assert run(tmp_path, "make-splits", "seed=7") == 1  # a different split is refused
    assert run(tmp_path, "make-splits", "--resplit", "seed=7") == 0
    manifest = read_manifest(tmp_path / "manifests" / "manifest.csv")
    assert set(manifest.loc[manifest["dataset"] == "idrid", "split"]) == {"external_idrid"}
    assert (tmp_path / "manifests" / "idrid_overlap.csv").is_file()
    assert (tmp_path / "results" / "env" / "pip_freeze.txt").is_file()


def test_later_steps_report_their_milestone(tmp_path: Path) -> None:
    assert run(tmp_path, "train") == 2
