"""Tester probes against the M0 code. Read-only with respect to src/; works in a temp folder.

Run from the repository root:  python docs/evidence/probes/m0_probes.py
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests"))

import xaidr.data.dataset as dataset_mod  # noqa: E402
import xaidr.data.preprocess as preprocess_mod  # noqa: E402
import xaidr.data.splits as splits_mod  # noqa: E402
from conftest import fundus, write_image  # noqa: E402
from xaidr.cli import main  # noqa: E402
from xaidr.config import load_config  # noqa: E402
from xaidr.data.adapters import Aptos2019Adapter  # noqa: E402
from xaidr.data.augment import train_transform  # noqa: E402
from xaidr.data.dataset import make_train_loader  # noqa: E402
from xaidr.data.manifest import normalise, read_manifest, write_manifest  # noqa: E402
from xaidr.data.preprocess import crop_fov, read_image  # noqa: E402
from xaidr.data.splits import make_splits  # noqa: E402
from xaidr.seed import set_seed  # noqa: E402

CONFIG = str(ROOT / "configs" / "default.yaml")


def header(title: str) -> None:
    print(f"\n=== {title} ===")


def make_aptos(
    raw: Path, grades: list[int], duplicate_pairs: int = 0, hw: tuple[int, int] = (48, 64)
) -> None:
    ids = [f"img{i:03d}" for i in range(len(grades))]
    (raw / "train_images").mkdir(parents=True)
    pd.DataFrame({"id_code": ids, "diagnosis": grades}).to_csv(raw / "train.csv", index=False)
    for i, code in enumerate(ids):
        write_image(raw / "train_images" / f"{code}.png", fundus(i, *hw))
    # the first `duplicate_pairs` images of grade 0 are byte-copied over later grade-0 images
    zeros = [c for c, g in zip(ids, grades, strict=True) if g == 0]
    for a, b in zip(zeros[:duplicate_pairs], zeros[duplicate_pairs:2 * duplicate_pairs], strict=True):
        shutil.copyfile(raw / "train_images" / f"{a}.png", raw / "train_images" / f"{b}.png")


def cli(tmp: Path, *args: str) -> int:
    paths = {k: tmp / k for k in ("raw", "manifests", "processed", "splits", "checkpoints", "results")}
    step, *flags = args
    return main([step, "--config", CONFIG, *flags, *[f"paths.{k}={v.as_posix()}" for k, v in paths.items()],
                 "data.image_size=224", "train.num_workers=2"])


def p1_augmentation_reproducibility() -> None:
    header("P1 augmentation reproducibility under set_seed(42)")
    image = fundus(1, 224, 224)[..., ::-1].copy()
    outs = []
    for _ in range(2):
        set_seed(42)
        transform = train_transform(224)
        outs.append(torch.stack([transform(image=image)["image"] for _ in range(6)]))
    print("two seeded runs give identical augmented tensors:", bool(torch.equal(outs[0], outs[1])))


def p2_train_loader_reproducibility(tmp: Path) -> pd.DataFrame:
    header("P2 training-loader reproducibility (same config, same seed, two fresh loaders)")
    # images large enough for the CLAHE tiles to matter in P3
    make_aptos(tmp / "raw" / "aptos2019", [0] * 40 + [1] * 20 + [2] * 20 + [3] * 10 + [4] * 10, hw=(360, 480))
    assert cli(tmp, "prepare-data", "--datasets", "aptos2019") == 0
    assert cli(tmp, "make-splits") == 0
    manifest = read_manifest(tmp / "manifests" / "manifest.csv")
    cfg = load_config(CONFIG, ["data.image_size=224", "train.batch_size=4", "train.num_workers=0"])
    batches = []
    for _ in range(2):
        set_seed(cfg.seed)
        images, _, _, ids = next(iter(make_train_loader(manifest, cfg)))
        batches.append((images, ids))
    print("same image ids in first batch:", batches[0][1] == batches[1][1])
    print("same image tensors in first batch:", bool(torch.equal(batches[0][0], batches[1][0])))
    return manifest


def p3_stale_processed_after_config_change(tmp: Path) -> None:
    header("P3 processed images after a preprocessing-config change, without --force")
    target = next((tmp / "processed" / "aptos2019" / "images").glob("*.png"))
    before = hashlib.sha256(target.read_bytes()).hexdigest()
    code = cli(tmp, "prepare-data", "--datasets", "aptos2019", "data.clahe.clip_limit=8.0")
    after = hashlib.sha256(target.read_bytes()).hexdigest()
    print("exit code:", code, "| processed file unchanged after clip_limit 2.0 -> 8.0:", before == after)
    cli(tmp, "prepare-data", "--datasets", "aptos2019", "--force", "data.clahe.clip_limit=8.0")
    forced = hashlib.sha256(target.read_bytes()).hexdigest()
    print("with --force the file changes:", forced != before)
    cli(tmp, "prepare-data", "--datasets", "aptos2019", "--force")


def p4_manifest_split_edit_is_not_checked(tmp: Path, manifest: pd.DataFrame) -> None:
    header("P4 manifest split column edited after locking")
    cfg = load_config(CONFIG, ["data.image_size=224", "train.batch_size=4", "train.num_workers=0"])
    edited = manifest.copy()
    test_rows = edited.index[edited["split"] == "test"]
    edited.loc[test_rows, "split"] = "train"
    write_manifest(edited, tmp / "manifests" / "manifest.csv")
    try:
        loader = make_train_loader(read_manifest(tmp / "manifests" / "manifest.csv"), cfg)
        moved = set(manifest.loc[test_rows, "image_id"]) & set(loader.dataset.df["image_id"])
        print(f"training loader built without error; locked-test images now in training: {len(moved)}")
    except Exception as exc:  # noqa: BLE001
        print("rejected:", type(exc).__name__, exc)
    write_manifest(manifest, tmp / "manifests" / "manifest.csv")
    print("lock file is under an ignored path (cannot be committed):",
          (ROOT / ".gitignore").read_text().splitlines()[1], "vs default splits dir 'data/splits'")


def p5_duplicate_images_across_splits() -> None:
    header("P5 byte-identical images under different ids (reported for APTOS 2019)")
    with tempfile.TemporaryDirectory() as name:
        tmp = Path(name)
        make_aptos(tmp / "aptos2019", [0] * 200 + [1] * 40 + [2] * 40 + [3] * 20 + [4] * 20, duplicate_pairs=60)
        manifest = normalise(Aptos2019Adapter().build_manifest(tmp / "aptos2019"))
        manifest = make_splits(manifest, splits_dir=tmp / "splits")
        digest = manifest["source_path"].map(lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
        groups = manifest.assign(digest=digest).groupby("digest")["split"].agg(["size", "nunique"])
        dup = groups[groups["size"] > 1]
        print(f"planted duplicate pairs: {len(dup)}; pairs whose two copies are in different splits: "
              f"{int((dup['nunique'] > 1).sum())}; warnings or errors raised: none")


def p6_corrupt_image_and_tiny_class() -> None:
    header("P6 error handling")
    with tempfile.TemporaryDirectory() as name:
        tmp = Path(name)
        make_aptos(tmp / "raw" / "aptos2019", [0] * 10 + [1] * 10)
        (tmp / "raw" / "aptos2019" / "train_images" / "img003.png").write_bytes(b"not a png")
        try:
            print("corrupt image -> exit code", cli(tmp, "prepare-data", "--datasets", "aptos2019"))
        except Exception as exc:  # noqa: BLE001
            print("corrupt image -> uncaught", type(exc).__name__ + ":", exc)
        print("manifest written after failure:", (tmp / "manifests" / "manifest.csv").exists())
    with tempfile.TemporaryDirectory() as name:
        tmp = Path(name)
        make_aptos(tmp / "raw" / "aptos2019", [0] * 30 + [1] * 10 + [4] * 2)
        cli(tmp, "prepare-data", "--datasets", "aptos2019")
        try:
            print("class with 2 images -> exit code", cli(tmp, "make-splits"))
        except Exception as exc:  # noqa: BLE001
            print("class with 2 images -> uncaught", type(exc).__name__ + ":", str(exc)[:110])


def p7_aspect_ratio_of_sample_photos() -> None:
    header("P7 field-of-view crop aspect ratio on the supplied sample photos")
    for path in sorted((ROOT / "tests" / "fixtures" / "sample_fundus").glob("*.jpg")):
        img = read_image(path)
        _, (y0, x0, y1, x1) = crop_fov(img)
        h, w = y1 - y0, x1 - x0
        print(f"{path.name:14s} source {img.shape[1]}x{img.shape[0]}  crop {w}x{h}  "
              f"width/height {w / h:.3f}  vertical stretch at 512x512: {(w / h - 1) * 100:+.1f}%")


def p8_mutations() -> None:
    header("P8 mutation checks (does the suite notice a planted fault?)")

    def run(label: str, test: str) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            code = pytest.main([str(ROOT / "tests" / test), "-q", "-x", "--tb=no", "-p", "no:cacheprovider"])
        print(f"{label}: {'DETECTED (test failed)' if code != 0 else 'NOT DETECTED (test passed)'}")

    original = preprocess_mod.resize_mask
    preprocess_mod.resize_mask = lambda mask, size: np.roll(original(mask, size), 3, axis=1)
    run("M1 masks shifted 3 px after resize", "test_preprocess.py::test_mask_and_image_stay_aligned")
    preprocess_mod.resize_mask = original

    original_eval = dataset_mod.eval_transform
    dataset_mod.eval_transform = dataset_mod.train_transform
    run("M2 evaluation loaders given the training augmentation",
        "test_no_leakage.py::test_augmentation_only_in_training")
    dataset_mod.eval_transform = original_eval

    original_tts = splits_mod.train_test_split
    splits_mod.train_test_split = lambda *a, stratify=None, **k: original_tts(*a, **k)
    run("M3 split made without stratification", "test_splits.py::test_stratified_within_one_percentage_point")
    splits_mod.train_test_split = original_tts

    original_min = dataset_mod.minority_sampler
    dataset_mod.minority_sampler = lambda df, grades, factor, seed, mode: original_min(df, grades, factor, seed, "none")
    run("M4 oversampling silently disabled in the training loader", "test_no_leakage.py")
    dataset_mod.minority_sampler = original_min


if __name__ == "__main__":
    probes = [p1_augmentation_reproducibility, p5_duplicate_images_across_splits,
              p6_corrupt_image_and_tiny_class, p7_aspect_ratio_of_sample_photos]
    with tempfile.TemporaryDirectory() as name:
        tmp = Path(name)
        try:
            manifest = p2_train_loader_reproducibility(tmp)
            p3_stale_processed_after_config_change(tmp)
            p4_manifest_split_edit_is_not_checked(tmp, manifest)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
    for probe in probes:
        try:
            probe()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
    p8_mutations()
