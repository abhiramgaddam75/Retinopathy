"""Single entry point: ``python -m xaidr.cli <step>``."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from xaidr.config import Config, ConfigError, load_config
from xaidr.data.adapters import DataMissingError, get_adapter
from xaidr.data.manifest import CROP_COLUMNS, ManifestError, normalise, read_manifest, write_manifest
from xaidr.data.preprocess import PreprocessError, build_processed
from xaidr.data.splits import SplitError, make_splits
from xaidr.logging_utils import setup_logging, write_env
from xaidr.seed import set_seed

log = logging.getLogger("xaidr")

MANIFEST_NAME = "manifest.csv"
# Steps that land in later milestones (DESIGN.md Section 15.1).
PENDING_STEPS = {
    "train": "M1", "evaluate": "M1", "xai": "M2", "crossdataset": "M2",
    "ablation": "M3", "efficiency": "M3", "figures": "M3", "all": "M4",
}


def manifest_path(cfg: Config) -> Path:
    return cfg.paths.manifests / MANIFEST_NAME


def prepare_data(cfg: Config, datasets: list[str], force: bool) -> None:
    missing = [name for name in datasets if not (cfg.paths.raw / name).is_dir()]
    if missing:
        raise DataMissingError(
            f"raw dataset folder(s) not found under {cfg.paths.raw}: {missing}. Place the data as "
            "described in docs/DATA_ACCESS.md, or pass --datasets to prepare only the ones you have."
        )
    frames = []
    for name in datasets:
        adapter = get_adapter(name, cfg.data)
        frame = normalise(adapter.build_manifest(cfg.paths.raw / name))
        log.info("%s: %d images", name, len(frame))
        overlap = getattr(adapter, "overlap", None)
        if overlap is not None:
            cfg.paths.manifests.mkdir(parents=True, exist_ok=True)
            overlap.to_csv(cfg.paths.manifests / f"{name}_overlap.csv", index=False)
        frames.append(frame)
    manifest = pd.concat(frames, ignore_index=True)

    path = manifest_path(cfg)
    if path.is_file():
        # Keep datasets prepared earlier, and reuse stored transform records so reruns can skip work.
        previous = read_manifest(path)
        carry = ["split", "content_hash", "orig_h", "orig_w", *CROP_COLUMNS]
        keyed = previous.set_index(["dataset", "image_id"])[carry]
        joined = manifest.drop(columns=carry).join(keyed, on=["dataset", "image_id"])
        joined[["split", "content_hash"]] = joined[["split", "content_hash"]].fillna("")
        manifest = pd.concat(
            [previous[~previous["dataset"].isin(datasets)], normalise(joined)], ignore_index=True
        )
    manifest = build_processed(
        manifest, cfg.paths.processed, cfg.data.image_size, cfg.data.crop_threshold,
        cfg.data.clahe.clip_limit, cfg.data.clahe.tile_grid, force=force,
        num_workers=cfg.train.num_workers,
    )
    write_manifest(manifest, path)
    log.info("wrote %s (%d rows)", path, len(manifest))


def run_make_splits(cfg: Config, resplit: bool) -> None:
    manifest = make_splits(
        read_manifest(manifest_path(cfg)), cfg.data.split_ratios, cfg.seed,
        splits_dir=cfg.paths.splits, resplit=resplit,
    )
    write_manifest(manifest, manifest_path(cfg))
    log.info("split counts:\n%s", manifest.groupby(["dataset", "split"]).size())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xaidr", description="DR-XAI pipeline")
    sub = parser.add_subparsers(dest="step", required=True)

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", default="configs/default.yaml")
        p.add_argument("overrides", nargs="*", help="config overrides, key.subkey=value")
        return p

    p = add("prepare-data", "adapters -> manifest -> preprocessing")
    p.add_argument("--datasets", help="comma-separated subset of data.datasets")
    p.add_argument("--force", action="store_true", help="re-process images that already exist")
    p = add("make-splits", "stratified, locked APTOS 2019 split")
    p.add_argument("--resplit", action="store_true", help="replace a locked split")
    for name, milestone in PENDING_STEPS.items():
        p = add(name, f"(milestone {milestone})")
        p.add_argument("--run", help="run id under checkpoints/")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    try:
        cfg = load_config(args.config, args.overrides)
        set_seed(cfg.seed, cfg.deterministic)
        if args.step == "prepare-data":
            datasets = args.datasets.split(",") if args.datasets else list(cfg.data.datasets)
            prepare_data(cfg, datasets, args.force)
            write_env(cfg.paths.results, cfg.to_dict())
        elif args.step == "make-splits":
            run_make_splits(cfg, args.resplit)
        else:
            log.error("step '%s' is not implemented yet (milestone %s)", args.step, PENDING_STEPS[args.step])
            return 2
    except (ConfigError, DataMissingError, ManifestError, PreprocessError, SplitError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
