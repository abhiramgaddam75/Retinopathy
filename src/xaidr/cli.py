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
from xaidr.eval.ablation import AblationError
from xaidr.logging_utils import setup_logging, write_env
from xaidr.seed import set_seed

log = logging.getLogger("xaidr")

MANIFEST_NAME = "manifest.csv"
# Steps that land in later milestones (DESIGN.md Section 15.1).
PENDING_STEPS = {
    "all": "M4",
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
    p = add("train", "train the dual-head model; prints the run id")
    p.add_argument("--resume", metavar="RUN_ID", help="continue a run from its last.pt")
    p = add("evaluate", "Table 4 and Table 6 on the APTOS 2019 test split")
    p.add_argument("--run", required=True, help="run id under checkpoints/")
    p = add("ablation", "paired comparison of an oversampling run with a no-oversampling run")
    p.add_argument("--run", required=True, help="run id trained with minority.mode=oversample")
    p.add_argument("--against", required=True, help="run id trained with minority.mode=none")
    for name, text in (
        ("xai", "explanation fidelity against lesion masks (Table 5)"),
        ("crossdataset", "in-domain versus external test sets (Section 6.3)"),
        ("efficiency", "parameters, size, latency and memory of a run"),
        ("figures", "confusion matrix, training curves and class distribution"),
    ):
        p = add(name, text)
        p.add_argument("--run", required=True, help="run id under checkpoints/")
        if name == "xai":
            p.add_argument("--limit", type=int, help="only the first N lesion-annotated images")
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
        elif args.step == "train":
            from xaidr.training.run import train_run

            run_dir = train_run(cfg, manifest_path(cfg), args.resume)
            write_env(cfg.paths.results, cfg.to_dict())
            log.info("run id: %s", run_dir.name)
        elif args.step == "evaluate":
            from xaidr.eval.evaluate_grading import evaluate_run
            from xaidr.training.run import CONFIG_SNAPSHOT

            run_dir = cfg.paths.checkpoints / args.run
            if not (run_dir / "best.pt").is_file():
                raise DataMissingError(f"no trained model at {run_dir / 'best.pt'}")
            # evaluate with the config the run was trained with, not the current default
            run_cfg = load_config(run_dir / CONFIG_SNAPSHOT, args.overrides)
            evaluate_run(run_cfg, run_dir, manifest_path(run_cfg))
        elif args.step in ("xai", "crossdataset", "efficiency", "figures"):
            from xaidr.training.run import CONFIG_SNAPSHOT

            run_dir = cfg.paths.checkpoints / args.run
            if not (run_dir / "best.pt").is_file():
                raise DataMissingError(f"no trained model at {run_dir / 'best.pt'}")
            run_cfg = load_config(run_dir / CONFIG_SNAPSHOT, args.overrides)
            if args.step == "xai":
                from xaidr.eval.xai_fidelity import run_fidelity

                run_fidelity(run_cfg, run_dir, manifest_path(run_cfg), args.limit)
            elif args.step == "crossdataset":
                from xaidr.eval.crossdataset import cross_dataset

                cross_dataset(run_cfg, run_dir, manifest_path(run_cfg))
            elif args.step == "efficiency":
                from xaidr.eval.efficiency import benchmark_run

                benchmark_run(run_cfg, run_dir)
            else:
                from xaidr.eval.evaluate_grading import method_label
                from xaidr.viz.figures import write_run_figures

                manifest = read_manifest(manifest_path(run_cfg))
                label = method_label(run_cfg)
                for path in write_run_figures(run_dir, manifest, run_cfg.paths.results, label):
                    log.info("wrote %s", path)
        elif args.step == "ablation":
            from xaidr.eval.ablation import write_ablation
            from xaidr.eval.evaluate_grading import predict_test_split
            from xaidr.training.run import CONFIG_SNAPSHOT

            run_dirs = [cfg.paths.checkpoints / args.against, cfg.paths.checkpoints / args.run]
            for run_dir in run_dirs:
                if not (run_dir / "run.json").is_file():
                    raise DataMissingError(f"no finished run at {run_dir}")
                if not (run_dir / "predictions_test.csv").is_file():
                    run_cfg = load_config(run_dir / CONFIG_SNAPSHOT, args.overrides)
                    predict_test_split(run_cfg, run_dir, manifest_path(run_cfg))
            write_ablation(run_dirs[0], run_dirs[1], cfg.paths.results, cfg.eval.bootstrap_iterations)
        else:
            log.error("step '%s' is not implemented yet (milestone %s)", args.step, PENDING_STEPS[args.step])
            return 2
    except (ConfigError, DataMissingError, ManifestError, PreprocessError, SplitError, AblationError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
