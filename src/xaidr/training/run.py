"""One training run: build everything from the config, fit, pick the threshold, write the run record."""
from __future__ import annotations

import dataclasses
import json
import logging
import time
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from xaidr.config import Config, load_config
from xaidr.data.dataset import make_eval_loader, make_train_loader
from xaidr.data.manifest import read_manifest
from xaidr.logging_utils import env_snapshot, git_commit, new_run_dir, sha256_file
from xaidr.models.dual_head import DRDualHeadModel
from xaidr.training.losses import JointDRLoss, inverse_sqrt_class_weights
from xaidr.training.threshold import youden_threshold
from xaidr.training.trainer import Trainer, predict, warmup_cosine

log = logging.getLogger(__name__)

CONFIG_SNAPSHOT = "config.yaml"


def run_config(run_dir: Path) -> Config:
    """The exact config a run was trained with."""
    return load_config(run_dir / CONFIG_SNAPSHOT)


def load_best_model(run_dir: Path, cfg: Config, device: torch.device) -> DRDualHeadModel:
    # the checkpoint supplies every weight, so the ImageNet weights are not downloaded again
    model = DRDualHeadModel(dataclasses.replace(cfg.model, pretrained=False), cfg.data.image_size)
    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def train_run(cfg: Config, manifest_path: Path, resume: str | None = None) -> Path:
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = read_manifest(manifest_path)
    train_loader = make_train_loader(manifest, cfg)
    val_loader = make_eval_loader(manifest, "val", cfg)

    if resume:
        run_dir = cfg.paths.checkpoints / resume
        if not (run_dir / "last.pt").is_file():
            raise FileNotFoundError(f"cannot resume: {run_dir / 'last.pt'} does not exist")
    else:
        tag = f"{cfg.model.backbone}_{cfg.minority.mode}_seed{cfg.seed}"
        run_dir = new_run_dir(cfg.paths.checkpoints, tag)
        (run_dir / CONFIG_SNAPSHOT).write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")

    model = DRDualHeadModel(cfg.model, cfg.data.image_size)
    train_grades = manifest.loc[manifest["split"] == "train", "grade"].astype(int)
    weights = None
    if cfg.train.loss.grade_class_weights == "inverse_sqrt_freq":
        weights = inverse_sqrt_class_weights(train_grades.tolist())  # training split only
    loss = JointDRLoss(cfg.train.loss.lambda_det, cfg.train.loss.lambda_grade, weights)
    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = LambdaLR(
        optimizer, lambda epoch: warmup_cosine(epoch, cfg.train.warmup_epochs, cfg.train.epochs)
    )
    trainer = Trainer(model, loss, optimizer, scheduler, cfg, run_dir, device)
    if resume:
        trainer.resume()
    log.info(
        "run %s on %s: %d training images, %d batches per epoch, %d validation batches",
        run_dir.name, device, len(train_grades), len(train_loader), len(val_loader),
    )
    summary = trainer.fit(train_loader, val_loader)

    best = load_best_model(run_dir, cfg, device)
    val_predictions = predict(best, val_loader, device, trainer.amp)
    val_predictions.to_csv(run_dir / "predictions_val.csv", index=False)
    threshold = youden_threshold(
        val_predictions["dr_label"].to_numpy(), val_predictions["det_score"].to_numpy()
    )
    (run_dir / "threshold.json").write_text(
        json.dumps({"threshold": threshold, "rule": "youden_j", "chosen_on": "val"}, indent=2),
        encoding="utf-8",
    )
    snapshot = env_snapshot()
    split_lock = cfg.paths.splits / "aptos2019_split.sha256"
    record = {
        "run_id": run_dir.name, "git_commit": git_commit(), "seed": cfg.seed, "device": str(device),
        "gpu": snapshot.get("gpu"), "torch": snapshot.get("torch"), "python": snapshot["python"],
        "packages": snapshot["packages"], "manifest_sha256": sha256_file(manifest_path),
        "split_sha256": split_lock.read_text(encoding="utf-8").split()[0],
        "class_weights": None if weights is None else [round(float(w), 6) for w in weights],
        "threshold": threshold, **summary, "wall_clock_seconds": round(time.time() - started, 1),
    }
    (run_dir / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    log.info("run %s finished: best epoch %d, score %.4f, threshold %.4f",
             run_dir.name, summary["best_epoch"], summary["best_score"], threshold)
    return run_dir
