"""Epoch loop with AMP, gradient accumulation, early stopping and checkpoints (DESIGN.md 7.13)."""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from xaidr.config import Config
from xaidr.logging_utils import append_jsonl
from xaidr.models.heads import NUM_GRADES
from xaidr.training.losses import JointDRLoss
from xaidr.training.metrics import auc, quadratic_weighted_kappa

log = logging.getLogger(__name__)

PROB_COLUMNS = [f"p{g}" for g in range(NUM_GRADES)]


def warmup_cosine(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    """Learning-rate factor for an epoch: linear warm-up, then cosine decay to zero."""
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    span = max(1, total_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / span))


def selection_score(qwk: float, det_auc: float) -> float:
    """Model-selection score on validation data (D9)."""
    return 0.5 * qwk + 0.5 * det_auc


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = False) -> pd.DataFrame:
    """One row per image: labels, detection score and the five grade probabilities."""
    model.eval()
    frames = []
    for images, dr_label, grade, image_ids in tqdm(loader, desc="predict", leave=False):
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            out = model(images.to(device, non_blocking=True))
        probs = torch.softmax(out["grade_logits"].float(), dim=1).cpu().numpy()
        frame = pd.DataFrame(probs, columns=PROB_COLUMNS)
        frame.insert(0, "det_score", torch.sigmoid(out["det_logit"].float()).cpu().numpy())
        frame.insert(0, "grade", grade.numpy())
        frame.insert(0, "dr_label", dr_label.numpy())
        frame.insert(0, "image_id", list(image_ids))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def validation_metrics(predictions: pd.DataFrame, loss: JointDRLoss) -> dict[str, float]:
    probs = predictions[PROB_COLUMNS].to_numpy()
    grade = predictions["grade"].to_numpy()
    dr_label = predictions["dr_label"].to_numpy()
    det_score = predictions["det_score"].to_numpy().clip(1e-7, 1 - 1e-7)
    # The same loss as in training, evaluated on CPU from the stored probabilities:
    # cross-entropy of log-probabilities and BCE of the logit reproduce the per-image losses.
    weights = None if loss.grade_class_weights is None else loss.grade_class_weights.detach().cpu()
    _, parts = JointDRLoss(loss.lambda_det, loss.lambda_grade, weights)(
        torch.logit(torch.tensor(det_score)), torch.log(torch.tensor(probs).clamp_min(1e-12)),
        torch.tensor(dr_label), torch.tensor(grade),
    )
    qwk = quadratic_weighted_kappa(grade, probs.argmax(axis=1))
    det_auc = auc(dr_label, det_score)
    return {
        "val_loss": parts["loss"], "val_loss_det": parts["loss_det"], "val_loss_grade": parts["loss_grade"],
        "val_qwk": qwk, "val_det_auc": det_auc,
        "val_grade_acc": float((probs.argmax(axis=1) == grade).mean()),
        # a one-class validation set has no AUC; 0.5 keeps the score defined (only in tiny test runs)
        "selection_score": selection_score(qwk, 0.5 if np.isnan(det_auc) else det_auc),
    }


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss: JointDRLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        cfg: Config,
        run_dir: Path,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss = loss.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.amp = cfg.train.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.start_epoch = 0
        self.best_score = -math.inf
        self.best_epoch = -1
        self.bad_epochs = 0

    def _state(self, epoch: int) -> dict:
        return {
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(), "scaler": self.scaler.state_dict(),
            "epoch": epoch, "best_score": self.best_score, "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
        }

    def resume(self) -> None:
        state = torch.load(self.run_dir / "last.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.start_epoch = state["epoch"] + 1
        self.best_score, self.best_epoch = state["best_score"], state["best_epoch"]
        self.bad_epochs = state["bad_epochs"]
        log.info("resumed from epoch %d (best score %.4f)", state["epoch"], self.best_score)

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        accum = self.cfg.train.grad_accum
        sums = {"loss": 0.0, "loss_det": 0.0, "loss_grade": 0.0}
        self.optimizer.zero_grad(set_to_none=True)
        steps = len(loader)
        for step, (images, dr_label, grade, _) in enumerate(tqdm(loader, desc="train", leave=False)):
            images = images.to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, enabled=self.amp):
                out = self.model(images)
            total, parts = self.loss(
                out["det_logit"], out["grade_logits"], dr_label.to(self.device), grade.to(self.device)
            )
            self.scaler.scale(total / accum).backward()
            if (step + 1) % accum == 0 or step + 1 == steps:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            for key in sums:
                sums[key] += parts[key]
        return {f"train_{key}": value / max(1, steps) for key, value in sums.items()}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict[str, float]:
        patience = self.cfg.train.early_stopping.patience
        for epoch in range(self.start_epoch, self.cfg.train.epochs):
            started = time.time()
            record: dict[str, float] = {"epoch": epoch, "lr": self.optimizer.param_groups[0]["lr"]}
            record.update(self.train_epoch(train_loader))
            val_predictions = predict(self.model, val_loader, self.device, self.amp)
            record.update(validation_metrics(val_predictions, self.loss))
            self.scheduler.step()
            improved = record["selection_score"] > self.best_score
            if improved:
                self.best_score, self.best_epoch, self.bad_epochs = record["selection_score"], epoch, 0
            else:
                self.bad_epochs += 1
            record.update(best_epoch=self.best_epoch, seconds=round(time.time() - started, 1))
            append_jsonl(self.run_dir / "metrics.jsonl", record)
            state = self._state(epoch)
            torch.save(state, self.run_dir / "last.pt")
            if improved:
                torch.save(state, self.run_dir / "best.pt")
            log.info(
                "epoch %d: train loss %.4f, val loss %.4f, QWK %.4f, det AUC %.4f, score %.4f%s",
                epoch, record["train_loss"], record["val_loss"], record["val_qwk"], record["val_det_auc"],
                record["selection_score"], " (best)" if improved else "",
            )
            if self.bad_epochs >= patience:
                log.info("early stopping: no improvement for %d epochs", patience)
                break
        return {"best_epoch": self.best_epoch, "best_score": self.best_score}
