"""Computational efficiency of a trained run (paper Section 6.5, DESIGN.md Section 7.23).

Batch-1 latency on CPU (fp32) and, when a GPU is present, on GPU in fp32 and fp16; parameter
counts; weight and checkpoint size; peak GPU memory. Latency is the median of timed forward
passes after warm-up, with the interquartile range.
"""
from __future__ import annotations

import logging
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from xaidr.config import Config
from xaidr.eval.evaluate_grading import method_label
from xaidr.training.run import load_best_model

log = logging.getLogger(__name__)

OUTPUT = "efficiency.csv"


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    return {
        "params_total": sum(p.numel() for p in model.parameters()),
        "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def weights_megabytes(model: torch.nn.Module) -> float:
    """Size of the parameters and buffers as stored (what a weights-only file would hold)."""
    tensors = [*model.parameters(), *model.buffers()]
    return sum(t.numel() * t.element_size() for t in tensors) / 1e6


@torch.no_grad()
def time_forward(
    model: torch.nn.Module, x: torch.Tensor, warmup: int = 20, timed: int = 200, half: bool = False
) -> dict[str, float]:
    """Median and quartiles of one forward pass, in milliseconds."""
    device = x.device
    cuda = device.type == "cuda"
    for _ in range(warmup):
        with torch.autocast(device.type, enabled=half):
            model(x)
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(timed):
        start = time.perf_counter()
        with torch.autocast(device.type, enabled=half):
            model(x)
        if cuda:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    q25, median, q75 = np.percentile(samples, [25, 50, 75])
    return {
        "latency_median_ms": float(median), "latency_q25_ms": float(q25), "latency_q75_ms": float(q75),
        "latency_iqr_ms": float(q75 - q25), "n_timed": timed, "n_warmup": warmup,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1e6 if cuda else float("nan"),
    }


def benchmark_run(cfg: Config, run_dir: Path, warmup: int = 20, timed: int = 200) -> pd.DataFrame:
    size = cfg.data.image_size
    settings = [("cpu", "fp32")]
    if torch.cuda.is_available():
        settings += [("cuda", "fp32"), ("cuda", "fp16")]
    rows = []
    for device_name, precision in settings:
        device = torch.device(device_name)
        model = load_best_model(run_dir, cfg, device)
        x = torch.randn(1, 3, size, size, device=device)
        row: dict[str, object] = {
            "model": method_label(cfg), "run_id": run_dir.name, "input_px": size, "batch_size": 1,
            "device": torch.cuda.get_device_name(0) if device_name == "cuda" else platform.processor(),
            "device_type": device_name, "precision": precision,
            "cpu_threads": torch.get_num_threads() if device_name == "cpu" else np.nan,
            **parameter_counts(model), "weights_mb": weights_megabytes(model),
            "checkpoint_mb": (run_dir / "best.pt").stat().st_size / 1e6,
        }
        row.update(time_forward(model, x, warmup, timed, half=precision == "fp16"))
        rows.append(row)
        del model
    table = pd.DataFrame(rows)
    cfg.paths.results.mkdir(parents=True, exist_ok=True)
    table.to_csv(cfg.paths.results / OUTPUT, index=False)
    shown = table[["device_type", "precision", "latency_median_ms", "latency_iqr_ms", "params_total"]]
    log.info("wrote %s\n%s", cfg.paths.results / OUTPUT, shown.round(2).to_string(index=False))
    return table
