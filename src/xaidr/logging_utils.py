"""Run folders, JSON metric logs and environment snapshots."""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

import yaml


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo_root: Path = Path(".")) -> str:
    """Current commit hash; read from .git directly so it works without git on PATH."""
    head = repo_root / ".git" / "HEAD"
    if not head.is_file():
        return "unknown"
    ref = head.read_text(encoding="utf-8").strip()
    if not ref.startswith("ref:"):
        return ref
    ref_path = repo_root / ".git" / ref.split(" ", 1)[1]
    if ref_path.is_file():
        return ref_path.read_text(encoding="utf-8").strip()
    packed = repo_root / ".git" / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line.endswith(ref.split(" ", 1)[1]):
                return line.split(" ", 1)[0]
    return "unknown"


def env_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()),
    }
    try:
        import torch

        snap["torch"] = torch.__version__
        snap["cuda_available"] = torch.cuda.is_available()
        snap["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        snap["torch"] = None
    return snap


def write_env(results_dir: Path, config: Mapping[str, Any]) -> None:
    """Write pip freeze, config snapshot, git commit and GPU info to ``results/env``."""
    env_dir = results_dir / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    snap = env_snapshot()
    (env_dir / "pip_freeze.txt").write_text("\n".join(snap.pop("packages")) + "\n", encoding="utf-8")
    snap["git_commit"] = git_commit()
    (env_dir / "env.json").write_text(json.dumps(snap, indent=2), encoding="utf-8")
    (env_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(dict(config)), encoding="utf-8")


def new_run_dir(checkpoints: Path, tag: str) -> Path:
    run_dir = checkpoints / f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
