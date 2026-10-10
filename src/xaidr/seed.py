"""Seeding for python, numpy and torch."""
from __future__ import annotations

import logging
import os
import random

import numpy as np

log = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # PYTHONHASHSEED only takes effect if set before the interpreter starts.
    if os.environ.get("PYTHONHASHSEED") != str(seed):
        log.info("for full hash determinism start python with PYTHONHASHSEED=%d", seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
