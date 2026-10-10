"""EyePACS adapter stub, disabled for this build (D4)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class EyepacsAdapter:
    name = "eyepacs"

    def build_manifest(self, raw_dir: Path) -> pd.DataFrame:
        raise NotImplementedError("EyePACS is out of scope for this build (D4)")
