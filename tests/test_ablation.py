from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import REPO_ROOT
from xaidr.config import load_config
from xaidr.eval.ablation import (
    AblationError,
    ablation_table,
    check_arms,
    minority_sensitivity,
    write_ablation,
)


def predictions(grade: np.ndarray, predicted: np.ndarray, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    probs = np.full((len(grade), 5), 0.05)
    probs[np.arange(len(grade)), predicted] = 0.8
    dr = (grade >= 1).astype(int)
    frame = pd.DataFrame(probs, columns=[f"p{g}" for g in range(5)])
    frame.insert(0, "det_score", np.clip(0.2 + 0.6 * dr + rng.normal(scale=0.1, size=len(grade)), 0, 1))
    frame.insert(0, "grade", grade)
    frame.insert(0, "dr_label", dr)
    frame.insert(0, "image_id", [f"img{i:03d}" for i in range(len(grade))])
    return frame


@pytest.fixture
def arms() -> tuple[pd.DataFrame, pd.DataFrame]:
    grade = np.repeat([0, 1, 2, 3, 4], [80, 30, 50, 20, 20])
    none_pred, over_pred = grade.copy(), grade.copy()
    none_pred[np.flatnonzero(grade == 3)[:10]] = 2  # 10 of 20 severe missed without oversampling
    none_pred[np.flatnonzero(grade == 4)[:14]] = 2  # 14 of 20 proliferative missed
    over_pred[np.flatnonzero(grade == 3)[:2]] = 2   # 2 of 20 missed with oversampling
    over_pred[np.flatnonzero(grade == 4)[:4]] = 2   # 4 of 20 missed
    over_pred[np.flatnonzero(grade == 2)[:5]] = 3   # and a few moderate over-called
    return predictions(grade, none_pred, 1), predictions(grade, over_pred, 2)


def test_minority_sensitivity() -> None:
    y = np.array([3, 3, 3, 3, 4, 4, 0, 1])
    p = np.array([3, 3, 3, 2, 4, 2, 0, 3])
    assert minority_sensitivity(y, p) == pytest.approx((0.75 + 0.5) / 2)


def test_ablation_table_known_differences(arms: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    none, oversample = arms
    table = ablation_table(none, oversample.sample(frac=1.0, random_state=0), 0.5, 0.5, n_boot=300)
    row = table.set_index("metric")
    assert row.loc["sensitivity_grade_3", ["none", "oversample"]].tolist() == [0.5, 0.9]
    assert row.loc["sensitivity_grade_4", ["none", "oversample"]].tolist() == pytest.approx([0.3, 0.8])
    minority = row.loc["minority_sensitivity_grades_3_4"]
    assert minority["diff_oversample_minus_none"] == pytest.approx(0.85 - 0.40)
    assert minority["ci_lo"] > 0 and minority["ci_excludes_zero"]
    assert minority["ci_lo"] < minority["diff_oversample_minus_none"] < minority["ci_hi"]
    assert (table["n"] == 200).all() and len(table) == 9
    # the detection scores of the two arms come from the same distribution: no clear difference
    assert not row.loc["detection_auc", "ci_excludes_zero"]
    same = ablation_table(none, none, 0.5, 0.5, n_boot=50)
    assert (same["diff_oversample_minus_none"] == 0).all() and not same["ci_excludes_zero"].any()
    with pytest.raises(AblationError, match="same test images"):
        ablation_table(none, oversample.iloc[1:], 0.5, 0.5, n_boot=10)


def make_run(root: Path, name: str, overrides: list[str], record: dict, frame: pd.DataFrame) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml", overrides)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict()))
    base = {"seed": 42, "split_sha256": "a" * 64, "manifest_sha256": "b" * 64}
    (run_dir / "run.json").write_text(json.dumps({**base, **record}))
    (run_dir / "threshold.json").write_text(json.dumps({"threshold": 0.5}))
    frame.to_csv(run_dir / "predictions_test.csv", index=False)
    return run_dir


def test_write_ablation_and_arm_checks(arms: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path) -> None:
    none, oversample = arms
    run_none = make_run(tmp_path, "none", ["minority.mode=none"], {}, none)
    run_over = make_run(tmp_path, "over", [], {}, oversample)
    assert check_arms(run_none, run_over) == {"seed": 42, "split_sha256": "a" * 64}
    table = write_ablation(run_none, run_over, tmp_path / "results", n_boot=100)
    written = pd.read_csv(tmp_path / "results" / "ablation_minority.csv")
    assert list(written["metric"]) == list(table["metric"])
    assert written["metric"].iloc[-1] == "mean_idrid_iou_k10"
    assert written["note"].iloc[-1].startswith("not computed") and np.isnan(written["none"].iloc[-1])
    assert set(written["run_none"]) == {"none"} and set(written["run_oversample"]) == {"over"}

    with pytest.raises(AblationError, match="none then oversample"):
        check_arms(run_over, run_none)
    other_lr = make_run(tmp_path, "lr", ["train.lr=0.001"], {}, oversample)
    with pytest.raises(AblationError, match="more than minority.mode.*train"):
        check_arms(run_none, other_lr)
    other_split = make_run(tmp_path, "split", [], {"split_sha256": "c" * 64}, oversample)
    with pytest.raises(AblationError, match="split_sha256"):
        check_arms(run_none, other_split)
    other_seed = make_run(tmp_path, "seed", [], {"seed": 43}, oversample)
    with pytest.raises(AblationError, match="seed"):
        check_arms(run_none, other_seed)
