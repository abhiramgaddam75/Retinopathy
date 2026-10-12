"""Copy ten APTOS test-split images into data/demo_samples for the demo app.

Run from the repository root:  python tools/make_demo_samples.py <run id>

Selection is by rule, not by hand: the first two test images of each grade in id order. If
all ten happen to be graded correctly by the model, the last one is swapped for the first
misgraded test image, so the demo always contains an honest miss. Images are resized to at
most 1,200 px on the long side to keep loading quick.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "demo_samples"


def main(run_id: str) -> None:
    predictions = pd.read_csv(ROOT / "checkpoints" / run_id / "predictions_test.csv", dtype={"image_id": str})
    predictions["predicted"] = predictions[[f"p{g}" for g in range(5)]].to_numpy().argmax(axis=1)
    predictions = predictions.sort_values("image_id")
    chosen = predictions.groupby("grade").head(2)
    if (chosen["predicted"] == chosen["grade"]).all():
        miss = predictions[predictions["predicted"] != predictions["grade"]].head(1)
        chosen = pd.concat([chosen.iloc[:-1], miss])
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, (_, row) in enumerate(chosen.sort_values(["grade", "image_id"]).iterrows(), start=1):
        source = ROOT / "data" / "raw" / "aptos2019" / "train_images" / f"{row['image_id']}.png"
        image = cv2.imdecode(np.fromfile(str(source), dtype=np.uint8), cv2.IMREAD_COLOR)
        scale = 1200 / max(image.shape[:2])
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        name = f"sample_{number:02d}.png"
        cv2.imencode(".png", image)[1].tofile(str(OUT / name))
        rows.append({"file": name, "image_id": row["image_id"], "true_grade": int(row["grade"]),
                     "model_grade_on_original": int(row["predicted"])})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "samples.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
