"""Write the six visualisation notebooks (DESIGN.md Section 12).

Run from the repository root:  python tools/make_notebooks.py
Execute them with:             python tools/make_notebooks.py --execute

Every notebook reads from results/, checkpoints/ and data/manifests/. None trains a model. Each
starts by checking that the files it needs exist and says what to run if they do not.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]

SETUP = '''import json, os, sys
from pathlib import Path
ROOT = Path.cwd() if (Path.cwd() / "src").is_dir() else Path.cwd().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from IPython.display import Image, display
pd.set_option("display.max_columns", 40, "display.width", 200, "display.precision", 4)
RESULTS, CHECKPOINTS = ROOT / "results", ROOT / "checkpoints"

def require(*paths, hint):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        print("MISSING:", *missing, sep="\\n  ")
        print("To create:", hint)
    return not missing

def runs():
    return sorted(p for p in CHECKPOINTS.iterdir() if (p / "run.json").is_file()) if CHECKPOINTS.is_dir() else []
'''

NOTEBOOKS: dict[str, list[tuple[str, str]]] = {
    "00_architecture_and_graph": [
        ("md", "# 00 Architecture\n\nModel summary, module tree and tensor shapes at each stage, for the "
               "specified model and for the CPU fallback model. Weights are not downloaded here."),
        ("code", SETUP),
        ("code", '''import dataclasses, torch
from torchinfo import summary
from xaidr.config import load_config
from xaidr.models.dual_head import DRDualHeadModel

def describe(config_file):
    cfg = load_config(ROOT / "configs" / config_file)
    size = cfg.data.image_size
    model = DRDualHeadModel(dataclasses.replace(cfg.model, pretrained=False), size).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, size, size))
    print(f"{cfg.model.backbone}, input 1 x 3 x {size} x {size}")
    for key, value in out.items():
        print(f"  {key:13s} {tuple(value.shape)}")
    return summary(model, input_size=(1, 3, size, size), depth=3, verbose=0,
                   col_names=("output_size", "num_params"))

describe("default.yaml")'''),
        ("code", 'describe("cpu_fallback.yaml")'),
        ("md", "```\nimage 3 x S x S -> shared backbone -> feature map C x h x w -> pooled features\n"
               "                                   |-> detection head (Linear-ReLU-Dropout-Linear) -> 1 logit\n"
               "                                   |-> grading head   (Linear-ReLU-Dropout-Linear) -> 5 logits\n"
               "loss = 0.4 * BCE(detection) + 0.6 * class-weighted CE(grading)\n```"),
    ],
    "01_data_audit": [
        ("md", "# 01 Data audit\n\nCounts per dataset, grade and split; duplicate handling; source image "
               "sizes; preprocessing example."),
        ("code", SETUP),
        ("code", '''MANIFEST = ROOT / "data" / "manifests" / "manifest.csv"
ok = require(MANIFEST, hint="python -m xaidr.cli prepare-data && python -m xaidr.cli make-splits")
if ok:
    from xaidr.data.manifest import read_manifest
    manifest = read_manifest(MANIFEST)
    display(manifest.groupby(["dataset", "split"]).size().rename("images").to_frame())
    used = manifest[manifest["split"].isin(["train", "val", "test"])]
    display(pd.crosstab(used["split"], used["grade"], margins=True))'''),
        ("code", '''DUPLICATES = ROOT / "data" / "splits" / "aptos2019_duplicates.csv"
if ok and require(DUPLICATES, hint="python -m xaidr.cli make-splits"):
    dup = pd.read_csv(DUPLICATES, keep_default_na=False)
    print("groups of byte-identical images:", dup["content_hash"].nunique(), "| images in them:", len(dup))
    display(dup["action"].value_counts().rename("images").to_frame())'''),
        ("code", '''if ok:
    sizes = manifest.assign(size=manifest["orig_w"].astype(str) + " x " + manifest["orig_h"].astype(str))
    display(sizes.groupby(["dataset", "size"]).size().sort_values(ascending=False).head(12).rename("images").to_frame())
    crop = manifest.assign(aspect=(manifest["crop_x1"] - manifest["crop_x0"]) / (manifest["crop_y1"] - manifest["crop_y0"]))
    print("crop width / height before padding:")
    display(crop.groupby("dataset")["aspect"].describe())'''),
        ("code", '''if ok:
    import matplotlib.pyplot as plt
    from xaidr.data.preprocess import read_image
    row = manifest[manifest["split"] == "test"].iloc[0]
    panels = [("processed (crop, CLAHE, pad, resize)", row["processed_path"])]
    if Path(row["source_path"]).is_file():
        panels.insert(0, ("source", row["source_path"]))
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5), squeeze=False)
    for ax, (title, path) in zip(axes[0], panels):
        ax.imshow(read_image(path)[..., ::-1]); ax.set_title(title, loc="left", fontsize=9); ax.set_axis_off()
    plt.show()'''),
        ("code", '''FIG = RESULTS / "figures" / "class_distribution.png"
if require(FIG, hint="python -m xaidr.cli figures --run <run id>"):
    display(Image(str(FIG), width=520))'''),
    ],
    "02_training_curves": [
        ("md", "# 02 Training curves\n\nLoss, validation QWK and detection AUC per epoch for every finished "
               "run; the dashed line marks the selected epoch."),
        ("code", SETUP),
        ("code", '''found = runs()
if not found:
    print("No finished run under checkpoints/. To create: python -m xaidr.cli train")
rows = []
for run in found:
    record = json.loads((run / "run.json").read_text())
    log = pd.DataFrame([json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()])
    best = log.loc[log["epoch"] == record["best_epoch"]].iloc[0]
    rows.append({"run": run.name, "device": record["device"], "epochs_run": len(log),
                 "best_epoch": record["best_epoch"] + 1, "val_qwk": best["val_qwk"],
                 "val_det_auc": best["val_det_auc"], "selection_score": record["best_score"],
                 "threshold": record["threshold"], "hours": record["wall_clock_seconds"] / 3600})
display(pd.DataFrame(rows))'''),
        ("code", '''from xaidr.viz.figures import training_curves_fig
import matplotlib.pyplot as plt
for run in found:
    fig = training_curves_fig(run / "metrics.jsonl", run.name)
    display(fig); plt.close(fig)'''),
    ],
    "03_results_tables": [
        ("md", "# 03 Results tables\n\nTables 4, 5 and 6 and the cross-dataset summary, with confidence "
               "intervals, and LaTeX snippets written to `results/tables_tex/`. A table that does not "
               "exist yet is reported as missing, never filled in."),
        ("code", SETUP),
        ("code", '''TEX = RESULTS / "tables_tex"; TEX.mkdir(parents=True, exist_ok=True)
def with_ci(table, value="value", lo="ci_lo", hi="ci_hi"):
    out = table.copy()
    out["value (95% CI)"] = [f"{v:.3f} ({a:.3f}-{b:.3f})" for v, a, b in zip(out[value], out[lo], out[hi])]
    return out

T4 = RESULTS / "table4_detection_grading_aptos_test.csv"
if require(T4, hint="python -m xaidr.cli evaluate --run <run id>"):
    t4 = pd.read_csv(T4)
    print("Model:", t4["model"].iloc[0]); print("Run:", t4["run_id"].iloc[0], "| threshold:", round(t4["detection_threshold"].iloc[0], 4))
    shown = with_ci(t4)[["task", "metric", "value (95% CI)", "n"]]
    display(shown); (TEX / "table4.tex").write_text(shown.to_latex(index=False))'''),
        ("code", '''T6 = RESULTS / "table6_comparative.csv"
if require(T6, hint="python -m xaidr.cli evaluate --run <run id>"):
    t6 = pd.read_csv(T6)
    display(t6[["method", "dataset", "grading_accuracy", "qwk", "source", "note"]])
    (TEX / "table6.tex").write_text(t6[["method", "dataset", "grading_accuracy", "qwk"]].to_latex(index=False, float_format="%.3f"))'''),
        ("code", '''T5 = RESULTS / "table5_xai_fidelity.csv"
if require(T5, hint="prepare IDRiD, then python -m xaidr.cli xai --run <run id>"):
    t5 = pd.read_csv(T5)
    primary = t5[t5["primary"]]
    cols = ["method", "lesion", "n_images", "mean_iou", "iou_ci_lo", "iou_ci_hi", "chance_iou", "max_iou",
            "mean_recall", "chance_recall", "pointing_acc", "chance_pointing", "p_holm", "exploratory"]
    display(primary[cols]); (TEX / "table5.tex").write_text(primary[cols].to_latex(index=False, float_format="%.3f"))'''),
        ("code", '''CD = RESULTS / "crossdataset_summary.csv"
if require(CD, hint="python -m xaidr.cli crossdataset --run <run id>"):
    cd = pd.read_csv(CD)
    display(cd[["dataset", "split", "n", "status"]].drop_duplicates())
    done = cd[cd["status"] == "computed"]
    display(with_ci(done)[["dataset", "task", "metric", "value (95% CI)", "n", "detection_definition"]])
    (TEX / "crossdataset.tex").write_text(with_ci(done)[["dataset", "task", "metric", "value (95% CI)", "n"]].to_latex(index=False))'''),
    ],
    "04_xai_gallery": [
        ("md", "# 04 Explanation gallery\n\nSaliency panels and the confusion matrix. Panels on APTOS images "
               "have no lesion masks: they show what the maps look like, not whether they are right. "
               "Good / average / poor cases by IoU need the IDRiD fidelity run."),
        ("code", SETUP),
        ("code", '''FIGS = RESULTS / "figures"
matrices = sorted(FIGS.glob("confusion_matrix_*.png")) if FIGS.is_dir() else []
if not matrices:
    print("No confusion matrix. To create: python -m xaidr.cli figures --run <run id>")
for path in matrices:
    display(Image(str(path), width=560))'''),
        ("code", '''panels = sorted(FIGS.glob("saliency_*.png")) if FIGS.is_dir() else []
if not panels:
    print("No saliency panels. To create: python -m xaidr.cli figures --run <run id> --gallery 1")
for path in panels:
    display(Image(str(path), width=900))'''),
        ("code", '''per_image = [p for run in runs() for p in [run / "fidelity_per_image.csv"] if p.is_file()]
if not per_image:
    print("No per-image fidelity file: the IDRiD fidelity study has not been run (IDRiD is required).")
for path in per_image:
    frame = pd.read_csv(path)
    primary = frame[(frame["lesion"] == "any") & (frame["k_percent"] == frame["k_percent"].iloc[0])]
    for method, group in primary.groupby("method"):
        ranked = group.sort_values("iou")
        picks = ranked.iloc[[len(ranked) - 1, len(ranked) // 2, 0]].assign(case=["good", "average", "poor"])
        print(method); display(picks[["case", "image_id", "iou", "chance_iou", "max_iou", "pointing"]])'''),
    ],
    "05_ablation_and_efficiency": [
        ("md", "# 05 Ablation and efficiency\n\nPaired differences (oversampling minus none) with 95% "
               "intervals, and the efficiency benchmark."),
        ("code", SETUP),
        ("code", '''AB = RESULTS / "ablation_minority.csv"
if require(AB, hint="python -m xaidr.cli ablation --run <oversample run> --against <none run>"):
    ab = pd.read_csv(AB)
    print("oversample run:", ab["run_oversample"].iloc[0]); print("none run:      ", ab["run_none"].iloc[0])
    print("interval:", ab["interval"].iloc[0])
    display(ab[["metric", "none", "oversample", "diff_oversample_minus_none", "ci_lo", "ci_hi", "ci_excludes_zero", "n", "note"]])'''),
        ("code", '''if AB.exists():
    import matplotlib.pyplot as plt
    done = ab.dropna(subset=["diff_oversample_minus_none"]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(done) + 1), facecolor="#fcfcfb")
    y = np.arange(len(done))
    ax.hlines(y, done["ci_lo"], done["ci_hi"], color="#2a78d6", linewidth=2)
    ax.plot(done["diff_oversample_minus_none"], y, "o", color="#2a78d6", markersize=6)
    ax.axvline(0, color="#898781", linewidth=1)
    ax.set_yticks(y, done["metric"], fontsize=8, color="#52514e"); ax.tick_params(length=0, labelsize=8, colors="#898781")
    ax.set_xlabel("oversampling minus none (95% paired bootstrap interval)", fontsize=8, color="#52514e")
    ax.set_facecolor("#fcfcfb"); ax.grid(axis="x", color="#e1e0d9", linewidth=0.6); ax.set_axisbelow(True)
    for side in ("top", "right", "left"): ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    fig.savefig(RESULTS / "figures" / "ablation_differences.png", dpi=220, facecolor="#fcfcfb", bbox_inches="tight")
    plt.show()'''),
        ("code", '''EF = RESULTS / "efficiency.csv"
if require(EF, hint="python -m xaidr.cli efficiency --run <run id>"):
    ef = pd.read_csv(EF)
    display(ef[["model", "device", "precision", "input_px", "params_total", "weights_mb", "checkpoint_mb",
                "latency_median_ms", "latency_iqr_ms", "peak_gpu_memory_mb", "n_timed"]])'''),
    ],
}


def build() -> list[Path]:
    paths = []
    for name, cells in NOTEBOOKS.items():
        notebook = nbformat.v4.new_notebook()
        for kind, source in cells:
            make = nbformat.v4.new_markdown_cell if kind == "md" else nbformat.v4.new_code_cell
            notebook.cells.append(make(source))
        notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
        path = ROOT / "notebooks" / f"{name}.ipynb"
        path.parent.mkdir(exist_ok=True)
        nbformat.write(notebook, path)
        paths.append(path)
    return paths


def execute(paths: list[Path]) -> None:
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(
            notebook, {"metadata": {"path": str(ROOT / "notebooks")}}
        )
        nbformat.write(notebook, path)
        print(f"executed {path.name}")


if __name__ == "__main__":
    written = build()
    print(f"wrote {len(written)} notebooks")
    if "--execute" in sys.argv:
        execute(written)
