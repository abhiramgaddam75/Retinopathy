"""Static figures for the paper, saved at 220 dpi to ``results/figures`` (DESIGN.md Section 7.24).

Conventions: one y-scale per panel (two measures get two panels, never a second axis); a single
blue ramp for magnitude; the first two hues of a fixed categorical order for the two series in
the loss panel, each also labelled at its line end; grid and axes kept recessive; text in ink
colours rather than series colours.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

from xaidr.config import Config

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

DPI = 220
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID, AXIS = "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834")  # fixed categorical order: blue, orange
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"]
)
LESION_COLOR = "#eda100"
GRADE_NAMES = ("0 No DR", "1 Mild", "2 Moderate", "3 Severe", "4 Proliferative")


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def confusion_matrix_fig(
    matrix: np.ndarray, labels: tuple[str, ...] = GRADE_NAMES, title: str = ""
) -> Figure:
    """Counts in each cell; colour is the share of the true grade, so rare grades stay readable."""
    matrix = np.asarray(matrix, dtype=float)
    share = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6), facecolor=SURFACE)
    image = ax.imshow(share, cmap=BLUE_RAMP, vmin=0, vmax=1)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            count = int(matrix[row, col])
            text = f"{count}\n{share[row, col]:.0%}" if count else "0"
            ink = "#ffffff" if share[row, col] > 0.55 else INK
            ax.text(col, row, text, ha="center", va="center", fontsize=8, color=ink)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted grade", color=INK_2, fontsize=9)
    ax.set_ylabel("True grade", color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # white gaps between cells
    ax.set_xticks(np.arange(-0.5, len(labels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label("Share of true grade", color=INK_2, fontsize=8)
    bar.ax.tick_params(colors=MUTED, labelsize=7, length=0)
    bar.outline.set_visible(False)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    return fig


def training_curves_fig(metrics_jsonl: Path, title: str = "") -> Figure:
    """Three panels sharing the epoch axis: loss (train, validation), validation QWK, detection AUC."""
    lines = Path(metrics_jsonl).read_text(encoding="utf-8").splitlines()
    frame = pd.DataFrame([json.loads(line) for line in lines if line])
    epochs = frame["epoch"] + 1
    best = int(frame["epoch"].to_numpy()[int(frame["selection_score"].to_numpy().argmax())]) + 1
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), facecolor=SURFACE, sharex=True)
    panels = [
        ("Loss", [("train_loss", "train", SERIES[0]), ("val_loss", "validation", SERIES[1])]),
        ("Validation QWK", [("val_qwk", "", SERIES[0])]),
        ("Validation detection AUC", [("val_det_auc", "", SERIES[0])]),
    ]
    for ax, (name, series) in zip(axes, panels, strict=True):
        _style(ax)
        for column, label, color in series:
            ax.plot(epochs, frame[column], color=color, linewidth=2, label=label or None)
            if label:  # direct label at the line end, in ink, next to the coloured line
                ax.annotate(label, (epochs.iloc[-1], frame[column].iloc[-1]), xytext=(5, 0),
                            textcoords="offset points", fontsize=8, color=INK_2, va="center")
        ax.axvline(best, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_title(name, color=INK, fontsize=9, loc="left")
        ax.set_xlabel("Epoch", color=INK_2, fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.margins(x=0.02)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper center")
    axes[1].annotate(f"selected epoch {best}", (best, frame["val_qwk"].min()), xytext=(-4, 2),
                     textcoords="offset points", fontsize=8, color=INK_2, ha="right")
    if title:
        fig.suptitle(title, color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def class_distribution_fig(manifest_df: pd.DataFrame) -> Figure:
    """Images per ICDR grade, one panel per dataset, from the rows that have a grade.

    Bars start at zero on a linear scale, each panel with its own axis. (DESIGN 7.24 asked for a
    log scale; a bar's length on a log axis has no fixed meaning, so counts are labelled instead.)
    """
    graded = manifest_df[manifest_df["grade"].notna() & ~manifest_df["image_id"].str.startswith("seg_")]
    datasets = list(dict.fromkeys(graded["dataset"]))
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(4.2 * len(datasets), 3.4), facecolor=SURFACE, squeeze=False
    )
    for ax, dataset in zip(axes[0], datasets, strict=True):
        grades = graded.loc[graded["dataset"] == dataset, "grade"].astype(int)
        counts = grades.value_counts().reindex(range(5), fill_value=0)
        _style(ax)
        bars = ax.bar(range(5), counts.to_numpy(), color=SERIES[0], width=0.62)
        ax.bar_label(bars, labels=[f"{v:,}" for v in counts], fontsize=8, color=INK_2, padding=2)
        ax.set_xticks(range(5), [name.split(" ", 1)[1] for name in GRADE_NAMES], rotation=20, ha="right")
        ax.set_title(f"{dataset} (n = {int(counts.sum()):,})", color=INK, fontsize=9, loc="left")
        ax.set_ylabel("Images", color=INK_2, fontsize=8)
        ax.margins(y=0.12)
    fig.tight_layout()
    return fig


def saliency_panel(
    image_rgb: np.ndarray, saliency: np.ndarray, mask: np.ndarray | None, title: str = ""
) -> Figure:
    """Image, saliency overlay, and expert lesion outline (when a mask is given)."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), facecolor=SURFACE)
    axes[0].imshow(image_rgb)
    axes[0].set_title("Image", color=INK, fontsize=9, loc="left")
    axes[1].imshow(image_rgb)
    alpha = np.clip(saliency, 0, 1) * 0.85
    overlay = axes[1].imshow(saliency, cmap=BLUE_RAMP, vmin=0, vmax=1, alpha=alpha)
    axes[1].set_title("Saliency", color=INK, fontsize=9, loc="left")
    axes[2].imshow(image_rgb)
    if mask is not None and mask.any():
        axes[2].contour(mask.astype(float), levels=[0.5], colors=[LESION_COLOR], linewidths=1.2)
    caption = "Expert lesion outline" if mask is not None else "No lesion mask"
    axes[2].set_title(caption, color=INK, fontsize=9, loc="left")
    for ax in axes:
        ax.set_axis_off()
    bar = fig.colorbar(overlay, ax=axes[1], fraction=0.046, pad=0.02)
    bar.ax.tick_params(colors=MUTED, labelsize=7, length=0)
    bar.outline.set_visible(False)
    if title:
        fig.suptitle(title, color=INK, fontsize=10, x=0.01, ha="left")
    return fig


def write_run_figures(
    run_dir: Path, manifest_df: pd.DataFrame, results_dir: Path, label: str
) -> list[Path]:
    """Figures that need no explainer: confusion matrix, training curves, class distribution."""
    out = results_dir / "figures"
    written = []
    matrix_file = run_dir / "confusion_matrix_test.csv"
    if matrix_file.is_file():
        matrix = pd.read_csv(matrix_file, index_col=0).to_numpy()
        fig = confusion_matrix_fig(matrix, title=f"APTOS 2019 test split (n = {int(matrix.sum())})\n{label}")
        written.append(save(fig, out / f"confusion_matrix_{run_dir.name}.png"))
    if (run_dir / "metrics.jsonl").is_file():
        curves = training_curves_fig(run_dir / "metrics.jsonl", label)
        written.append(save(curves, out / f"training_curves_{run_dir.name}.png"))
    written.append(save(class_distribution_fig(manifest_df), out / "class_distribution.png"))
    return written


def write_saliency_gallery(
    cfg: Config, run_dir: Path, manifest_df: pd.DataFrame, per_grade: int = 1, split: str = "test"
) -> list[Path]:
    """Qualitative panels for a few images of each DR grade, one file per image and method.

    When an image has lesion masks their union is outlined; APTOS images have none, so these
    panels illustrate the maps and are not evidence of their correctness.
    """
    import torch

    from xaidr.data.augment import eval_transform
    from xaidr.data.preprocess import preprocess_record, read_image
    from xaidr.training.run import load_best_model
    from xaidr.xai import build_explainer
    from xaidr.xai.attention import AttentionUnavailableError
    from xaidr.xai.base import denormalise
    from xaidr.xai.fidelity import FIDELITY_LESIONS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_best_model(run_dir, cfg, device)
    rows = manifest_df[(manifest_df["split"] == split) & manifest_df["grade"].notna()]
    chosen = rows[rows["grade"] >= 1].sort_values("image_id").groupby("grade").head(per_grade)
    transform = eval_transform(cfg.data.image_size)
    written = []
    for name in cfg.xai.methods:
        try:
            explainer = build_explainer(name, model, cfg.xai, cfg.seed)
        except AttentionUnavailableError:
            continue
        for _, row in chosen.iterrows():
            rgb = read_image(row["processed_path"])[..., ::-1].copy()
            tensor = transform(image=rgb)["image"]
            with torch.no_grad():
                predicted = int(model(tensor[None].to(device))["grade_logits"].argmax(dim=1))
            saliency = explainer.explain(tensor, cfg.xai.target if cfg.xai.target == "disease" else predicted)
            mask = None
            if row["has_masks"]:
                masks = preprocess_record(row.to_dict(), cfg.data.image_size)[1]
                lesions = [masks[k] for k in FIDELITY_LESIONS if k in masks]
                mask = np.logical_or.reduce(lesions) if lesions else None
            title = (f"{name}: {row['dataset']} {row['image_id']}, true grade {int(row['grade'])}, "
                     f"predicted {predicted}, target {cfg.xai.target}")
            fig = saliency_panel(denormalise(tensor), saliency, mask, title)
            target = results_figures(cfg) / f"saliency_{name}_{row['image_id']}_{run_dir.name}.png"
            written.append(save(fig, target))
    return written


def results_figures(cfg: Config) -> Path:
    return cfg.paths.results / "figures"
