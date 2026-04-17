"""
plots.py
========
Plotting utilities: confusion matrices, comparison charts, per-class recall bars.
All plots are saved to disk as PNG files.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/Colab environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

log = logging.getLogger(__name__)

# ── Colour palette ─────────────────────────────────────────────────────────────
PALETTE = {
    "lr": "#4C72B0",
    "gb": "#DD8452",
    "benign": "#2ca02c",
    "malicious": "#d62728",
}


def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    class_names: list[str],
    title: str,
    save_path: str,
    normalize: bool = True,
    figsize: tuple[int, int] | None = None,
) -> None:
    """
    Plot and save a confusion matrix as a heatmap.

    Parameters
    ----------
    cm          : confusion matrix (n×n)
    class_names : list of class name strings
    title       : plot title
    save_path   : where to save the PNG
    normalize   : if True, show row-normalized values (recall per class)
    """
    cm = np.array(cm)
    n = len(class_names)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = cm.astype(float) / (row_sums + 1e-10)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    if figsize is None:
        figsize = (max(8, n), max(6, n))

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        annot_kws={"size": max(6, 10 - n // 5)},
        linewidths=0.3,
        linecolor="lightgray",
        vmin=0,
        vmax=1 if normalize else None,
    )

    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=max(6, 10 - n // 5))
    plt.yticks(rotation=0, fontsize=max(6, 10 - n // 5))
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Confusion matrix saved to {save_path}")


def plot_per_class_recall(
    per_class: dict,
    title: str,
    save_path: str,
    figsize: tuple[int, int] = (14, 5),
) -> None:
    """
    Bar chart of per-class recall values.
    Rare classes with low recall stand out visually.
    """
    names = list(per_class.keys())
    recalls = [per_class[n]["recall"] for n in names]
    supports = [per_class[n]["support"] for n in names]

    # Colour bars: green if recall ≥ 0.7, amber if 0.4–0.7, red below
    colors = [
        "#2ca02c" if r >= 0.70 else "#ff7f0e" if r >= 0.40 else "#d62728"
        for r in recalls
    ]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(names, recalls, color=colors, edgecolor="white", linewidth=0.5)

    # Annotate each bar with recall value and support count
    for bar, rec, sup in zip(bars, recalls, supports):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{rec:.2f}\n(n={sup:,})",
            ha="center", va="bottom",
            fontsize=max(5, 9 - len(names) // 8),
        )

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Recall", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axhline(y=0.7, color="gray", linestyle="--", linewidth=0.8, label="0.7 threshold")
    ax.legend(fontsize=9)
    plt.xticks(rotation=45, ha="right", fontsize=max(5, 9 - len(names) // 6))
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Per-class recall plot saved to {save_path}")


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save_path: str,
    metric: str = "macro_f1",
    figsize: tuple[int, int] = (12, 5),
) -> None:
    """
    Grouped bar chart comparing LR vs GB across all three tasks.
    """
    tasks = ["binary", "8class", "34class"]
    models = ["lr", "gb"]
    model_labels = {"lr": "Logistic Regression", "gb": "Gradient Boosting"}
    colors = [PALETTE["lr"], PALETTE["gb"]]

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    for i, (model, color) in enumerate(zip(models, colors)):
        vals = []
        for task in tasks:
            row = comparison_df[(comparison_df["model"] == model) &
                                (comparison_df["task"] == task)]
            vals.append(float(row[metric].values[0]) if len(row) > 0 else 0.0)

        offset = (i - 0.5) * width
        rects = ax.bar(x + offset, vals, width, label=model_labels[model],
                       color=color, alpha=0.85, edgecolor="white")

        for rect, val in zip(rects, vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    ax.set_title(f"Model Comparison — {metric.replace('_', ' ').title()}",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("class", "-class") for t in tasks], fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.axhline(y=0.9, color="green", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.axhline(y=0.7, color="orange", linestyle=":", linewidth=0.8, alpha=0.7)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Model comparison plot saved to {save_path}")


def plot_class_distribution(
    label_series: pd.Series,
    title: str,
    save_path: str,
    figsize: tuple[int, int] = (14, 5),
) -> None:
    """Bar chart of class sample counts (log-scale y-axis for imbalanced data)."""
    counts = label_series.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color="#4878d0", edgecolor="white", linewidth=0.4)

    ax.set_yscale("log")
    ax.set_ylabel("Sample count (log scale)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.xticks(rotation=60, ha="right", fontsize=7)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Class distribution plot saved to {save_path}")
