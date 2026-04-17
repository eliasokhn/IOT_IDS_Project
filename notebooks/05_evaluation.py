# %% [markdown]
# # Notebook 05 — Evaluation & Model Comparison
#
# **Goal:** Load all saved metrics, build comparison tables,
# and generate final report-quality plots.
# ---

# %% — Setup
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import os

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

import json, glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# %% — Load all saved metrics
metrics_files = sorted(glob.glob(str(PROJECT_ROOT / "reports/metrics_*.json")))
print(f"Found {len(metrics_files)} metrics files:")
for f in metrics_files:
    print(f"  {f}")

all_metrics = []
for path in metrics_files:
    with open(path) as f:
        m = json.load(f)
    all_metrics.append(m)
    print(f"\n{m['model'].upper()} | {m['task']} | "
          f"Macro F1={m['macro_f1']:.4f} | "
          f"Accuracy={m['accuracy']:.4f} | "
          f"FPR_benign={m.get('fpr_benign', 'N/A')}")

# %% — Build comparison table
from src.evaluation.metrics import build_comparison_table

comparison_df = build_comparison_table(all_metrics)
print("\n" + "="*70)
print("MODEL COMPARISON TABLE (Primary metric: Macro F1)")
print("="*70)
print(comparison_df.to_string(index=False))

comparison_df.to_csv(str(PROJECT_ROOT / "reports/model_comparison.csv"), index=False)
print("\nComparison table saved to reports/model_comparison.csv")

# %% — Comparison bar chart
from src.evaluation.plots import plot_model_comparison

plot_model_comparison(
    comparison_df,
    save_path=str(PROJECT_ROOT / "reports/comparison_macro_f1.png"),
    metric="macro_f1",
)

plot_model_comparison(
    comparison_df,
    save_path=str(PROJECT_ROOT / "reports/comparison_accuracy.png"),
    metric="accuracy",
)
print("Comparison plots saved.")

# %% — FPR on Benign comparison
fpr_rows = comparison_df[comparison_df["fpr_benign"] > 0].copy()
if len(fpr_rows) > 0:
    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(fpr_rows))
    colors = ["#4C72B0" if m == "lr" else "#DD8452" for m in fpr_rows["model"]]
    bars = ax.bar(x, fpr_rows["fpr_benign"], color=colors, edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [f"{r['model'].upper()}\n{r['task']}" for _, r in fpr_rows.iterrows()],
        fontsize=9,
    )
    ax.set_ylabel("False Positive Rate (Benign traffic)")
    ax.set_title("FPR on Benign Traffic — Lower is Better", fontweight="bold")
    ax.axhline(0.02, color="red", linestyle="--", linewidth=1.0, label="2% threshold")
    ax.legend(fontsize=9)
    for bar, val in zip(bars, fpr_rows["fpr_benign"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(str(PROJECT_ROOT / "reports/fpr_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("FPR comparison plot saved.")

# %% — Per-class recall analysis for rare classes
print("\n" + "="*60)
print("PER-CLASS RECALL — RARE CLASSES (Web, BruteForce)")
print("="*60)

rare_classes = ["Web", "BruteForce", "Spoofing"]
for m in all_metrics:
    per_class = m.get("per_class", {})
    print(f"\n{m['model'].upper()} | {m['task']}:")
    for cls in rare_classes:
        if cls in per_class:
            r = per_class[cls]
            print(f"  {cls:<15} | recall={r['recall']:.4f} | "
                  f"precision={r['precision']:.4f} | support={r['support']:,}")

# %% — Print best model recommendation
print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)

best_rows = comparison_df.loc[comparison_df.groupby("task")["macro_f1"].idxmax()]
for _, row in best_rows.iterrows():
    print(f"  Best for {row['task']:<10}: {row['model'].upper()} "
          f"(Macro F1 = {row['macro_f1']:.4f})")

gb_avg = comparison_df[comparison_df["model"] == "gb"]["macro_f1"].mean()
lr_avg = comparison_df[comparison_df["model"] == "lr"]["macro_f1"].mean()
print(f"\n  Average Macro F1 — GB: {gb_avg:.4f} | LR: {lr_avg:.4f}")
print(f"  GB outperforms LR by {(gb_avg - lr_avg)*100:.1f} percentage points on average.")

print("\nEvaluation complete. Run notebook 06 next (streaming simulation).")
