# %% [markdown]
# # Notebook 01 — Data Exploration
#
# Merges 63 CSV files into one Parquet (once only, skips if exists).
# Loads the FULL dataset with NO sampling.
# Balanced sampling happens in notebook 02 AFTER splitting, on train only.
# ---

# %% — Setup
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import os

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

import pandas as pd
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# SET USE_DEMO = False to use real CICIoT2023 data
# SET USE_DEMO = True  to run quickly with synthetic data
USE_DEMO = False

# %% — Load config
with open(str(PROJECT_ROOT / "configs/data_config.yaml"), encoding="utf-8") as f:
    data_cfg = yaml.safe_load(f)
RANDOM_STATE = data_cfg["dataset"]["random_state"]

# %% — Load data
from src.data.loader import load_dataset, generate_demo_data, build_merged_parquet
from src.data.label_mapping import add_all_label_columns
from src.data.validation import validate_dataset

if USE_DEMO:
    print("Using synthetic demo data")
    df_raw = generate_demo_data(n_samples=50_000, random_state=RANDOM_STATE)
else:
    print("Using real CICIoT2023 dataset")
    build_merged_parquet(
        raw_dir=str(PROJECT_ROOT / "data/raw"),
        output_path=str(PROJECT_ROOT / "data/processed/merged.parquet"),
    )
    df_raw = load_dataset(
        parquet_path=str(PROJECT_ROOT / "data/processed/merged.parquet"),
        sample_frac=None,   # load all rows — sampling done in notebook 02
        random_state=RANDOM_STATE,
    )

print(f"\nDataset shape (full, no sampling): {df_raw.shape}")

# %% — Validate
report = validate_dataset(df_raw, label_col="label")
print(f"Rows:            {report['n_rows']:,}")
print(f"Classes:         {report['n_classes']} / 34")
print(f"Imbalance ratio: {report['imbalance_ratio']}x  (raw)")
print(f"  -> Will be fixed in notebook 02 after splitting")

# %% — Show data quality issues
print("\n=== DATA QUALITY ISSUES (fixed in preprocessing) ===")

n_exact_dups = df_raw.duplicated().sum()
print(f"Exact duplicate rows: {n_exact_dups:,}  (~5% per file)")

feature_cols  = [c for c in df_raw.columns if c != "label"]
n_conflict    = df_raw.duplicated(subset=feature_cols, keep=False).sum()
print(f"Rows with duplicate feature vectors: {n_conflict:,}")
print("  -> May have DIFFERENT labels (DDOS-UDP_FLOOD vs DOS-UDP_FLOOD)")
print("  -> Fixed by deduplicate() in notebook 02")

if "Protocol Type" in df_raw.columns:
    pt_vals = sorted(df_raw["Protocol Type"].dropna().unique())
    print(f"\nProtocol Type values: {pt_vals}")
    print("  -> IP protocol numbers, not ordinal")
    print("  -> Fixed: one-hot encoded into proto_0..proto_47")

if "Variance" in df_raw.columns and "Std" in df_raw.columns:
    mask = df_raw["Std"].notna() & df_raw["Variance"].notna() & (df_raw["Std"] > 0)
    if mask.sum() > 0:
        diff = (df_raw.loc[mask, "Variance"] - df_raw.loc[mask, "Std"]**2).abs().max()
        print(f"\nVariance vs Std^2 max diff: {diff:.2e}  (Variance = Std^2 exactly)")
        print(f"  Variance max: {df_raw['Variance'].max():,.0f}  (extreme outlier)")
        print("  -> Fixed: Variance dropped")

# %% — Class distribution
dist = df_raw["label"].value_counts().sort_values(ascending=False)
print(f"\nRaw class distribution:")
for cls, cnt in dist.items():
    pct = 100 * cnt / len(df_raw)
    bar = "█" * max(1, int(cnt / dist.max() * 40))
    print(f"  {cls:<42} {cnt:>9,}  ({pct:5.2f}%)  {bar}")

fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(range(len(dist)), dist.values, color="#2196F3", edgecolor="white", linewidth=0.3)
ax.set_yscale("log")
ax.set_xticks(range(len(dist)))
ax.set_xticklabels(dist.index, rotation=60, ha="right", fontsize=7)
ax.set_ylabel("Sample count (log scale)")
ax.set_title("CICIoT2023 -- Raw Class Distribution (before balancing)")
plt.tight_layout()
os.makedirs(str(PROJECT_ROOT / "reports"), exist_ok=True)
fig.savefig(str(PROJECT_ROOT / "reports/class_distribution_raw.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved: reports/class_distribution_raw.png")

# %% — Add label columns and save
df = add_all_label_columns(df_raw, raw_col="label")

from src.data.label_mapping import CATEGORY_CLASS_NAMES
cat_dist = df["label_category"].value_counts().sort_index()
print("\n8-class family distribution (raw):")
for idx, cnt in cat_dist.items():
    name = CATEGORY_CLASS_NAMES[int(idx)]
    print(f"  {int(idx)}. {name:<12}: {cnt:>9,}")

os.makedirs(str(PROJECT_ROOT / "data/processed"), exist_ok=True)
import pickle
out_path = str(PROJECT_ROOT / "data/processed/dataset_labelled.pkl")
with open(out_path, "wb") as f:
    pickle.dump(df, f)

print(f"\nSaved: {out_path}")
print(f"Shape: {df.shape}  (real rows, no sampling)")
print("\n=== Notebook 01 complete. Run notebook 02 next. ===")
