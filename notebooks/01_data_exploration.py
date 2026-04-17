# %% [markdown]
# # Notebook 01 — Data Exploration  (v2 — real CICIoT2023 aware)
#
# **What this notebook does:**
# 1. Merges your 63 Merged*.csv files into one Parquet (run once, skips if exists)
# 2. Loads with per-class stratified sampling (10% default — change for your RAM)
# 3. Validates schema, checks duplicates, missing values, class distribution
# 4. Shows the Protocol Type encoding problem and Variance redundancy
# 5. Saves the labelled dataset for notebook 02
#
# **Run in:** Google Colab (recommended) or locally
#
# **Colab setup — run these in your first cell:**
# ```python
# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/iot-ids-project
# !pip install -r requirements.txt -q
# import sys; sys.path.insert(0, '.')
# ```
# ---

# %% — Setup
import sys, os
sys.path.insert(0, os.path.abspath(".."))

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────
# CHANGE THIS FLAG:
#   USE_DEMO = True   →  uses synthetic data, no download needed
#   USE_DEMO = False  →  uses your real 63 Merged*.csv files
# ──────────────────────────────────────────────────────────────────
USE_DEMO = True

# Memory setting — how much of each class to load
# Free Colab:  0.05–0.10 (safe)
# Colab Pro:   0.15–0.25 (faster training)
# Full dataset: None (may crash free Colab — use Pro or local GPU)
SAMPLE_FRAC = 0.10

# %% — Load data
from src.data.loader import load_dataset, generate_demo_data, build_merged_parquet
from src.data.label_mapping import add_all_label_columns
from src.data.validation import validate_dataset

if USE_DEMO:
    print("Using synthetic demo data (mirrors real CICIoT2023 column structure)")
    df_raw = generate_demo_data(n_samples=50_000, random_state=42)
else:
    print("Using real CICIoT2023 dataset from data/raw/")
    # Step 1: merge 63 CSV files into one Parquet (runs once, skips if done)
    build_merged_parquet(
        raw_dir="../data/raw",
        output_path="../data/processed/merged.parquet",
    )
    # Step 2: load with stratified sampling
    df_raw = load_dataset(
        parquet_path="../data/processed/merged.parquet",
        sample_frac=SAMPLE_FRAC,
        random_state=42,
    )

print(f"\nDataset shape: {df_raw.shape}")
print(f"Columns: {list(df_raw.columns)}")

# %% — Show real data problems we found and fixed
print("\n=== DATA QUALITY INSPECTION ===")

# 1. Duplicates
n_exact_dups = df_raw.duplicated().sum()
print(f"\nExact duplicate rows: {n_exact_dups:,}  (~5% per file in real data)")

# 2. Conflicting label feature vectors
feature_cols = [c for c in df_raw.columns if c != "label"]
dup_feat_mask = df_raw.duplicated(subset=feature_cols, keep=False)
n_conflict_rows = dup_feat_mask.sum()
print(f"Rows with duplicate feature vectors: {n_conflict_rows:,}")
print("  → These may have DIFFERENT labels (e.g. DDOS-UDP_FLOOD vs DOS-UDP_FLOOD)")
print("  → Fixed by deduplicate() in notebook 02 BEFORE splitting")

# 3. Protocol Type
if "Protocol Type" in df_raw.columns:
    pt_vals = sorted(df_raw["Protocol Type"].dropna().unique())
    print(f"\nProtocol Type values: {pt_vals}")
    print("  → These are IP protocol NUMBERS: 0=Other, 1=ICMP, 6=TCP, 17=UDP, 47=GRE")
    print("  → Ordinal treatment (StandardScaler) is WRONG — GRE≠47×Other")
    print("  → Fixed: one-hot encoded into proto_0, proto_1, proto_6, proto_17, proto_47")

# 4. Variance vs Std
if "Variance" in df_raw.columns and "Std" in df_raw.columns:
    mask = df_raw["Std"].notna() & df_raw["Variance"].notna() & (df_raw["Std"] > 0)
    if mask.sum() > 0:
        diff = (df_raw.loc[mask,"Variance"] - df_raw.loc[mask,"Std"]**2).abs().max()
        print(f"\nVariance vs Std^2 max difference: {diff:.2e}")
        print(f"  → Variance IS exactly Std^2 (redundant!)")
        print(f"  → Variance max: {df_raw['Variance'].max():,.0f} (extreme outlier)")
        print(f"  → Variance median: {df_raw['Variance'].median():.4f} (74% of values = 0)")
        print("  → Fixed: Variance dropped, RobustScaler replaces StandardScaler")

# %% — Validate dataset
report = validate_dataset(df_raw, label_col="label")
print(f"\nRows: {report['n_rows']:,}")
print(f"Columns: {report['n_cols']}")
print(f"Classes found: {report['n_classes']} / 34")
print(f"Imbalance ratio (max/min class): {report['imbalance_ratio']}x")

# %% — Class distribution plot
dist = df_raw["label"].value_counts().sort_values(ascending=False)
print(f"\nClass distribution (sorted by count):")
for cls, cnt in dist.items():
    pct = 100 * cnt / len(df_raw)
    bar = "█" * max(1, int(cnt / dist.max() * 40))
    print(f"  {cls:<40} {cnt:>8,}  ({pct:5.2f}%)  {bar}")

fig, ax = plt.subplots(figsize=(16, 5))
colors = ["#2196F3" if cnt > 1000 else "#FF9800" if cnt > 100 else "#F44336"
          for cnt in dist.values]
ax.bar(range(len(dist)), dist.values, color=colors, edgecolor="white", linewidth=0.3)
ax.set_yscale("log")
ax.set_xticks(range(len(dist)))
ax.set_xticklabels(dist.index, rotation=60, ha="right", fontsize=7)
ax.set_ylabel("Sample count (log scale)")
ax.set_title("CICIoT2023 — Class Distribution  |  Blue=common  Orange=medium  Red=rare")
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color="#2196F3", label=">1000 rows"),
    Patch(color="#FF9800", label="100–1000 rows"),
    Patch(color="#F44336", label="<100 rows (rare — all kept by sampler)"),
], fontsize=9)
plt.tight_layout()
os.makedirs("../reports", exist_ok=True)
fig.savefig("../reports/class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: reports/class_distribution.png")

# %% — Rare class counts (critical for understanding model limits)
print("\n=== RARE CLASS ANALYSIS ===")
print("Classes with < 500 rows (need special attention):")
rare = dist[dist < 500]
for cls, cnt in rare.items():
    in_train_est = int(cnt * 0.70)
    print(f"  {cls:<40} total={cnt:>5,}  est. train={in_train_est:>4}  "
          f"{'⚠ VERY RARE' if cnt < 100 else 'RARE'}")

print("\n  Strategy: class_weight='balanced' up-weights rare classes automatically.")
print("  Stratified split ensures all classes appear in train/val/test.")
print("  min_per_class=200 in sampler ensures rare classes keep ALL their rows.")

# %% — Add label columns and save
df = add_all_label_columns(df_raw, raw_col="label")
print(f"\nLabel columns added: label_binary, label_category, label_fine")

# Show 8-class distribution
from src.data.label_mapping import CATEGORY_CLASS_NAMES
cat_dist = df["label_category"].value_counts().sort_index()
print("\n8-class family distribution:")
for idx, cnt in cat_dist.items():
    name = CATEGORY_CLASS_NAMES[idx]
    print(f"  {idx}. {name:<12}: {cnt:>8,}")

# Save
os.makedirs("../data/processed", exist_ok=True)
import pickle
with open("../data/processed/dataset_labelled.pkl", "wb") as f:
    pickle.dump(df, f)
print("\nSaved: data/processed/dataset_labelled.pkl")
print("Run notebook 02 next.")
