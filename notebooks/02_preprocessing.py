# %% [markdown]
# # Notebook 02 — Deduplication, Preprocessing & Splits
# ---

# %% — Setup
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import os

import logging, json, pickle
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────
# SET USE_DEMO = False  to use your real CICIoT2023 data
# SET USE_DEMO = True   to run quickly with synthetic data
# ──────────────────────────────────────────────────────
USE_DEMO = False

# %% — Load labelled dataset (output of notebook 01)
if USE_DEMO:
    from src.data.loader import generate_demo_data
    from src.data.label_mapping import add_all_label_columns
    df_raw = generate_demo_data(n_samples=50_000, random_state=42)
    df = add_all_label_columns(df_raw, raw_col="label")
else:
    pkl_path = str(PROJECT_ROOT / "data/processed/dataset_labelled.pkl")
    print(f"Loading from: {pkl_path}")
    with open(pkl_path, "rb") as f:
        df = pickle.load(f)

print(f"Loaded: {df.shape}")
print(f"Columns: {list(df.columns)}")

# %% — FIX 1: Deduplicate BEFORE splitting
print("\n=== FIX 1: Deduplication ===")
from src.features.preprocessing import deduplicate

n_before = len(df)
df = deduplicate(df, label_col="label", random_state=42)
n_after = len(df)
print(f"Rows before: {n_before:,}  |  Rows after: {n_after:,}  |  Removed: {n_before - n_after:,}")

feature_cols = [c for c in df.columns if c not in
                ["label", "label_binary", "label_category", "label_fine", "Label"]]
n_conflicts = df.duplicated(subset=feature_cols).sum()
print(f"Remaining conflicting feature vectors: {n_conflicts}  (should be 0)")

# %% — Label distribution after dedup
print("\nLabel distribution after dedup:")
dist = df["label"].value_counts()
for label, cnt in dist.items():
    bar = "█" * min(int(cnt / dist.max() * 30), 30)
    print(f"  {label:<40} {cnt:>8,}  {bar}")

# %% — Split ONCE on 34-class labels
print("\n=== Creating 70/15/15 Stratified Split ===")
from src.data.splitter import create_splits, get_X_y
from src.data.validation import check_split_integrity

os.makedirs(str(PROJECT_ROOT / "data/processed"), exist_ok=True)

train_df, val_df, test_df = create_splits(
    df,
    train_frac=0.70,
    val_frac=0.15,
    test_frac=0.15,
    stratify_col="label_fine",
    random_state=42,
    save_path=str(PROJECT_ROOT / "data/processed/split_indices.pkl"),
)

print(f"\nTrain: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")
check_split_integrity(train_df, val_df, test_df, label_col="label_fine")

# %% — Verify rare classes appear in ALL splits
print("\nRare class coverage:")
rare_classes = ["XSS", "SQLINJECTION", "COMMANDINJECTION",
                "BROWSERHIJACKING", "UPLOADING_ATTACK",
                "BACKDOOR_MALWARE", "DICTIONARYBRUTEFORCE", "RECON-PINGSWEEP"]
for cls in rare_classes:
    from src.data.label_mapping import FINE_MAP
    idx = FINE_MAP.get(cls)
    if idx is None:
        continue
    in_train = (train_df["label_fine"] == idx).sum()
    in_val   = (val_df["label_fine"]   == idx).sum()
    in_test  = (test_df["label_fine"]  == idx).sum()
    status = "✓" if (in_train > 0 and in_val > 0 and in_test > 0) else "⚠ MISSING"
    print(f"  {cls:<35}: train={in_train:>4} val={in_val:>4} test={in_test:>4}  {status}")

# %% — Extract X, y for all tasks
print("\n=== Extracting X, y for all tasks ===")
X_train_raw, y_train_binary  = get_X_y(train_df, "binary")
X_val_raw,   y_val_binary    = get_X_y(val_df,   "binary")
X_test_raw,  y_test_binary   = get_X_y(test_df,  "binary")

_, y_train_8class  = get_X_y(train_df, "8class")
_, y_val_8class    = get_X_y(val_df,   "8class")
_, y_test_8class   = get_X_y(test_df,  "8class")

_, y_train_34class = get_X_y(train_df, "34class")
_, y_val_34class   = get_X_y(val_df,   "34class")
_, y_test_34class  = get_X_y(test_df,  "34class")

print(f"Raw feature matrix: {X_train_raw.shape}")
print(f"Protocol Type in columns: {'Protocol Type' in X_train_raw.columns}")
print(f"Variance in columns:      {'Variance' in X_train_raw.columns}")

# %% — FIX 2+3: Fit preprocessor (one-hot Protocol Type + drop Variance + RobustScaler)
print("\n=== FIX 2+3: Fitting Preprocessor ===")
from src.features.preprocessing import build_and_fit_preprocessor

os.makedirs(str(PROJECT_ROOT / "models"), exist_ok=True)
preprocessor = build_and_fit_preprocessor(
    X_train_raw,
    save_path=str(PROJECT_ROOT / "models/preprocessor.pkl"),
)

print(f"Input features:  {X_train_raw.shape[1]}")
print(f"Output features: {len(preprocessor.get_feature_names())}")
proto_cols = [n for n in preprocessor.get_feature_names() if n.startswith("proto_")]
print(f"Protocol one-hot columns: {proto_cols}")
assert len(proto_cols) > 0, "ERROR: Protocol Type not one-hot encoded!"
assert "Variance" not in preprocessor.get_feature_names(), "ERROR: Variance still present!"
print("Protocol Type one-hot: ✓")
print("Variance removed:      ✓")

# %% — Apply preprocessor to all splits
X_train = preprocessor.transform(X_train_raw)
X_val   = preprocessor.transform(X_val_raw)
X_test  = preprocessor.transform(X_test_raw)

print(f"\nTrain scaled: {X_train.shape}")
print(f"Val   scaled: {X_val.shape}")
print(f"Test  scaled: {X_test.shape}")

assert not np.isnan(X_train).any(), "NaN in train!"
assert not np.isinf(X_train).any(), "Inf in train!"
print("No NaN or Inf. ✓")

# %% — Save all splits and class name files
print("\n=== Saving splits ===")
from src.models.model_utils import save_class_names
from src.data.label_mapping import BINARY_CLASS_NAMES, CATEGORY_CLASS_NAMES, FINE_CLASS_NAMES

splits_data = {
    "X_train":         X_train,
    "X_val":           X_val,
    "X_test":          X_test,
    "y_train_binary":  y_train_binary.values,
    "y_val_binary":    y_val_binary.values,
    "y_test_binary":   y_test_binary.values,
    "y_train_8class":  y_train_8class.values,
    "y_val_8class":    y_val_8class.values,
    "y_test_8class":   y_test_8class.values,
    "y_train_34class": y_train_34class.values,
    "y_val_34class":   y_val_34class.values,
    "y_test_34class":  y_test_34class.values,
    "feature_names":   preprocessor.get_feature_names(),
    "X_test_raw":      X_test_raw,
}

splits_path = str(PROJECT_ROOT / "data/processed/splits.pkl")
with open(splits_path, "wb") as f:
    pickle.dump(splits_data, f)
print(f"Saved: {splits_path}")

save_class_names(BINARY_CLASS_NAMES,   str(PROJECT_ROOT / "models/class_names_binary.json"))
save_class_names(CATEGORY_CLASS_NAMES, str(PROJECT_ROOT / "models/class_names_8class.json"))
save_class_names(FINE_CLASS_NAMES,     str(PROJECT_ROOT / "models/class_names_34class.json"))

print(f"Saved: {str(PROJECT_ROOT / 'models/preprocessor.pkl')}")
print(f"Saved: {str(PROJECT_ROOT / 'models/class_names_*.json')}")
print("\n=== Notebook 02 complete. Run notebook 03 next. ===")