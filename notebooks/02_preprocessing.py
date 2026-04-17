# %% [markdown]
# # Notebook 02 — Deduplication, Preprocessing & Splits  (v2 — fixed)
#
# **Three fixes applied in this notebook:**
#
# **FIX 1 — DUPLICATES (deduplicate BEFORE splitting)**
# - ~5% of rows per file are exact duplicates
# - 997 feature vectors map to two different labels (e.g. DDOS-UDP_FLOOD vs DOS-UDP_FLOOD)
# - Duplicates in the test set inflate binary accuracy and confuse 34-class learning
# - We remove them GLOBALLY before the split so no identical row appears
#   in both train and test
#
# **FIX 2 — PROTOCOL TYPE (one-hot instead of ordinal)**
# - Protocol Type values are {0,1,6,17,47} = IP protocol numbers
# - Treating them as numeric implies GRE(47) > UDP(17) > TCP(6) — meaningless
# - We one-hot encode into 5 binary columns: proto_0, proto_1, proto_6, proto_17, proto_47
#
# **FIX 3 — VARIANCE + SCALER**
# - Variance = Std^2 exactly — pure redundancy, extreme outlier source
# - Dropped. RobustScaler (median+IQR) replaces StandardScaler for outlier resilience
#
# **KEY RULE (unchanged):**
# Split ONCE on fine-grained (34-class) labels. Reuse same rows for binary and 8-class.
# ---

# %% — Setup
import sys, os
sys.path.insert(0, os.path.abspath(".."))

import logging, json, pickle
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# %% — Choose data source
USE_DEMO = True   # ← Set to False when using real CICIoT2023 data

# %% — Load labelled dataset (output of notebook 01)
if USE_DEMO:
    from src.data.loader import generate_demo_data
    from src.data.label_mapping import add_all_label_columns
    df_raw = generate_demo_data(n_samples=50_000, random_state=42)
    df = add_all_label_columns(df_raw, raw_col="label")
else:
    with open("../data/processed/dataset_labelled.pkl", "rb") as f:
        df = pickle.load(f)

print(f"Loaded: {df.shape}")
print(f"Columns: {list(df.columns)}")

# %% ─────────────────────────────────────────────────────────────
# FIX 1 — DEDUPLICATE BEFORE SPLITTING
# ─────────────────────────────────────────────────────────────────
print("\n=== FIX 1: Deduplication ===")
from src.features.preprocessing import deduplicate

n_before = len(df)
df = deduplicate(df, label_col="label", random_state=42)
n_after = len(df)
print(f"Rows before: {n_before:,}  |  Rows after: {n_after:,}  |  Removed: {n_before - n_after:,}")

# Verify: no duplicate feature vectors remain
feature_cols = [c for c in df.columns if c not in ["label","label_binary","label_category","label_fine","Label"]]
n_conflicts = df.duplicated(subset=feature_cols).sum()
print(f"Remaining conflicting feature vectors: {n_conflicts}  (should be 0)")

# %% — Check label distribution after dedup (rare classes still present?)
print("\nLabel distribution after dedup:")
dist = df["label"].value_counts()
for label, cnt in dist.items():
    bar = "█" * min(int(cnt / dist.max() * 30), 30)
    print(f"  {label:<40} {cnt:>8,}  {bar}")

# %% ─────────────────────────────────────────────────────────────
# SPLIT ONCE on 34-class labels
# ─────────────────────────────────────────────────────────────────
print("\n=== Creating 70/15/15 Stratified Split ===")
from src.data.splitter import create_splits, get_X_y
from src.data.validation import check_split_integrity

train_df, val_df, test_df = create_splits(
    df,
    train_frac=0.70,
    val_frac=0.15,
    test_frac=0.15,
    stratify_col="label_fine",
    random_state=42,
    save_path="../data/processed/split_indices.pkl",
)

print(f"\nTrain: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")
check_split_integrity(train_df, val_df, test_df, label_col="label_fine")

# Verify rare classes appear in ALL splits
print("\nRare class coverage:")
rare_classes = ["XSS", "SQLINJECTION", "COMMANDINJECTION",
                "BROWSERHIJACKING", "UPLOADING_ATTACK",
                "BACKDOOR_MALWARE", "DICTIONARYBRUTEFORCE", "RECON-PINGSWEEP"]
fine_map = df.set_index("label")["label_fine"].to_dict()

for cls in rare_classes:
    from src.data.label_mapping import FINE_MAP
    idx = FINE_MAP.get(cls)
    if idx is None:
        continue
    in_train = (train_df["label_fine"] == idx).sum()
    in_val   = (val_df["label_fine"]   == idx).sum()
    in_test  = (test_df["label_fine"]  == idx).sum()
    status = "✓" if (in_train > 0 and in_val > 0 and in_test > 0) else "⚠ MISSING in some split"
    print(f"  {cls:<35}: train={in_train:>4} val={in_val:>4} test={in_test:>4}  {status}")

# %% — Extract features and labels for each task
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
print(f"Columns include 'Protocol Type': {'Protocol Type' in X_train_raw.columns}")
print(f"Columns include 'Variance': {'Variance' in X_train_raw.columns}")

# %% ─────────────────────────────────────────────────────────────
# FIX 2 + FIX 3 — Fit Preprocessor on TRAINING data only
# (Protocol Type one-hot + Variance drop + RobustScaler)
# ─────────────────────────────────────────────────────────────────
print("\n=== FIX 2+3: Fitting Preprocessor (one-hot + RobustScaler) ===")
from src.features.preprocessing import build_and_fit_preprocessor

os.makedirs("../models", exist_ok=True)
preprocessor = build_and_fit_preprocessor(
    X_train_raw,
    save_path="../models/preprocessor.pkl",
)

print(f"Input features:  {X_train_raw.shape[1]}")
print(f"Output features: {len(preprocessor.get_feature_names())}")
print(f"Features after preprocessing:")
for name in preprocessor.get_feature_names():
    proto = " ← one-hot" if name.startswith("proto_") else ""
    print(f"  {name}{proto}")

# %% — Apply preprocessor to all splits
X_train = preprocessor.transform(X_train_raw)
X_val   = preprocessor.transform(X_val_raw)
X_test  = preprocessor.transform(X_test_raw)

print(f"\nScaled train: {X_train.shape}  mean={X_train.mean():.4f}  std={X_train.std():.4f}")
print(f"Scaled val:   {X_val.shape}")
print(f"Scaled test:  {X_test.shape}")

# Sanity checks
assert not np.isnan(X_train).any(), "NaN in train after preprocessing!"
assert not np.isinf(X_train).any(), "Inf in train after preprocessing!"
print("No NaN or Inf after preprocessing. ✓")

# Check Protocol Type columns are present in output
proto_cols = [n for n in preprocessor.get_feature_names() if n.startswith("proto_")]
print(f"Protocol Type one-hot columns: {proto_cols}")
assert len(proto_cols) > 0, "ERROR: Protocol Type was not one-hot encoded!"
print("Protocol Type one-hot encoding: ✓")

# Check Variance is gone
assert "Variance" not in preprocessor.get_feature_names(), "ERROR: Variance still present!"
print("Variance column removed: ✓")

# %% — Save all split arrays
print("\n=== Saving splits to disk ===")
from src.models.model_utils import save_class_names
from src.data.label_mapping import BINARY_CLASS_NAMES, CATEGORY_CLASS_NAMES, FINE_CLASS_NAMES

splits_data = {
    "X_train": X_train,
    "X_val":   X_val,
    "X_test":  X_test,
    "y_train_binary":   y_train_binary.values,
    "y_val_binary":     y_val_binary.values,
    "y_test_binary":    y_test_binary.values,
    "y_train_8class":   y_train_8class.values,
    "y_val_8class":     y_val_8class.values,
    "y_test_8class":    y_test_8class.values,
    "y_train_34class":  y_train_34class.values,
    "y_val_34class":    y_val_34class.values,
    "y_test_34class":   y_test_34class.values,
    "feature_names":    preprocessor.get_feature_names(),
    "X_test_raw":       X_test_raw,   # unscaled, for streaming simulation
}

os.makedirs("../data/processed", exist_ok=True)
with open("../data/processed/splits.pkl", "wb") as f:
    pickle.dump(splits_data, f)

save_class_names(BINARY_CLASS_NAMES,   "../models/class_names_binary.json")
save_class_names(CATEGORY_CLASS_NAMES, "../models/class_names_8class.json")
save_class_names(FINE_CLASS_NAMES,     "../models/class_names_34class.json")

print("Saved: data/processed/splits.pkl")
print("Saved: models/preprocessor.pkl")
print("Saved: models/class_names_*.json")
print("\n=== Notebook 02 complete. Run notebook 03 next. ===")
