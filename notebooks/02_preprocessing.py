# %% [markdown]
# # Notebook 02 -- Deduplication, Splits, Balanced Sampling, Preprocessing
#
# Correct pipeline order:
#   1. Load real deduplicated data from notebook 01
#   2. deduplicate() globally before any split
#   3. create_splits() -- 70/15/15 stratified on REAL rows
#   4. sample_balanced() on TRAIN ONLY (val and test keep real rows forever)
#   5. Fit preprocessor on balanced train
#   6. Transform all three splits
#
# Why this order matters:
#   Oversampling before splitting causes data leakage -- the same duplicate
#   row appears in both train and test, giving inflated metrics.
#   Val and test must always reflect the real-world distribution.
# ---

# %% -- Setup
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import os

import logging, pickle
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

import numpy as np
import pandas as pd
import yaml
import warnings
warnings.filterwarnings("ignore")

USE_DEMO = False

# %% -- Load config
with open(str(PROJECT_ROOT / "configs/data_config.yaml"), encoding="utf-8") as f:
    data_cfg = yaml.safe_load(f)

BALANCED_FLOOR   = data_cfg["dataset"]["balanced_floor"]
BALANCED_CEILING = data_cfg["dataset"]["balanced_ceiling"]
RANDOM_STATE     = data_cfg["dataset"]["random_state"]

print(f"Balanced sampling: floor={BALANCED_FLOOR:,}  ceiling={BALANCED_CEILING:,}")
print(f"Applied to TRAIN ONLY -- val and test keep real rows")

# %% -- Load labelled dataset (output of notebook 01)
if USE_DEMO:
    from src.data.loader import generate_demo_data
    from src.data.label_mapping import add_all_label_columns
    df_raw = generate_demo_data(n_samples=50_000, random_state=RANDOM_STATE)
    df = add_all_label_columns(df_raw, raw_col="label")
else:
    pkl_path = str(PROJECT_ROOT / "data/processed/dataset_labelled.pkl")
    print(f"\nLoading from: {pkl_path}")
    with open(pkl_path, "rb") as f:
        df = pickle.load(f)

print(f"Loaded: {df.shape}  (real rows before dedup)")

# %% -- STEP 1: Deduplicate BEFORE splitting
print("\n=== STEP 1: Deduplication ===")
from src.features.preprocessing import deduplicate

n_before = len(df)
df       = deduplicate(df, label_col="label", random_state=RANDOM_STATE)
n_after  = len(df)
print(f"Before: {n_before:,}  After: {n_after:,}  Removed: {n_before - n_after:,}")

feature_cols = [c for c in df.columns if c not in
                ["label","label_binary","label_category","label_fine","Label"]]
assert df.duplicated(subset=feature_cols).sum() == 0, "Conflicts remain after dedup!"
print("No conflicts remaining. OK")

# %% -- STEP 2: Split on REAL rows first
print("\n=== STEP 2: Stratified Split (on real rows) ===")
from src.data.splitter import create_splits, get_X_y
from src.data.validation import check_split_integrity

os.makedirs(str(PROJECT_ROOT / "data/processed"), exist_ok=True)

train_df, val_df, test_df = create_splits(
    df,
    train_frac=0.70,
    val_frac=0.15,
    test_frac=0.15,
    stratify_col="label_fine",
    random_state=RANDOM_STATE,
    save_path=str(PROJECT_ROOT / "data/processed/split_indices.pkl"),
)

print(f"\nBEFORE balancing:")
print(f"  Train: {len(train_df):,}  (real)")
print(f"  Val:   {len(val_df):,}  (real -- never changes)")
print(f"  Test:  {len(test_df):,}  (real -- never changes)")
check_split_integrity(train_df, val_df, test_df, label_col="label_fine")

# %% -- STEP 3: Balanced sampling on TRAIN ONLY
print("\n=== STEP 3: Balanced sampling on TRAIN only ===")
from src.data.loader import sample_balanced

train_df = sample_balanced(
    train_df,
    label_col="label",
    random_state=RANDOM_STATE,
    floor=BALANCED_FLOOR,
    ceiling=BALANCED_CEILING,
)

print(f"\nAFTER balancing:")
print(f"  Train: {len(train_df):,}  (balanced)")
print(f"  Val:   {len(val_df):,}  (real)")
print(f"  Test:  {len(test_df):,}  (real)")

rare_classes = ["XSS","SQLINJECTION","COMMANDINJECTION","BROWSERHIJACKING",
                "UPLOADING_ATTACK","BACKDOOR_MALWARE","DICTIONARYBRUTEFORCE",
                "RECON-PINGSWEEP"]
print("\nRare class coverage after balancing:")
for cls in rare_classes:
    from src.data.label_mapping import FINE_MAP
    idx = FINE_MAP.get(cls)
    if idx is None: continue
    in_train = (train_df["label_fine"] == idx).sum()
    in_val   = (val_df["label_fine"]   == idx).sum()
    in_test  = (test_df["label_fine"]  == idx).sum()
    ok = "OK" if in_train >= BALANCED_FLOOR else "LOW"
    print(f"  {cls:<35}: train={in_train:>5,}[{ok}]  val={in_val:>4}  test={in_test:>4}")

# %% -- Extract X, y for all tasks
print("\n=== Extracting X, y ===")
X_train_raw, y_train_binary  = get_X_y(train_df, "binary")
X_val_raw,   y_val_binary    = get_X_y(val_df,   "binary")
X_test_raw,  y_test_binary   = get_X_y(test_df,  "binary")

_, y_train_8class  = get_X_y(train_df, "8class")
_, y_val_8class    = get_X_y(val_df,   "8class")
_, y_test_8class   = get_X_y(test_df,  "8class")

_, y_train_34class = get_X_y(train_df, "34class")
_, y_val_34class   = get_X_y(val_df,   "34class")
_, y_test_34class  = get_X_y(test_df,  "34class")

print(f"X_train (balanced): {X_train_raw.shape}")
print(f"X_val   (real):     {X_val_raw.shape}")
print(f"X_test  (real):     {X_test_raw.shape}")

# %% -- STEP 4: Fit preprocessor on balanced train only
print("\n=== STEP 4: Fitting Preprocessor ===")
print("Pipeline: one-hot Protocol Type -> drop Variance -> impute ->")
print("          Yeo-Johnson (skewed features) -> RobustScaler -> clip[-20,20]")
from src.features.preprocessing import build_and_fit_preprocessor

os.makedirs(str(PROJECT_ROOT / "models"), exist_ok=True)
preprocessor = build_and_fit_preprocessor(
    X_train_raw,
    save_path=str(PROJECT_ROOT / "models/preprocessor.pkl"),
)

proto_cols  = [n for n in preprocessor.get_feature_names() if n.startswith("proto_")]
skewed_cols = preprocessor.get_skewed_features()
print(f"\nInput features:          {X_train_raw.shape[1]}")
print(f"Output features:         {len(preprocessor.get_feature_names())}")
print(f"Protocol one-hot cols:   {proto_cols}")
print(f"Yeo-Johnson applied to:  {len(skewed_cols)} features")
if skewed_cols:
    for c in skewed_cols:
        print(f"  {c}")

assert len(proto_cols) == 5, "Protocol Type not one-hot encoded!"
assert "Variance" not in preprocessor.get_feature_names(), "Variance still present!"
print("Protocol Type one-hot: OK")
print("Variance removed:      OK")

# %% -- Apply preprocessor to all splits
X_train = preprocessor.transform(X_train_raw)
X_val   = preprocessor.transform(X_val_raw)
X_test  = preprocessor.transform(X_test_raw)

print(f"\nX_train scaled: {X_train.shape}")
print(f"X_val   scaled: {X_val.shape}")
print(f"X_test  scaled: {X_test.shape}")

assert not np.isnan(X_train).any(), "NaN in train!"
assert not np.isinf(X_train).any(), "Inf in train!"
assert not np.isnan(X_val).any(),   "NaN in val!"
assert not np.isnan(X_test).any(),  "NaN in test!"
print("No NaN or Inf in any split. OK")

# %% -- Save everything
print("\n=== Saving ===")
from src.models.model_utils import save_class_names
from src.data.label_mapping import BINARY_CLASS_NAMES, CATEGORY_CLASS_NAMES, FINE_CLASS_NAMES

splits_data = {
    "X_train":          X_train,
    "X_val":            X_val,
    "X_test":           X_test,
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
    "X_test_raw":       X_test_raw,
}

splits_path = str(PROJECT_ROOT / "data/processed/splits.pkl")
with open(splits_path, "wb") as f:
    pickle.dump(splits_data, f)

save_class_names(BINARY_CLASS_NAMES,   str(PROJECT_ROOT / "models/class_names_binary.json"))
save_class_names(CATEGORY_CLASS_NAMES, str(PROJECT_ROOT / "models/class_names_8class.json"))
save_class_names(FINE_CLASS_NAMES,     str(PROJECT_ROOT / "models/class_names_34class.json"))

print(f"Saved: {splits_path}")
print(f"Train size: {X_train.shape[0]:,} rows (balanced)")
print(f"Val size:   {X_val.shape[0]:,} rows (real)")
print(f"Test size:  {X_test.shape[0]:,} rows (real)")
print("\n=== Notebook 02 complete. Run notebook 03 next. ===")
