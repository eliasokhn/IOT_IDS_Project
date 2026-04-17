"""
splitter.py
===========
Create reproducible train / validation / test splits.

KEY RULE: Split ONCE on fine-grained (34-class) labels, then reuse
the same row assignments for binary and 8-class experiments.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


def create_splits(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    stratify_col: str = "label_fine",
    random_state: int = 42,
    save_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train / val / test using stratified sampling on
    the 34-class label column.

    All three binary/8-class/34-class tasks reuse EXACTLY these row assignments.

    Parameters
    ----------
    df            : dataframe with label columns already applied
    train_frac    : fraction for training (0.70)
    val_frac      : fraction for validation (0.15)
    test_frac     : fraction for test (0.15)
    stratify_col  : column to stratify on (must be 'label_fine' for correctness)
    random_state  : reproducibility
    save_path     : optional path to save split indices as pickle

    Returns
    -------
    (train_df, val_df, test_df)
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        "Split fractions must sum to 1.0"
    assert stratify_col in df.columns, \
        f"Stratify column '{stratify_col}' not found in dataframe."

    log.info(
        f"Creating stratified split: "
        f"{train_frac*100:.0f}% / {val_frac*100:.0f}% / {test_frac*100:.0f}%"
    )

    # Step 1: Split off test set
    remaining_frac = val_frac + test_frac
    train_idx, temp_idx = train_test_split(
        df.index,
        test_size=remaining_frac,
        stratify=df[stratify_col],
        random_state=random_state,
    )

    # Step 2: Split remaining into val and test
    val_relative = val_frac / remaining_frac
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1.0 - val_relative,
        stratify=df.loc[temp_idx, stratify_col],
        random_state=random_state,
    )

    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)

    log.info(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # Log per-class distribution in each split
    _log_split_distribution(train_df, val_df, test_df, stratify_col)

    # Save indices for reproducibility
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(
                {
                    "train_idx": list(train_idx),
                    "val_idx": list(val_idx),
                    "test_idx": list(test_idx),
                    "random_state": random_state,
                    "stratify_col": stratify_col,
                },
                f,
            )
        log.info(f"Split indices saved to {save_path}")

    return train_df, val_df, test_df


def load_splits_from_indices(
    df: pd.DataFrame,
    indices_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reconstruct exact same splits from saved indices.
    Use this to guarantee reproducibility across team members.
    """
    with open(indices_path, "rb") as f:
        saved = pickle.load(f)

    train_df = df.loc[saved["train_idx"]].reset_index(drop=True)
    val_df = df.loc[saved["val_idx"]].reset_index(drop=True)
    test_df = df.loc[saved["test_idx"]].reset_index(drop=True)

    log.info(
        f"Loaded splits from {indices_path}: "
        f"Train={len(train_df):,} Val={len(val_df):,} Test={len(test_df):,}"
    )
    return train_df, val_df, test_df


def _log_split_distribution(train, val, test, label_col: str) -> None:
    """Log per-class counts across splits to verify stratification."""
    all_classes = sorted(train[label_col].unique())
    log.info(f"\nClass distribution per split (top 5 classes shown):")
    log.info(f"  {'Class':<40} {'Train':>8} {'Val':>8} {'Test':>8}")
    log.info(f"  {'-'*64}")

    train_counts = train[label_col].value_counts()
    val_counts = val[label_col].value_counts()
    test_counts = test[label_col].value_counts()

    for cls in all_classes[:5]:
        tr = train_counts.get(cls, 0)
        v = val_counts.get(cls, 0)
        te = test_counts.get(cls, 0)
        log.info(f"  {str(cls):<40} {tr:>8,} {v:>8,} {te:>8,}")

    if len(all_classes) > 5:
        log.info(f"  ... ({len(all_classes) - 5} more classes)")


def get_X_y(
    df: pd.DataFrame,
    task: str,
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix X and label vector y for a given task.

    Parameters
    ----------
    df        : dataframe with all label columns
    task      : one of 'binary', '8class', '34class'
    drop_cols : additional columns to drop (non-feature columns)

    Returns
    -------
    (X, y) — X is a float DataFrame, y is an integer Series
    """
    from src.data.label_mapping import get_label_col

    label_col = get_label_col(task)
    y = df[label_col].astype(int)

    # Drop all label columns + any specified extras
    all_label_cols = ["label", "label_binary", "label_category", "label_fine"]
    cols_to_drop = set(all_label_cols)
    if drop_cols:
        cols_to_drop.update(drop_cols)

    feature_cols = [c for c in df.columns if c not in cols_to_drop]
    X = df[feature_cols].copy()

    # Ensure all features are numeric
    X = X.select_dtypes(include=[np.number])

    return X, y


import numpy as np  # noqa: F811 (already imported above, kept for clarity)
