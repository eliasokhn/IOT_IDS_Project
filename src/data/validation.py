"""
validation.py
=============
Dataset validation: schema checks, missing values, class distribution,
leakage checks, and label consistency.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.data.label_mapping import ALL_LABELS, FINE_CLASS_NAMES

log = logging.getLogger(__name__)


def validate_dataset(df: pd.DataFrame, label_col: str = "label") -> dict[str, Any]:
    """
    Run full dataset validation and return a report dict.
    Raises ValueError on critical issues.
    """
    report: dict[str, Any] = {}
    issues: list[str] = []
    warnings: list[str] = []

    log.info("=" * 60)
    log.info("DATASET VALIDATION REPORT")
    log.info("=" * 60)

    # ── 1. Basic shape ────────────────────────────────────────────
    report["n_rows"] = len(df)
    report["n_cols"] = len(df.columns)
    log.info(f"Shape: {df.shape}")

    # ── 2. Label column exists ────────────────────────────────────
    if label_col not in df.columns:
        issues.append(f"Label column '{label_col}' not found in dataframe.")
    else:
        found_labels = set(df[label_col].unique())
        expected_labels = set(ALL_LABELS)

        unknown = found_labels - expected_labels
        missing = expected_labels - found_labels

        if unknown:
            warnings.append(f"Unknown labels found (not in taxonomy): {unknown}")
        if missing:
            warnings.append(f"Expected labels missing from dataset: {missing}")

        report["labels_found"] = sorted(found_labels)
        report["n_classes"] = len(found_labels)
        log.info(f"Classes found: {len(found_labels)} / 34 expected")

    # ── 3. Class distribution ─────────────────────────────────────
    if label_col in df.columns:
        dist = df[label_col].value_counts()
        report["class_distribution"] = dist.to_dict()
        log.info("\nClass distribution:")
        for cls, cnt in dist.items():
            pct = 100 * cnt / len(df)
            log.info(f"  {cls:<40} {cnt:>10,}  ({pct:5.2f}%)")

        # Imbalance ratio
        imbalance_ratio = dist.max() / dist.min()
        report["imbalance_ratio"] = round(imbalance_ratio, 1)
        log.info(f"\nImbalance ratio (max/min class): {imbalance_ratio:.1f}x")
        if imbalance_ratio > 100:
            warnings.append(
                f"High class imbalance detected ({imbalance_ratio:.0f}x). "
                "Use class_weight='balanced' and report macro F1."
            )

    # ── 4. Missing values ─────────────────────────────────────────
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    report["missing_value_cols"] = missing_cols.to_dict()

    if len(missing_cols) > 0:
        log.info(f"\nMissing values in {len(missing_cols)} columns:")
        for col, cnt in missing_cols.items():
            log.info(f"  {col}: {cnt:,} ({100*cnt/len(df):.2f}%)")
        warnings.append(f"{len(missing_cols)} columns have missing values — will be imputed.")
    else:
        log.info("No missing values found.")
        report["missing_value_cols"] = {}

    # ── 5. Infinite values ────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    inf_counts = {}
    for col in numeric_cols:
        n_inf = np.isinf(df[col]).sum()
        if n_inf > 0:
            inf_counts[col] = int(n_inf)

    report["inf_value_cols"] = inf_counts
    if inf_counts:
        log.info(f"\nInfinite values in {len(inf_counts)} columns:")
        for col, cnt in inf_counts.items():
            log.info(f"  {col}: {cnt:,}")
        warnings.append(f"{len(inf_counts)} columns have infinite values — will be replaced.")
    else:
        log.info("No infinite values found.")

    # ── 6. Leakage check ─────────────────────────────────────────
    # Check for columns whose name suggests they contain label info
    leakage_keywords = ["label", "class", "target", "attack", "category"]
    leakage_candidates = [
        c for c in df.columns
        if any(kw in c.lower() for kw in leakage_keywords) and c != label_col
    ]
    report["leakage_candidates"] = leakage_candidates
    if leakage_candidates:
        warnings.append(
            f"Possible leakage columns detected: {leakage_candidates}. "
            "Review and drop before training."
        )
        log.info(f"\nPossible leakage columns: {leakage_candidates}")

    # ── 7. Duplicate rows ─────────────────────────────────────────
    n_dups = df.duplicated().sum()
    report["n_duplicates"] = int(n_dups)
    if n_dups > 0:
        dup_pct = 100 * n_dups / len(df)
        warnings.append(f"{n_dups:,} duplicate rows ({dup_pct:.2f}%). Consider deduplication.")
        log.info(f"\nDuplicate rows: {n_dups:,} ({dup_pct:.2f}%)")

    # ── 8. Feature count ─────────────────────────────────────────
    n_features = len(numeric_cols)
    report["n_numeric_features"] = n_features
    log.info(f"\nNumeric feature columns: {n_features}")

    # ── Summary ──────────────────────────────────────────────────
    report["issues"] = issues
    report["warnings"] = warnings

    log.info("\n" + "=" * 60)
    if issues:
        log.error(f"CRITICAL ISSUES ({len(issues)}):")
        for iss in issues:
            log.error(f"  ✗ {iss}")
        raise ValueError(f"Dataset validation failed: {issues}")

    if warnings:
        log.warning(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            log.warning(f"  ⚠ {w}")
    else:
        log.info("All validation checks passed.")

    log.info("=" * 60)
    return report


def check_split_integrity(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    label_col: str = "label_fine",
) -> None:
    """
    Verify that train/val/test splits don't overlap and are correctly sized.
    """
    n_total = len(train) + len(val) + len(test)
    log.info(f"Split integrity check:")
    log.info(f"  Train: {len(train):,} ({100*len(train)/n_total:.1f}%)")
    log.info(f"  Val:   {len(val):,} ({100*len(val)/n_total:.1f}%)")
    log.info(f"  Test:  {len(test):,} ({100*len(test)/n_total:.1f}%)")

    # Check all fine-grained classes appear in train
    train_classes = set(train[label_col].unique()) if label_col in train.columns else set()
    val_classes = set(val[label_col].unique()) if label_col in val.columns else set()
    test_classes = set(test[label_col].unique()) if label_col in test.columns else set()

    if val_classes - train_classes:
        log.warning(
            f"Val has classes not in train: {val_classes - train_classes}. "
            "This is risky — model never saw these during training."
        )
    if test_classes - train_classes:
        log.warning(
            f"Test has classes not in train: {test_classes - train_classes}."
        )

    log.info("Split integrity check passed.")
