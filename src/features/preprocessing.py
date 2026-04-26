"""
preprocessing.py
================
Feature preprocessing pipeline for IoT IDS.

Pipeline (fit on training data only):
  - Drop Variance column (redundant: Variance = Std^2 exactly)
  - One-hot encode Protocol Type (values are nominal IP protocol IDs, not ordinal)
  - Replace inf values, median-impute NaN
  - PowerTransformer (Yeo-Johnson) on skewed continuous features
  - RobustScaler on all features (median+IQR, robust to outliers)
  - Clip output to [-20, 20]
"""

import hashlib
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, RobustScaler

log = logging.getLogger(__name__)

PROTOCOL_TYPE_COL = "Protocol Type"
VARIANCE_COL      = "Variance"
ALL_LABEL_COLS    = ["label", "label_binary", "label_category",
                     "label_fine", "Label", "_feat_hash"]
CLIP_AFTER_SCALE  = 20.0

# Yeo-Johnson thresholds — only apply to features that pass ALL three gates
# (same logic as the external pipeline we reviewed)
SKEW_THRESHOLD          = 2.0   # |skew| must exceed this
SKEW_MIN_UNIQUE         = 50    # must have at least this many unique values
SKEW_MIN_NONZERO_FRAC   = 0.05  # at least 5% of values must be non-zero


def _detect_skewed_features(
    X: pd.DataFrame,
    exclude: list[str] | None = None,
) -> list[str]:
    """
    Identify genuinely skewed continuous features that will benefit from
    Yeo-Johnson power transform.

    Three gates (all must pass):
      1. |skew| > SKEW_THRESHOLD        — heavy tail exists
      2. nunique >= SKEW_MIN_UNIQUE      — enough distinct values (not binary/sparse)
      3. nonzero fraction >= SKEW_MIN_NONZERO_FRAC — not a sparse indicator

    Columns like SSH, Telnet, ece_flag_number are mostly zero — they register
    huge mathematical skew but Yeo-Johnson cannot normalize them and applying
    it just wastes time.
    """
    exclude = set(exclude or [])
    skewed = []
    for col in X.columns:
        if col in exclude:
            continue
        series = X[col].dropna()
        if len(series) == 0:
            continue
        skewness      = float(series.skew())
        n_unique      = series.nunique()
        nonzero_frac  = float((series != 0).mean())
        if (abs(skewness) > SKEW_THRESHOLD
                and n_unique >= SKEW_MIN_UNIQUE
                and nonzero_frac >= SKEW_MIN_NONZERO_FRAC):
            skewed.append(col)
    return skewed


class Preprocessor:
    """
    Stateful preprocessor. Fit on training data only, transform any split.

    Pipeline:
      1. Drop Variance (= Std^2, redundant + extreme outlier)
      2. One-hot encode Protocol Type (categorical, not ordinal)
      3. Replace +/-inf with column finite-max
      4. Median-impute NaN
      5. Yeo-Johnson PowerTransformer on skewed continuous features only
      6. RobustScaler (median + IQR) on all features
      7. Clip to [-20, 20]

    Save/load with save()/load() to guarantee identical preprocessing
    at training time and inference time.
    """

    def __init__(self):
        self.scaler:            RobustScaler          = RobustScaler()
        self.power_transformer: PowerTransformer | None = None
        self.skewed_features:   list[str]             = []
        self.col_medians:       dict[str, float]      = {}
        self.col_maxes:         dict[str, float]      = {}
        self.protocol_values:   list[int]             = []
        self.feature_names:     list[str]             = []
        self._fitted:           bool                  = False

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        """Fit all preprocessing steps on training data only."""
        log.info(f"Fitting preprocessor on {X.shape[0]:,} rows, "
                 f"{X.shape[1]} raw features.")

        X_prep = self._structural_fixes(X, fit_mode=True)
        self.feature_names = list(X_prep.columns)

        # Compute finite column maxes for inf-replacement
        for col in self.feature_names:
            finite = X_prep[col].replace([np.inf, -np.inf], np.nan).dropna()
            self.col_maxes[col] = float(finite.max()) if len(finite) else 0.0

        X_no_inf = self._replace_inf(X_prep)

        # Compute medians for NaN imputation
        for col in self.feature_names:
            self.col_medians[col] = float(X_no_inf[col].median())

        X_imputed = self._impute(X_no_inf)

        # FIX 5 — Detect skewed features and fit Yeo-Johnson
        # Proto one-hot columns are excluded (binary, not continuous)
        proto_cols = [c for c in self.feature_names if c.startswith("proto_")]
        self.skewed_features = _detect_skewed_features(
            X_imputed, exclude=proto_cols
        )

        if self.skewed_features:
            log.info(f"Yeo-Johnson will be applied to {len(self.skewed_features)} "
                     f"skewed features: {self.skewed_features}")
            self.power_transformer = PowerTransformer(
                method="yeo-johnson", standardize=False
            )
            self.power_transformer.fit(X_imputed[self.skewed_features])
        else:
            log.info("No skewed features detected — Yeo-Johnson skipped.")
            self.power_transformer = None

        # Fit RobustScaler on fully prepared data
        X_transformed = self._apply_power_transform(X_imputed)
        self.scaler.fit(X_transformed)
        self._fitted = True

        log.info(
            f"Preprocessor fitted. Output features: {len(self.feature_names)}  "
            f"(Protocol Type one-hot, Variance dropped, "
            f"Yeo-Johnson on {len(self.skewed_features)} features, "
            f"RobustScaler, clip [{-CLIP_AFTER_SCALE}, {CLIP_AFTER_SCALE}])."
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply fitted preprocessing. Returns numpy array for sklearn."""
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        X_prep        = self._structural_fixes(X, fit_mode=False)
        X_no_inf      = self._replace_inf(X_prep)
        X_imputed     = self._impute(X_no_inf)
        X_transformed = self._apply_power_transform(X_imputed)
        scaled        = self.scaler.transform(X_transformed)
        return np.clip(scaled, -CLIP_AFTER_SCALE, CLIP_AFTER_SCALE)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step. Use on training data only."""
        self.fit(X)
        return self.transform(X)

    def get_feature_names(self) -> list[str]:
        return self.feature_names

    def get_skewed_features(self) -> list[str]:
        return self.skewed_features

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info(f"Preprocessor saved -> {path}")

    @classmethod
    def load(cls, path: str) -> "Preprocessor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        log.info(f"Preprocessor loaded <- {path}")
        return obj

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _structural_fixes(self, X: pd.DataFrame, fit_mode: bool) -> pd.DataFrame:
        """Drop Variance, one-hot Protocol Type, enforce column order."""
        X = X.copy()

        # Drop label columns that leaked in
        drop_these = [c for c in ALL_LABEL_COLS if c in X.columns]
        if drop_these:
            X = X.drop(columns=drop_these)

        # FIX 3 — Drop Variance (= Std^2)
        if VARIANCE_COL in X.columns:
            X = X.drop(columns=[VARIANCE_COL])

        # FIX 2 — One-hot encode Protocol Type
        if PROTOCOL_TYPE_COL in X.columns:
            if fit_mode:
                self.protocol_values = sorted(
                    int(v) for v in X[PROTOCOL_TYPE_COL].dropna().unique()
                )
                log.info(f"Protocol Type values in training: {self.protocol_values}")

            proto_dummies = pd.DataFrame(
                0.0, index=X.index,
                columns=[f"proto_{v}" for v in self.protocol_values],
            )
            for v in self.protocol_values:
                proto_dummies[f"proto_{v}"] = (X[PROTOCOL_TYPE_COL] == v).astype(float)

            X = X.drop(columns=[PROTOCOL_TYPE_COL])
            X = pd.concat([X, proto_dummies], axis=1)

        # Enforce column order (critical at inference time)
        if not fit_mode and self.feature_names:
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[self.feature_names]

        return X.select_dtypes(include=[np.number])

    def _replace_inf(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in X.columns:
            X[col] = X[col].replace(
                [np.inf, -np.inf], self.col_maxes.get(col, 0.0)
            )
        return X

    def _impute(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in X.columns:
            X[col] = X[col].fillna(self.col_medians.get(col, 0.0))
        return X

    def _apply_power_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply Yeo-Johnson to skewed features only. Leave others unchanged."""
        if self.power_transformer is None or not self.skewed_features:
            return X
        X = X.copy()
        present = [c for c in self.skewed_features if c in X.columns]
        if present:
            X[present] = self.power_transformer.transform(X[present])
        return X


def build_and_fit_preprocessor(
    X_train: pd.DataFrame,
    save_path: str | None = None,
) -> "Preprocessor":
    """Convenience wrapper: build, fit, optionally save."""
    p = Preprocessor()
    p.fit(X_train)
    if save_path:
        p.save(save_path)
    return p


# ── Deduplication ──────────────────────────────────────────────────────────────

def deduplicate(
    df: pd.DataFrame,
    label_col: str = "label",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Remove duplicates BEFORE the train/val/test split.

    Stage A: exact duplicates (same features AND same label) — drop all but one.
    Stage B: conflicting duplicates (same features, DIFFERENT labels) — resolve
             by majority-label vote, tie-break by rarer class.

    Uses hash-based grouping to avoid pandas 3.x NaN-in-groupby issues.
    """
    n_before     = len(df)
    feature_cols = [c for c in df.columns
                    if c != label_col and c not in ALL_LABEL_COLS]

    null_mask = df[label_col].isna() | (df[label_col].astype(str) == "NAN")
    if null_mask.sum() > 0:
        log.warning(f"Dropping {null_mask.sum()} rows with null labels.")
        df = df[~null_mask].reset_index(drop=True)

    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Stage A: exact duplicates
    df        = df.drop_duplicates(subset=feature_cols + [label_col], keep="first")
    n_after_A = len(df)
    log.info(f"Dedup Stage A (exact):       {n_before:>10,} -> {n_after_A:>10,}  "
             f"(-{n_before - n_after_A:,})")

    # Stage B: conflicting feature vectors (hash-based)
    def _row_hash(row: pd.Series) -> str:
        return hashlib.md5(str(row.values.tolist()).encode()).hexdigest()

    class_freq      = df[label_col].value_counts().to_dict()
    df["_feat_hash"] = df[feature_cols].apply(_row_hash, axis=1)
    hash_counts      = df["_feat_hash"].value_counts()
    conflict_hashes  = hash_counts[hash_counts > 1].index

    clean_df    = df[~df["_feat_hash"].isin(conflict_hashes)].drop(columns=["_feat_hash"])
    conflict_df = df[df["_feat_hash"].isin(conflict_hashes)].copy()

    resolved_rows = []
    for h, group in conflict_df.groupby("_feat_hash", sort=False):
        group  = group.drop(columns=["_feat_hash"])
        counts = group[label_col].value_counts()
        if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
            tied   = counts[counts == counts.iloc[0]].index.tolist()
            winner = min(tied, key=lambda l: class_freq.get(l, 0))
        else:
            winner = counts.index[0]
        resolved_rows.append(group[group[label_col] == winner].iloc[0])

    if resolved_rows:
        df = pd.concat([clean_df, pd.DataFrame(resolved_rows).reset_index(drop=True)],
                       ignore_index=True)
    else:
        df = clean_df

    n_after_B = len(df)
    log.info(f"Dedup Stage B (conflicting): {n_after_A:>10,} -> {n_after_B:>10,}  "
             f"(-{n_after_A - n_after_B:,})")
    log.info(f"Total removed: {n_before - n_after_B:,}  "
             f"({100*(n_before-n_after_B)/n_before:.1f}%)")
    return df.reset_index(drop=True)
