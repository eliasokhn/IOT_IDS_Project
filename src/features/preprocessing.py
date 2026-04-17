"""
preprocessing.py  (v2 — fixed)
================================
Three bugs fixed based on real CICIoT2023 data inspection:

FIX 1 — DUPLICATES
  Problem : ~5% exact duplicate rows per file, plus 997 feature vectors
            per file that map to two different labels (DDOS-UDP_FLOOD vs
            DOS-UDP_FLOOD with identical features). Duplicates in the test
            set inflate binary accuracy and confuse 34-class boundaries.
  Fix     : deduplicate() removes exact dups (Stage A) and resolves
            conflicting-label dups by majority vote / rarer-class tie-break
            (Stage B). Uses hash-based grouping — safe for pandas 3.x.
            Must be called BEFORE create_splits().

FIX 2 — PROTOCOL TYPE (ordinal → one-hot)
  Problem : Values {0,1,6,17,47} are IP protocol numbers.
            StandardScaler treats them as continuous — implies
            GRE(47) = 47x Other(0). Dominant spurious feature
            that misleads LR and hurts 34-class separation.
  Fix     : One-hot encode into 5 binary columns: proto_0, proto_1,
            proto_6, proto_17, proto_47. Numeric value dropped.

FIX 3 — VARIANCE COLUMN + SCALER
  Problem : Variance = Std^2 exactly (max diff < 1e-8, verified on
            Merged52.csv). Pure redundancy. Extreme outlier source:
            range 0 to 46,402,130 with median 0. StandardScaler
            produces near-infinite scaled values that distort LR.
  Fix     : Drop Variance. Switch to RobustScaler (median + IQR)
            which is resilient to remaining outliers in Rate, IAT, etc.
            GB is immune either way; LR benefits significantly.
"""

import hashlib
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

log = logging.getLogger(__name__)

PROTOCOL_TYPE_COL = "Protocol Type"
VARIANCE_COL      = "Variance"
ALL_LABEL_COLS    = ["label", "label_binary", "label_category",
                     "label_fine", "Label", "_feat_hash"]


class Preprocessor:
    """
    Stateful preprocessor. Fit on training data only, transform any split.

    Pipeline (in order):
      1. Drop Variance column (= Std^2, redundant + extreme outlier source)
      2. One-hot encode Protocol Type  (categorical, NOT ordinal)
      3. Replace ±inf with column finite-max  (computed from training data)
      4. Median-impute remaining NaNs         (computed from training data)
      5. RobustScaler (median + IQR — resilient to outliers)

    Save / load with save() / load() to guarantee identical preprocessing
    at training time and inference time.
    """

    def __init__(self):
        self.scaler:            RobustScaler       = RobustScaler()
        self.col_medians:       dict[str, float]   = {}
        self.col_maxes:         dict[str, float]   = {}
        self.protocol_values:   list[int]          = []
        self.feature_names:     list[str]          = []
        self._fitted:           bool               = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        """Fit on training data only."""
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
        self.scaler.fit(X_imputed)
        self._fitted = True
        log.info(
            f"Preprocessor fitted. Output features: {len(self.feature_names)}  "
            f"(Protocol Type one-hot added, Variance dropped, RobustScaler)."
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply fitted preprocessing. Returns numpy array for sklearn."""
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        X_prep    = self._structural_fixes(X, fit_mode=False)
        X_no_inf  = self._replace_inf(X_prep)
        X_imputed = self._impute(X_no_inf)
        return self.scaler.transform(X_imputed)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step. Use on training data only."""
        self.fit(X)
        return self.transform(X)

    def get_feature_names(self) -> list[str]:
        return self.feature_names

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
        """Apply FIX 2 (one-hot) and FIX 3a (drop Variance)."""
        X = X.copy()

        # Drop label / utility columns that may have leaked in
        drop_these = [c for c in ALL_LABEL_COLS if c in X.columns]
        if drop_these:
            X = X.drop(columns=drop_these)

        # FIX 3a — Drop Variance (= Std^2, extreme outlier source)
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
                proto_dummies[f"proto_{v}"] = (
                    X[PROTOCOL_TYPE_COL] == v
                ).astype(float)

            X = X.drop(columns=[PROTOCOL_TYPE_COL])
            X = pd.concat([X, proto_dummies], axis=1)

        # Ensure column order matches training (critical at inference time)
        if not fit_mode and self.feature_names:
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0.0          # unseen protocol at inference
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


# ── Standalone deduplication (call BEFORE create_splits) ──────────────────────

def deduplicate(
    df: pd.DataFrame,
    label_col: str = "label",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    FIX 1 — Remove duplicates BEFORE the train/val/test split.

    Must be called on the full merged dataset before create_splits().
    Uses hash-based grouping to avoid pandas 3.x NaN-in-groupby issues.

    Stage A — Exact duplicates (same features AND same label):
        Drop all but one copy.

    Stage B — Conflicting duplicates (same features, DIFFERENT labels):
        For each conflicting feature vector, keep one row by:
          - Majority-label vote
          - Tie-break: choose the rarer class (lower global frequency)
            This protects rare classes (XSS, BruteForce, etc.)

    Parameters
    ----------
    df           : full merged dataframe with 'label' column
    label_col    : name of the raw label column (default: 'label')
    random_state : shuffle seed for reproducible tie-breaking

    Returns
    -------
    Deduplicated DataFrame, reset index.
    """
    n_before     = len(df)
    feature_cols = [c for c in df.columns
                    if c != label_col and c not in ALL_LABEL_COLS]

    # Drop rows with null labels (cannot be used for training or testing)
    null_mask = df[label_col].isna() | (df[label_col].astype(str) == "NAN")
    if null_mask.sum() > 0:
        log.warning(f"Dropping {null_mask.sum()} rows with null/NaN labels.")
        df = df[~null_mask].reset_index(drop=True)

    # Shuffle for reproducible tie-breaking
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # ── Stage A: exact duplicates (features + label) ──────────────────────────
    df        = df.drop_duplicates(subset=feature_cols + [label_col], keep="first")
    n_after_A = len(df)
    log.info(
        f"Dedup Stage A (exact):       {n_before:>10,} -> {n_after_A:>10,}  "
        f"(-{n_before - n_after_A:,} rows)"
    )

    # ── Stage B: conflicting feature vectors ──────────────────────────────────
    # Hash each row's feature values to group without pandas groupby float issues
    def _row_hash(row: pd.Series) -> str:
        return hashlib.md5(str(row.values.tolist()).encode()).hexdigest()

    class_freq = df[label_col].value_counts().to_dict()

    df["_feat_hash"] = df[feature_cols].apply(_row_hash, axis=1)

    # Identify hashes that appear more than once (conflicting vectors)
    hash_counts  = df["_feat_hash"].value_counts()
    conflict_hashes = hash_counts[hash_counts > 1].index

    clean_df    = df[~df["_feat_hash"].isin(conflict_hashes)].drop(
                        columns=["_feat_hash"])
    conflict_df = df[df["_feat_hash"].isin(conflict_hashes)].copy()

    resolved_rows = []
    for h, group in conflict_df.groupby("_feat_hash", sort=False):
        group = group.drop(columns=["_feat_hash"])
        if len(group) == 1:
            resolved_rows.append(group.iloc[0])
            continue
        counts = group[label_col].value_counts()
        top_count = counts.iloc[0]
        if len(counts) > 1 and top_count == counts.iloc[1]:
            # Tie → pick rarer class (lower global frequency)
            tied   = counts[counts == top_count].index.tolist()
            winner = min(tied, key=lambda l: class_freq.get(l, 0))
        else:
            winner = counts.index[0]
        resolved_rows.append(group[group[label_col] == winner].iloc[0])

    if resolved_rows:
        resolved = pd.DataFrame(resolved_rows).reset_index(drop=True)
        df       = pd.concat([clean_df, resolved], ignore_index=True)
    else:
        df = clean_df

    n_after_B = len(df)
    log.info(
        f"Dedup Stage B (conflicting): {n_after_A:>10,} -> {n_after_B:>10,}  "
        f"(-{n_after_A - n_after_B:,} rows)"
    )
    log.info(
        f"Total removed: {n_before - n_after_B:,}  "
        f"({100*(n_before-n_after_B)/n_before:.1f}%)"
    )
    return df.reset_index(drop=True)
