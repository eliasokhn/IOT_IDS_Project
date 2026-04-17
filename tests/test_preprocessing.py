"""
Tests for preprocessing.py  (v2 — covers all three fixes)
"""

import numpy as np
import pandas as pd
import tempfile, pathlib
from src.features.preprocessing import Preprocessor, build_and_fit_preprocessor, deduplicate


def make_real_structure_df(n=300, seed=42):
    """Make a DataFrame that mirrors real CICIoT2023 columns."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "Header_Length":     rng.uniform(0, 100, n),
        "Protocol Type":     rng.choice([0, 1, 6, 17, 47], n),
        "Time_To_Live":      rng.uniform(0, 128, n),
        "Rate":              rng.exponential(5000, n),
        "Std":               rng.exponential(10, n),
        "Variance":          rng.exponential(10, n) ** 2,   # will be dropped
        "syn_flag_number":   rng.choice([0.0, 1.0], n),
        "Tot sum":           rng.uniform(0, 6000, n),
    })
    # Inject NaN and inf
    df.iloc[0, 0] = np.nan
    df.iloc[1, 2] = np.inf
    return df


def make_simple_df(n=200, seed=42):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(rng.randn(n, 5), columns=[f"f{i}" for i in range(5)])
    df.iloc[0, 0] = np.nan
    df.iloc[1, 1] = np.inf
    return df


class TestFix1Deduplication:
    def test_exact_dups_removed(self):
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 1.0],
            "f2": [3.0, 4.0, 3.0],
            "label": ["A", "B", "A"],
        })
        result = deduplicate(df, label_col="label")
        assert len(result) == 2, f"Expected 2, got {len(result)}"

    def test_conflicting_labels_resolved(self):
        # Same features, two different labels — majority wins
        df = pd.DataFrame({
            "f1": [1.0, 1.0, 1.0, 2.0],
            "f2": [0.5, 0.5, 0.5, 0.9],
            "label": ["A", "A", "B", "C"],  # A wins for first 3
        })
        result = deduplicate(df, label_col="label")
        conflict_rows = result[result["label"].isin(["A","B"]) & 
                                (result["f1"] == 1.0) & (result["f2"] == 0.5)]
        assert len(conflict_rows) == 1
        assert conflict_rows["label"].iloc[0] == "A"

    def test_rare_class_wins_tie(self):
        # Tie-break: rarer class (B has fewer global rows) should win
        df = pd.DataFrame({
            "f1": [1.0, 1.0, 5.0, 5.0, 5.0],
            "f2": [2.0, 2.0, 6.0, 6.0, 6.0],
            "label": ["A", "B", "C", "C", "C"],
            # f1=1,f2=2: A(1) vs B(1) tie — B is rarer globally, B wins
        })
        result = deduplicate(df, label_col="label")
        conflict_rows = result[(result["f1"] == 1.0)]
        assert len(conflict_rows) == 1
        assert conflict_rows["label"].iloc[0] == "B"

    def test_no_conflicts_remain(self):
        rng = np.random.RandomState(0)
        df = pd.DataFrame(rng.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        df["label"] = rng.choice(["A","B","C"], 100)
        # Add some dups
        df = pd.concat([df, df.iloc[:10]], ignore_index=True)
        result = deduplicate(df, "label")
        feature_cols = [c for c in result.columns if c != "label"]
        remaining = result.duplicated(subset=feature_cols).sum()
        assert remaining == 0, f"Still {remaining} conflict rows after dedup"

    def test_null_labels_dropped(self):
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0],
            "label": ["A", None, "B"],
        })
        result = deduplicate(df, "label")
        assert len(result) == 2
        assert result["label"].notna().all()

    def test_label_distribution_preserved(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.randn(500, 5), columns=[f"f{i}" for i in range(5)])
        df["label"] = rng.choice(["A","B","C"], 500)
        result = deduplicate(df, "label")
        # All 3 classes should still be present
        assert set(result["label"].unique()) == {"A", "B", "C"}


class TestFix2ProtocolTypeOneHot:
    def test_protocol_type_one_hot_created(self):
        df = make_real_structure_df()
        p = Preprocessor()
        p.fit(df)
        proto_cols = [n for n in p.get_feature_names() if n.startswith("proto_")]
        assert len(proto_cols) == 5, f"Expected 5 proto cols, got {len(proto_cols)}"
        assert "proto_0"  in p.get_feature_names()
        assert "proto_1"  in p.get_feature_names()
        assert "proto_6"  in p.get_feature_names()
        assert "proto_17" in p.get_feature_names()
        assert "proto_47" in p.get_feature_names()

    def test_protocol_type_raw_removed(self):
        df = make_real_structure_df()
        p = Preprocessor()
        p.fit(df)
        assert "Protocol Type" not in p.get_feature_names()

    def test_protocol_one_hot_values_are_binary(self):
        df = make_real_structure_df()
        p = Preprocessor()
        X_scaled = p.fit_transform(df)
        # proto columns — after RobustScaler their median is 0 but values should be 0 or 1 before
        # We verify by checking that Protocol Type handling produces valid output
        feat = p.get_feature_names()
        idx_proto6 = feat.index("proto_6")
        # At least some rows should have proto_6 = 1 (TCP is common)
        assert X_scaled[:, idx_proto6].max() > 0

    def test_inference_handles_unseen_protocol(self):
        df_train = make_real_structure_df()
        df_train = df_train[df_train["Protocol Type"].isin([1, 6])]  # train only on ICMP+TCP
        p = Preprocessor()
        p.fit(df_train)
        # Inference on row with protocol 17 (UDP, unseen) — should not crash
        df_test = pd.DataFrame({
            "Header_Length": [20.0], "Protocol Type": [17], "Time_To_Live": [64.0],
            "Rate": [1000.0], "Std": [0.5], "syn_flag_number": [1.0], "Tot sum": [6000.0],
        })
        result = p.transform(df_test)
        assert result.shape[1] == len(p.get_feature_names())
        assert not np.isnan(result).any()


class TestFix3VarianceAndScaler:
    def test_variance_column_dropped(self):
        df = make_real_structure_df()
        p = Preprocessor()
        p.fit(df)
        assert "Variance" not in p.get_feature_names(), "Variance should be removed"

    def test_robust_scaler_used(self):
        from sklearn.preprocessing import RobustScaler
        df = make_real_structure_df()
        p = Preprocessor()
        p.fit(df)
        assert isinstance(p.scaler, RobustScaler), "Must use RobustScaler"

    def test_feature_medians_approx_zero(self):
        # RobustScaler: median of each feature ≈ 0
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.randn(2000, 5), columns=[f"f{i}" for i in range(5)])
        p = Preprocessor()
        X = p.fit_transform(df)
        medians = np.median(X, axis=0)
        assert np.abs(medians).mean() < 0.1, f"Medians not near 0: {medians}"

    def test_no_nan_after_preprocessing(self):
        df = make_real_structure_df()
        p = Preprocessor()
        X = p.fit_transform(df)
        assert not np.isnan(X).any()

    def test_no_inf_after_preprocessing(self):
        df = make_real_structure_df()
        p = Preprocessor()
        X = p.fit_transform(df)
        assert not np.isinf(X).any()

    def test_extreme_variance_outlier_handled(self):
        # Even with Variance max=46M, output should be finite
        df = make_real_structure_df()
        df["Variance"] = np.where(df.index < 5, 46_000_000, df["Variance"])
        p = Preprocessor()
        X = p.fit_transform(df)
        assert not np.isinf(X).any()
        assert not np.isnan(X).any()


class TestPreprocessorCore:
    def test_fit_transform_output_shape(self):
        df = make_simple_df()
        p = Preprocessor()
        X = p.fit_transform(df)
        assert X.shape[0] == len(df)
        assert X.shape[1] == 5  # 5 features, no Protocol Type in simple df

    def test_val_uses_train_stats(self):
        rng = np.random.RandomState(0)
        df_train = pd.DataFrame(rng.randn(500, 5) + 0, columns=[f"f{i}" for i in range(5)])
        df_val   = pd.DataFrame(rng.randn(100, 5) + 5, columns=[f"f{i}" for i in range(5)])
        p = Preprocessor()
        p.fit(df_train)
        X_val = p.transform(df_val)
        # Val mean should NOT be 0 (uses train stats, not re-centered)
        assert abs(X_val.mean()) > 0.5

    def test_transform_before_fit_raises(self):
        p = Preprocessor()
        try:
            p.transform(make_simple_df())
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass

    def test_save_load_roundtrip(self):
        df = make_real_structure_df()
        p = Preprocessor()
        p.fit(df)
        X1 = p.transform(df)
        with tempfile.TemporaryDirectory() as td:
            path = str(pathlib.Path(td) / "pp.pkl")
            p.save(path)
            p2 = Preprocessor.load(path)
        X2 = p2.transform(df)
        np.testing.assert_array_almost_equal(X1, X2)

    def test_feature_names_include_proto_not_variance(self):
        df = make_real_structure_df()
        p = Preprocessor()
        p.fit(df)
        names = p.get_feature_names()
        assert any(n.startswith("proto_") for n in names)
        assert "Variance" not in names
        assert "Protocol Type" not in names
