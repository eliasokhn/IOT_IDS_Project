"""
Tests for the Predictor inference pipeline.
Uses a small in-memory model to avoid needing saved artifacts.
"""

import numpy as np
import pandas as pd

from src.features.preprocessing import Preprocessor
from src.serving.predictor import Predictor


def _make_mock_predictor(task="binary"):
    """Build a Predictor with a tiny in-memory LogisticRegression."""
    from sklearn.linear_model import LogisticRegression
    from src.data.label_mapping import get_class_names

    rng = np.random.RandomState(42)
    n_features = 5
    feature_names = [f"f{i}" for i in range(n_features)]
    class_names = get_class_names(task)
    n_classes = len(class_names)

    # Train a tiny model
    X = rng.randn(200, n_features)
    y = rng.randint(0, n_classes, size=200)
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X, y)

    # Fit preprocessor
    preprocessor = Preprocessor()
    df = pd.DataFrame(X, columns=feature_names)
    preprocessor.fit(df)

    return Predictor(
        model=model,
        preprocessor=preprocessor,
        class_names=class_names,
        task=task,
        model_name="lr",
    )


class TestPredictorSingleRecord:
    def test_returns_expected_keys(self):
        p = _make_mock_predictor("binary")
        features = {f"f{i}": float(i) for i in range(5)}
        result = p.predict(features)
        assert "predicted_class" in result
        assert "predicted_label" in result
        assert "probabilities" in result
        assert "is_malicious" in result

    def test_predicted_class_is_int(self):
        p = _make_mock_predictor("binary")
        features = {f"f{i}": 1.0 for i in range(5)}
        result = p.predict(features)
        assert isinstance(result["predicted_class"], int)

    def test_probabilities_sum_to_one(self):
        p = _make_mock_predictor("binary")
        features = {f"f{i}": 1.0 for i in range(5)}
        result = p.predict(features)
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-5

    def test_is_malicious_is_bool(self):
        p = _make_mock_predictor("binary")
        features = {f"f{i}": 1.0 for i in range(5)}
        result = p.predict(features)
        assert isinstance(result["is_malicious"], bool)

    def test_benign_prediction_not_malicious(self):
        """If predicted class is 0 (Benign), is_malicious should be False."""
        p = _make_mock_predictor("binary")
        features = {f"f{i}": 1.0 for i in range(5)}
        result = p.predict(features)
        if result["predicted_class"] == 0:
            assert result["is_malicious"] is False


class TestPredictorBatch:
    def test_batch_length_matches_input(self):
        p = _make_mock_predictor("binary")
        records = [{f"f{i}": float(j) for i in range(5)} for j in range(20)]
        results = p.predict_batch(records)
        assert len(results) == 20

    def test_empty_batch_returns_empty(self):
        p = _make_mock_predictor("binary")
        assert p.predict_batch([]) == []

    def test_batch_probabilities_all_sum_to_one(self):
        p = _make_mock_predictor("8class")
        records = [{f"f{i}": float(j) for i in range(5)} for j in range(10)]
        results = p.predict_batch(records)
        for r in results:
            total = sum(r["probabilities"].values())
            assert abs(total - 1.0) < 1e-5


class TestPredictorFeatureNames:
    def test_feature_names_returned(self):
        p = _make_mock_predictor("binary")
        names = p.get_feature_names()
        assert isinstance(names, list)
        assert len(names) == 5
