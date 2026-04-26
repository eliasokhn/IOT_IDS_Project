"""
predictor.py
============
Inference pipeline: load saved artifacts and predict on new traffic records.

CRITICAL: Uses the EXACT same preprocessor that was fitted during training.
This guarantees that inference-time feature processing matches training.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class Predictor:
    """
    Wraps a trained model + preprocessor for inference.

    Usage
    -----
        predictor = Predictor.load("models/", task="binary", model_name="gb")
        result = predictor.predict({"flow_duration": 1.2, "total_fwd_packets": 5, ...})
    """

    def __init__(
        self,
        model,
        preprocessor,
        class_names: list[str],
        task: str,
        model_name: str,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.class_names = class_names
        self.task = task
        self.model_name = model_name

    @classmethod
    def load(
        cls,
        artifacts_dir: str,
        task: str,
        model_name: str,
    ) -> "Predictor":
        """
        Load a trained predictor from saved artifacts.

        Parameters
        ----------
        artifacts_dir : directory containing .pkl model files
        task          : 'binary', '8class', or '34class'
        model_name    : 'lr' or 'gb'
        """
        from src.features.preprocessing import Preprocessor
        from src.models.model_utils import load_model

        artifacts_dir = Path(artifacts_dir)

        model_path = artifacts_dir / f"{model_name}_{task}.pkl"
        preprocessor_path = artifacts_dir / "preprocessor.pkl"
        class_names_path = artifacts_dir / f"class_names_{task}.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Run the training notebook first to generate model artifacts."
            )

        model = load_model(str(model_path))
        preprocessor = Preprocessor.load(str(preprocessor_path))

        if class_names_path.exists():
            with open(class_names_path) as f:
                class_names = json.load(f)
        else:
            class_names = [str(i) for i in range(100)]

        log.info(f"Predictor loaded: model={model_name}, task={task}")
        return cls(
            model=model,
            preprocessor=preprocessor,
            class_names=class_names,
            task=task,
            model_name=model_name,
        )

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        """
        Predict class and probabilities for a single traffic record.

        Parameters
        ----------
        features : dict mapping feature name → value

        Returns
        -------
        dict with keys: predicted_class, predicted_label, probabilities, is_malicious
        """
        X = pd.DataFrame([features])
        X_scaled = self.preprocessor.transform(X)

        pred_idx = int(self.model.predict(X_scaled)[0])
        proba = self.model.predict_proba(X_scaled)[0].tolist()

        predicted_label = (
            self.class_names[pred_idx]
            if pred_idx < len(self.class_names)
            else str(pred_idx)
        )

        is_malicious = predicted_label.upper() != "BENIGN"

        return {
            "predicted_class": pred_idx,
            "predicted_label": predicted_label,
            "probabilities": {
                self.class_names[i]: round(p, 6)
                for i, p in enumerate(proba)
                if i < len(self.class_names)
            },
            "is_malicious": is_malicious,
            "model": self.model_name,
            "task": self.task,
        }

    def predict_batch(self, records: list[dict[str, float]]) -> list[dict[str, Any]]:
        """
        Predict on a batch of traffic records.

        Parameters
        ----------
        records : list of feature dicts

        Returns
        -------
        list of prediction dicts (same format as predict())
        """
        if not records:
            return []

        X = pd.DataFrame(records)
        X_scaled = self.preprocessor.transform(X)

        preds = self.model.predict(X_scaled)
        probas = self.model.predict_proba(X_scaled)

        results = []
        for i, (pred_idx, proba) in enumerate(zip(preds, probas)):
            pred_idx = int(pred_idx)
            predicted_label = (
                self.class_names[pred_idx]
                if pred_idx < len(self.class_names)
                else str(pred_idx)
            )
            results.append({
                "predicted_class": pred_idx,
                "predicted_label": predicted_label,
                "probabilities": {
                    self.class_names[j]: round(float(p), 6)
                    for j, p in enumerate(proba)
                    if j < len(self.class_names)
                },
                "is_malicious": predicted_label.upper() != "BENIGN",
                "model": self.model_name,
                "task": self.task,
            })
        return results

    def get_feature_names(self) -> list[str]:
        return self.preprocessor.get_feature_names()
