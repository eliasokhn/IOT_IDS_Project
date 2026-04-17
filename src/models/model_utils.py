"""
model_utils.py
==============
Save, load, and list trained model artifacts.
"""

import json
import logging
import pickle
from pathlib import Path

log = logging.getLogger(__name__)

VALID_MODEL_KEYS = [
    "lr_binary", "lr_8class", "lr_34class",
    "gb_binary", "gb_8class", "gb_34class",
]


def save_model(model, path: str) -> None:
    """Save a trained sklearn model to disk using joblib."""
    import joblib
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    log.info(f"Model saved to {path}")


def load_model(path: str):
    """Load a trained sklearn model from disk."""
    import joblib
    if not Path(path).exists():
        raise FileNotFoundError(f"Model not found at {path}")
    model = joblib.load(path)
    log.info(f"Model loaded from {path}")
    return model


def save_class_names(class_names: list[str], path: str) -> None:
    """Save class name list to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(class_names, f, indent=2)
    log.info(f"Class names saved to {path}")


def load_class_names(path: str) -> list[str]:
    """Load class name list from JSON."""
    with open(path) as f:
        return json.load(f)


def list_saved_models(artifacts_dir: str = "models") -> list[str]:
    """Return list of all .pkl files in artifacts directory."""
    p = Path(artifacts_dir)
    if not p.exists():
        return []
    return [str(f) for f in p.glob("*.pkl")]
