"""
metrics.py
==========
Compute all classification metrics for intrusion detection evaluation.

Primary metric: Macro F1 (treats every class equally regardless of size)
Secondary:      Per-class recall, FPR on benign traffic
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

log = logging.getLogger(__name__)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    class_names: list[str],
    task: str,
    model_name: str,
    benign_class_idx: int = 0,
) -> dict[str, Any]:
    """
    Compute comprehensive metrics for one model / one task.

    Parameters
    ----------
    y_true          : true integer labels
    y_pred          : predicted integer labels
    y_proba         : predicted probabilities (n_samples × n_classes) or None
    class_names     : list of class name strings
    task            : 'binary', '8class', or '34class'
    model_name      : 'lr' or 'gb'
    benign_class_idx: integer index of the benign class (default 0)

    Returns
    -------
    dict with all computed metrics
    """
    results: dict[str, Any] = {
        "model": model_name,
        "task": task,
        "n_samples": int(len(y_true)),
    }

    # ── Accuracy ──────────────────────────────────────────────────
    results["accuracy"] = float(accuracy_score(y_true, y_pred))

    # ── Macro F1 (primary metric) ─────────────────────────────────
    results["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # ── Weighted F1 ───────────────────────────────────────────────
    results["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # ── Per-class precision, recall, f1 ───────────────────────────
    present_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    present_names = [class_names[i] for i in present_labels if i < len(class_names)]

    per_class_prec = precision_score(y_true, y_pred, labels=present_labels,
                                     average=None, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, labels=present_labels,
                                  average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=present_labels,
                             average=None, zero_division=0)

    results["per_class"] = {
        name: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(np.sum(y_true == label)),
        }
        for name, label, p, r, f in zip(
            present_names, present_labels,
            per_class_prec, per_class_rec, per_class_f1
        )
    }

    # ── False Positive Rate on Benign traffic ─────────────────────
    # FPR = FP / (FP + TN) = benign samples predicted as malicious / total benign
    benign_mask = y_true == benign_class_idx
    if benign_mask.sum() > 0:
        benign_correct = ((y_true == benign_class_idx) & (y_pred == benign_class_idx)).sum()
        benign_fp = ((y_true == benign_class_idx) & (y_pred != benign_class_idx)).sum()
        fpr_benign = float(benign_fp / (benign_fp + benign_correct + 1e-10))
        results["fpr_benign"] = round(fpr_benign, 6)
        log.info(f"FPR on benign traffic: {fpr_benign:.4f} "
                 f"({benign_fp:,} benign samples mis-classified as malicious)")
    else:
        results["fpr_benign"] = None

    # ── ROC-AUC (binary only) ─────────────────────────────────────
    if task == "binary" and y_proba is not None:
        try:
            proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            results["roc_auc"] = float(roc_auc_score(y_true, proba_pos))
        except Exception as e:
            log.warning(f"Could not compute ROC-AUC: {e}")
            results["roc_auc"] = None

    # ── Confusion matrix ──────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    results["confusion_matrix"] = cm.tolist()
    results["confusion_matrix_labels"] = present_names

    # ── Classification report string (for logging) ────────────────
    report_str = classification_report(
        y_true, y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0,
    )
    results["classification_report"] = report_str

    # ── Log summary ───────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info(f"Results: {model_name.upper()} | {task}")
    log.info(f"  Accuracy:    {results['accuracy']:.4f}")
    log.info(f"  Macro F1:    {results['macro_f1']:.4f}  ← PRIMARY METRIC")
    log.info(f"  Weighted F1: {results['weighted_f1']:.4f}")
    if results.get("fpr_benign") is not None:
        log.info(f"  FPR Benign:  {results['fpr_benign']:.4f}")
    if results.get("roc_auc"):
        log.info(f"  ROC-AUC:     {results['roc_auc']:.4f}")
    log.info(f"{'='*60}")

    return results


def save_metrics(results: dict, save_path: str) -> None:
    """Save metrics dict to JSON file."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy types to native Python for JSON serialisation
    results_serialisable = _make_serialisable(results)
    with open(save_path, "w") as f:
        json.dump(results_serialisable, f, indent=2)
    log.info(f"Metrics saved to {save_path}")


def load_metrics(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_comparison_table(metrics_list: list[dict]) -> pd.DataFrame:
    """
    Build a comparison DataFrame from a list of metrics dicts.
    One row per (model, task) combination.
    """
    rows = []
    for m in metrics_list:
        rows.append({
            "model": m.get("model", ""),
            "task": m.get("task", ""),
            "accuracy": round(m.get("accuracy", 0), 4),
            "macro_f1": round(m.get("macro_f1", 0), 4),
            "weighted_f1": round(m.get("weighted_f1", 0), 4),
            "fpr_benign": round(m.get("fpr_benign") or 0, 4),
            "roc_auc": round(m.get("roc_auc") or 0, 4),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(["task", "model"]).reset_index(drop=True)
    return df


def _make_serialisable(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
