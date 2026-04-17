"""
drift_monitor.py
================
Monitors feature distribution drift and alert rate spikes in streaming micro-batches.

How it works
------------
1.  Compute baseline statistics from training data (mean, std per feature).
2.  For each incoming micro-batch, compute current stats.
3.  Use Z-score to detect features that have drifted significantly.
4.  Track alert rate (fraction of batch predicted malicious) and raise warnings.
5.  Log everything to a JSON file for later analysis.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class DriftMonitor:
    """
    Detects feature drift and alert rate spikes in streaming micro-batches.

    Parameters
    ----------
    baseline_feature_stats : dict {feature_name: {"mean": float, "std": float}}
                             Computed from training data.
    baseline_alert_rate    : expected fraction of malicious predictions (from validation set)
    z_threshold            : Z-score threshold to flag feature drift (default 3.0)
    alert_rate_warning     : alert rate above this triggers WARNING
    alert_rate_critical    : alert rate above this triggers CRITICAL
    top_features           : list of feature names to monitor (monitor all if None)
    log_dir                : directory to save monitoring logs
    """

    def __init__(
        self,
        baseline_feature_stats: dict[str, dict],
        baseline_alert_rate: float = 0.05,
        z_threshold: float = 3.0,
        alert_rate_warning: float = 0.30,
        alert_rate_critical: float = 0.60,
        top_features: list[str] | None = None,
        log_dir: str = "reports/monitoring_logs",
    ):
        self.baseline_feature_stats = baseline_feature_stats
        self.baseline_alert_rate = baseline_alert_rate
        self.z_threshold = z_threshold
        self.alert_rate_warning = alert_rate_warning
        self.alert_rate_critical = alert_rate_critical
        self.top_features = top_features or list(baseline_feature_stats.keys())
        self.log_dir = log_dir
        self.history: list[dict] = []
        self.batch_idx = 0

    @classmethod
    def from_training_data(
        cls,
        X_train: pd.DataFrame | np.ndarray,
        feature_names: list[str] | None = None,
        n_top_features: int = 10,
        **kwargs,
    ) -> "DriftMonitor":
        """
        Build a DriftMonitor from training data.
        Selects the top-N highest-variance features to monitor.
        """
        if isinstance(X_train, np.ndarray):
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
            X_df = pd.DataFrame(X_train, columns=feature_names)
        else:
            X_df = X_train
            feature_names = list(X_df.columns)

        # Select top-N features by variance (most informative for drift detection)
        variances = X_df.var()
        top_features = variances.nlargest(n_top_features).index.tolist()

        baseline_stats = {}
        for feat in top_features:
            baseline_stats[feat] = {
                "mean": float(X_df[feat].mean()),
                "std": float(X_df[feat].std()) + 1e-9,
            }

        log.info(f"DriftMonitor initialized with {len(top_features)} features: {top_features}")
        return cls(baseline_feature_stats=baseline_stats, top_features=top_features, **kwargs)

    def check_batch(
        self,
        X_batch: pd.DataFrame | np.ndarray,
        y_pred: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Check a micro-batch for drift and alert rate spikes.

        Parameters
        ----------
        X_batch      : feature matrix for the batch
        y_pred       : integer predictions for the batch (0 = benign)
        feature_names: column names (needed if X_batch is numpy array)

        Returns
        -------
        dict with drift warnings, alert rate, z-scores per feature
        """
        self.batch_idx += 1

        # Convert to DataFrame if needed
        if isinstance(X_batch, np.ndarray):
            cols = feature_names or [f"feature_{i}" for i in range(X_batch.shape[1])]
            X_df = pd.DataFrame(X_batch, columns=cols)
        else:
            X_df = X_batch

        warnings = []
        drift_details = {}

        # ── Feature drift check (Z-score) ─────────────────────────────
        monitored = [f for f in self.top_features if f in X_df.columns]
        for feat in monitored:
            current_mean = float(X_df[feat].mean())
            baseline = self.baseline_feature_stats[feat]
            z_score = abs(current_mean - baseline["mean"]) / baseline["std"]

            drift_details[feat] = {
                "current_mean": round(current_mean, 4),
                "baseline_mean": round(baseline["mean"], 4),
                "z_score": round(z_score, 3),
                "drifted": z_score > self.z_threshold,
            }

            if z_score > self.z_threshold:
                msg = (
                    f"FEATURE DRIFT | {feat} | "
                    f"current_mean={current_mean:.3f} | "
                    f"baseline_mean={baseline['mean']:.3f} | "
                    f"z={z_score:.2f}"
                )
                warnings.append(msg)
                log.warning(f"Batch {self.batch_idx}: {msg}")

        # ── Alert rate check ──────────────────────────────────────────
        n_malicious = int((y_pred != 0).sum())
        alert_rate = n_malicious / max(len(y_pred), 1)

        alert_level = "normal"
        if alert_rate >= self.alert_rate_critical:
            alert_level = "CRITICAL"
            msg = (
                f"CRITICAL ALERT SPIKE | batch={self.batch_idx} | "
                f"alert_rate={alert_rate:.1%} (threshold={self.alert_rate_critical:.0%})"
            )
            warnings.append(msg)
            log.critical(f"Batch {self.batch_idx}: {msg}")
        elif alert_rate >= self.alert_rate_warning:
            alert_level = "WARNING"
            msg = (
                f"HIGH ALERT RATE | batch={self.batch_idx} | "
                f"alert_rate={alert_rate:.1%} (threshold={self.alert_rate_warning:.0%})"
            )
            warnings.append(msg)
            log.warning(f"Batch {self.batch_idx}: {msg}")

        # ── Prediction class distribution drift ───────────────────────
        unique, counts = np.unique(y_pred, return_counts=True)
        pred_distribution = {int(cls): int(cnt) for cls, cnt in zip(unique, counts)}

        # ── Build batch record ────────────────────────────────────────
        batch_record = {
            "batch_idx": self.batch_idx,
            "batch_size": len(y_pred),
            "n_malicious": n_malicious,
            "alert_rate": round(alert_rate, 4),
            "alert_level": alert_level,
            "n_drift_warnings": sum(1 for d in drift_details.values() if d["drifted"]),
            "drift_details": drift_details,
            "pred_distribution": pred_distribution,
            "warnings": warnings,
        }

        self.history.append(batch_record)
        return batch_record

    def get_summary(self) -> dict[str, Any]:
        """Return summary statistics across all monitored batches."""
        if not self.history:
            return {"message": "No batches processed yet."}

        alert_rates = [b["alert_rate"] for b in self.history]
        drift_counts = [b["n_drift_warnings"] for b in self.history]
        critical_batches = [b for b in self.history if b["alert_level"] == "CRITICAL"]
        warning_batches = [b for b in self.history if b["alert_level"] == "WARNING"]

        return {
            "total_batches": len(self.history),
            "total_records": sum(b["batch_size"] for b in self.history),
            "mean_alert_rate": round(float(np.mean(alert_rates)), 4),
            "max_alert_rate": round(float(np.max(alert_rates)), 4),
            "mean_drift_warnings_per_batch": round(float(np.mean(drift_counts)), 2),
            "n_critical_batches": len(critical_batches),
            "n_warning_batches": len(warning_batches),
        }

    def save_log(self, filename: str | None = None) -> str:
        """Save monitoring history to JSON file."""
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        fname = filename or f"monitoring_log_batch_{self.batch_idx}.json"
        path = Path(self.log_dir) / fname
        with open(path, "w") as f:
            json.dump({"history": self.history, "summary": self.get_summary()}, f, indent=2)
        log.info(f"Monitoring log saved to {path}")
        return str(path)

    def plot_alert_rate(self, save_path: str) -> None:
        """Plot alert rate over streaming batches."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.history:
            log.warning("No history to plot.")
            return

        batch_idxs = [b["batch_idx"] for b in self.history]
        alert_rates = [b["alert_rate"] for b in self.history]

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(batch_idxs, alert_rates, color="#2196F3", linewidth=1.5, label="Alert rate")
        ax.axhline(self.alert_rate_warning, color="orange", linestyle="--",
                   linewidth=1.0, label=f"Warning ({self.alert_rate_warning:.0%})")
        ax.axhline(self.alert_rate_critical, color="red", linestyle="--",
                   linewidth=1.0, label=f"Critical ({self.alert_rate_critical:.0%})")
        ax.axhline(self.baseline_alert_rate, color="green", linestyle=":",
                   linewidth=1.0, label=f"Baseline ({self.baseline_alert_rate:.0%})")
        ax.fill_between(batch_idxs, alert_rates,
                        alpha=0.15, color="#2196F3")
        ax.set_xlabel("Micro-batch index", fontsize=11)
        ax.set_ylabel("Alert rate (fraction malicious)", fontsize=11)
        ax.set_title("Streaming Alert Rate Monitor", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        plt.tight_layout()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Alert rate plot saved to {save_path}")

    def plot_feature_drift(self, feature: str, save_path: str) -> None:
        """Plot drift of a specific feature across batches."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        batch_idxs = [b["batch_idx"] for b in self.history
                      if feature in b["drift_details"]]
        current_means = [b["drift_details"][feature]["current_mean"]
                         for b in self.history if feature in b["drift_details"]]
        z_scores = [b["drift_details"][feature]["z_score"]
                    for b in self.history if feature in b["drift_details"]]

        if not batch_idxs:
            log.warning(f"Feature '{feature}' not found in monitoring history.")
            return

        baseline_mean = self.baseline_feature_stats[feature]["mean"]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

        ax1.plot(batch_idxs, current_means, color="#4C72B0", linewidth=1.2, label="Batch mean")
        ax1.axhline(baseline_mean, color="green", linestyle="--", linewidth=1.0,
                    label=f"Baseline mean ({baseline_mean:.2f})")
        ax1.set_ylabel(f"{feature}\nmean", fontsize=9)
        ax1.legend(fontsize=8)
        ax1.set_title(f"Feature Drift Monitor: {feature}", fontsize=12, fontweight="bold")

        ax2.plot(batch_idxs, z_scores, color="#DD8452", linewidth=1.2, label="Z-score")
        ax2.axhline(self.z_threshold, color="red", linestyle="--",
                    linewidth=1.0, label=f"Threshold (z={self.z_threshold})")
        ax2.set_ylabel("Z-score", fontsize=9)
        ax2.set_xlabel("Micro-batch index", fontsize=10)
        ax2.legend(fontsize=8)

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Feature drift plot saved to {save_path}")
