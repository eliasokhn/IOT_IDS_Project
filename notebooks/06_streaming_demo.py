# %% [markdown]
# # Notebook 06 — Streaming Simulation & Drift Monitoring
#
# **Goal:** Simulate real-time IoT traffic arriving in micro-batches.
# Monitor for feature drift and alert rate spikes across 3 scenarios:
# 1. Normal traffic
# 2. Sudden attack surge (injected at batch 30)
# 3. Gradual drift (slowly increasing attack fraction)
# ---

# %% — Setup
import sys, os
sys.path.insert(0, os.path.abspath(".."))

import logging, pickle, json
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# %% — Load split data and trained model
with open("../data/processed/splits.pkl", "rb") as f:
    splits = pickle.load(f)

X_test_raw = splits["X_test_raw"]     # unscaled, as DataFrame
X_test     = splits["X_test"]         # scaled numpy array
y_test     = splits["y_test_binary"]  # binary labels for monitoring

print(f"Test set: {X_test_raw.shape[0]:,} records, {X_test_raw.shape[1]} features")

# %% — Load predictor (binary GB model)
from src.serving.predictor import Predictor

try:
    predictor = Predictor.load(
        artifacts_dir="../models",
        task="binary",
        model_name="gb",
    )
    print("Predictor loaded successfully.")
    USE_PREDICTOR = True
except FileNotFoundError:
    print("WARNING: Model not found. Streaming demo will use random predictions.")
    print("Run notebooks 02-04 first to train models.")
    USE_PREDICTOR = False

# %% — Build predict function for streaming simulation
def predict_fn(records: list[dict]) -> list[dict]:
    """Wrapper: takes list of feature dicts, returns list of prediction dicts."""
    if USE_PREDICTOR:
        return predictor.predict_batch(records)
    else:
        # Fallback: random predictions for demo purposes
        import random
        return [
            {
                "predicted_class": random.choices([0, 1], weights=[0.9, 0.1])[0],
                "is_malicious": False,
                "predicted_label": "Benign",
                "probabilities": {"Benign": 0.9, "Malicious": 0.1},
            }
            for _ in records
        ]

# %% — Build DriftMonitor from training data
from src.monitoring.drift_monitor import DriftMonitor

X_train_raw_sample = X_test_raw.iloc[:5000]  # Use a subset as "training baseline"

monitor = DriftMonitor.from_training_data(
    X_train=X_train_raw_sample,
    n_top_features=5,
    baseline_alert_rate=0.05,
    z_threshold=3.0,
    alert_rate_warning=0.30,
    alert_rate_critical=0.60,
    log_dir="../reports/monitoring_logs",
)
print(f"DriftMonitor tracking features: {monitor.top_features}")

# %% ─────────────────────────────────────────────────────────────
# SCENARIO 1: Normal Traffic
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SCENARIO 1: Normal Traffic")
print("="*60)

from src.monitoring.streaming_sim import run_simulation

monitor_normal = DriftMonitor.from_training_data(
    X_train=X_train_raw_sample,
    n_top_features=5,
    baseline_alert_rate=0.05,
    log_dir="../reports/monitoring_logs",
)

results_normal = run_simulation(
    predict_fn=predict_fn,
    monitor=monitor_normal,
    X_test=X_test_raw,
    y_test=y_test,
    batch_size=500,
    n_batches=50,
    scenario="normal",
    verbose=True,
)

print(f"\nNormal scenario summary:")
print(json.dumps(results_normal["drift_summary"], indent=2))

monitor_normal.plot_alert_rate(
    save_path="../reports/monitoring_plots/alert_rate_normal.png"
)

# %% ─────────────────────────────────────────────────────────────
# SCENARIO 2: Attack Surge (injected at batch 20)
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SCENARIO 2: Attack Surge at batch 20")
print("="*60)

monitor_surge = DriftMonitor.from_training_data(
    X_train=X_train_raw_sample,
    n_top_features=5,
    baseline_alert_rate=0.05,
    log_dir="../reports/monitoring_logs",
)

results_surge = run_simulation(
    predict_fn=predict_fn,
    monitor=monitor_surge,
    X_test=X_test_raw,
    y_test=y_test,
    batch_size=500,
    n_batches=50,
    scenario="attack_surge",
    attack_class_idx=1,
    inject_at_batch=20,
    inject_for_n_batches=10,
    verbose=True,
)

print(f"\nSurge scenario summary:")
print(json.dumps(results_surge["drift_summary"], indent=2))

monitor_surge.plot_alert_rate(
    save_path="../reports/monitoring_plots/alert_rate_surge.png"
)

# %% ─────────────────────────────────────────────────────────────
# SCENARIO 3: Gradual Drift
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SCENARIO 3: Gradual Drift starting at batch 10")
print("="*60)

monitor_drift = DriftMonitor.from_training_data(
    X_train=X_train_raw_sample,
    n_top_features=5,
    baseline_alert_rate=0.05,
    log_dir="../reports/monitoring_logs",
)

results_drift = run_simulation(
    predict_fn=predict_fn,
    monitor=monitor_drift,
    X_test=X_test_raw,
    y_test=y_test,
    batch_size=500,
    n_batches=60,
    scenario="gradual_drift",
    attack_class_idx=1,
    inject_at_batch=10,
    verbose=True,
)

print(f"\nGradual drift summary:")
print(json.dumps(results_drift["drift_summary"], indent=2))

monitor_drift.plot_alert_rate(
    save_path="../reports/monitoring_plots/alert_rate_gradual.png"
)

# Plot feature drift for one tracked feature
if monitor_drift.top_features:
    feat = monitor_drift.top_features[0]
    monitor_drift.plot_feature_drift(
        feature=feat,
        save_path=f"../reports/monitoring_plots/feature_drift_{feat}.png",
    )

# %% — Side-by-side comparison of all 3 scenarios
fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)

for ax, results, monitor_obj, title, color in zip(
    axes,
    [results_normal, results_surge, results_drift],
    [monitor_normal, monitor_surge, monitor_drift],
    ["Normal", "Attack Surge", "Gradual Drift"],
    ["#2196F3", "#F44336", "#FF9800"],
):
    batches = [b["batch_num"] for b in results["per_batch"]]
    alert_rates = [b["pred_alert_rate"] for b in results["per_batch"]]

    ax.plot(batches, alert_rates, color=color, linewidth=1.5)
    ax.fill_between(batches, alert_rates, alpha=0.2, color=color)
    ax.axhline(0.30, color="orange", linestyle="--", linewidth=0.8, label="Warning")
    ax.axhline(0.60, color="red", linestyle="--", linewidth=0.8, label="Critical")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Micro-batch")
    ax.legend(fontsize=7)

axes[0].set_ylabel("Alert Rate")
fig.suptitle("Streaming Alert Rate — 3 Scenarios", fontsize=13, fontweight="bold")
plt.tight_layout()

os.makedirs("../reports/monitoring_plots", exist_ok=True)
fig.savefig("../reports/monitoring_plots/scenarios_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nScenario comparison plot saved.")

# %% — Save all monitoring logs
monitor_normal.save_log("monitoring_log_normal.json")
monitor_surge.save_log("monitoring_log_surge.json")
monitor_drift.save_log("monitoring_log_gradual.json")

print("\nStreaming simulation complete. All results saved to reports/monitoring_plots/")
