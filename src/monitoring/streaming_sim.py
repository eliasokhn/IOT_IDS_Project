"""
streaming_sim.py
================
Simulates real-time IoT traffic streaming using micro-batches from the test set.

Scenarios
---------
1. Normal traffic  — mostly benign records
2. Attack surge    — inject a wave of attack traffic mid-stream
3. Gradual drift   — slowly increasing proportion of one attack type

This simulates what a deployed IDS would see in production.
"""

import logging
import time
from typing import Callable

import numpy as np
import pandas as pd

from src.monitoring.drift_monitor import DriftMonitor

log = logging.getLogger(__name__)


def run_simulation(
    predict_fn: Callable[[list[dict]], list[dict]],
    monitor: DriftMonitor,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    batch_size: int = 500,
    n_batches: int | None = None,
    scenario: str = "normal",
    attack_class_idx: int = 1,
    inject_at_batch: int = 30,
    inject_for_n_batches: int = 10,
    sleep_between_batches: float = 0.0,
    verbose: bool = True,
) -> dict:
    """
    Run the streaming simulation.

    Parameters
    ----------
    predict_fn          : callable that takes list of dicts and returns list of prediction dicts
    monitor             : DriftMonitor instance (already initialized from training data)
    X_test              : test feature DataFrame (never seen during training)
    y_test              : true labels for X_test
    batch_size          : number of records per micro-batch
    n_batches           : max batches to process (None = process all test data)
    scenario            : 'normal', 'attack_surge', or 'gradual_drift'
    attack_class_idx    : class index to inject in attack scenarios
    inject_at_batch     : which batch number to start injection
    inject_for_n_batches: how many batches to keep injection active
    sleep_between_batches: seconds to sleep between batches (0 for max speed)
    verbose             : log progress

    Returns
    -------
    dict with simulation results and per-batch statistics
    """
    log.info(f"Starting streaming simulation | scenario={scenario} | batch_size={batch_size}")

    total_rows = len(X_test)
    max_batches = n_batches or (total_rows // batch_size)
    feature_names = list(X_test.columns)

    simulation_results = {
        "scenario": scenario,
        "batch_size": batch_size,
        "n_batches_processed": 0,
        "total_records": 0,
        "total_malicious_predicted": 0,
        "total_true_malicious": 0,
        "per_batch": [],
    }

    start_time = time.time()

    for batch_num in range(max_batches):
        # ── Slice this micro-batch ────────────────────────────────────
        start = (batch_num * batch_size) % total_rows
        end = min(start + batch_size, total_rows)
        X_batch = X_test.iloc[start:end].reset_index(drop=True)
        y_batch_true = y_test[start:end]

        # ── Scenario: inject attack traffic ──────────────────────────
        if scenario == "attack_surge":
            if inject_at_batch <= batch_num < inject_at_batch + inject_for_n_batches:
                X_batch, y_batch_true = _inject_attack_traffic(
                    X_batch, y_batch_true, X_test, y_test,
                    attack_class_idx, inject_frac=0.7
                )

        elif scenario == "gradual_drift":
            drift_frac = min(
                0.0 + 0.05 * max(0, batch_num - inject_at_batch),
                0.8
            )
            if drift_frac > 0:
                X_batch, y_batch_true = _inject_attack_traffic(
                    X_batch, y_batch_true, X_test, y_test,
                    attack_class_idx, inject_frac=drift_frac
                )

        # ── Predict ───────────────────────────────────────────────────
        records = X_batch.to_dict(orient="records")
        predictions = predict_fn(records)
        y_pred = np.array([p["predicted_class"] for p in predictions])

        # ── Monitor ───────────────────────────────────────────────────
        batch_record = monitor.check_batch(X_batch, y_pred, feature_names=feature_names)

        # ── True metrics for this batch ───────────────────────────────
        n_true_malicious = int((y_batch_true != 0).sum())
        n_pred_malicious = int((y_pred != 0).sum())

        batch_summary = {
            "batch_num": batch_num + 1,
            "n_records": len(y_pred),
            "n_true_malicious": n_true_malicious,
            "n_pred_malicious": n_pred_malicious,
            "true_alert_rate": round(n_true_malicious / max(len(y_batch_true), 1), 4),
            "pred_alert_rate": round(n_pred_malicious / max(len(y_pred), 1), 4),
            "n_drift_warnings": batch_record["n_drift_warnings"],
            "alert_level": batch_record["alert_level"],
        }

        simulation_results["per_batch"].append(batch_summary)
        simulation_results["total_records"] += len(y_pred)
        simulation_results["total_malicious_predicted"] += n_pred_malicious
        simulation_results["total_true_malicious"] += n_true_malicious

        if verbose and (batch_num + 1) % 10 == 0:
            elapsed = time.time() - start_time
            log.info(
                f"  Batch {batch_num+1}/{max_batches} | "
                f"alert_rate={batch_record['alert_rate']:.1%} | "
                f"level={batch_record['alert_level']} | "
                f"drift_warnings={batch_record['n_drift_warnings']} | "
                f"elapsed={elapsed:.1f}s"
            )

        if sleep_between_batches > 0:
            time.sleep(sleep_between_batches)

    simulation_results["n_batches_processed"] = max_batches
    simulation_results["elapsed_seconds"] = round(time.time() - start_time, 2)
    simulation_results["drift_summary"] = monitor.get_summary()

    log.info(
        f"\nSimulation complete: {max_batches} batches | "
        f"{simulation_results['total_records']:,} records | "
        f"{simulation_results['elapsed_seconds']:.1f}s"
    )
    log.info(f"Drift summary: {monitor.get_summary()}")

    return simulation_results


def _inject_attack_traffic(
    X_batch: pd.DataFrame,
    y_batch: np.ndarray,
    X_pool: pd.DataFrame,
    y_pool: np.ndarray,
    attack_class_idx: int,
    inject_frac: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Replace inject_frac of the batch with attack-class records from the pool.
    """
    n_inject = int(len(X_batch) * inject_frac)
    attack_mask = y_pool == attack_class_idx
    attack_pool = X_pool[attack_mask]

    if len(attack_pool) == 0:
        return X_batch, y_batch

    n_inject = min(n_inject, len(attack_pool))
    attack_sample = attack_pool.sample(n=n_inject, replace=True)
    attack_labels = np.full(n_inject, attack_class_idx)

    # Keep remaining benign rows
    n_keep = len(X_batch) - n_inject
    keep_rows = X_batch.iloc[:n_keep]
    keep_labels = y_batch[:n_keep]

    X_new = pd.concat([keep_rows, attack_sample], ignore_index=True)
    y_new = np.concatenate([keep_labels, attack_labels])

    return X_new, y_new
