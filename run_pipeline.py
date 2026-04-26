"""
run_pipeline.py
===============
Runs the full IoT IDS training pipeline in one command.

Usage:
    python run_pipeline.py                      # demo data (no dataset needed)
    python run_pipeline.py --real-data          # 40% sample of CICIoT2023
    python run_pipeline.py --real-data --lr-sample-frac 0.30
    python run_pipeline.py --skip-train         # re-evaluate saved models
    python run_pipeline.py --real-data --models gb --tasks binary 8class
"""

import argparse, logging, os, sys, pickle, json, glob
import numpy as np
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# BENIGN class index per task.
# 34-class labels are sorted alphabetically: BACKDOOR_MALWARE=0, BENIGN=1.
# Must be 1 for 34class — using 0 would measure BACKDOOR_MALWARE's error rate.
BENIGN_IDX = {"binary": 0, "8class": 0, "34class": 1}


def parse_args():
    p = argparse.ArgumentParser(description="IoT IDS Full Pipeline v2")
    p.add_argument("--real-data",       action="store_true")
    p.add_argument("--skip-train",      action="store_true")
    p.add_argument("--sample-frac",     type=float, default=0.40,
                   help="Sample fraction for GB (and LR if --lr-sample-frac not set)")
    p.add_argument("--lr-sample-frac",  type=float, default=None,
                   help="Sample fraction for LR only. If smaller than --sample-frac, "
                        "LR training data is subsampled from the already-loaded dataset. "
                        "Data loading/dedup uses --sample-frac (the larger value).")
    p.add_argument("--tasks",   nargs="+", default=["binary","8class","34class"])
    p.add_argument("--models",  nargs="+", default=["lr","gb"])
    return p.parse_args()


def _subsample_train(X_train, splits_data, tasks, target_n, random_state=42):
    """
    Stratified subsample of X_train + all y_train arrays down to target_n rows.
    Stratifies on the 34-class labels for maximum representativeness.
    """
    y_34 = splits_data["y_train_34class"]
    rng  = np.random.RandomState(random_state)

    # Stratified sampling: keep proportional class counts
    classes, counts = np.unique(y_34, return_counts=True)
    ratio = target_n / len(X_train)
    keep_idx = []
    for cls, cnt in zip(classes, counts):
        cls_idx = np.where(y_34 == cls)[0]
        n_keep  = max(1, int(round(cnt * ratio)))
        n_keep  = min(n_keep, len(cls_idx))
        chosen  = rng.choice(cls_idx, size=n_keep, replace=False)
        keep_idx.extend(chosen.tolist())

    keep_idx = np.array(sorted(keep_idx))
    X_sub = X_train[keep_idx]
    y_sub = {task: splits_data[f"y_train_{task}"][keep_idx] for task in tasks}
    log.info(
        f"LR training data subsampled: {len(X_train):,} → {len(X_sub):,} rows "
        f"({100*len(X_sub)/len(X_train):.1f}% of full train set)"
    )
    return X_sub, y_sub


def step_load_data(use_demo: bool, sample_frac: float):
    log.info("="*60 + "\nSTEP 1: Load and validate data\n" + "="*60)
    from src.data.loader import load_dataset, generate_demo_data, build_merged_parquet
    from src.data.label_mapping import add_all_label_columns
    from src.data.validation import validate_dataset

    if use_demo:
        log.info("Using synthetic demo data")
        df_raw = generate_demo_data(n_samples=50_000, random_state=42)
    else:
        build_merged_parquet("data/raw", "data/processed/merged.parquet")
        df_raw = load_dataset("data/processed/merged.parquet",
                              sample_frac=sample_frac, random_state=42)

    validate_dataset(df_raw, label_col="label")
    df = add_all_label_columns(df_raw, raw_col="label")
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/dataset_labelled.pkl", "wb") as f:
        pickle.dump(df, f)
    log.info(f"Dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def step_preprocess(df):
    log.info("="*60 + "\nSTEP 2: Dedup + Preprocess + Split\n" + "="*60)
    from src.features.preprocessing import deduplicate, build_and_fit_preprocessor
    from src.data.splitter import create_splits, get_X_y
    from src.models.model_utils import save_class_names
    from src.data.label_mapping import BINARY_CLASS_NAMES, CATEGORY_CLASS_NAMES, FINE_CLASS_NAMES

    df = deduplicate(df, label_col="label", random_state=42)

    train_df, val_df, test_df = create_splits(
        df, train_frac=0.70, val_frac=0.15, test_frac=0.15,
        stratify_col="label_fine", random_state=42,
        save_path="data/processed/split_indices.pkl",
    )

    X_train_raw, _ = get_X_y(train_df, "binary")
    os.makedirs("models", exist_ok=True)

    preprocessor = build_and_fit_preprocessor(X_train_raw, save_path="models/preprocessor.pkl")

    splits_data = {}
    for sname, sdf in [("train",train_df),("val",val_df),("test",test_df)]:
        X_raw, _ = get_X_y(sdf, "binary")
        splits_data[f"X_{sname}"] = preprocessor.transform(X_raw)
        for task in ["binary","8class","34class"]:
            _, y = get_X_y(sdf, task)
            splits_data[f"y_{sname}_{task}"] = y.values

    X_test_raw, _ = get_X_y(test_df, "binary")
    splits_data["X_test_raw"]    = X_test_raw
    splits_data["feature_names"] = preprocessor.get_feature_names()

    with open("data/processed/splits.pkl","wb") as f:
        pickle.dump(splits_data, f)

    save_class_names(BINARY_CLASS_NAMES,   "models/class_names_binary.json")
    save_class_names(CATEGORY_CLASS_NAMES, "models/class_names_8class.json")
    save_class_names(FINE_CLASS_NAMES,     "models/class_names_34class.json")

    proto_cols = [n for n in preprocessor.get_feature_names() if n.startswith("proto_")]
    log.info(f"Output features: {len(preprocessor.get_feature_names())}  "
             f"(proto one-hot: {proto_cols}, Variance: removed)")
    return splits_data


def step_train(splits_data, tasks, models_to_train, lr_sample_frac=None, gb_sample_frac=None):
    log.info("="*60 + "\nSTEP 3: Model training\n" + "="*60)
    from src.models.train_lr import train_logistic_regression
    from src.models.train_gb import train_gradient_boosting
    from src.evaluation.metrics import compute_all_metrics, save_metrics
    from src.evaluation.plots import plot_confusion_matrix, plot_per_class_recall
    from src.data.label_mapping import get_class_names

    os.makedirs("reports", exist_ok=True)
    all_metrics = []
    X_train_full = splits_data["X_train"]
    X_test       = splits_data["X_test"]

    # Precompute subsampled LR training data if a smaller frac was requested
    if lr_sample_frac and gb_sample_frac and lr_sample_frac < gb_sample_frac:
        target_n = int(len(X_train_full) * lr_sample_frac / gb_sample_frac)
        log.info(
            f"LR will train on {lr_sample_frac*100:.0f}% data "
            f"({target_n:,} rows), GB on {gb_sample_frac*100:.0f}% "
            f"({len(X_train_full):,} rows)."
        )
        X_train_lr, y_train_lr_map = _subsample_train(
            X_train_full, splits_data, tasks, target_n
        )
    else:
        X_train_lr, y_train_lr_map = None, None

    fns = {}
    if "lr" in models_to_train: fns["lr"] = train_logistic_regression
    if "gb" in models_to_train: fns["gb"] = train_gradient_boosting

    for model_key, train_fn in fns.items():
        # Pick the right training set for this model
        if model_key == "lr" and X_train_lr is not None:
            X_train = X_train_lr
            y_train_src = y_train_lr_map
        else:
            X_train = X_train_full
            y_train_src = None   # will read from splits_data below

        for task in tasks:
            log.info(f"\n--- {model_key.upper()} | {task} ---")
            y_train = y_train_src[task] if y_train_src else splits_data[f"y_train_{task}"]
            y_test  = splits_data[f"y_test_{task}"]
            cn = get_class_names(task)

            model   = train_fn(X_train, y_train, task=task,
                               config_path="configs/model_config.yaml", save=True)
            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            m = compute_all_metrics(y_test, y_pred, y_proba, cn,
                                    task=task, model_name=model_key,
                                    benign_class_idx=BENIGN_IDX.get(task, 0))
            save_metrics(m, f"reports/metrics_{model_key}_{task}.json")

            fs = (16,14) if task == "34class" else (10,8)
            plot_confusion_matrix(m["confusion_matrix"], m["confusion_matrix_labels"],
                title=f"{model_key.upper()} — {task}", normalize=True,
                save_path=f"reports/cm_{model_key}_{task}.png", figsize=fs)
            if task != "binary":
                plot_per_class_recall(m["per_class"],
                    title=f"{model_key.upper()} {task} recall",
                    save_path=f"reports/recall_{model_key}_{task}.png")

            log.info(f"  macro_f1={m['macro_f1']:.4f}  "
                     f"fpr_benign={m.get('fpr_benign',0):.4f}  "
                     f"accuracy={m['accuracy']:.4f}")
            all_metrics.append(m)
    return all_metrics


def step_evaluate(all_metrics):
    log.info("="*60 + "\nSTEP 4: Evaluation comparison\n" + "="*60)
    from src.evaluation.metrics import build_comparison_table
    from src.evaluation.plots import plot_model_comparison

    df = build_comparison_table(all_metrics)
    df.to_csv("reports/model_comparison.csv", index=False)
    log.info("\nMODEL COMPARISON (Macro F1):\n" + df.to_string(index=False))
    if len(df) >= 2:
        plot_model_comparison(df, save_path="reports/comparison_macro_f1.png",
                              metric="macro_f1")
    return df


def step_streaming():
    log.info("="*60 + "\nSTEP 5: Streaming simulation\n" + "="*60)
    with open("data/processed/splits.pkl","rb") as f:
        splits = pickle.load(f)
    X_test_raw = splits["X_test_raw"]
    y_test     = splits["y_test_binary"]

    from src.monitoring.drift_monitor import DriftMonitor
    from src.monitoring.streaming_sim import run_simulation

    try:
        from src.serving.predictor import Predictor
        predictor = Predictor.load("models/", task="binary", model_name="gb")
        def predict_fn(records): return predictor.predict_batch(records)
        log.info("Using trained GB binary model")
    except FileNotFoundError:
        import random
        def predict_fn(records):
            return [{"predicted_class": random.choices([0,1],weights=[0.85,0.15])[0],
                     "is_malicious":False,"predicted_label":"Benign",
                     "probabilities":{"Benign":0.85,"Malicious":0.15}}
                    for _ in records]
        log.warning("Model not found — using random predictions for demo")

    os.makedirs("reports/monitoring_plots", exist_ok=True)
    os.makedirs("reports/monitoring_logs",  exist_ok=True)
    baseline = X_test_raw.iloc[:min(5000, len(X_test_raw))]

    for scenario in ["normal","attack_surge","gradual_drift"]:
        monitor = DriftMonitor.from_training_data(X_train=baseline, n_top_features=5,
            baseline_alert_rate=0.05, log_dir="reports/monitoring_logs")
        results = run_simulation(predict_fn, monitor, X_test_raw, y_test,
            batch_size=500, n_batches=50, scenario=scenario,
            inject_at_batch=20, inject_for_n_batches=10, verbose=False)
        monitor.plot_alert_rate(f"reports/monitoring_plots/alert_rate_{scenario}.png")
        monitor.save_log(f"monitoring_log_{scenario}.json")
        log.info(f"  {scenario}: {results['drift_summary']}")


def main():
    args = parse_args()
    log.info("IoT IDS Pipeline starting...")

    lr_frac = args.lr_sample_frac if args.lr_sample_frac else args.sample_frac
    gb_frac = args.sample_frac

    if not args.skip_train:
        df = step_load_data(not args.real_data, gb_frac)
        splits_data = step_preprocess(df)
        all_metrics = step_train(
            splits_data, args.tasks, args.models,
            lr_sample_frac=lr_frac,
            gb_sample_frac=gb_frac,
        )
        step_evaluate(all_metrics)
    else:
        log.info("Skipping training — loading saved metrics...")
        all_metrics = [json.load(open(p)) for p in glob.glob("reports/metrics_*.json")]
        if all_metrics:
            step_evaluate(all_metrics)

    if os.path.exists("data/processed/splits.pkl"):
        step_streaming()

    log.info("\n" + "="*60)
    log.info("PIPELINE COMPLETE")
    log.info("  models/          ← trained .pkl artifacts")
    log.info("  reports/         ← metrics JSON + PNG plots")
    log.info("  reports/monitoring_plots/  ← drift charts")
    log.info("  Start API: uvicorn src.serving.api:app --reload")
    log.info("="*60)

if __name__ == "__main__":
    main()
