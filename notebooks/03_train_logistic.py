# %% [markdown]
# # Notebook 03 — Logistic Regression Training  (v3 — GPU-aware)
#
# **GPU acceleration:** Uses cuML if available, otherwise sklearn CPU.
# **No changes needed** — GPU detection is fully automatic.
#
# **To force CPU:** Set `use_gpu: false` in `configs/model_config.yaml`
# **To check GPU:** Run the GPU status cell below before training.
#
# **Colab GPU setup:**
# Runtime → Change runtime type → T4 GPU
# Then install cuML:
# ```
# !pip install cuml-cu12  # for CUDA 12
# ```
# ---

# %% — Setup
import sys, os
sys.path.insert(0, os.path.abspath(".."))

import logging, pickle
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

import numpy as np
import warnings
warnings.filterwarnings("ignore")

# %% — GPU status check (run this first)
from src.models.gpu_utils import detect_gpu, print_gpu_status, GpuConfig, benchmark_gpu

cfg = detect_gpu(GpuConfig(use_gpu=True))
print_gpu_status(cfg)

# Optionally run a quick benchmark to check GPU speed:
# benchmark_gpu(cfg)

# %% — Load preprocessed splits
with open("../data/processed/splits.pkl", "rb") as f:
    splits = pickle.load(f)

X_train = splits["X_train"]
X_val   = splits["X_val"]
X_test  = splits["X_test"]

print(f"X_train: {X_train.shape}  dtype: {X_train.dtype}")
feat_names = splits["feature_names"]
proto_cols = [f for f in feat_names if f.startswith("proto_")]
print(f"Protocol one-hot cols: {proto_cols}")
assert "Variance" not in feat_names, "Variance should be dropped!"
print("Feature checks passed ✓")

# %% — Import tools
from src.models.train_lr import train_logistic_regression
from src.evaluation.metrics import compute_all_metrics, save_metrics
from src.evaluation.plots import plot_confusion_matrix, plot_per_class_recall
from src.data.label_mapping import get_class_names

os.makedirs("../reports", exist_ok=True)
os.makedirs("../models", exist_ok=True)
all_metrics = []

# %% ════════════════════════════════════════════════════════════
# TASK 1: BINARY
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TRAINING: LR | Binary (2-class)")
print("="*60)

y_train = splits["y_train_binary"]
y_val   = splits["y_val_binary"]
y_test  = splits["y_test_binary"]

# Pass pre-detected cfg to avoid re-running detection 6 times
lr_binary = train_logistic_regression(
    X_train, y_train, task="binary",
    config_path="../configs/model_config.yaml",
    save=True, gpu_cfg=cfg,
)

y_val_pred  = lr_binary.predict(X_val)
y_val_proba = lr_binary.predict_proba(X_val)
# Ensure predictions are CPU numpy (safe even if model was on GPU)
if hasattr(y_val_pred, "to_numpy"):
    y_val_pred = y_val_pred.to_numpy()
if hasattr(y_val_proba, "to_numpy"):
    y_val_proba = y_val_proba.to_numpy()

class_names = get_class_names("binary")
val_m = compute_all_metrics(y_val, y_val_pred, y_val_proba, class_names,
                             task="binary", model_name="lr")
print(f"\n[VAL]  Macro F1: {val_m['macro_f1']:.4f} | FPR Benign: {val_m['fpr_benign']:.4f}")

y_test_pred  = lr_binary.predict(X_test)
y_test_proba = lr_binary.predict_proba(X_test)
if hasattr(y_test_pred, "to_numpy"):  y_test_pred  = y_test_pred.to_numpy()
if hasattr(y_test_proba, "to_numpy"): y_test_proba = y_test_proba.to_numpy()

test_m = compute_all_metrics(y_test, y_test_pred, y_test_proba, class_names,
                              task="binary", model_name="lr")
print(f"[TEST] Macro F1: {test_m['macro_f1']:.4f} | FPR Benign: {test_m['fpr_benign']:.4f}")

save_metrics(test_m, "../reports/metrics_lr_binary.json")
plot_confusion_matrix(
    test_m["confusion_matrix"], test_m["confusion_matrix_labels"],
    title="LR — Binary — Confusion Matrix (Test Set)",
    save_path="../reports/cm_lr_binary.png", normalize=True,
)
all_metrics.append(test_m)

# %% ════════════════════════════════════════════════════════════
# TASK 2: 8-CLASS
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TRAINING: LR | 8-class")
print("="*60)

y_train = splits["y_train_8class"]
y_val   = splits["y_val_8class"]
y_test  = splits["y_test_8class"]

lr_8class = train_logistic_regression(
    X_train, y_train, task="8class",
    config_path="../configs/model_config.yaml",
    save=True, gpu_cfg=cfg,
)

y_test_pred  = lr_8class.predict(X_test)
y_test_proba = lr_8class.predict_proba(X_test)
if hasattr(y_test_pred,  "to_numpy"): y_test_pred  = y_test_pred.to_numpy()
if hasattr(y_test_proba, "to_numpy"): y_test_proba = y_test_proba.to_numpy()

class_names = get_class_names("8class")
test_m = compute_all_metrics(y_test, y_test_pred, y_test_proba, class_names,
                              task="8class", model_name="lr")
print(f"[TEST] Macro F1: {test_m['macro_f1']:.4f}")

save_metrics(test_m, "../reports/metrics_lr_8class.json")
plot_confusion_matrix(
    test_m["confusion_matrix"], test_m["confusion_matrix_labels"],
    title="LR — 8-class — Confusion Matrix (Test Set)",
    save_path="../reports/cm_lr_8class.png", normalize=True,
)
plot_per_class_recall(test_m["per_class"],
    title="LR — 8-class — Per-Class Recall",
    save_path="../reports/recall_lr_8class.png")
all_metrics.append(test_m)

# %% ════════════════════════════════════════════════════════════
# TASK 3: 34-CLASS
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TRAINING: LR | 34-class")
print("="*60)

y_train = splits["y_train_34class"]
y_val   = splits["y_val_34class"]
y_test  = splits["y_test_34class"]

lr_34class = train_logistic_regression(
    X_train, y_train, task="34class",
    config_path="../configs/model_config.yaml",
    save=True, gpu_cfg=cfg,
)

y_test_pred  = lr_34class.predict(X_test)
y_test_proba = lr_34class.predict_proba(X_test)
if hasattr(y_test_pred,  "to_numpy"): y_test_pred  = y_test_pred.to_numpy()
if hasattr(y_test_proba, "to_numpy"): y_test_proba = y_test_proba.to_numpy()

class_names = get_class_names("34class")
test_m = compute_all_metrics(y_test, y_test_pred, y_test_proba, class_names,
                              task="34class", model_name="lr")
print(f"[TEST] Macro F1: {test_m['macro_f1']:.4f}")

save_metrics(test_m, "../reports/metrics_lr_34class.json")
plot_confusion_matrix(
    test_m["confusion_matrix"], test_m["confusion_matrix_labels"],
    title="LR — 34-class — Confusion Matrix (Test Set)",
    save_path="../reports/cm_lr_34class.png", normalize=True, figsize=(16,14),
)
plot_per_class_recall(test_m["per_class"],
    title="LR — 34-class — Per-Class Recall",
    save_path="../reports/recall_lr_34class.png", figsize=(20, 5))
all_metrics.append(test_m)

# %% — Summary
print("\n" + "="*60)
print("LR SUMMARY")
print("="*60)
print(f"{'Task':<12} {'Macro F1':>10} {'Accuracy':>10} {'FPR Benign':>12}")
print("-"*46)
for m in all_metrics:
    print(f"{m['task']:<12} {m['macro_f1']:>10.4f} {m['accuracy']:>10.4f} "
          f"{m.get('fpr_benign',0):>12.4f}")
print("\nAll LR models saved. Run notebook 04 next.")
