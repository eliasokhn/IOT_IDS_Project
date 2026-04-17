# %% [markdown]
# # Notebook 04 — Gradient Boosting Training  (v3 — GPU-aware)
#
# **GPU acceleration:**
# - LightGBM GPU (device='gpu') if LightGBM is installed
# - XGBoost GPU (device='cuda') as fallback
# - sklearn HistGBM CPU as final fallback (original behaviour)
#
# **To install GPU libraries locally:**
# ```bash
# pip install lightgbm    # GPU auto-detected if CUDA toolkit present
# pip install xgboost     # GPU built-in since 1.6
# ```
#
# **To install in Colab (T4 GPU runtime):**
# ```python
# !pip install lightgbm xgboost -q
# ```
#
# **To force CPU:** Set `use_gpu: false` in `configs/model_config.yaml`
# ---

# %% — Setup
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import os

import logging, pickle
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

import numpy as np
import warnings
warnings.filterwarnings("ignore")

# %% — GPU status check
from src.models.gpu_utils import detect_gpu, print_gpu_status, GpuConfig, benchmark_gpu

cfg = detect_gpu(GpuConfig(use_gpu=True))
print_gpu_status(cfg)

# %% — Load preprocessed splits
with open(str(PROJECT_ROOT / "data/processed/splits.pkl"), "rb") as f:
    splits = pickle.load(f)

X_train = splits["X_train"]
X_val   = splits["X_val"]
X_test  = splits["X_test"]

print(f"X_train: {X_train.shape}  dtype: {X_train.dtype}")
feat_names = splits["feature_names"]
assert "Variance" not in feat_names
print("Feature checks passed ✓")

# %% — Import tools
from src.models.train_gb import train_gradient_boosting
from src.evaluation.metrics import compute_all_metrics, save_metrics
from src.evaluation.plots import plot_confusion_matrix, plot_per_class_recall
from src.data.label_mapping import get_class_names

os.makedirs(str(PROJECT_ROOT / "reports"), exist_ok=True)
os.makedirs(str(PROJECT_ROOT / "models"), exist_ok=True)
all_metrics = []

# ── Helper: safely move predictions to CPU ───────────────────────────────────
def to_cpu(arr):
    """Ensure predictions are CPU numpy — works for sklearn, LightGBM, XGBoost."""
    if hasattr(arr, "to_numpy"):
        return arr.to_numpy()
    if hasattr(arr, "get"):       # cupy array
        return arr.get()
    return np.asarray(arr)

# %% ════════════════════════════════════════════════════════════
# TASK 1: BINARY
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TRAINING: GBM | Binary (2-class)")
print("="*60)

y_train = splits["y_train_binary"]
y_val   = splits["y_val_binary"]
y_test  = splits["y_test_binary"]

gb_binary = train_gradient_boosting(
    X_train, y_train, task="binary",
    config_path=str(PROJECT_ROOT / "configs/model_config.yaml"),
    save=True, gpu_cfg=cfg,
)

y_test_pred  = to_cpu(gb_binary.predict(X_test))
y_test_proba = to_cpu(gb_binary.predict_proba(X_test))
class_names  = get_class_names("binary")

test_m = compute_all_metrics(y_test, y_test_pred, y_test_proba, class_names,
                              task="binary", model_name="gb")
print(f"[TEST] Macro F1: {test_m['macro_f1']:.4f} | "
      f"FPR Benign: {test_m['fpr_benign']:.4f} | "
      f"ROC-AUC: {test_m.get('roc_auc','N/A')}")

save_metrics(test_m, str(PROJECT_ROOT / "reports/metrics_gb_binary.json"))
plot_confusion_matrix(
    test_m["confusion_matrix"], test_m["confusion_matrix_labels"],
    title="GB — Binary — Confusion Matrix (Test Set)",
    save_path=str(PROJECT_ROOT / "reports/cm_gb_binary.png"), normalize=True,
)
all_metrics.append(test_m)

# %% ════════════════════════════════════════════════════════════
# TASK 2: 8-CLASS
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TRAINING: GBM | 8-class")
print("="*60)

y_train = splits["y_train_8class"]
y_val   = splits["y_val_8class"]
y_test  = splits["y_test_8class"]

gb_8class = train_gradient_boosting(
    X_train, y_train, task="8class",
    config_path=str(PROJECT_ROOT / "configs/model_config.yaml"),
    save=True, gpu_cfg=cfg,
)

y_test_pred  = to_cpu(gb_8class.predict(X_test))
y_test_proba = to_cpu(gb_8class.predict_proba(X_test))
class_names  = get_class_names("8class")

test_m = compute_all_metrics(y_test, y_test_pred, y_test_proba, class_names,
                              task="8class", model_name="gb")
print(f"[TEST] Macro F1: {test_m['macro_f1']:.4f}")

save_metrics(test_m, str(PROJECT_ROOT / "reports/metrics_gb_8class.json"))
plot_confusion_matrix(
    test_m["confusion_matrix"], test_m["confusion_matrix_labels"],
    title="GB — 8-class — Confusion Matrix (Test Set)",
    save_path=str(PROJECT_ROOT / "reports/cm_gb_8class.png"), normalize=True,
)
plot_per_class_recall(test_m["per_class"],
    title="GB — 8-class — Per-Class Recall",
    save_path=str(PROJECT_ROOT / "reports/recall_gb_8class.png"))

print("\nPer-class recall:")
for cls, m in test_m["per_class"].items():
    flag = " ⚠ LOW" if m["recall"] < 0.5 else ""
    print(f"  {cls:<35} recall={m['recall']:.4f}{flag}")
all_metrics.append(test_m)

# %% ════════════════════════════════════════════════════════════
# TASK 3: 34-CLASS
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TRAINING: GBM | 34-class")
print("="*60)

y_train = splits["y_train_34class"]
y_val   = splits["y_val_34class"]
y_test  = splits["y_test_34class"]

gb_34class = train_gradient_boosting(
    X_train, y_train, task="34class",
    config_path=str(PROJECT_ROOT / "configs/model_config.yaml"),
    save=True, gpu_cfg=cfg,
)

y_test_pred  = to_cpu(gb_34class.predict(X_test))
y_test_proba = to_cpu(gb_34class.predict_proba(X_test))
class_names  = get_class_names("34class")

test_m = compute_all_metrics(y_test, y_test_pred, y_test_proba, class_names,
                              task="34class", model_name="gb")
print(f"[TEST] Macro F1: {test_m['macro_f1']:.4f}")

save_metrics(test_m, str(PROJECT_ROOT / "reports/metrics_gb_34class.json"))
plot_confusion_matrix(
    test_m["confusion_matrix"], test_m["confusion_matrix_labels"],
    title="GB — 34-class — Confusion Matrix (Test Set)",
    save_path=str(PROJECT_ROOT / "reports/cm_gb_34class.png"), normalize=True, figsize=(16,14),
)
plot_per_class_recall(test_m["per_class"],
    title="GB — 34-class — Per-Class Recall",
    save_path=str(PROJECT_ROOT / "reports/recall_gb_34class.png"), figsize=(20, 5))

print("\nRare class recall:")
for cls in ["XSS","SQLINJECTION","COMMANDINJECTION","BROWSERHIJACKING",
            "UPLOADING_ATTACK","BACKDOOR_MALWARE","DICTIONARYBRUTEFORCE","RECON-PINGSWEEP"]:
    if cls in test_m["per_class"]:
        m = test_m["per_class"][cls]
        print(f"  {cls:<35} recall={m['recall']:.4f}  support={m['support']:,}")
all_metrics.append(test_m)

# %% — Summary + GPU cleanup
from src.models.gpu_utils import clear_gpu_memory

print("\n" + "="*60)
print("GBM SUMMARY")
print("="*60)
print(f"{'Task':<12} {'Macro F1':>10} {'Accuracy':>10} {'FPR Benign':>12}")
print("-"*46)
for m in all_metrics:
    print(f"{m['task']:<12} {m['macro_f1']:>10.4f} {m['accuracy']:>10.4f} "
          f"{m.get('fpr_benign',0):>12.4f}")

# Final GPU memory cleanup — important for notebooks that stay open
if cfg.backend != "none":
    clear_gpu_memory(cfg.gpu_id)
    print("\nGPU memory cleared.")

print("\nAll GBM models saved. Run notebook 05 next.")
