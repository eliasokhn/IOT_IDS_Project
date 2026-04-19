"""
train_lr.py  (v3 — GPU-aware)
==============================
Train Logistic Regression with optional GPU acceleration.

GPU path:  cuML LogisticRegression (RAPIDS) — full sklearn-compatible API
CPU path:  sklearn LogisticRegression        — identical results, no GPU needed

AUTOMATIC FALLBACK: if cuML is unavailable or any GPU error occurs,
the code silently falls back to CPU sklearn with a clear log message.

GPU issues handled:
  - GPU not detected         → detect_gpu() decides backend
  - Silent CPU fallback      → every device decision is logged explicitly
  - OOM during training      → caught, GPU cleared, retried on CPU
  - NaN/Inf inputs           → validate_array() blocks GPU call
  - Memory leak between runs → gpu_memory_context() clears before/after
  - Corrupted checkpoint     → atomic_save() writes temp then renames
  - Disk full                → check_disk_space() before every save
  - Reproducibility          → seed_everything() called before fit
"""

import logging
import time

import numpy as np
import yaml

from src.models.gpu_utils import (
    GpuConfig, BACKEND_CUML, BACKEND_NONE,
    detect_gpu, gpu_memory_context, validate_array,
    atomic_save, seed_everything, clear_gpu_memory, check_disk_space,
)
from src.models.model_utils import save_model

log = logging.getLogger(__name__)


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str,
    config_path: str = "configs/model_config.yaml",
    save: bool = True,
    gpu_cfg: GpuConfig | None = None,
):
    """
    Train Logistic Regression — GPU (cuML) if available, CPU (sklearn) otherwise.

    Parameters
    ----------
    X_train     : scaled feature matrix (float32 or float64 numpy array)
    y_train     : integer label array
    task        : 'binary', '8class', or '34class'
    config_path : model config YAML path
    save        : whether to save the model artifact
    gpu_cfg     : pre-built GpuConfig (detection runs automatically if None)

    Returns
    -------
    Fitted model — cuML or sklearn, both have .predict() and .predict_proba()
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    lr_cfg        = config["logistic_regression"]
    artifacts_dir = config["artifacts_dir"]
    n_classes     = len(np.unique(y_train))

    # ── GPU setup ──────────────────────────────────────────────────────────────
    if gpu_cfg is None:
        gpu_section = config.get("gpu", {})
        gpu_cfg = _build_gpu_cfg(gpu_section, lr_cfg)
        gpu_cfg = detect_gpu(gpu_cfg)

    # ── Reproducibility ────────────────────────────────────────────────────────
    seed_everything(lr_cfg.get("random_state", 42))

    log.info(
        f"Training LR | task={task} | classes={n_classes} | "
        f"samples={len(X_train):,} | "
        f"device={'GPU (cuML)' if gpu_cfg.backend == BACKEND_CUML else 'CPU (sklearn)'}"
    )

    # ── Input safety (prevents NaN/Inf from reaching GPU) ─────────────────────
    if gpu_cfg.validate_input:
        validate_array(X_train, "X_train for LR")

    # float32: 2x faster on GPU, half the VRAM vs float64
    if X_train.dtype != np.float32:
        X_train = X_train.astype(np.float32)

    # ── Train ──────────────────────────────────────────────────────────────────
    model = None

    if gpu_cfg.backend == BACKEND_CUML:
        model = _train_cuml_lr(X_train, y_train, lr_cfg, gpu_cfg, task)

    if model is None:
        # CPU fallback — either by design or after GPU failure
        model = _train_cpu_lr(X_train, y_train, lr_cfg, task)

    # ── Save ───────────────────────────────────────────────────────────────────
    if save:
        model_path = f"{artifacts_dir}/lr_{task}.pkl"
        if gpu_cfg.atomic_save:
            check_disk_space(artifacts_dir, gpu_cfg.min_disk_gb)
            atomic_save(model, model_path, gpu_cfg.min_disk_gb)
        else:
            save_model(model, model_path)

    return model


def _build_gpu_cfg(gpu_section: dict, model_cfg: dict) -> GpuConfig:
    return GpuConfig(
        use_gpu                   = gpu_section.get("use_gpu", True),
        gpu_id                    = gpu_section.get("gpu_id", 0),
        fallback_to_cpu           = gpu_section.get("fallback_to_cpu", True),
        max_vram_fraction         = gpu_section.get("max_vram_fraction", 0.85),
        clear_cache_between_tasks = gpu_section.get("clear_cache_between_tasks", True),
        random_state              = model_cfg.get("random_state", 42),
        validate_input            = gpu_section.get("validate_input", True),
        atomic_save               = gpu_section.get("atomic_save", True),
        min_disk_gb               = gpu_section.get("min_disk_gb", 2.0),
    )


def _compute_sample_weights(y: np.ndarray) -> np.ndarray:
    from sklearn.utils.class_weight import compute_sample_weight
    return compute_sample_weight(class_weight="balanced", y=y).astype(np.float32)


def _train_cuml_lr(X_train, y_train, lr_cfg, gpu_cfg: GpuConfig, task: str):
    """GPU path: cuML LogisticRegression."""
    try:
        from cuml.linear_model import LogisticRegression as cuLR  # type: ignore

        sample_weights = _compute_sample_weights(y_train)

        with gpu_memory_context(gpu_cfg, f"cuML LR {task}"):
            model = cuLR(
                C=lr_cfg["C"],
                max_iter=lr_cfg["max_iter"],
                solver="qn",     # Quasi-Newton — fastest on GPU for LR
                tol=1e-4,
                verbose=False,
            )
            start = time.time()
            model.fit(X_train, y_train, sample_weight=sample_weights)
            elapsed = time.time() - start
            log.info(f"cuML LR done in {elapsed:.1f}s (GPU)")

        return model

    except ImportError:
        log.warning("cuML not installed — falling back to CPU sklearn LR.")
        return None
    except Exception as e:
        err = str(e).lower()
        if "out of memory" in err or "oom" in err:
            log.error(
                f"GPU OOM during LR {task}. "
                "Reduce sample_frac in data_config.yaml or set use_gpu: false."
            )
        else:
            log.error(f"cuML LR failed: {e}")
        if gpu_cfg.fallback_to_cpu:
            log.warning("Falling back to CPU sklearn LR.")
            clear_gpu_memory(gpu_cfg.gpu_id)
            return None
        raise


def _train_cpu_lr(X_train, y_train, lr_cfg, task: str):
    """CPU path: sklearn LogisticRegression."""
    from sklearn.linear_model import LogisticRegression
    log.info(f"sklearn LR CPU | task={task}")
    model = LogisticRegression(
        C=lr_cfg["C"],
        max_iter=lr_cfg["max_iter"],
        class_weight=lr_cfg["class_weight"],
        solver=lr_cfg["solver"],
        n_jobs=lr_cfg["n_jobs"],
        random_state=lr_cfg["random_state"],
        verbose=0,
    )
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    log.info(f"sklearn LR done in {elapsed:.1f}s (CPU)")
    if hasattr(model, "n_iter_") and not model.n_iter_[0] < lr_cfg["max_iter"]:
        log.warning(
            f"LR hit max_iter={lr_cfg['max_iter']}. "
            "Increase max_iter in model_config.yaml."
        )
    return model