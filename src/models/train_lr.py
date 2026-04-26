"""
train_lr.py
===========
Train Logistic Regression with optional GPU acceleration.

GPU path:  cuML LogisticRegression (RAPIDS)
CPU path:  sklearn LogisticRegression

Uses class_weight='balanced' as a base, then applies an extra multiplier
on BENIGN samples from model_config.yaml to reduce the false positive rate.
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

# BENIGN class integer index for each task (mirrors train_gb.py)
BENIGN_IDX = {"binary": 0, "8class": 0, "34class": 1}


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

    if gpu_cfg is None:
        gpu_section = config.get("gpu", {})
        gpu_cfg = _build_gpu_cfg(gpu_section, lr_cfg)
        gpu_cfg = detect_gpu(gpu_cfg)

    seed_everything(lr_cfg.get("random_state", 42))

    log.info(
        f"Training LR | task={task} | classes={n_classes} | "
        f"samples={len(X_train):,} | "
        f"device={'GPU (cuML)' if gpu_cfg.backend == BACKEND_CUML else 'CPU (sklearn)'}"
    )

    if gpu_cfg.validate_input:
        validate_array(X_train, "X_train for LR")

    if X_train.dtype != np.float32:
        X_train = X_train.astype(np.float32)

    model = None

    if gpu_cfg.backend == BACKEND_CUML:
        model = _train_cuml_lr(X_train, y_train, lr_cfg, gpu_cfg, task)

    if model is None:
        model = _train_cpu_lr(X_train, y_train, lr_cfg, task)

    if save:
        model_path = f"{artifacts_dir}/lr_{task}.pkl"
        if gpu_cfg.atomic_save:
            check_disk_space(artifacts_dir, gpu_cfg.min_disk_gb)
            atomic_save(model, model_path, gpu_cfg.min_disk_gb)
        else:
            save_model(model, model_path)

    return model


def _resolve_multiplier(multiplier_cfg, task: str) -> float:
    """Resolve benign_weight_multiplier — supports scalar or per-task dict."""
    if isinstance(multiplier_cfg, dict):
        return float(multiplier_cfg.get(task, 1.0))
    return float(multiplier_cfg) if multiplier_cfg is not None else 1.0


def _compute_sample_weights(
    y: np.ndarray,
    task: str,
    benign_multiplier: float = 1.0,
) -> np.ndarray:
    """
    Compute per-sample weights: class_weight='balanced' base plus an optional
    extra multiplier on BENIGN samples.
    """
    from sklearn.utils.class_weight import compute_sample_weight
    weights = compute_sample_weight(class_weight="balanced", y=y).astype(np.float32)
    if benign_multiplier != 1.0:
        benign_idx = BENIGN_IDX.get(task, 0)
        weights[y == benign_idx] *= benign_multiplier
    return weights


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


def _train_cuml_lr(X_train, y_train, lr_cfg, gpu_cfg: GpuConfig, task: str):
    """GPU path: cuML LogisticRegression with BENIGN-boosted sample weights."""
    try:
        from cuml.linear_model import LogisticRegression as cuLR  # type: ignore

        with gpu_memory_context(gpu_cfg, f"cuML LR {task}"):
            model = cuLR(
                C=lr_cfg["C"],
                max_iter=lr_cfg["max_iter"],
                solver="qn",
                tol=1e-4,
                verbose=False,
            )
            benign_multiplier = _resolve_multiplier(lr_cfg.get("benign_weight_multiplier", 1.0), task)
            sample_weights    = _compute_sample_weights(y_train, task, benign_multiplier)
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
            log.error(f"GPU OOM during LR {task}. Reduce sample_frac or set use_gpu: false.")
        else:
            log.error(f"cuML LR failed: {e}")
        if gpu_cfg.fallback_to_cpu:
            log.warning("Falling back to CPU sklearn LR.")
            clear_gpu_memory(gpu_cfg.gpu_id)
            return None
        raise


def _train_cpu_lr(X_train, y_train, lr_cfg, task: str):
    """CPU path: sklearn LogisticRegression with BENIGN-boosted sample weights."""
    from sklearn.linear_model import LogisticRegression
    log.info(f"sklearn LR CPU | task={task}")

    benign_multiplier = _resolve_multiplier(lr_cfg.get("benign_weight_multiplier", 1.0), task)
    sample_weights    = _compute_sample_weights(y_train, task, benign_multiplier)

    # class_weight=None because sample_weight already includes the balanced correction
    model = LogisticRegression(
        C=lr_cfg["C"],
        max_iter=lr_cfg["max_iter"],
        class_weight=None,
        solver=lr_cfg["solver"],
        n_jobs=lr_cfg["n_jobs"],
        random_state=lr_cfg["random_state"],
        verbose=0,
    )
    start = time.time()
    model.fit(X_train, y_train, sample_weight=sample_weights)
    elapsed = time.time() - start
    log.info(f"sklearn LR done in {elapsed:.1f}s (CPU)")
    if hasattr(model, "n_iter_") and not model.n_iter_[0] < lr_cfg["max_iter"]:
        log.warning(
            f"LR hit max_iter={lr_cfg['max_iter']}. "
            "Increase max_iter in model_config.yaml if time allows."
        )
    return model
