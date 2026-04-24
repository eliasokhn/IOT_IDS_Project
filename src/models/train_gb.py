"""
train_gb.py  (v4 — GPU-aware + checkpoint/resume)
==================================================
Train Gradient Boosting with optional GPU acceleration and pause/resume.

GPU path A: LightGBM  device='gpu'                      — fastest, recommended
GPU path B: XGBoost   device='cuda'                      — fallback
CPU path:   sklearn HistGradientBoostingClassifier        — always works

PAUSE / RESUME
--------------
LightGBM supports resuming from a checkpoint via init_model.
Every `checkpoint_every` trees, a checkpoint file is saved to:
    models/checkpoints/gb_{task}_checkpoint.txt

If training is interrupted (Ctrl+C, power loss, crash), the next run
automatically detects the checkpoint and continues from that point.
When training completes successfully, the checkpoint is deleted.

To force a fresh start (ignore existing checkpoint):
    Delete models/checkpoints/gb_{task}_checkpoint.txt
    Or set checkpoint_resume: false in model_config.yaml

sklearn CPU fallback has no checkpoint support — it trains from scratch.
If interrupted, re-run the notebook and it will restart from zero.
This is acceptable because CPU GBM on balanced ~680K rows takes
roughly the same time as LightGBM GPU, so checkpointing is less critical.
"""

import logging
import os
import time
from pathlib import Path

import numpy as np
import yaml

from src.models.gpu_utils import (
    GpuConfig, BACKEND_NONE, BACKEND_LGBM, BACKEND_XGB, BACKEND_CUML,
    detect_gpu, gpu_memory_context, validate_array,
    atomic_save, seed_everything, clear_gpu_memory, check_disk_space,
    get_gpu_memory_info,
)
from src.models.model_utils import save_model

log = logging.getLogger(__name__)


def train_gradient_boosting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str,
    config_path: str = "configs/model_config.yaml",
    save: bool = True,
    gpu_cfg: GpuConfig | None = None,
):
    """
    Train Gradient Boosting — GPU (LightGBM/XGBoost) if available, CPU otherwise.

    Supports pause/resume for LightGBM GPU path.
    If a checkpoint exists from a previous interrupted run, training
    automatically continues from that point.

    Parameters
    ----------
    X_train     : scaled feature matrix
    y_train     : integer label array
    task        : 'binary', '8class', or '34class'
    config_path : model config YAML path
    save        : whether to save the final model artifact
    gpu_cfg     : pre-built GpuConfig (detection runs automatically if None)
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    gb_cfg        = config["gradient_boosting"]
    artifacts_dir = config["artifacts_dir"]
    ckpt_cfg      = config.get("checkpoint", {})
    n_classes     = len(np.unique(y_train))

    # ── GPU setup ──────────────────────────────────────────────────────────────
    if gpu_cfg is None:
        gpu_section = config.get("gpu", {})
        gpu_cfg     = _build_gpu_cfg(gpu_section, gb_cfg)
        gpu_cfg     = detect_gpu(gpu_cfg)

    seed_everything(gb_cfg.get("random_state", 42))

    log.info(
        f"Training GBM | task={task} | classes={n_classes} | "
        f"samples={len(X_train):,} | backend={gpu_cfg.backend}"
    )

    if gpu_cfg.validate_input:
        validate_array(X_train, "X_train for GBM")

    if X_train.dtype != np.float32:
        X_train = X_train.astype(np.float32)

    if gpu_cfg.backend != BACKEND_NONE:
        _check_data_fits_vram(X_train, gpu_cfg, task)

    # ── Train ──────────────────────────────────────────────────────────────────
    model = None

    if gpu_cfg.backend in (BACKEND_LGBM, BACKEND_CUML):
        model = _train_lightgbm_gpu(
            X_train, y_train, gb_cfg, gpu_cfg, task, n_classes,
            artifacts_dir, ckpt_cfg,
        )

    if model is None and gpu_cfg.backend == BACKEND_XGB:
        model = _train_xgboost_gpu(X_train, y_train, gb_cfg, gpu_cfg, task, n_classes)

    if model is None:
        model = _train_cpu_gbm(X_train, y_train, gb_cfg, task)

    # ── Save final model ───────────────────────────────────────────────────────
    if save:
        model_path = f"{artifacts_dir}/gb_{task}.pkl"
        if gpu_cfg.atomic_save:
            check_disk_space(artifacts_dir, gpu_cfg.min_disk_gb)
            atomic_save(model, model_path, gpu_cfg.min_disk_gb)
        else:
            save_model(model, model_path)

    return model


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _checkpoint_path(artifacts_dir: str, task: str) -> Path:
    """Return the path for a LightGBM checkpoint file."""
    ckpt_dir = Path(artifacts_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir / f"gb_{task}_checkpoint.txt"


def _delete_checkpoint(artifacts_dir: str, task: str) -> None:
    """Delete checkpoint after successful training completion."""
    ckpt = _checkpoint_path(artifacts_dir, task)
    if ckpt.exists():
        ckpt.unlink()
        log.info(f"Checkpoint deleted (training complete): {ckpt}")


# ── Training functions ─────────────────────────────────────────────────────────

def _build_gpu_cfg(gpu_section: dict, model_cfg: dict) -> GpuConfig:
    return GpuConfig(
        use_gpu                   = gpu_section.get("use_gpu", True),
        gpu_id                    = gpu_section.get("gpu_id", 0),
        fallback_to_cpu           = gpu_section.get("fallback_to_cpu", True),
        max_vram_fraction         = gpu_section.get("max_vram_fraction", 0.85),
        batch_size_mb             = gpu_section.get("batch_size_mb", 1024),
        clear_cache_between_tasks = gpu_section.get("clear_cache_between_tasks", True),
        random_state              = model_cfg.get("random_state", 42),
        validate_input            = gpu_section.get("validate_input", True),
        atomic_save               = gpu_section.get("atomic_save", True),
        min_disk_gb               = gpu_section.get("min_disk_gb", 2.0),
    )


def _check_data_fits_vram(X: np.ndarray, cfg: GpuConfig, task: str) -> None:
    data_mb           = X.nbytes / (1024 ** 2)
    mem               = get_gpu_memory_info(cfg.gpu_id)
    free_gb           = mem.get("free_gb", cfg.free_vram_gb)
    available_mb      = free_gb * 1024 * cfg.max_vram_fraction
    estimated_need_mb = data_mb * 3    # LightGBM needs ~3x for histograms

    log.info(
        f"[GPU] Data: {data_mb:.0f} MB | "
        f"Available VRAM: {available_mb:.0f} MB | "
        f"Estimated needed: {estimated_need_mb:.0f} MB"
    )

    if estimated_need_mb > available_mb:
        log.warning(
            f"[GPU] Data may exceed VRAM for task={task}. "
            "Reduce balanced_ceiling in data_config.yaml if OOM occurs."
        )


def _compute_sample_weights(y: np.ndarray) -> np.ndarray:
    from sklearn.utils.class_weight import compute_sample_weight
    return compute_sample_weight(class_weight="balanced", y=y).astype(np.float32)


def _train_lightgbm_gpu(
    X_train, y_train, gb_cfg, gpu_cfg: GpuConfig,
    task: str, n_classes: int,
    artifacts_dir: str, ckpt_cfg: dict,
):
    """
    GPU path: LightGBM with device='gpu' and checkpoint/resume support.

    Checkpoint behaviour:
      - Every `checkpoint_every` trees, saves a .txt checkpoint file
      - On next run, if checkpoint exists, resumes from it automatically
      - On successful completion, checkpoint is deleted
      - To restart from scratch: delete models/checkpoints/gb_{task}_checkpoint.txt
    """
    try:
        import lightgbm as lgb  # type: ignore

        # ── Checkpoint config ─────────────────────────────────────────────────
        checkpoint_enabled = ckpt_cfg.get("enabled", True)
        checkpoint_every   = ckpt_cfg.get("checkpoint_every", 50)
        checkpoint_resume  = ckpt_cfg.get("resume", True)
        ckpt_file          = _checkpoint_path(artifacts_dir, task)

        # ── Check for existing checkpoint ─────────────────────────────────────
        init_model    = None
        trees_done    = 0
        resumed       = False

        if checkpoint_resume and ckpt_file.exists():
            log.info(f"[CHECKPOINT] Found checkpoint: {ckpt_file}")
            try:
                init_model = lgb.Booster(model_file=str(ckpt_file))
                trees_done = init_model.num_trees()
                remaining  = gb_cfg["max_iter"] - trees_done
                if remaining <= 0:
                    log.info(
                        f"[CHECKPOINT] Checkpoint has {trees_done} trees — "
                        f"already at max_iter={gb_cfg['max_iter']}. "
                        f"Loading checkpoint as final model."
                    )
                    _delete_checkpoint(artifacts_dir, task)
                    return LightGBMWrapper(init_model, n_classes, task)
                log.info(
                    f"[CHECKPOINT] Resuming from tree {trees_done}. "
                    f"Remaining: {remaining} trees."
                )
                resumed = True
            except Exception as e:
                log.warning(
                    f"[CHECKPOINT] Could not load checkpoint ({e}). "
                    f"Starting from scratch."
                )
                init_model = None
                trees_done = 0
        elif checkpoint_enabled:
            log.info(
                f"[CHECKPOINT] No checkpoint found. Starting fresh. "
                f"Checkpoints saved every {checkpoint_every} trees to {ckpt_file}"
            )

        # ── Sample weights ────────────────────────────────────────────────────
        sample_weights = _compute_sample_weights(y_train)

        # ── LightGBM objective ────────────────────────────────────────────────
        if task == "binary":
            objective = "binary"
            num_class = 1
            metric    = "binary_logloss"
        else:
            objective = "multiclass"
            num_class = n_classes
            metric    = "multi_logloss"

        params = {
            "objective":         objective,
            "metric":            metric,
            "learning_rate":     gb_cfg["learning_rate"],
            "num_leaves":        2 ** gb_cfg["max_depth"] - 1,
            "max_depth":         gb_cfg["max_depth"],
            "min_child_samples": gb_cfg["min_samples_leaf"],
            "lambda_l2":         gb_cfg["l2_regularization"],
            "random_state":      gb_cfg["random_state"],
            "device":            "gpu",
            "gpu_device_id":     gpu_cfg.gpu_id,
            "gpu_use_dp":        False,
            "verbose":           -1,
            "min_gain_to_split": 0.0,
            "max_bin":           255,
        }
        if objective == "multiclass":
            params["num_class"] = num_class

        # ── Build datasets ────────────────────────────────────────────────────
        rng     = np.random.RandomState(gb_cfg["random_state"])
        val_n   = max(100, int(len(X_train) * gb_cfg["validation_fraction"]))
        val_idx = rng.choice(len(X_train), size=val_n, replace=False)
        tr_idx  = np.setdiff1d(np.arange(len(X_train)), val_idx)

        ds_train = lgb.Dataset(
            X_train[tr_idx], label=y_train[tr_idx],
            weight=sample_weights[tr_idx], free_raw_data=True,
        )
        ds_val = lgb.Dataset(
            X_train[val_idx], label=y_train[val_idx],
            weight=sample_weights[val_idx], free_raw_data=True,
            reference=ds_train,
        )

        # ── Callbacks ─────────────────────────────────────────────────────────
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=gb_cfg["n_iter_no_change"],
                verbose=False,
            ),
            lgb.log_evaluation(period=50),
        ]

        # Checkpoint callback — saves every N trees during training
        if checkpoint_enabled:
            class CheckpointCallback:
                """
                LightGBM callback that saves a checkpoint every N trees.
                Allows pause and resume if training is interrupted.
                """
                def __init__(self, every: int, path: Path):
                    self.every = every
                    self.path  = path
                    self.order = 20    # run after other callbacks

                def __call__(self, env):
                    if env.iteration % self.every == 0 and env.iteration > 0:
                        tmp = self.path.with_suffix(".tmp")
                        env.model.save_model(str(tmp))
                        tmp.rename(self.path)
                        log.info(
                            f"[CHECKPOINT] Saved at tree {env.iteration} -> {self.path}"
                        )

            callbacks.append(CheckpointCallback(every=checkpoint_every, path=ckpt_file))

        # ── Train ──────────────────────────────────────────────────────────────
        remaining_iters = gb_cfg["max_iter"] - trees_done

        with gpu_memory_context(gpu_cfg, f"LightGBM GPU {task}"):
            start   = time.time()
            booster = lgb.train(
                params,
                ds_train,
                num_boost_round=remaining_iters,
                valid_sets=[ds_val],
                callbacks=callbacks,
                init_model=init_model,    # None = fresh start, or loaded checkpoint
            )
            elapsed = time.time() - start
            total_trees = booster.num_trees()

        log.info(
            f"LightGBM GPU done in {elapsed:.1f}s | "
            f"total trees: {total_trees} "
            f"({'resumed from ' + str(trees_done) if resumed else 'fresh start'})"
        )

        # Training completed successfully — delete checkpoint
        if checkpoint_enabled:
            _delete_checkpoint(artifacts_dir, task)

        return LightGBMWrapper(booster, n_classes, task)

    except ImportError:
        log.warning("LightGBM not installed — trying XGBoost GPU.")
        return None
    except KeyboardInterrupt:
        # User pressed Ctrl+C — checkpoint is already saved by the callback
        log.warning(
            f"\n[CHECKPOINT] Training interrupted by user (Ctrl+C).\n"
            f"Checkpoint saved at: {ckpt_file}\n"
            f"Re-run the notebook to resume from where you stopped."
        )
        raise
    except Exception as e:
        err = str(e).lower()
        if "out of memory" in err or "oom" in err or "cuda" in err:
            log.error(
                f"LightGBM GPU OOM for task={task}: {e}\n"
                "Reduce balanced_ceiling in data_config.yaml or set use_gpu: false."
            )
        else:
            log.error(f"LightGBM GPU failed: {e}")
        if gpu_cfg.fallback_to_cpu:
            log.warning("Falling back to CPU.")
            clear_gpu_memory(gpu_cfg.gpu_id)
            return None
        raise


def _train_xgboost_gpu(
    X_train, y_train, gb_cfg, gpu_cfg: GpuConfig, task: str, n_classes: int
):
    """GPU path: XGBoost with device='cuda'."""
    try:
        import xgboost as xgb  # type: ignore

        sample_weights = _compute_sample_weights(y_train)

        if task == "binary":
            objective = "binary:logistic"
            num_class = None
        else:
            objective = "multi:softprob"
            num_class = n_classes

        params = {
            "objective":       objective,
            "eval_metric":     "mlogloss" if task != "binary" else "logloss",
            "learning_rate":   gb_cfg["learning_rate"],
            "max_depth":       gb_cfg["max_depth"],
            "min_child_weight":gb_cfg["min_samples_leaf"],
            "reg_lambda":      gb_cfg["l2_regularization"],
            "seed":            gb_cfg["random_state"],
            "device":          f"cuda:{gpu_cfg.gpu_id}",
            "tree_method":     "hist",
            "verbosity":       0,
        }
        if num_class:
            params["num_class"] = num_class

        dm_train = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)

        with gpu_memory_context(gpu_cfg, f"XGBoost GPU {task}"):
            start   = time.time()
            booster = xgb.train(
                params, dm_train,
                num_boost_round=gb_cfg["max_iter"],
                evals=[(dm_train, "train")],
                verbose_eval=50,
                early_stopping_rounds=gb_cfg["n_iter_no_change"],
            )
            elapsed = time.time() - start
            log.info(f"XGBoost GPU done in {elapsed:.1f}s")

        return XGBoostWrapper(booster, n_classes, task)

    except ImportError:
        log.warning("XGBoost not installed — falling back to CPU.")
        return None
    except Exception as e:
        err = str(e).lower()
        if "out of memory" in err or "oom" in err:
            log.error(f"XGBoost GPU OOM for task={task}: {e}")
        else:
            log.error(f"XGBoost GPU failed: {e}")
        if gpu_cfg.fallback_to_cpu:
            clear_gpu_memory(gpu_cfg.gpu_id)
            return None
        raise


def _train_cpu_gbm(X_train, y_train, gb_cfg, task: str):
    """CPU fallback: sklearn HistGradientBoostingClassifier."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    log.info(f"sklearn HistGBM CPU | task={task}")
    log.info(
        "NOTE: CPU path has no checkpoint/resume support. "
        "If interrupted, training restarts from scratch."
    )
    model = HistGradientBoostingClassifier(
        max_iter=gb_cfg["max_iter"],
        learning_rate=gb_cfg["learning_rate"],
        max_depth=gb_cfg["max_depth"],
        min_samples_leaf=gb_cfg["min_samples_leaf"],
        l2_regularization=gb_cfg["l2_regularization"],
        class_weight=gb_cfg["class_weight"],
        early_stopping=gb_cfg["early_stopping"],
        validation_fraction=gb_cfg["validation_fraction"],
        n_iter_no_change=gb_cfg["n_iter_no_change"],
        random_state=gb_cfg["random_state"],
        verbose=1,
    )
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    log.info(
        f"sklearn HistGBM done in {elapsed:.1f}s (CPU) | "
        f"iterations: {model.n_iter_} / {gb_cfg['max_iter']}"
    )
    return model


# ── Sklearn-compatible wrappers ────────────────────────────────────────────────

class LightGBMWrapper:
    """
    Wraps a LightGBM Booster to expose .predict() / .predict_proba() API.
    Predictions always returned as CPU numpy arrays.
    """

    def __init__(self, booster, n_classes: int, task: str):
        self.booster   = booster
        self.n_classes = n_classes
        self.task      = task

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        if self.task == "binary":
            return (proba[:, 1] >= 0.5).astype(int)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float32)
        raw = self.booster.predict(X, num_iteration=self.booster.best_iteration)
        if self.task == "binary":
            raw = np.clip(raw, 1e-7, 1 - 1e-7)
            return np.column_stack([1 - raw, raw]).astype(np.float32)
        return raw.reshape(-1, self.n_classes).astype(np.float32)

    def __repr__(self):
        return (f"LightGBMWrapper(n_classes={self.n_classes}, "
                f"task={self.task}, trees={self.booster.num_trees()})")


class XGBoostWrapper:
    """
    Wraps an XGBoost Booster to expose .predict() / .predict_proba() API.
    Predictions always returned as CPU numpy arrays.
    """

    def __init__(self, booster, n_classes: int, task: str):
        self.booster   = booster
        self.n_classes = n_classes
        self.task      = task

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        if self.task == "binary":
            return (proba[:, 1] >= 0.5).astype(int)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import xgboost as xgb  # type: ignore
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float32)
        dm  = xgb.DMatrix(X)
        raw = self.booster.predict(dm)
        if self.task == "binary":
            raw = np.clip(raw, 1e-7, 1 - 1e-7)
            return np.column_stack([1 - raw, raw]).astype(np.float32)
        return raw.reshape(-1, self.n_classes).astype(np.float32)

    def __repr__(self):
        return f"XGBoostWrapper(n_classes={self.n_classes}, task={self.task})"
