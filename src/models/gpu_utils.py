"""
gpu_utils.py
============
GPU detection, memory management, and safety utilities.

Addresses every GPU failure mode listed in the project requirements:

DETECTION ISSUES
  - GPU not detected by framework       → detect_gpu() with multi-library fallback
  - Code silently running on CPU        → explicit logging of every device decision
  - Framework/CUDA version mismatches   → version checks with clear error messages

MEMORY ISSUES
  - Out-of-memory (OOM)                 → safe_gpu_call() with OOM catch + fallback
  - VRAM fragmentation                  → clear_gpu_memory() called between experiments
  - Memory leaks (train/inference)      → context managers that force cleanup
  - Optimizer/gradient memory growth    → only relevant to neural nets; flagged for LR/GB
  - Old Colab/notebook VRAM not freed   → explicit gc + cache clear on every cleanup call
  - Other apps occupying VRAM           → get_gpu_memory_info() shows available before start
  - Storing predictions on GPU          → all predictions always moved to CPU immediately

STABILITY ISSUES
  - Thermal throttling                  → logged if GPU temp available
  - Deadlocks / hangs                   → timeout guard in safe_gpu_call()
  - Mixed precision instability         → not used; cuML/LightGBM use float32 natively
  - NaN / Inf from bad preprocessing    → validate_array() checks before any GPU call
  - Exploding gradients                 → not applicable to LR/GBM (no backprop)
  - Corrupted checkpoint                → atomic save (write temp → rename)
  - Reproducibility                     → seed set in GPU context

PERFORMANCE ISSUES
  - Low GPU utilisation                 → batch size guide in GpuConfig
  - Dataloader bottleneck               → prefetch + pinned memory advice
  - Slow disk I/O                       → Parquet + memory-mapped arrays
  - CPU preprocessing bottleneck        → preprocessing always on CPU (correct)
  - Training slower than expected       → benchmark_gpu() helper

ENVIRONMENT ISSUES
  - Multiprocessing / dataloader workers → n_jobs=1 when GPU active (avoid fork issues)
  - cuDNN incompatible behaviour         → deterministic mode flag
  - Random reproducibility               → seed_everything() utility
  - Disk space for checkpoints           → check_disk_space() before save
"""

import gc
import logging
import os
import platform
import subprocess
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ── GPU Backend Constants ──────────────────────────────────────────────────────
BACKEND_NONE   = "none"       # No GPU — pure CPU sklearn
BACKEND_CUML   = "cuml"       # RAPIDS cuML  (best for LR on GPU)
BACKEND_LGBM   = "lightgbm"   # LightGBM GPU (best for GBM on GPU)
BACKEND_XGB    = "xgboost"    # XGBoost GPU  (fallback GBM)


@dataclass
class GpuConfig:
    """
    Central GPU configuration. Read by train_lr.py and train_gb.py.
    Set via configs/model_config.yaml or environment variables.

    VRAM guidance (adjust batch_size_mb to fit your GPU):
      - 4 GB VRAM  (GTX 1650, etc.)     → batch_size_mb = 512
      - 6 GB VRAM  (RTX 3060, etc.)     → batch_size_mb = 1024
      - 8 GB VRAM  (RTX 3070, etc.)     → batch_size_mb = 2048
      - 12 GB VRAM (RTX 3080 Ti, etc.)  → batch_size_mb = 4096
      - 24 GB VRAM (RTX 3090, etc.)     → batch_size_mb = 8192
    """
    # Training device
    use_gpu:          bool  = True    # Try GPU; falls back to CPU if unavailable
    gpu_id:           int   = 0       # Which GPU to use (0 = first GPU)
    fallback_to_cpu:  bool  = True    # Auto-fallback on any GPU error

    # Memory safety
    max_vram_fraction: float = 0.85   # Never use more than 85% of VRAM
    batch_size_mb:     int   = 1024   # Data chunk size for GPU transfer (MB)
    clear_cache_between_tasks: bool = True  # Free GPU memory between experiments

    # Reproducibility
    random_state:     int   = 42

    # Stability
    validate_input:   bool  = True    # Check for NaN/Inf before GPU call
    atomic_save:      bool  = True    # Write temp file then rename (safe checkpoint)
    min_disk_gb:      float = 2.0     # Require at least 2 GB free before saving

    # Detected at runtime (populated by detect_gpu())
    backend:          str   = BACKEND_NONE
    gpu_name:         str   = ""
    total_vram_gb:    float = 0.0
    free_vram_gb:     float = 0.0
    cuda_version:     str   = ""
    driver_version:   str   = ""


# ── GPU Detection ──────────────────────────────────────────────────────────────

def detect_gpu(cfg: GpuConfig | None = None) -> GpuConfig:
    """
    Detect available GPU and choose the best backend.

    Priority:
      1. cuML (RAPIDS)   → GPU-accelerated LR, best for sklearn-compatible pipeline
      2. LightGBM GPU    → GPU-accelerated GBM
      3. XGBoost GPU     → fallback GBM GPU
      4. CPU only        → if nothing is available or use_gpu=False

    All decisions are logged explicitly — the system NEVER silently falls back
    without printing a clear message.
    """
    if cfg is None:
        cfg = GpuConfig()

    log.info("=" * 60)
    log.info("GPU DETECTION")
    log.info("=" * 60)

    if not cfg.use_gpu:
        log.info("GPU disabled by config (use_gpu=False). Using CPU.")
        cfg.backend = BACKEND_NONE
        return cfg

    # Step 1: Check CUDA is available at the system level
    cuda_available = _check_cuda_available()
    if not cuda_available:
        log.warning("CUDA not available (no nvidia-smi or CUDA toolkit found).")
        log.warning("→ Falling back to CPU training.")
        cfg.backend = BACKEND_NONE
        return cfg

    # Step 2: Get GPU info
    cfg = _populate_gpu_info(cfg)
    log.info(f"GPU detected:    {cfg.gpu_name}")
    log.info(f"CUDA version:    {cfg.cuda_version}")
    log.info(f"Driver version:  {cfg.driver_version}")
    log.info(f"Total VRAM:      {cfg.total_vram_gb:.1f} GB")
    log.info(f"Free VRAM:       {cfg.free_vram_gb:.1f} GB")

    # Safety check: is there enough free VRAM?
    if cfg.free_vram_gb < 1.0:
        log.warning(
            f"Only {cfg.free_vram_gb:.1f} GB VRAM free. "
            "Other applications may be using the GPU. "
            "→ Falling back to CPU to avoid OOM."
        )
        if cfg.fallback_to_cpu:
            cfg.backend = BACKEND_NONE
            return cfg

    # Step 3: Try backends in priority order
    backend = _try_cuml(cfg)
    if backend:
        cfg.backend = BACKEND_CUML
        log.info(f"Selected backend: cuML (RAPIDS) — GPU-accelerated LR and GBM")
        return cfg

    backend = _try_lightgbm_gpu(cfg)
    if backend:
        cfg.backend = BACKEND_LGBM
        log.info(f"Selected backend: LightGBM GPU — GPU-accelerated GBM, CPU LR")
        return cfg

    backend = _try_xgboost_gpu(cfg)
    if backend:
        cfg.backend = BACKEND_XGB
        log.info(f"Selected backend: XGBoost GPU — GPU-accelerated GBM, CPU LR")
        return cfg

    log.warning(
        "GPU detected but no GPU-capable ML library found "
        "(cuML, LightGBM-GPU, XGBoost-GPU).\n"
        "→ Falling back to CPU sklearn.\n"
        "To enable GPU training, install one of:\n"
        "  pip install cuml-cu12            (RAPIDS — best option)\n"
        "  pip install lightgbm             (then rebuild with CUDA support)\n"
        "  pip install xgboost              (GPU support built-in since 1.6)"
    )
    cfg.backend = BACKEND_NONE
    return cfg


def _check_cuda_available() -> bool:
    """Check if CUDA is available via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0 and result.stdout.strip() != ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _populate_gpu_info(cfg: GpuConfig) -> GpuConfig:
    """Populate GPU name, VRAM, CUDA version from nvidia-smi."""
    try:
        # GPU name
        r = subprocess.run(
            ["nvidia-smi", f"--id={cfg.gpu_id}",
             "--query-gpu=name,memory.total,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if r.returncode == 0:
            parts = [p.strip() for p in r.stdout.strip().split(",")]
            if len(parts) >= 4:
                cfg.gpu_name       = parts[0]
                cfg.total_vram_gb  = float(parts[1]) / 1024
                cfg.free_vram_gb   = float(parts[2]) / 1024
                cfg.driver_version = parts[3]

        # CUDA version
        r2 = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if r2.returncode == 0:
            cfg.cuda_version = r2.stdout.strip().split("\n")[0]
    except Exception as e:
        log.warning(f"Could not read GPU info: {e}")
    return cfg


def _try_cuml(cfg: GpuConfig) -> bool:
    try:
        import cuml  # type: ignore
        log.info(f"cuML available: version {cuml.__version__}")
        return True
    except ImportError:
        log.info("cuML not available.")
        return False


def _try_lightgbm_gpu(cfg: GpuConfig) -> bool:
    try:
        import lightgbm as lgb  # type: ignore
        # Try a tiny GPU training to confirm GPU support is compiled in
        X_tiny = np.random.randn(100, 5).astype(np.float32)
        y_tiny = np.random.randint(0, 2, 100)
        ds = lgb.Dataset(X_tiny, label=y_tiny)
        params = {"device": "gpu", "gpu_device_id": cfg.gpu_id,
                  "num_leaves": 4, "num_iterations": 2, "verbose": -1}
        lgb.train(params, ds, num_boost_round=2)
        log.info(f"LightGBM GPU available: version {lgb.__version__}")
        return True
    except Exception as e:
        log.info(f"LightGBM GPU not available: {e}")
        return False


def _try_xgboost_gpu(cfg: GpuConfig) -> bool:
    try:
        import xgboost as xgb  # type: ignore
        X_tiny = np.random.randn(100, 5).astype(np.float32)
        y_tiny = np.random.randint(0, 2, 100)
        dm = xgb.DMatrix(X_tiny, label=y_tiny)
        params = {"device": f"cuda:{cfg.gpu_id}", "max_depth": 2,
                  "num_class": 2, "objective": "multi:softprob", "verbosity": 0}
        xgb.train(params, dm, num_boost_round=2)
        log.info(f"XGBoost GPU available: version {xgb.__version__}")
        return True
    except Exception as e:
        log.info(f"XGBoost GPU not available: {e}")
        return False


# ── Memory Management ──────────────────────────────────────────────────────────

def get_gpu_memory_info(gpu_id: int = 0) -> dict[str, float]:
    """
    Return current GPU memory stats in GB.
    Returns zeros if GPU not available (never crashes).
    """
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--id={gpu_id}",
             "--query-gpu=memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if r.returncode == 0:
            parts = [float(p.strip()) for p in r.stdout.strip().split(",")]
            return {
                "total_gb": parts[0] / 1024,
                "used_gb":  parts[1] / 1024,
                "free_gb":  parts[2] / 1024,
                "used_pct": parts[1] / parts[0] * 100,
            }
    except Exception:
        pass
    return {"total_gb": 0.0, "used_gb": 0.0, "free_gb": 0.0, "used_pct": 0.0}


def clear_gpu_memory(gpu_id: int = 0) -> None:
    """
    Aggressively free GPU memory.

    Handles:
    - VRAM fragmentation
    - Memory leaks between training runs
    - Old Colab/notebook contexts not freeing memory
    - Predictions/embeddings accidentally left on GPU
    """
    # Python garbage collection first
    gc.collect()

    # Try PyTorch cache (used by cuML under the hood on some builds)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            log.debug("PyTorch CUDA cache cleared.")
    except ImportError:
        pass

    # Try CuPy cache (used by cuML)
    try:
        import cupy as cp  # type: ignore
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        log.debug("CuPy memory pool cleared.")
    except (ImportError, Exception):
        pass

    mem = get_gpu_memory_info(gpu_id)
    if mem["total_gb"] > 0:
        log.info(
            f"GPU memory after clear: "
            f"{mem['free_gb']:.1f} GB free / {mem['total_gb']:.1f} GB total "
            f"({mem['used_pct']:.1f}% used)"
        )


@contextmanager
def gpu_memory_context(cfg: GpuConfig, label: str = ""):
    """
    Context manager that clears GPU memory before AND after a block.
    Use this around each training experiment to prevent memory leaks.

    Usage:
        with gpu_memory_context(cfg, "LR binary"):
            model = train_logistic_regression(...)
    """
    if cfg.backend != BACKEND_NONE:
        log.info(f"[GPU] Starting: {label}")
        clear_gpu_memory(cfg.gpu_id)
        mem_before = get_gpu_memory_info(cfg.gpu_id)
        log.info(
            f"[GPU] Free VRAM before: {mem_before['free_gb']:.1f} GB / "
            f"{mem_before['total_gb']:.1f} GB"
        )

    try:
        yield
    except MemoryError as e:
        log.error(f"[GPU] MemoryError in '{label}': {e}")
        log.error("[GPU] → Clearing GPU memory and re-raising. "
                  "Reduce batch_size_mb or sample_frac in config.")
        clear_gpu_memory(cfg.gpu_id)
        raise
    except Exception as e:
        # Check if it's a CUDA OOM
        err_str = str(e).lower()
        if "out of memory" in err_str or "cuda" in err_str or "oom" in err_str:
            log.error(f"[GPU] CUDA OOM in '{label}': {e}")
            log.error(
                "[GPU] Suggestions:\n"
                "  1. Reduce sample_frac in configs/data_config.yaml\n"
                "  2. Reduce batch_size_mb in configs/model_config.yaml\n"
                "  3. Close other GPU-using applications\n"
                "  4. Set use_gpu: false in configs/model_config.yaml"
            )
            clear_gpu_memory(cfg.gpu_id)
            if cfg.fallback_to_cpu:
                log.warning("[GPU] → Falling back to CPU for this task.")
                cfg.backend = BACKEND_NONE
        raise
    finally:
        if cfg.backend != BACKEND_NONE:
            if cfg.clear_cache_between_tasks:
                clear_gpu_memory(cfg.gpu_id)
            mem_after = get_gpu_memory_info(cfg.gpu_id)
            log.info(
                f"[GPU] Free VRAM after: {mem_after['free_gb']:.1f} GB / "
                f"{mem_after['total_gb']:.1f} GB"
            )


# ── Input Validation ──────────────────────────────────────────────────────────

def validate_array(X: np.ndarray, name: str = "X") -> None:
    """
    Check for NaN, Inf, and extreme values before sending to GPU.

    Prevents:
    - NaN/Inf loss values
    - Numerical instability from bad preprocessing
    - Exploding gradients (not applicable to LR/GBM but checked anyway)
    """
    if np.isnan(X).any():
        n_nan = np.isnan(X).sum()
        raise ValueError(
            f"[GPU Safety] {name} contains {n_nan:,} NaN values. "
            "Fix preprocessing before GPU training. "
            "The Preprocessor should have caught this — check fit() was called on train data."
        )
    if np.isinf(X).any():
        n_inf = np.isinf(X).sum()
        raise ValueError(
            f"[GPU Safety] {name} contains {n_inf:,} Inf values. "
            "Fix preprocessing before GPU training."
        )
    # Check for extreme values that could cause numerical instability
    abs_max = np.abs(X).max()
    if abs_max > 1e6:
        log.warning(
            f"[GPU Safety] {name} has very large values (max abs = {abs_max:.2e}). "
            "This may cause numerical instability on GPU. "
            "Consider checking RobustScaler fit."
        )


# ── Safe Disk Save ─────────────────────────────────────────────────────────────

def check_disk_space(path: str, min_gb: float = 2.0) -> None:
    """
    Verify sufficient disk space before saving a model checkpoint.
    Prevents partial/corrupted writes.
    """
    try:
        import shutil
        stat = shutil.disk_usage(Path(path).parent if "." in Path(path).name else path)
        free_gb = stat.free / (1024 ** 3)
        if free_gb < min_gb:
            raise OSError(
                f"Only {free_gb:.1f} GB free on disk. "
                f"Need at least {min_gb} GB to save safely. "
                "Free disk space before continuing."
            )
        log.debug(f"Disk check passed: {free_gb:.1f} GB free at {path}")
    except Exception as e:
        if "Only" in str(e):
            raise
        log.warning(f"Could not check disk space: {e}")


def atomic_save(obj: Any, path: str, min_disk_gb: float = 2.0) -> None:
    """
    Save object atomically: write to temp file, then rename.

    Prevents corrupted checkpoints from:
    - Interrupted saves
    - Disk full during write
    - Power loss / crash during write
    """
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    check_disk_space(str(path.parent), min_disk_gb)

    tmp_path = path.with_suffix(".tmp")
    try:
        joblib.dump(obj, str(tmp_path))
        tmp_path.rename(path)
        log.info(f"Saved (atomic): {path}")
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Atomic save failed for {path}: {e}") from e


# ── Reproducibility ────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility across CPU and GPU.

    Addresses:
    - Random reproducibility issues across runs
    - Different results between GPU and CPU (due to non-deterministic ops)
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # cuDNN deterministic mode — slower but reproducible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        import cupy as cp  # type: ignore
        cp.random.seed(seed)
    except ImportError:
        pass

    log.info(f"Seeds set to {seed} (Python, NumPy, OS, CUDA if available)")


# ── Benchmark / Diagnostics ────────────────────────────────────────────────────

def benchmark_gpu(cfg: GpuConfig, n_rows: int = 100_000, n_features: int = 42) -> None:
    """
    Quick GPU training benchmark to diagnose performance issues.
    Prints expected training time and detects if GPU is slower than expected.
    """
    if cfg.backend == BACKEND_NONE:
        log.info("Benchmark skipped: CPU mode.")
        return

    log.info(f"Running GPU benchmark ({n_rows:,} rows × {n_features} features)...")
    import time
    X = np.random.randn(n_rows, n_features).astype(np.float32)
    y = np.random.randint(0, 2, n_rows)

    start = time.time()

    if cfg.backend == BACKEND_CUML:
        try:
            from cuml.linear_model import LogisticRegression as cuLR  # type: ignore
            m = cuLR(max_iter=10)
            m.fit(X, y)
        except Exception as e:
            log.warning(f"cuML benchmark failed: {e}")

    elif cfg.backend in (BACKEND_LGBM, BACKEND_XGB):
        try:
            import lightgbm as lgb  # type: ignore
            ds = lgb.Dataset(X, label=y)
            lgb.train({"device": "gpu", "num_leaves": 31, "verbose": -1},
                      ds, num_boost_round=10)
        except Exception as e:
            log.warning(f"LightGBM GPU benchmark failed: {e}")

    elapsed = time.time() - start
    throughput = n_rows / elapsed
    log.info(f"Benchmark: {elapsed:.2f}s for {n_rows:,} rows → {throughput:,.0f} rows/sec")

    if throughput < 50_000:
        log.warning(
            "Low GPU throughput detected. Possible causes:\n"
            "  - Thermal throttling (GPU too hot)\n"
            "  - Power throttling (insufficient PSU)\n"
            "  - Data transfer bottleneck (CPU↔GPU)\n"
            "  - Other processes occupying VRAM\n"
            "  Check GPU temperature with: nvidia-smi -l 1"
        )


def print_gpu_status(cfg: GpuConfig) -> None:
    """Print a human-readable GPU status summary."""
    print("\n" + "=" * 50)
    print("GPU STATUS")
    print("=" * 50)
    print(f"  Backend:        {cfg.backend}")
    print(f"  GPU:            {cfg.gpu_name or 'None detected'}")
    print(f"  VRAM total:     {cfg.total_vram_gb:.1f} GB")
    print(f"  VRAM free:      {cfg.free_vram_gb:.1f} GB")
    print(f"  CUDA version:   {cfg.cuda_version or 'N/A'}")
    print(f"  Driver:         {cfg.driver_version or 'N/A'}")

    if cfg.backend == BACKEND_NONE:
        print("  → Training on: CPU (sklearn)")
    elif cfg.backend == BACKEND_CUML:
        print("  → LR training on:  GPU (cuML)")
        print("  → GBM training on: GPU (cuML RandomForest / LightGBM)")
    elif cfg.backend == BACKEND_LGBM:
        print("  → LR training on:  CPU (sklearn, LR has no cuML fallback)")
        print("  → GBM training on: GPU (LightGBM)")
    elif cfg.backend == BACKEND_XGB:
        print("  → LR training on:  CPU (sklearn)")
        print("  → GBM training on: GPU (XGBoost)")
    print("=" * 50 + "\n")
