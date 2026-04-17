"""
api.py
======
FastAPI inference service for IoT Intrusion Detection.

Endpoints
---------
GET  /health              — liveness check
GET  /info                — model and task info
POST /predict             — predict one traffic record
POST /predict_batch       — predict a batch of records
GET  /classes             — list class names for the loaded task

Run locally
-----------
    uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8000

Docker
------
    docker run -p 8000:8000 yourteam/iot-ids:latest
    Then visit http://localhost:8000/docs
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# ── Configuration from environment variables (with sensible defaults) ─────────
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "models")
MODEL_NAME = os.getenv("MODEL_NAME", "gb")       # 'lr' or 'gb'
TASK = os.getenv("TASK", "binary")               # 'binary', '8class', '34class'

# ── Global predictor (loaded once at startup) ─────────────────────────────────
predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts at startup."""
    global predictor
    log.info(f"Loading predictor: model={MODEL_NAME}, task={TASK}, dir={ARTIFACTS_DIR}")
    try:
        from src.serving.predictor import Predictor
        predictor = Predictor.load(ARTIFACTS_DIR, task=TASK, model_name=MODEL_NAME)
        log.info("Predictor loaded successfully.")
    except FileNotFoundError as e:
        log.warning(f"Model artifacts not found: {e}")
        log.warning("API will start in demo mode — predictions will return placeholder data.")
        predictor = None
    yield
    log.info("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="IoT Intrusion Detection API",
    description=(
        "ML-powered intrusion detection for IoT network traffic. "
        "Returns predictions and probabilities for binary, 8-class, and 34-class tasks."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────

class TrafficRecord(BaseModel):
    features: dict[str, float] = Field(
        ...,
        example={
            "flow_duration": 1.23,
            "total_fwd_packets": 10.0,
            "total_bwd_packets": 8.0,
            "total_length_fwd_packets": 1500.0,
            "total_length_bwd_packets": 900.0,
            "flow_bytes_per_sec": 2400.5,
            "flow_packets_per_sec": 14.6,
            "fwd_packet_length_mean": 150.0,
            "bwd_packet_length_mean": 112.5,
            "syn_flag_count": 1.0,
        },
        description="Dictionary mapping feature names to their float values.",
    )


class BatchRequest(BaseModel):
    records: list[dict[str, float]] = Field(
        ...,
        max_items=10_000,
        description="List of feature dicts. Max 10,000 records per request.",
    )


class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    probabilities: dict[str, float]
    is_malicious: bool
    model: str
    task: str
    latency_ms: float


class BatchResponse(BaseModel):
    predictions: list[dict]
    n_records: int
    n_malicious: int
    alert_rate: float
    latency_ms: float
    model: str
    task: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Liveness check — always returns 200 if the server is running."""
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "model": MODEL_NAME,
        "task": TASK,
    }


@app.get("/info", tags=["System"])
def info():
    """Return model configuration and class names."""
    if predictor is None:
        return {
            "status": "demo_mode",
            "model": MODEL_NAME,
            "task": TASK,
            "artifacts_dir": ARTIFACTS_DIR,
            "message": "Model not loaded. Run training notebooks first.",
        }
    return {
        "status": "ready",
        "model": MODEL_NAME,
        "task": TASK,
        "n_classes": len(predictor.class_names),
        "class_names": predictor.class_names,
        "n_features": len(predictor.get_feature_names()),
        "feature_names": predictor.get_feature_names(),
    }


@app.get("/classes", tags=["Model"])
def list_classes():
    """List all class names for the loaded task."""
    if predictor is None:
        raise HTTPException(503, "Model not loaded. See /health for details.")
    return {"task": TASK, "classes": predictor.class_names}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(record: TrafficRecord):
    """
    Predict class and probabilities for a single network traffic record.

    Send a JSON body with a 'features' dict mapping feature names to float values.
    Returns the predicted class label, full probability vector, and malicious flag.
    """
    if predictor is None:
        # Demo mode: return a dummy response
        return _demo_prediction()

    t0 = time.time()
    try:
        result = predictor.predict(record.features)
    except Exception as e:
        log.error(f"Prediction error: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    result["latency_ms"] = round((time.time() - t0) * 1000, 2)
    return result


@app.post("/predict_batch", response_model=BatchResponse, tags=["Inference"])
def predict_batch(request: BatchRequest):
    """
    Predict on a batch of traffic records.

    Returns all individual predictions plus batch-level statistics:
    alert rate (fraction malicious), total malicious count, latency.
    """
    if predictor is None:
        return _demo_batch(len(request.records))

    t0 = time.time()
    try:
        predictions = predictor.predict_batch(request.records)
    except Exception as e:
        log.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    n_malicious = sum(1 for p in predictions if p["is_malicious"])
    latency_ms = round((time.time() - t0) * 1000, 2)

    return {
        "predictions": predictions,
        "n_records": len(predictions),
        "n_malicious": n_malicious,
        "alert_rate": round(n_malicious / max(len(predictions), 1), 4),
        "latency_ms": latency_ms,
        "model": MODEL_NAME,
        "task": TASK,
    }


# ── Demo mode helpers ──────────────────────────────────────────────────────────

def _demo_prediction() -> dict:
    return {
        "predicted_class": 0,
        "predicted_label": "Benign",
        "probabilities": {"Benign": 0.97, "Malicious": 0.03},
        "is_malicious": False,
        "model": MODEL_NAME,
        "task": TASK,
        "latency_ms": 0.1,
    }


def _demo_batch(n: int) -> dict:
    return {
        "predictions": [_demo_prediction() for _ in range(n)],
        "n_records": n,
        "n_malicious": 0,
        "alert_rate": 0.0,
        "latency_ms": 0.5,
        "model": MODEL_NAME,
        "task": TASK,
    }
