"""
app.py
────────────────────────────────────────────────────────────────────────────────
FastAPI application — the REST API layer for RiskTraceML.

Acts as the bridge between the Spring Boot backend and the ML inference engine.

Endpoints:
  GET  /health          → liveness / readiness probe
  POST /predict         → single session anomaly score
  POST /predict/batch   → batch of sessions anomaly scores

Run locally:
  uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
────────────────────────────────────────────────────────────────────────────────
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ml.predict import load_artifacts, predict_session, predict_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────


class SessionFeatures(BaseModel):
    """
    Input schema for a single session behavior window.
    All fields are derived from the aggregated log data.
    """
    request_count: float = Field(..., ge=0, description="Total HTTP requests in window")
    error_rate: float = Field(..., ge=0.0, le=1.0, description="Ratio of 4xx/5xx status codes")
    auth_failure_count: float = Field(..., ge=0, description="Count of 401/403 status codes")
    avg_response_time_ms: float = Field(..., ge=0.0, description="Average response time in ms")
    p95_response_time_ms: float = Field(..., ge=0.0, description="95th percentile response time in ms")
    unique_endpoints: float = Field(..., ge=0, description="Count of unique URLs accessed")
    unique_ips: float = Field(..., ge=0, description="Count of unique source IPs (usually 1)")
    anomalous_path_count: float = Field(..., ge=0, description="Count of probes to sensitive paths like /admin, /.env")
    post_ratio: float = Field(..., ge=0.0, le=1.0, description="Ratio of POST requests")
    js_error_count: float = Field(..., ge=0, description="Browser-side JavaScript errors tracked")
    request_rate: float = Field(..., ge=0.0, description="Requests per second")
    session_duration_s: float = Field(..., ge=0.0, description="Time between first and last request in window")

    class Config:
        json_schema_extra = {
            "example": {
                "request_count": 150.0,
                "error_rate": 0.05,
                "auth_failure_count": 2.0,
                "avg_response_time_ms": 120.5,
                "p95_response_time_ms": 450.0,
                "unique_endpoints": 12.0,
                "unique_ips": 1.0,
                "anomalous_path_count": 0.0,
                "post_ratio": 0.1,
                "js_error_count": 0.0,
                "request_rate": 2.5,
                "session_duration_s": 60.0
            }
        }


class BatchSessionFeatures(BaseModel):
    """Input schema for a batch of session behavior windows."""
    sessions: list[SessionFeatures] = Field(..., min_length=1, description="List of session feature objects.")


class PredictionResponse(BaseModel):
    """Output schema for a single prediction result."""
    anomalyScore: float = Field(..., description="Normalized anomaly score [0.0 = safe, 1.0 = highly anomalous].")
    prediction: str = Field(..., description="'NORMAL' or 'ANOMALY'.")
    confidence: str = Field(..., description="Model confidence level ('LOW', 'MEDIUM', or 'HIGH').")


class BatchPredictionResponse(BaseModel):
    """Output schema for a batch prediction results."""
    results: list[PredictionResponse]
    total: int


# ─── Application Lifespan ─────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artifacts once at startup."""
    logger.info("🚀 RiskTraceML service starting — loading artifacts …")

    try:
        app.state.artifacts = load_artifacts()
        app.state.model_ready = True
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        app.state.artifacts = None
        app.state.model_ready = False

    logger.info("✅ RiskTraceML service ready.")
    yield
    logger.info("🛑 RiskTraceML service shutting down.")


# ─── FastAPI App ──────────────────────────────────────────────────────────────


app = FastAPI(
    title="RiskTraceML",
    description=(
        "Anomaly detection microservice for the RiskTrace platform. "
        "Uses an Isolation Forest model trained on application behavioral data."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ─── Endpoints ────────────────────────────────────────────────────────────────


@app.get(
    "/health",
    tags=["System"],
    summary="Liveness & readiness probe",
    response_description="Service and model status.",
)
async def health_check() -> dict[str, Any]:
    """
    Returns the current health status of the service.
    """
    return {
        "status": "UP",
        "model": "LOADED" if getattr(app.state, "model_ready", False) else "NOT_LOADED",
        "service": "RiskTraceML",
        "version": "0.1.0",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict anomaly score for a single session",
)
async def predict(request: SessionFeatures) -> PredictionResponse:
    """
    Accept a single session behavior window and return an anomaly prediction.
    """
    if not app.state.model_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run python src/train.py first.",
        )

    from ml.feature_engineering import format_for_prediction
    features_dict = request.model_dump()
    formatted = format_for_prediction(features_dict)
    
    result = predict_session(formatted, app.state.artifacts)
    return PredictionResponse(**result)


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Predict anomaly scores for a batch of sessions",
)
async def predict_batch_endpoint(request: BatchSessionFeatures) -> BatchPredictionResponse:
    """
    Accept multiple session behavior windows and return a prediction for each.
    """
    if not app.state.model_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run python src/train.py first.",
        )

    from ml.feature_engineering import format_for_prediction
    formatted_sessions = [format_for_prediction(s.model_dump()) for s in request.sessions]
    
    results = predict_batch(formatted_sessions, app.state.artifacts)
    response_items = [PredictionResponse(**r) for r in results]
    return BatchPredictionResponse(results=response_items, total=len(results))
