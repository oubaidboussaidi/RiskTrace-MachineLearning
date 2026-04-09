"""
app.py
────────────────────────────────────────────────────────────────────────────────
FastAPI application — the REST API layer for RiskTraceML.

Acts as the bridge between the Spring Boot backend and the ML inference engine.

Endpoints:
  GET  /health          → liveness / readiness probe
  POST /predict         → single session anomaly score
  POST /predict/batch   → batch of sessions anomaly scores

Startup:
  Artifacts (model, scaler, encoders) are loaded ONCE via a lifespan handler
  and cached in app.state to avoid repeated disk I/O per request.

Run locally:
  uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
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
    Input schema for a single session prediction request.

    TODO:
        - Add all 15 feature fields with proper types, descriptions, and
          example values once FEATURE_COLUMNS is finalised
        - Use Field(ge=0) / Field(ge=0.0, le=1.0) for range validation
        - Example:
            request_count:        int   = Field(..., ge=0, description="Total HTTP requests in window")
            error_rate:           float = Field(..., ge=0.0, le=1.0)
            request_rate:         float = Field(..., ge=0.0)
            avg_response_time_ms: float = Field(..., ge=0.0)
            ...
    """

    # TODO: Replace with actual feature fields
    features: dict[str, float] = Field(
        ...,
        description="Temporary catch-all dict. Replace with typed fields.",
        example={
            "request_count": 150,
            "error_rate": 0.12,
            "auth_failure_count": 5,
            "avg_response_time_ms": 340.0,
            "p95_response_time_ms": 820.0,
            "unique_endpoints": 8,
            "unique_ips": 1,
            "anomalous_path_count": 1,
            "post_ratio": 0.2,
            "js_error_count": 0,
            "request_rate": 2.5,
            "session_duration_s": 60.0,
        },
    )


class BatchSessionFeatures(BaseModel):
    """
    Input schema for a batch prediction request.

    TODO:
        - Replace sessions list type with list[SessionFeatures] once fields
          are properly typed in SessionFeatures
        - Add a max length validator to protect the service
    """

    sessions: list[dict[str, float]] = Field(
        ...,
        description="List of session feature dicts.",
        min_length=1,
    )


class PredictionResponse(BaseModel):
    """
    Output schema for a single prediction result.

    TODO:
        - Add sessionId field if tracking context is passed in the request
        - Add modelVersion field once versioning is implemented
    """

    anomalyScore: float = Field(..., description="Normalized anomaly score [0, 1].")
    prediction: str = Field(..., description="'NORMAL' or 'ANOMALY'.")
    confidence: str = Field(..., description="'LOW', 'MEDIUM', or 'HIGH'.")


class BatchPredictionResponse(BaseModel):
    """Output schema for a batch prediction result."""

    results: list[PredictionResponse]
    total: int


# ─── Application Lifespan ─────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load ML artifacts once at startup; release on shutdown.

    TODO:
        - Call predict.load_artifacts() and store in app.state.artifacts
        - Log model loading time
        - If artifacts are missing, log a warning but don't crash the process
          (allows /health to respond with a degraded status)
    """
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
        "Uses an Isolation Forest model trained on UNSW-NB15 network traffic data."
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

    Used by:
      - Spring Boot Actuator / gateway health aggregation
      - Kubernetes liveness / readiness probes

    TODO:
        - Return {"status": "UP", "model": "LOADED"} when model is ready
        - Return {"status": "DEGRADED", "model": "NOT_LOADED"} if model failed
        - Add "version" and "uptime_seconds" to the response
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
    Accept a single session feature vector and return an anomaly prediction.

    Called by:
      Spring Boot log-service after aggregating a session window.

    TODO:
        - Check app.state.model_ready; raise 503 if model is not loaded
        - Call feature_engineering.format_for_prediction(request.features)
        - Call predict.predict_session(formatted_features, app.state.artifacts)
        - Return the result as a PredictionResponse
        - Add request/response logging with a correlation ID
    """
    if not app.state.model_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run python src/train.py first.",
        )

    from ml.feature_engineering import format_for_prediction
    formatted = format_for_prediction(request.features)
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
    Accept multiple session feature vectors and return a prediction for each.

    Useful for:
      - Replaying historical log windows through the model
      - Bulk analysis tasks triggered by the admin dashboard

    TODO:
        - Check app.state.model_ready; raise 503 if model is not loaded
        - Format each session: [format_for_prediction(s) for s in request.sessions]
        - Call predict.predict_batch(formatted_sessions, app.state.artifacts)
        - Wrap results in BatchPredictionResponse
        - Add pagination support for very large batches (future)
    """
    if not app.state.model_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run python src/train.py first.",
        )

    from ml.feature_engineering import format_for_prediction
    formatted_sessions = [format_for_prediction(s) for s in request.sessions]
    results = predict_batch(formatted_sessions, app.state.artifacts)
    response_items = [PredictionResponse(**r) for r in results]
    return BatchPredictionResponse(results=response_items, total=len(results))
