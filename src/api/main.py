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
from .schemas import SessionFeatures, BatchSessionFeatures, PredictionResponse, BatchPredictionResponse

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model.predict import load_artifacts, predict_session, predict_batch
import py_eureka_client.eureka_client as eureka_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


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

    import os
    try:
        await eureka_client.init_async(
            eureka_server=os.getenv("EUREKA_SERVER", "http://localhost:8761/eureka"),
            app_name="ML-SERVICE",
            instance_port=int(os.getenv("ML_SERVICE_PORT", 8000)),
            instance_ip=os.getenv("ML_INSTANCE_IP", "127.0.0.1")
        )
        logger.info("✅ Registered with Eureka!")
    except Exception as e:
        logger.error(f"❌ Failed to register with Eureka: {e}")

    logger.info("✅ RiskTraceML service ready.")
    yield
    
    try:
        await eureka_client.stop_async()
        logger.info("🛑 Unregistered from Eureka!")
    except Exception as e:
        pass
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

    from src.model.engineering import format_for_prediction
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

    from src.model.engineering import format_for_prediction
    formatted_sessions = [format_for_prediction(s.model_dump()) for s in request.sessions]
    
    results = predict_batch(formatted_sessions, app.state.artifacts)
    response_items = [PredictionResponse(**r) for r in results]
    return BatchPredictionResponse(results=response_items, total=len(results))
