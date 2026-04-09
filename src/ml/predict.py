"""
predict.py
────────────────────────────────────────────────────────────────────────────────
Core inference engine for the RiskTraceML anomaly detection service.

Responsibilities:
  - Load all trained artifacts from disk into memory (once, at startup)
  - Accept a feature vector (single or batch)
  - Apply scaling
  - Run Isolation Forest scoring via decision_function()
  - Normalize raw anomaly scores to a [0, 1] range
  - Apply a configurable threshold to classify NORMAL vs ANOMALY
  - Return a structured prediction response

Output schema (per session):
  {
    "anomalyScore":  float,   # 0 = normal, 1 = highly anomalous
    "prediction":    str,     # "NORMAL" | "ANOMALY"
    "confidence":    str,     # "LOW" | "MEDIUM" | "HIGH"
  }
────────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ─── Default Artifact Paths ───────────────────────────────────────────────────

DEFAULT_MODEL_PATH = "models/isolation_forest_model.pkl"
DEFAULT_SCALER_PATH = "models/scaler.pkl"
DEFAULT_ENCODERS_PATH = "models/label_encoders.pkl"
DEFAULT_FEATURE_COLUMNS_PATH = "models/feature_columns.json"

# ─── Decision Threshold ───────────────────────────────────────────────────────

# Raw decision_function() scores are negative for anomalies; we invert and
# normalize. Samples whose normalized score exceeds this threshold are flagged.
ANOMALY_THRESHOLD: float = 0.5

# Confidence band boundaries (applied to normalized score)
CONFIDENCE_HIGH_THRESHOLD: float = 0.75
CONFIDENCE_MEDIUM_THRESHOLD: float = 0.5


# ─── Artifact Management ──────────────────────────────────────────────────────


def load_artifacts(
    model_path: str = DEFAULT_MODEL_PATH,
    scaler_path: str = DEFAULT_SCALER_PATH,
    encoders_path: str = DEFAULT_ENCODERS_PATH,
    feature_columns_path: str = DEFAULT_FEATURE_COLUMNS_PATH,
) -> dict[str, Any]:
    from pathlib import Path
    
    base_dir = Path(__file__).resolve().parent.parent.parent
    def resolve_path(p):
        path = Path(p)
        return path if path.is_absolute() else base_dir / path

    model_p = resolve_path(model_path)
    scaler_p = resolve_path(scaler_path)
    
    if not model_p.exists() or not scaler_p.exists():
        raise RuntimeError(f"Model or Scaler not found at {model_p} / {scaler_p}. Please run train.py")
        
    artifacts = {
        "model": joblib.load(model_p),
        "scaler": joblib.load(scaler_p),
        "feature_columns": None,
        "encoders": None
    }
    
    feat_p = resolve_path(feature_columns_path)
    if feat_p.exists():
        with open(feat_p, 'r') as f:
            artifacts["feature_columns"] = json.load(f)
            
    logger.info("Artifacts successfully loaded into memory.")
    return artifacts


# ─── Score Normalization ──────────────────────────────────────────────────────


def normalize_score(raw_scores) -> list[float]:
    import numpy as np
    scores = np.array(raw_scores)
    # IsolationForest decision_function: positive is normal, negative is anomaly.
    # Sigmoid inversion to make [0, 1] scale where 1 = anomaly
    normalized = 1.0 / (1.0 + np.exp(scores))
    return normalized.tolist()


def _map_confidence(score: float) -> str:
    if score >= CONFIDENCE_HIGH_THRESHOLD:
        return "HIGH"
    elif score >= CONFIDENCE_MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


# ─── Single Prediction ────────────────────────────────────────────────────────


def predict_session(
    features: dict[str, float],
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent))
    from preprocessing import FEATURE_COLUMNS
    
    feature_cols = artifacts.get("feature_columns") or FEATURE_COLUMNS
    
    feature_array = []
    for col in feature_cols:
        feature_array.append(float(features.get(col, 0.0)))
        
    X = np.array([feature_array])
    X_scaled = artifacts["scaler"].transform(X)
    
    raw_score = artifacts["model"].decision_function(X_scaled)[0]
    anomaly_score = normalize_score([raw_score])[0]
    is_anomaly = anomaly_score >= ANOMALY_THRESHOLD
    
    return {
        "anomalyScore": float(anomaly_score),
        "prediction": "ANOMALY" if is_anomaly else "NORMAL",
        "confidence": _map_confidence(anomaly_score)
    }


# ─── Batch Prediction ─────────────────────────────────────────────────────────


def predict_batch(
    feature_list: list[dict[str, float]],
    artifacts: dict[str, Any],
) -> list[dict[str, Any]]:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent))
    from preprocessing import FEATURE_COLUMNS
    
    feature_cols = artifacts.get("feature_columns") or FEATURE_COLUMNS
    
    X_list = []
    for f in feature_list:
        X_list.append([float(f.get(col, 0.0)) for col in feature_cols])
        
    X = np.array(X_list)
    X_scaled = artifacts["scaler"].transform(X)
    
    raw_scores = artifacts["model"].decision_function(X_scaled)
    anomaly_scores = normalize_score(raw_scores)
    
    results = []
    for score in anomaly_scores:
        is_anomaly = score >= ANOMALY_THRESHOLD
        results.append({
            "anomalyScore": float(score),
            "prediction": "ANOMALY" if is_anomaly else "NORMAL",
            "confidence": _map_confidence(score)
        })
    return results
