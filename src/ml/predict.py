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
import sys
from pathlib import Path
from typing import Any

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Ensure the src/ directory is on sys.path so sibling modules are importable
sys.path.append(str(Path(__file__).resolve().parent))

logger = logging.getLogger(__name__)

# ─── Default Artifact Paths ───────────────────────────────────────────────────

DEFAULT_MODEL_PATH = "models/isolation_forest_model.pkl"
DEFAULT_SCALER_PATH = "models/scaler.pkl"
DEFAULT_ENCODERS_PATH = "models/label_encoders.pkl"
DEFAULT_FEATURE_COLUMNS_PATH = "models/feature_columns.json"

# --- Decision Threshold ---

# Sensitivity multiplier — no longer used (kept for reference)
SENSITIVITY_FACTOR: float = 8.0

# Raw decision_function() score range observed from the trained model.
# Normal sessions cluster near +0.07, attacks cluster near -0.25.
# We use min-max normalization against this range so the final score
# maps intuitively: 0% = perfectly normal, 100% = clear attack.
SCORE_MIN: float = -0.35   # widened slightly
SCORE_MAX: float = +0.15   # widened slightly

# Samples whose normalized score exceeds this threshold are flagged ANOMALY.
ANOMALY_THRESHOLD: float = 0.5

# Confidence band boundaries
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
    """
    Convert raw IsolationForest decision_function() scores to [0, 1].

    decision_function() convention:
      positive  -> normal  (far from anomaly boundary)
      negative  -> anomaly (close to or past boundary)

    We invert and min-max scale so that:
      SCORE_MAX (most normal) -> 0.0
      SCORE_MIN (most anomalous) -> 1.0
      
    Then apply a sigmoid transformation to push normal scores lower
    and attack scores higher, creating a clear separation.
    """
    scores = np.array(raw_scores, dtype=float)
    # 1. Linear Min-Max Scale
    normalized = (SCORE_MAX - scores) / (SCORE_MAX - SCORE_MIN)
    normalized = np.clip(normalized, 0.0, 1.0)
    
    # 2. S-Curve (Sigmoid) to stretch the extremes
    # Centered at 0.5, k=10 controls the steepness of the curve
    k = 10.0
    s_curve = 1.0 / (1.0 + np.exp(-k * (normalized - 0.5)))
    
    # Debug print to console
    for i, s in enumerate(scores):
        print(f"[ML-DEBUG] Raw: {s:.4f} -> Linear: {normalized[i]:.4f} -> Sigmoid: {s_curve[i]:.4f}")
        
    return s_curve.tolist()


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
    import pandas as pd
    from preprocessing import FEATURE_COLUMNS

    feature_cols = artifacts.get("feature_columns") or FEATURE_COLUMNS

    # Use DataFrame so the scaler/model receive named columns (avoids sklearn warnings)
    row = {col: float(features.get(col, 0.0)) for col in feature_cols}
    X = pd.DataFrame([row], columns=feature_cols)
    X_scaled = artifacts["scaler"].transform(X)

    raw_score    = artifacts["model"].decision_function(X_scaled)[0]
    anomaly_score = normalize_score([raw_score])[0]
    is_anomaly    = anomaly_score >= ANOMALY_THRESHOLD

    logger.debug("[predict_session] raw=%.4f  normalized=%.4f  label=%s",
                 raw_score, anomaly_score, "ANOMALY" if is_anomaly else "NORMAL")

    return {
        "anomalyScore": float(anomaly_score),
        "prediction":   "ANOMALY" if is_anomaly else "NORMAL",
        "confidence":   _map_confidence(anomaly_score)
    }


# ─── Batch Prediction ─────────────────────────────────────────────────────────


def predict_batch(
    feature_list: list[dict[str, float]],
    artifacts: dict[str, Any],
) -> list[dict[str, Any]]:
    import pandas as pd
    from preprocessing import FEATURE_COLUMNS

    feature_cols = artifacts.get("feature_columns") or FEATURE_COLUMNS

    # Use DataFrame so the scaler/model receive named columns (avoids sklearn warnings)
    rows = [{col: float(f.get(col, 0.0)) for col in feature_cols} for f in feature_list]
    X = pd.DataFrame(rows, columns=feature_cols)
    X_scaled = artifacts["scaler"].transform(X)

    raw_scores    = artifacts["model"].decision_function(X_scaled)
    anomaly_scores = normalize_score(raw_scores)

    logger.info("[predict_batch] %d sessions scored. raw range [%.4f, %.4f]",
                len(raw_scores), float(raw_scores.min()), float(raw_scores.max()))

    results = []
    for score in anomaly_scores:
        is_anomaly = score >= ANOMALY_THRESHOLD
        results.append({
            "anomalyScore": float(score),
            "prediction":   "ANOMALY" if is_anomaly else "NORMAL",
            "confidence":   _map_confidence(score)
        })
    return results
