"""
feature_engineering.py
────────────────────────────────────────────────────────────────────────────────
Transforms raw RiskTrace application logs into ML-ready feature vectors.

This module is the bridge between what the Spring Boot backend sends and what
the Isolation Forest model expects.

Responsibilities:
  - Aggregate per-session / per-time-window log entries
  - Compute the 15 behavioural features used by the model
  - Ensure a consistent, schema-validated feature dictionary at output

Output (per session):
  {
    "request_count":        float,
    "error_rate":           float,
    "auth_failure_count":   float,
    "avg_response_time_ms": float,
    "p95_response_time_ms": float,
    "unique_endpoints":     float,
    "unique_ips":           float,
    "anomalous_path_count": float,
    "post_ratio":           float,
    "js_error_count":       float,
    "request_rate":         float,
    "session_duration_s":   float,
  }
────────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

# The exact 12 features the model is trained on — ORDER matters for numpy arrays
FEATURE_COLUMNS: list[str] = [
    "request_count",
    "error_rate",
    "auth_failure_count",
    "avg_response_time_ms",
    "p95_response_time_ms",
    "unique_endpoints",
    "unique_ips",
    "anomalous_path_count",
    "post_ratio",
    "js_error_count",
    "request_rate",
    "session_duration_s",
]

# Default time-window size used when aggregating a stream of logs
DEFAULT_WINDOW_SECONDS: int = 60

# HTTP status codes considered errors
ERROR_STATUS_CODES: set[int] = {400, 401, 403, 404, 405, 429, 500, 502, 503, 504}

# Paths that are typically probed during reconnaissance / attacks
ANOMALOUS_PATH_PATTERNS: list[str] = [
    "/admin",
    "/actuator",
    "/.env",
    "/wp-admin",
    "/phpmyadmin",
    "/.git",
    "/config",
]


# ─── Session Aggregation ──────────────────────────────────────────────────────


def aggregate_session_logs(logs: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate a list of raw log dictionaries representing one session window
    into a single feature dictionary.

    Args:
        logs: List of log entry dicts. Each dict is expected to have at minimum:
              {
                "timestamp":       str (ISO-8601),
                "method":          str (GET, POST, ...),
                "endpoint":        str,
                "status_code":     int,
                "response_time":   float (ms),
                "bytes_sent":      int,
                "bytes_received":  int,
                "ip_address":      str,
                "user_agent":      str,
              }

    Returns:
        Feature dict with FEATURE_COLUMNS as keys.

    TODO:
        - Parse timestamps and compute session_duration_s as max-min
        - Count request_count as len(logs)
        - Compute error_rate = count(status in ERROR_STATUS_CODES) / request_count
        - Compute request_rate = request_count / session_duration_s
        - Compute avg and p95 of response_time values
        - Count unique endpoints, unique IPs, unique user agents
        - Sum bytes_sent and bytes_received across log entries
        - Compute get_ratio and post_ratio
        - Count auth_failure_count (status_code == 401 or 403)
        - Count anomalous_path_count by matching endpoint against ANOMALOUS_PATH_PATTERNS
        - Handle edge case: single log entry (duration = 0 → request_rate = 0)
    """
    if not logs:
        return {col: 0.0 for col in FEATURE_COLUMNS}
        
    req_count = len(logs)
    
    errors = 0
    auth_fails = 0
    response_times = []
    endpoints = set()
    ips = set()
    posts = 0
    js_errors = 0
    anomalous = 0
    
    for log in logs:
        # Cross-compatibility between Offline Dataset (snake_case) & SpringBoot API (camelCase)
        sc = log.get("status_code") if "status_code" in log else log.get("statusCode", 200)
        sc = int(sc) if sc is not None else 200
        
        rt = log.get("response_time") if "response_time" in log else log.get("responseTime", 0.0)
        rt = float(rt) if rt is not None else 0.0
        
        ep = str(log.get("endpoint") or log.get("url") or "")
        ip = str(log.get("ip_address") or log.get("ipAddress") or "")
        method = str(log.get("method", "")).upper()
        log_type = str(log.get("type", ""))
        
        if sc in ERROR_STATUS_CODES:
            errors += 1
        if sc in {401, 403}:
            auth_fails += 1
            
        response_times.append(rt)
        endpoints.add(ep)
        ips.add(ip)
        
        if method == "POST":
            posts += 1
            
        if any(pat in ep for pat in ANOMALOUS_PATH_PATTERNS):
            anomalous += 1
            
        if log_type == "js_error":
            js_errors += 1

    # Mocking duration for simple batch if not parsed properly
    duration = float(DEFAULT_WINDOW_SECONDS)
    
    return {
        "request_count": float(req_count),
        "error_rate": float(errors / req_count) if req_count else 0.0,
        "auth_failure_count": float(auth_fails),
        "avg_response_time_ms": float(np.mean(response_times)) if response_times else 0.0,
        "p95_response_time_ms": float(np.percentile(response_times, 95)) if response_times else 0.0,
        "unique_endpoints": float(len(endpoints)),
        "unique_ips": float(len(ips)),
        "anomalous_path_count": float(anomalous),
        "post_ratio": float(posts / req_count) if req_count else 0.0,
        "js_error_count": float(js_errors),
        "request_rate": float(req_count / duration) if duration else 0.0,
        "session_duration_s": duration,
    }


# ─── Inference Formatting ─────────────────────────────────────────────────────


def format_for_prediction(raw_features: dict[str, Any]) -> dict[str, float]:
    """
    Validate and format a raw feature dictionary for model inference.

    Ensures:
      - All 15 FEATURE_COLUMNS are present
      - All values are numeric (float/int)
      - Missing features default to 0.0 with a warning

    Args:
        raw_features: A dict produced by aggregate_session_logs() or sent
                      directly by the Spring Boot backend.

    Returns:
        Clean, ordered feature dict ready to be passed to predict.py.

    TODO:
        - Iterate over FEATURE_COLUMNS; check each key exists in raw_features
        - Cast each value to float; catch TypeError / ValueError → default 0.0
        - Log a warning for each missing or malformed feature
        - Optionally enforce value ranges (e.g., rates must be >= 0)
    """
    formatted = {}
    for col in FEATURE_COLUMNS:
        if col not in raw_features:
            logger.warning(f"Feature '{col}' missing from raw_features, defaulting to 0.0")
            formatted[col] = 0.0
        else:
            try:
                formatted[col] = float(raw_features[col])
            except (TypeError, ValueError):
                logger.warning(f"Feature '{col}' malformed ('{raw_features[col]}'), defaulting to 0.0")
                formatted[col] = 0.0
    return formatted


# ─── Batch Utilities ──────────────────────────────────────────────────────────


def logs_to_feature_matrix(
    session_groups: list[list[dict[str, Any]]],
) -> np.ndarray:
    """
    Convert multiple session groups into a 2-D feature matrix for batch prediction.

    Args:
        session_groups: List of log groups, one per session window.

    Returns:
        numpy array of shape (n_sessions, len(FEATURE_COLUMNS)).

    TODO:
        - Call aggregate_session_logs() for each group
        - Call format_for_prediction() on the result
        - Stack all feature dicts into a numpy array in FEATURE_COLUMNS order
    """
    matrix = []
    for group in session_groups:
        agg = aggregate_session_logs(group)
        fmt = format_for_prediction(agg)
        matrix.append([fmt[col] for col in FEATURE_COLUMNS])
    return np.array(matrix) if matrix else np.array([])
