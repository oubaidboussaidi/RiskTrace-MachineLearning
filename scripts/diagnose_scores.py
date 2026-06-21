"""
diagnose_scores.py
------------------
Loads the trained Isolation Forest model and prints the raw decision_function()
scores for typical NORMAL and ATTACK sessions.

Run from the RiskTraceML directory:
    python diagnose_scores.py
"""
import sys
import numpy as np
import joblib
from pathlib import Path

# Setup paths
BASE = Path(__file__).resolve().parent
sys.path.append(str(BASE / "src"))

from src.ml.preprocessing import FEATURE_COLUMNS

model  = joblib.load(BASE / "models" / "isolation_forest_model.pkl")
scaler = joblib.load(BASE / "models" / "scaler.pkl")

# Synthetic sessions
# [request_count, error_rate, auth_failure_count, avg_rt_ms, p95_rt_ms,
#  unique_endpoints, unique_ips, anomalous_path_count, post_ratio,
#  js_error_count, request_rate, session_duration_s]
sessions = {
    "Normal browsing (5 pages, 60s)":
        [5,   0.0,  0,  150, 300,  5, 1,  0, 0.1,  0, 0.08, 60],
    "Normal browsing (20 pages, 120s)":
        [20,  0.02, 0,  180, 400, 12, 1,  0, 0.15, 0, 0.17, 120],
    "Brute Force (50 failed logins, 30s)":
        [50,  1.0,  50, 100, 150,  1, 1,  0, 1.0,  0, 1.67, 30],
    "Brute Force (100 failed logins, 30s)":
        [100, 1.0, 100,  90, 120,  1, 1,  0, 1.0,  0, 3.33, 30],
    "Path Scanning (admin probes, 20s)":
        [30,  0.9,  10, 120, 200, 30, 1, 30, 0.05, 0, 1.5,  20],
    "Traffic Spike (200 req, 10s)":
        [200, 0.05,  0, 100, 180, 15, 1,  0, 0.1,  0, 20.0, 10],
    "Single 404 (1 request)":
        [1,   1.0,   0, 200, 200,  1, 1,  0, 0.0,  0, 1.0,   1],
}

SEP = "-" * 85
print(SEP)
print(f"{'Session':<45} {'Raw Score':>12} {'Sigmoid*8':>12} {'IsAnomaly':>10}")
print(SEP)

raw_all = []
for label, feat in sessions.items():
    X   = np.array([feat])
    X_s = scaler.transform(X)
    raw = model.decision_function(X_s)[0]
    raw_all.append(raw)
    sig8      = 1.0 / (1.0 + np.exp(np.clip(raw * 8.0, -20, 20)))
    is_anom   = "ANOMALY" if sig8 >= 0.5 else "normal"
    print(f"{label:<45} {raw:>12.4f} {sig8:>12.4f} {is_anom:>10}")

print(SEP)
print(f"Raw score range:  min={min(raw_all):.4f}  max={max(raw_all):.4f}")
print(f"Model offset (contamination boundary): {model.offset_:.4f}")
print(f"  -> Scores below this offset are classified as ANOMALY by sklearn itself")

# Min-max normalization stats
mn, mx = min(raw_all), max(raw_all)
print(SEP)
print("Min-max normalization preview (score=(raw-max)/(min-max), clipped to [0,1]):")
print(f"{'Session':<45} {'Min-Max Score':>14} {'IsAnomaly':>10}")
print(SEP)
for label, feat in sessions.items():
    X   = np.array([feat])
    X_s = scaler.transform(X)
    raw = model.decision_function(X_s)[0]
    mm_score = np.clip((raw - mx) / (mn - mx + 1e-9), 0.0, 1.0)
    is_anom  = "ANOMALY" if mm_score >= 0.5 else "normal"
    print(f"{label:<45} {mm_score:>14.4f} {is_anom:>10}")

print(SEP)
print("DONE. Use these results to pick the best normalization strategy.")
