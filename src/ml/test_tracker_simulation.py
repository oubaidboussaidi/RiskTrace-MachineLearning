import pandas as pd
import numpy as np
import joblib
from predict import DEFAULT_SCALER_PATH, DEFAULT_MODEL_PATH, normalize_score, ANOMALY_THRESHOLD, _map_confidence
import json

scaler = joblib.load(DEFAULT_SCALER_PATH)
model = joblib.load(DEFAULT_MODEL_PATH)

# Feature columns must match EXACTLY
FEATURE_COLUMNS = [
    'anomalous_path_count', 'auth_failure_count', 'avg_response_time_ms', 
    'error_rate', 'js_error_count', 'p95_response_time_ms', 
    'post_ratio', 'request_count', 'request_rate', 
    'session_duration_s', 'unique_endpoints', 'unique_ips'
]

def score_features(user, features):
    row_df = pd.DataFrame([features])
    row_scaled = scaler.transform(row_df)
    raw_score = model.decision_function(row_scaled)[0]
    score = normalize_score([raw_score])[0]
    
    pred = "ANOMALY" if score >= ANOMALY_THRESHOLD else "NORMAL"
    print(f"[{user}]")
    print(f"  Score: {score*100:.2f}% ({pred})")
    print(f"  Features: {json.dumps({k: round(v, 2) for k,v in features.items()})}")
    print()

print("="*50)
print("  TEST TRACKER REALISTIC SIMULATION")
print("="*50 + "\n")

# Alice: Normal Browsing (8 requests over 20s)
alice_features = {
    'request_count': 8,
    'error_rate': 0.0,
    'auth_failure_count': 0,
    'avg_response_time_ms': 250.0,
    'p95_response_time_ms': 350.0,
    'unique_endpoints': 8,
    'unique_ips': 1,
    'anomalous_path_count': 0,
    'post_ratio': 0.125,
    'js_error_count': 0,
    'request_rate': 8 / 20.0,
    'session_duration_s': 20.0
}
score_features("Alice (Normal User)", alice_features)

# Bob: Power User (25 requests over 40s)
bob_features = {
    'request_count': 25,
    'error_rate': 0.04,
    'auth_failure_count': 0,
    'avg_response_time_ms': 180.0,
    'p95_response_time_ms': 220.0,
    'unique_endpoints': 12,
    'unique_ips': 1,
    'anomalous_path_count': 0,
    'post_ratio': 0.2,
    'js_error_count': 0,
    'request_rate': 25 / 40.0,
    'session_duration_s': 40.0
}
score_features("Bob (Power User)", bob_features)

# Eve: Brute Force (50 login requests over 1.5s, all 401)
eve_features = {
    'request_count': 50,
    'error_rate': 1.0,
    'auth_failure_count': 50,
    'avg_response_time_ms': 100.0,
    'p95_response_time_ms': 120.0,
    'unique_endpoints': 1,
    'unique_ips': 1,
    'anomalous_path_count': 0,
    'post_ratio': 1.0,
    'js_error_count': 0,
    'request_rate': 50 / 1.5,
    'session_duration_s': 1.5
}
score_features("Eve (Brute Force)", eve_features)

# Mallory: Path Scanner (30 requests over 1.2s, 404s/403s, highly anomalous paths)
mallory_features = {
    'request_count': 30,
    'error_rate': 1.0,
    'auth_failure_count': 6, # some 403s
    'avg_response_time_ms': 50.0,
    'p95_response_time_ms': 80.0,
    'unique_endpoints': 30,
    'unique_ips': 1,
    'anomalous_path_count': 25,
    'post_ratio': 0.0,
    'js_error_count': 0,
    'request_rate': 30 / 1.2,
    'session_duration_s': 1.2
}
score_features("Mallory (Path Scanner)", mallory_features)

# Trent: Traffic Spike (100 rapid GET requests in 2 seconds)
trent_features = {
    'request_count': 100,
    'error_rate': 0.0,
    'auth_failure_count': 0,
    'avg_response_time_ms': 45.0,
    'p95_response_time_ms': 60.0,
    'unique_endpoints': 100, # with ?p=... 
    'unique_ips': 1,
    'anomalous_path_count': 0,
    'post_ratio': 0.0,
    'js_error_count': 0,
    'request_rate': 100 / 2.0,
    'session_duration_s': 2.0
}
score_features("Trent (Traffic Spike DDoS)", trent_features)
