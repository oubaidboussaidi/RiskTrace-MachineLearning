"""
full_audit_and_test.py

1. Audits CSV dataset columns vs ML FEATURE_COLUMNS
2. Trains the Isolation Forest on the dataset
3. Evaluates accuracy, precision, recall, F1
4. Runs inference on simulated live tracker logs (exactly like backend sends)

Run: python src/ml/full_audit_and_test.py
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#  Setup paths 
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(Path(__file__).resolve().parent))

from preprocessing import FEATURE_COLUMNS, LABEL_COLUMN, load_data, clean_data, scale_features, prepare_features
from feature_engineering import aggregate_session_logs, format_for_prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ====================================================================
# STEP 1: AUDIT  CSV vs ML Schema
# ====================================================================
print("\n" + "="*60)
print("STEP 1: SCHEMA AUDIT")
print("="*60)

DATA_PATH = BASE_DIR / "Data" / "risk_trace_training_data.csv"
df_raw = pd.read_csv(DATA_PATH)

csv_cols = set(df_raw.columns.tolist())
ml_cols  = set(FEATURE_COLUMNS)

print(f"  CSV columns    ({len(csv_cols)}): {sorted(csv_cols)}")
print(f"  ML  features   ({len(ml_cols)}): {sorted(ml_cols)}")

missing_in_csv = ml_cols - csv_cols
extra_in_csv   = csv_cols - ml_cols - {LABEL_COLUMN}

if missing_in_csv:
    print(f"\n  [FAIL] FEATURES MISSING FROM CSV: {missing_in_csv}")
else:
    print(f"\n  OK All ML features are present in the dataset CSV.")

if extra_in_csv:
    print(f"  [INFO]  Extra CSV cols (ignored by ML): {extra_in_csv}")
else:
    print(f"  OK No unexpected extra columns.")

print(f"\n  Dataset shape : {df_raw.shape}")
print(f"  Normal  (0)   : {(df_raw[LABEL_COLUMN]==0).sum():,}")
print(f"  Attack  (1)   : {(df_raw[LABEL_COLUMN]==1).sum():,}")
ratio = (df_raw[LABEL_COLUMN]==1).sum() / len(df_raw)
print(f"  Attack ratio  : {ratio:.1%}")


# ====================================================================
# STEP 2: TRAIN + EVALUATE
# ====================================================================
print("\n" + "="*60)
print("STEP 2: TRAIN & EVALUATE")
print("="*60)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
SCALER_PATH = MODEL_DIR / "scaler.pkl"
MODEL_PATH  = MODEL_DIR / "isolation_forest_model.pkl"
FEAT_PATH   = MODEL_DIR / "feature_columns.json"

# Preprocessing
df = load_data(str(DATA_PATH))
df = clean_data(df)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[LABEL_COLUMN])
logger.info(f"Train/Test split: {len(train_df):,} / {len(test_df):,}")

import os; os.chdir(str(BASE_DIR))
train_scaled = scale_features(train_df, training=True)
test_scaled  = scale_features(test_df,  training=False)

X_train, y_train = prepare_features(train_scaled)
X_test,  y_test  = prepare_features(test_scaled)

# Train  contamination set to actual attack ratio in training split
contamination = float((train_df[LABEL_COLUMN]==1).sum() / len(train_df))
contamination = round(min(max(contamination, 0.01), 0.49), 3)
logger.info(f"Contamination set to: {contamination}")

model = IsolationForest(
    contamination=contamination,
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train)

# Evaluate
raw_preds = model.predict(X_test)
y_pred    = [0 if p == 1 else 1 for p in raw_preds]  # IF: +1=normal, -1=anomaly

print("\n  Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))

# Save
joblib.dump(model, MODEL_PATH)
with open(FEAT_PATH, "w") as f:
    json.dump(FEATURE_COLUMNS, f)
logger.info(f"OK Model saved  {MODEL_PATH}")
logger.info(f"OK Feature columns saved  {FEAT_PATH}")


# ====================================================================
# STEP 3: SIMULATE LIVE TRACKER LOGS (exactly as Spring Boot sends)
# ====================================================================
print("\n" + "="*60)
print("STEP 3: SIMULATED LIVE TRACKER INFERENCE")
print("="*60)

# Load fresh artifacts
scaler = joblib.load(SCALER_PATH)
model2 = joblib.load(MODEL_PATH)

def predict_from_raw_logs(session_name, logs):
    """Full pipeline: raw Spring Boot logs  features  prediction."""
    features = aggregate_session_logs(logs)
    features = format_for_prediction(features)
    row = np.array([[features[col] for col in FEATURE_COLUMNS]])
    row_scaled = scaler.transform(row)
    raw_score = model2.decision_function(row_scaled)[0]
    score = float(1.0 / (1.0 + np.exp(raw_score)))
    prediction = " ANOMALY" if score >= 0.5 else "OK NORMAL"
    confidence = "HIGH" if score >= 0.75 else ("MEDIUM" if score >= 0.5 else "LOW")
    print(f"\n  [{session_name}]")
    print(f"    Features      : {features}")
    print(f"    Anomaly Score : {score:.4f}")
    print(f"    Prediction    : {prediction}   ({confidence} confidence)")

# --- Scenario A: Normal user browsing (camelCase  exactly what Spring Boot sends)
normal_session = [
    {"type": "page_load",      "url": "/dashboard",         "method": "GET",    "statusCode": 200, "responseTime": 210, "ipAddress": "192.168.1.5", "userAgent": "Mozilla/5.0"},
    {"type": "fetch_request",  "url": "/api/v1/risks",      "method": "GET",    "statusCode": 200, "responseTime": 180, "ipAddress": "192.168.1.5", "userAgent": "Mozilla/5.0"},
    {"type": "fetch_request",  "url": "/api/v1/sites",      "method": "GET",    "statusCode": 200, "responseTime": 220, "ipAddress": "192.168.1.5", "userAgent": "Mozilla/5.0"},
    {"type": "fetch_request",  "url": "/api/v1/users/me",   "method": "GET",    "statusCode": 200, "responseTime": 150, "ipAddress": "192.168.1.5", "userAgent": "Mozilla/5.0"},
    {"type": "form_submit",    "url": "/api/v1/reports",    "method": "POST",   "statusCode": 201, "responseTime": 310, "ipAddress": "192.168.1.5", "userAgent": "Mozilla/5.0"},
]

# --- Scenario B: Brute force attack (as triggered by test-tracker.html)
brute_force_session = [
    {"type": "fetch_request", "url": "/api/auth/login", "method": "POST", "statusCode": 401, "responseTime": 95, "ipAddress": "10.0.0.99", "userAgent": "python-requests/2.28"},
    {"type": "fetch_request", "url": "/api/auth/login", "method": "POST", "statusCode": 401, "responseTime": 88, "ipAddress": "10.0.0.99", "userAgent": "python-requests/2.28"},
    {"type": "fetch_request", "url": "/api/auth/login", "method": "POST", "statusCode": 401, "responseTime": 92, "ipAddress": "10.0.0.99", "userAgent": "python-requests/2.28"},
    {"type": "fetch_request", "url": "/api/auth/login", "method": "POST", "statusCode": 403, "responseTime": 90, "ipAddress": "10.0.0.99", "userAgent": "python-requests/2.28"},
    {"type": "fetch_request", "url": "/api/auth/login", "method": "POST", "statusCode": 401, "responseTime": 91, "ipAddress": "10.0.0.99", "userAgent": "python-requests/2.28"},
    {"type": "fetch_request", "url": "/api/auth/login", "method": "POST", "statusCode": 401, "responseTime": 89, "ipAddress": "10.0.0.99", "userAgent": "python-requests/2.28"},
    {"type": "fetch_request", "url": "/api/auth/login", "method": "POST", "statusCode": 401, "responseTime": 93, "ipAddress": "10.0.0.99", "userAgent": "python-requests/2.28"},
    {"type": "fetch_request", "url": "/api/auth/login", "method": "POST", "statusCode": 401, "responseTime": 94, "ipAddress": "10.0.0.99", "userAgent": "python-requests/2.28"},
    {"type": "fetch_request", "url": "/api/auth/login", "method": "POST", "statusCode": 401, "responseTime": 87, "ipAddress": "10.0.0.99", "userAgent": "python-requests/2.28"},
    {"type": "fetch_request", "url": "/api/auth/login", "method": "POST", "statusCode": 403, "responseTime": 96, "ipAddress": "10.0.0.99", "userAgent": "python-requests/2.28"},
]

# --- Scenario C: Reconnaissance / path scanning
recon_session = [
    {"type": "fetch_request", "url": "/.env",           "method": "GET", "statusCode": 404, "responseTime": 50,  "ipAddress": "185.0.0.1", "userAgent": "Nikto/2.1"},
    {"type": "fetch_request", "url": "/.git/config",    "method": "GET", "statusCode": 404, "responseTime": 45,  "ipAddress": "185.0.0.1", "userAgent": "Nikto/2.1"},
    {"type": "fetch_request", "url": "/phpmyadmin",     "method": "GET", "statusCode": 404, "responseTime": 48,  "ipAddress": "185.0.0.1", "userAgent": "Nikto/2.1"},
    {"type": "fetch_request", "url": "/admin",          "method": "GET", "statusCode": 403, "responseTime": 60,  "ipAddress": "185.0.0.1", "userAgent": "Nikto/2.1"},
    {"type": "fetch_request", "url": "/wp-admin",       "method": "GET", "statusCode": 404, "responseTime": 52,  "ipAddress": "185.0.0.1", "userAgent": "Nikto/2.1"},
    {"type": "fetch_request", "url": "/config",         "method": "GET", "statusCode": 403, "responseTime": 55,  "ipAddress": "185.0.0.1", "userAgent": "Nikto/2.1"},
    {"type": "fetch_request", "url": "/backup.zip",     "method": "GET", "statusCode": 404, "responseTime": 47,  "ipAddress": "185.0.0.1", "userAgent": "Nikto/2.1"},
]

# --- Scenario D: JS Error flood (triggered by test-tracker.html error simulator)
js_error_session = [
    {"type": "page_load",                "url": "/dashboard", "method": "GET",  "statusCode": 200, "responseTime": 200, "ipAddress": "192.168.1.10", "userAgent": "Mozilla/5.0"},
    {"type": "js_error",                  "url": "/dashboard", "method": "GET",  "statusCode": 500, "responseTime": 0,   "ipAddress": "192.168.1.10", "userAgent": "Mozilla/5.0"},
    {"type": "js_error",                  "url": "/dashboard", "method": "GET",  "statusCode": 500, "responseTime": 0,   "ipAddress": "192.168.1.10", "userAgent": "Mozilla/5.0"},
    {"type": "unhandled_promise_rejection","url": "/dashboard", "method": "GET",  "statusCode": 500, "responseTime": 0,   "ipAddress": "192.168.1.10", "userAgent": "Mozilla/5.0"},
    {"type": "js_error",                  "url": "/dashboard", "method": "GET",  "statusCode": 500, "responseTime": 0,   "ipAddress": "192.168.1.10", "userAgent": "Mozilla/5.0"},
]

predict_from_raw_logs("Normal User Session",    normal_session)
predict_from_raw_logs("Brute Force Attack",      brute_force_session)
predict_from_raw_logs("Recon / Path Scanning",   recon_session)
predict_from_raw_logs("JS Error Flood",           js_error_session)

print("\n" + "="*60)
print("AUDIT COMPLETE")
print("="*60 + "\n")
