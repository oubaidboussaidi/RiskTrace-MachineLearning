"""
threshold_sweeping.py
Perform a full threshold sweep from 0.01 to 0.99 to find the argmax of the F1-Score.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(Path(__file__).resolve().parent))

from preprocessing import FEATURE_COLUMNS, LABEL_COLUMN, load_data, clean_data, scale_features, prepare_features
from predict import normalize_score

# ── Load data & model ─────────────────────────────────────────────
import os; os.chdir(str(BASE_DIR))

DATA_PATH = BASE_DIR / "Data" / "risk_trace_training_data.csv"
MODEL_PATH = BASE_DIR / "models" / "isolation_forest_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

print("Loading data and model...")
df = load_data(str(DATA_PATH))
df = clean_data(df)
_, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[LABEL_COLUMN])

test_scaled = scale_features(test_df, training=False)
X_test, y_test = prepare_features(test_scaled)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ── Get normalized scores ─────────────────────────────────────────
raw_scores = model.decision_function(X_test)
normalized_scores = normalize_score(raw_scores)

# ── Threshold Sweeping ────────────────────────────────────────────
print("\n" + "="*60)
print("  THRESHOLD SWEEPING (0.01 to 0.99)")
print("="*60)

best_threshold = 0.0
best_f1 = 0.0
best_metrics = {}

thresholds = np.arange(0.01, 0.90, 0.01)
f1_scores = []
precisions = []
recalls = []
results = []

for thresh in thresholds:
    # y_test contains 1 for anomaly, 0 for normal
    y_pred = [1 if s >= thresh else 0 for s in normalized_scores]
    
    p = precision_score(y_test, y_pred, zero_division=0)
    r = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    f1_scores.append(f1)
    precisions.append(p)
    recalls.append(r)
    results.append((thresh, p, r, f1))
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh
        best_metrics = {"precision": p, "recall": r, "f1": f1}

print("\n  [+] Sweeping Complete!")
print(f"  [+] Maximum F1-Score found at Threshold: {best_threshold:.2f}")
print(f"      - Precision : {best_metrics['precision']:.4f}")
print(f"      - Recall    : {best_metrics['recall']:.4f}")
print(f"      - F1-Score  : {best_metrics['f1']:.4f}")

print("\n  Detailed snapshot around the best threshold:")
for thresh, p, r, f1 in results:
    if abs(thresh - best_threshold) <= 0.05:
        marker = " <--- MAX F1" if thresh == best_threshold else ""
        print(f"    Thresh {thresh:.2f} | P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f}{marker}")

# ── Plotting ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresholds, f1_scores, color='purple', linewidth=2.5, label='F1-Score')
ax.plot(thresholds, precisions, color='blue', linestyle='--', label='Precision')
ax.plot(thresholds, recalls, color='green', linestyle='--', label='Recall')
ax.axvline(x=best_threshold, color='red', linestyle=':', linewidth=2, label=f'Optimal Threshold = {best_threshold:.2f}')

ax.set_xlabel('Anomaly Score Threshold')
ax.set_ylabel('Score Metric')
ax.set_title('Isolation Forest Threshold Optimization (Argmax of F1-Score)')
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)

out_path = Path("c:/Users/hp/Desktop/PFE/app/RiskTrace/Rapport_ML/optimization_curve_steep_sigmoid.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"\n  [+] Saved diagram to: {out_path}")
