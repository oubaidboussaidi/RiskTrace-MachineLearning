"""
train.py
────────────────────────────────────────────────────────────────────────────────
End-to-end training pipeline for the RiskTraceML anomaly detection model.
"""
import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from preprocessing import load_data, clean_data, scale_features, prepare_features, FEATURE_COLUMNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = "Data/risk_trace_training_data.csv"
DEFAULT_MODEL_DIR = "models/"
DEFAULT_CONTAMINATION = 0.15

def main():
    parser = argparse.ArgumentParser(description="RiskTraceML — Training Pipeline")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--contamination", type=float, default=DEFAULT_CONTAMINATION)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("RiskTraceML — Isolation Forest Training Pipeline")
    logger.info("=" * 60)

    base_dir = Path(__file__).resolve().parent.parent.parent
    data_p = base_dir / args.data_path if not Path(args.data_path).is_absolute() else Path(args.data_path)
    
    if not data_p.exists():
        logger.error(f"Dataset not found at {data_p}!")
        return
        
    # 1. Load Data
    logger.info(f"Loading data from {data_p}...")
    df = load_data(str(data_p))
    df = clean_data(df)
    
    logger.info(f"Original dataset shape: {df.shape}")
    
    # Train / Test split
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["is_anomaly"]
    )
    
    # 2. Preprocess
    logger.info("Scaling features...")
    os.chdir(str(base_dir)) # To ensure relative model saves go to RiskTraceML/models
    
    train_scaled = scale_features(train_df, training=True)
    test_scaled = scale_features(test_df, training=False)
    
    X_train, y_train = prepare_features(train_scaled)
    X_test, y_test = prepare_features(test_scaled)
    
    # 3. Train
    logger.info(f"Training Isolation Forest (contamination={args.contamination})...")
    model = IsolationForest(
        contamination=args.contamination, 
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train)
    
    # 4. Evaluate
    logger.info("Evaluating on test set...")
    raw_preds = model.predict(X_test)
    y_pred = [0 if p == 1 else 1 for p in raw_preds]
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 5. Save Artifacts
    model_dir = base_dir / args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "isolation_forest_model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    feat_path = model_dir / "feature_columns.json"
    with open(feat_path, "w") as f:
        json.dump(FEATURE_COLUMNS, f)
    logger.info(f"Feature columns schema saved to {feat_path}")
    
    logger.info("✅ Training complete!")

if __name__ == "__main__":
    main()


