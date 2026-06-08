import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ml.preprocessing import clean_data, prepare_features, FEATURE_COLUMNS
from api.main import (
    app,
    SessionFeatures,
    BatchSessionFeatures,
    health_check,
    predict,
    predict_batch_endpoint,
)

def test_clean_data():
    # Test data cleaning
    dummy_data = {col: [1.0, np.nan, -5.0] for col in FEATURE_COLUMNS}
    dummy_data["is_anomaly"] = [0, 1, 0]
    df = pd.DataFrame(dummy_data)
    
    cleaned = clean_data(df)
    
    # NaN should be filled
    assert not cleaned.isnull().values.any()
    # Negative values in count/rate cols should be clipped to 0
    assert (cleaned["error_rate"] >= 0).all()
    assert (cleaned["request_count"] >= 0).all()
    # Check shape
    assert cleaned.shape[1] == len(FEATURE_COLUMNS) + 1

def test_prepare_features():
    dummy_data = {col: [1.0, 2.0] for col in FEATURE_COLUMNS}
    dummy_data["is_anomaly"] = [0, 1]
    df = pd.DataFrame(dummy_data)
    
    X, y = prepare_features(df)
    assert X.shape == (2, len(FEATURE_COLUMNS))
    assert len(y) == 2
    assert list(y) == [0, 1]

@pytest.mark.anyio
async def test_api_health():
    # Mock model_ready to True
    app.state.model_ready = True
    response = await health_check()
    assert response["status"] == "UP"
    assert response["model"] == "LOADED"
    assert response["service"] == "RiskTraceML"

@pytest.mark.anyio
@patch("api.main.predict_session")
async def test_api_predict(mock_predict):
    app.state.model_ready = True
    app.state.artifacts = {}
    
    mock_predict.return_value = {
        "anomalyScore": 0.12,
        "prediction": "NORMAL",
        "confidence": "LOW"
    }
    
    payload = SessionFeatures(
        request_count=150.0,
        error_rate=0.05,
        auth_failure_count=2.0,
        avg_response_time_ms=120.5,
        p95_response_time_ms=450.0,
        unique_endpoints=12.0,
        unique_ips=1.0,
        anomalous_path_count=0.0,
        post_ratio=0.1,
        js_error_count=0.0,
        request_rate=2.5,
        session_duration_s=60.0
    )
    
    response = await predict(payload)
    assert response.prediction == "NORMAL"
    assert response.anomalyScore == 0.12
    assert response.confidence == "LOW"

@pytest.mark.anyio
@patch("api.main.predict_batch")
async def test_api_predict_batch(mock_predict_batch):
    app.state.model_ready = True
    app.state.artifacts = {}
    
    mock_predict_batch.return_value = [
        {"anomalyScore": 0.12, "prediction": "NORMAL", "confidence": "LOW"},
        {"anomalyScore": 0.85, "prediction": "ANOMALY", "confidence": "HIGH"}
    ]
    
    payload = BatchSessionFeatures(
        sessions=[
            SessionFeatures(
                request_count=150.0,
                error_rate=0.05,
                auth_failure_count=2.0,
                avg_response_time_ms=120.5,
                p95_response_time_ms=450.0,
                unique_endpoints=12.0,
                unique_ips=1.0,
                anomalous_path_count=0.0,
                post_ratio=0.1,
                js_error_count=0.0,
                request_rate=2.5,
                session_duration_s=60.0
            ),
            SessionFeatures(
                request_count=10.0,
                error_rate=0.9,
                auth_failure_count=9.0,
                avg_response_time_ms=120.5,
                p95_response_time_ms=450.0,
                unique_endpoints=1.0,
                unique_ips=1.0,
                anomalous_path_count=5.0,
                post_ratio=0.1,
                js_error_count=0.0,
                request_rate=2.5,
                session_duration_s=60.0
            )
        ]
    )
    
    response = await predict_batch_endpoint(payload)
    assert response.total == 2
    assert response.results[0].prediction == "NORMAL"
    assert response.results[1].prediction == "ANOMALY"
