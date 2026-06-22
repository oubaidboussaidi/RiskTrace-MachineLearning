from pydantic import BaseModel, Field

class SessionFeatures(BaseModel):
    request_count: float = Field(..., ge=0, description="Total HTTP requests in window")
    error_rate: float = Field(..., ge=0.0, le=1.0, description="Ratio of 4xx/5xx status codes")
    auth_failure_count: float = Field(..., ge=0, description="Count of 401/403 status codes")
    avg_response_time_ms: float = Field(..., ge=0.0, description="Average response time in ms")
    p95_response_time_ms: float = Field(..., ge=0.0, description="95th percentile response time in ms")
    unique_endpoints: float = Field(..., ge=0, description="Count of unique URLs accessed")
    unique_ips: float = Field(..., ge=0, description="Count of unique source IPs (usually 1)")
    anomalous_path_count: float = Field(..., ge=0, description="Count of probes to sensitive paths like /admin, /.env")
    post_ratio: float = Field(..., ge=0.0, le=1.0, description="Ratio of POST requests")
    js_error_count: float = Field(..., ge=0, description="Browser-side JavaScript errors tracked")
    request_rate: float = Field(..., ge=0.0, description="Requests per second")
    session_duration_s: float = Field(..., ge=0.0, description="Time between first and last request in window")

class BatchSessionFeatures(BaseModel):
    sessions: list[SessionFeatures] = Field(..., min_length=1, description="List of session feature objects.")

class PredictionResponse(BaseModel):
    anomalyScore: float = Field(..., description="Normalized anomaly score [0.0 = safe, 1.0 = highly anomalous].")
    prediction: str = Field(..., description="'NORMAL' or 'ANOMALY'.")
    confidence: str = Field(..., description="Model confidence level ('LOW', 'MEDIUM', or 'HIGH').")

class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]
    total: int
