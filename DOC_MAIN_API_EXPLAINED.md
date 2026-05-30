# Technical Code Guide: main.py (FastAPI)

This is the "Bridge" that connects the Python model to the Java/Spring Boot backend.

---

## 1. Data Validation (Pydantic)
```python
# Lines 45-63: Defining the API Schema
class SessionFeatures(BaseModel):
    request_count: float
    error_rate: float
    auth_failure_count: float
    # ... (other 9 features)
```
*   **The Code**: Uses the `pydantic` library's `BaseModel`.
*   **Logic**: This tells the server exactly what the data from Spring Boot should look like. If Java sends a string when it should be a number, the server will block it automatically.

---

## 2. Lifespan (Memory Management)
```python
# Lines 120-135: Pre-loading the Model
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load scaler and model from 'models/' directory
    ml_models['scaler'] = joblib.load("models/scaler.pkl")
    ml_models['model'] = joblib.load("models/isolation_forest_model.pkl")
    yield
    ml_models.clear()
```
*   **The Code**: An asynchronous context manager.
*   **Logic**: This code runs **only once** when the server starts. It loads the AI into RAM so that every incoming request is processed instantly. `yield` keeps the model alive until the server is turned off.

---

## 3. The Batch Prediction Endpoint
```python
# Lines 210-230: Processing multiple users at once
@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    results = []
    for session in request.sessions:
        # Convert Pydantic model to raw dictionary
        data = session.model_dump()
        result = predict_anomaly(data)
        results.append(result)
    return {"results": results}
```
*   **The Code**: A `POST` endpoint that accepts a list of sessions.
*   **Logic**: Instead of sending one IP at a time, Spring Boot can send a "batch" of 100 users. The loop processes each one and returns a list of "Normal/Anomaly" verdicts.
*   **`model_dump()`**: Converts the validated data into a clean Python dictionary.
