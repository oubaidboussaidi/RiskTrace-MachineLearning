# Line-by-Line: main.py (FastAPI)

This file is the "Bridge." It allows the Spring Boot backend to send data to the Python model via the network.

---

### 1. Pydantic Models (The Data Guardians)
```python
class SessionFeatures(BaseModel):
    request_count: float
    error_rate: float
    # ... (10 other features)
```
*   **`BaseModel`**: This is from the `pydantic` library. It validates incoming data.
*   **`float`**: We define every feature as a number. If Spring Boot sends a string (like "ABC") instead of a number, FastAPI will automatically reject the request before it even touches the model.

---

### 2. Lifespan (Startup Logic)
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML artifacts
    ml_models['scaler'] = joblib.load("models/scaler.pkl")
    ml_models['model'] = joblib.load("models/isolation_forest_model.pkl")
    yield
    # Shutdown: Clear memory
    ml_models.clear()
```
*   **`@asynccontextmanager`**: A special Python feature that manages the "Life" of the app.
*   **`joblib.load`**: This loads the pre-trained model and scaler from the files we saved during training. 
*   **Why here?** We load them **once** when the server starts so that predictions are fast. If we loaded them inside the route, the server would be very slow.
*   **`yield`**: This separates the "Start" code from the "Stop" code.

---

### 3. The Prediction Route (`/predict`)
```python
@app.post("/predict")
async def predict_endpoint(request: PredictionRequest):
```
*   **`@app.post`**: This defines an HTTP POST endpoint at `http://localhost:8000/predict`.
*   **`request: PredictionRequest`**: The variable `request` will automatically contain the JSON data sent by Spring Boot.

```python
# 1. Feature Engineering
features = aggregate_session_logs(request.logs)
features = format_for_prediction(features)
```
*   These lines call our logic in `feature_engineering.py` to turn the raw logs into the 12 numbers.

```python
# 2. Inference
result = predict_anomaly(features)
```
*   This calls `predict.py` to get the final score and prediction.

```python
# 3. Response
return result
```
*   FastAPI turns the Python dictionary into a JSON object and sends it back to Spring Boot.

---

# Line-by-Line: predict.py (The Judge)

### 1. Score Calculation
```python
raw_score = model.decision_function(scaled_features)[0]
```
*   **`decision_function`**: This is a built-in Scikit-Learn function. It returns a number.
    *   **Negative number**: Likely an anomaly.
    *   **Positive number**: Likely normal.

```python
# The Sigmoid Normalization
score = float(1.0 / (1.0 + np.exp(raw_score)))
```
*   **`np.exp(raw_score)`**: This is the "Exponential" math function ($e^x$).
*   **Why this formula?** It's a standard "Sigmoid" curve. It squashes any number (from $-\infty$ to $+\infty$) into a beautiful range between **0.0 and 1.0**.
    *   This makes the results much easier to read for humans and the UI dashboard.

```python
prediction = "ANOMALY" if score >= 0.5 else "NORMAL"
```
*   This is the final "Verdict." If the score is above 0.5 (50%), we flag it.
