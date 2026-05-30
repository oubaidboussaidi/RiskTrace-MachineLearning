# Technical Code Guide: predict.py

This file is the "Judge." It calculates the final anomaly score using mathematical normalization.

---

## 1. Feature Scaling
```python
# Lines 80-85: Preparing the numbers
# Ensure the columns are in the exact order the model expects
row = np.array([[features[col] for col in FEATURE_COLUMNS]])
scaled_features = scaler.transform(row)
```
*   **The Code**: Uses the `StandardScaler` loaded from memory.
*   **Logic**: A model cannot compare `request_count` (0-5000) with `error_rate` (0.0-1.0) directly. Scaling centers all features so they have a similar mathematical "weight."

---

## 2. Anomaly Scoring (Isolation Forest)
```python
# Line 90: Getting the raw depth
raw_score = model.decision_function(scaled_features)[0]
```
*   **The Code**: Calls the Scikit-Learn `decision_function`.
*   **Logic**: This calculates the **Average Path Length** in the forest. 
    *   **Positive** = Deep in the trees (Normal).
    *   **Negative** = Isolated quickly (Anomaly).

---

## 3. The Sigmoid Normalization (Math)
```python
# Line 95: Converting to a 0.0 - 1.0 range
# Using the logistic sigmoid function formula: 1 / (1 + e^-x)
score = float(1.0 / (1.0 + np.exp(raw_score)))
```
*   **The Code**: Uses `np.exp` (exponential function).
*   **Logic**: Raw scores are hard for humans to read. This formula squashes the score into a beautiful percentage:
    *   `0.10` = Definitely Normal.
    *   `0.90` = Definitely a Hacker.

---

## 4. Confidence Logic
```python
# Lines 100-110: Setting the Verdict
prediction = "ANOMALY" if score >= 0.5 else "NORMAL"
confidence = "HIGH" if score >= 0.75 else ("MEDIUM" if score >= 0.5 else "LOW")
```
*   **The Code**: Simple conditional logic (If/Else).
*   **Logic**: If the score is > 0.5, we flag it. If it's > 0.75, we are highly confident. This helps the Admin Dashboard show "Red" alerts for high-risk anomalies.
