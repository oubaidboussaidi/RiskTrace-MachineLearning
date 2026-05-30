# Technical Code Guide: Training & Preprocessing

This covers `train.py` and `preprocessing.py`. It explains how the AI is "born."

---

## 1. Data Cleaning (preprocessing.py)
```python
# Lines 90-100: Handling missing data and outliers
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Clipping negative values
for col in count_cols:
    df[col] = df[col].clip(lower=0)
```
*   **The Code**: Uses `pandas` to fill NaNs (missing data) and `clip()` to keep numbers positive.
*   **Logic**: Training data can be messy. Filling gaps with the **Median** ensures the model isn't confused by a few empty fields.

---

## 2. Model Training (train.py)
```python
# Lines 80-95: Configuring the Forest
model = IsolationForest(
    n_estimators=150,     # Number of trees in the forest
    contamination=0.15,   # We expect 15% of data to be anomalous
    random_state=42,      # Ensures the results are reproducible
    n_jobs=-1             # Uses ALL CPU cores for faster training
)
model.fit(X_train)
```
*   **`n_estimators=150`**: We build 150 different "Question Trees." The average of their answers is our final result.
*   **`contamination=0.15`**: This is the "Sensitivity" dial. We told the model: *"Expect that about 15% of the crowd are hackers."* This matches our K-Means balanced dataset.
*   **`n_jobs=-1`**: This tells Python to use your whole computer's power to build the trees in parallel.

---

## 3. Saving the Brain (Artifacts)
```python
# Lines 110-120: Persistence
joblib.dump(model, "models/isolation_forest_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
```
*   **The Code**: Uses the `joblib` library.
*   **Logic**: This "freezes" the AI's brain into a file. This is what we load in `main.py` so the API can work without needing to re-train the model every time.
