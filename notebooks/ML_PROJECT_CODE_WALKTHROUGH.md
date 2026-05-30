# RiskTraceML: Line-by-Line Technical Walkthrough

This document explains the inner workings of every major file in the RiskTrace Machine Learning pipeline.

---

## 1. `src/ml/feature_engineering.py` (The Feature Builder)
**Purpose:** This file takes a list of raw HTTP logs and turns them into a single "Behavioral Snapshot" (the 12 features).

*   **Lines 30-50 (`ANOMALOUS_PATH_PATTERNS`):** This is a Regular Expression (Regex). It looks for strings like `/admin`, `.env`, or `union select`. If it finds them, it flags the request as a "path anomaly."
*   **Lines 105-150 (`aggregate_session_logs`):** This is the most important function.
    *   **Looping:** It iterates through every log in a user's session.
    *   **Counting:** It calculates the `error_rate` (how many 4xx/5xx errors) and `auth_failure_count` (how many 401/403 errors).
    *   **Timing:** it calculates `session_duration_s` by subtracting the first log's timestamp from the last.
    *   **Averaging:** It calculates the average and 95th percentile of response times to detect server stress.

---

## 2. `src/ml/dataset_aggregator.py` (The Data Distiller)
**Purpose:** This file merges NASA and ModSecurity logs and uses K-Means to clean the data.

*   **Lines 262-310 (`intelligent_sample_anomalies`):**
    *   **Deduplication:** `drop_duplicates()` removes identical attack logs so the model doesn't get "bored" or biased.
    *   **K-Means Initialization:** `KMeans(n_clusters=...)` tells Python to find 'X' different groups of attacks.
    *   **Clustering:** `fit_predict()` calculates which group every attack belongs to based on its features.
    *   **Selection:** `.groupby('cluster').first()` picks one "representative" from every group. This ensures 100% diversity in our training set.

---

## 3. `src/api/main.py` (The FastAPI Bridge)
**Purpose:** This is the web server that lets Spring Boot talk to our Python model.

*   **Lines 45-65 (`SessionFeatures` Pydantic Model):** This defines the schema. It tells FastAPI exactly which 12 fields it should expect. If Spring Boot sends the wrong data, it will return a `422 Unprocessable Entity` error.
*   **Lines 120-140 (`lifespan` handler):** This code runs **only once** when the server starts. It loads the `scaler.pkl` and `model.pkl` into memory so that every prediction is instant.
*   **Line 165 (`@app.post("/predict")`):** The endpoint Spring Boot calls. It takes raw logs, sends them to `feature_engineering.py`, then to `predict.py`, and returns the JSON result.

---

## 4. `src/ml/predict.py` (The Inference Engine)
**Purpose:** This file takes the features and makes the final Normal/Anomaly decision.

*   **Line 85 (`scaler.transform(row)`):** This is crucial. It "scales" the data. If a user has 1,000 requests, the scaler turns that into a number between 0 and 1 so it doesn't "overwhelm" other features like `error_rate`.
*   **Line 90 (`model.decision_function`):** This asks the Isolation Forest: *"How deep in the tree is this user?"*
*   **Line 95 (Score Normalization):** We use a **Sigmoid-like math formula** to turn the raw tree score into a user-friendly `0.0` to `1.0` score.
    *   `0.0` = Perfect normal user.
    *   `1.0` = Definite hacker.

---

## 5. `src/ml/full_audit_and_test.py` (The Validation Script)
**Purpose:** This is the script you run to prove to your teacher that the model works.

*   **Step 1:** It audits the CSV to make sure all 12 columns are there.
*   **Step 2:** It trains the model on 80% of the data and tests it on the "invisible" 20%.
*   **Step 3:** It generates the **Confusion Matrix** (TN, FP, FN, TP) and the **Classification Report** (Precision, Recall).
*   **Step 4:** It simulates 4 real-world scenarios (Brute Force, Scanning, JS Errors) to show the model catching them in real-time.

---

## 🎓 Summary for your Supervisor
*"Our code is modular. We have a **Feature Engineer** that extracts behavioral signatures, a **Data Distiller** that uses K-Means to ensure diversity, and an **Inference Engine** that uses an Isolation Forest to flag anomalies in milliseconds. Everything is bridged via a **FastAPI** microservice for seamless integration with our Spring Boot backend."*
