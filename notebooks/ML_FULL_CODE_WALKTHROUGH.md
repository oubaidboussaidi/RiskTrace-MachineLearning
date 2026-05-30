# RiskTraceML: The Master Line-by-Line Walkthrough

This document contains a complete, line-by-line technical explanation for every file in the RiskTrace Machine Learning project.

---

## PART 1: `feature_engineering.py` (The Behavioral Brain)
This file is responsible for turning raw HTTP logs into 12 behavioral features.

### 1.1 The Anomaly Detection Regex
```python
ANOMALOUS_PATH_PATTERNS = re.compile(r'(?i)(/admin|\.env|\.git|\.bak|phpmyadmin|union.*select|script>|<script|eval\()')
```
*   **`re.compile`**: Pre-loads the search pattern for maximum speed.
*   **`(?i)`**: Case-insensitive (it catches `/ADMIN` and `/admin`).
*   **`.env`, `.git`**: These are sensitive files hackers look for to steal secrets.
*   **`union.*select`**: This is a signature for SQL Injection.
*   **`<script`, `eval(`**: These are signatures for XSS (Cross-Site Scripting).

### 1.2 Feature Calculation (`aggregate_session_logs`)
*   **`duration = (max(timestamps) - min(timestamps)).total_seconds()`**: Calculates how long the user stayed on the site.
*   **`error_count = sum(1 for log in logs if status >= 400)`**: Counts how many times the server returned an error (4xx/5xx).
*   **`auth_fail = sum(1 for log in logs if status in [401, 403])`**: Specifically counts "Access Denied" events (detects Brute Force).
*   **`unique_endpoints = len(set(urls))`**: Counts how many different pages the user visited. A hacker usually visits many more pages than a human.

---

## PART 2: `dataset_aggregator.py` (The Data Distiller)
This file handles the ingestion of NASA/ModSecurity logs and performs the K-Means clustering.

### 2.1 Intelligent Sampling (`intelligent_sample_anomalies`)
*   **`attack_unique = attack_df.drop_duplicates(...)`**: Removes identical logs so the model isn't biased by redundant data.
*   **`kmeans = KMeans(n_clusters=n_attack_target)`**: Initializes the AI to find 'X' distinct groups of attacks.
*   **`attack_unique['cluster'] = kmeans.fit_predict(...)`**: Assigns every attack session to a specific "Behavioral Cluster."
*   **`attack_final = attack_unique.groupby('cluster').first()`**: Picks one representative from every cluster.
*   **Result:** This ensures your model learns **every** type of attack (diverse) rather than just the most common one.

---

## PART 3: `main.py` (The FastAPI Bridge)
This is the web server that lets the Java backend talk to the Python model.

### 3.1 Pydantic Validation
```python
class SessionFeatures(BaseModel):
    request_count: float
    # ...
```
*   **`BaseModel`**: Automatically validates that the data coming from Java is the correct type (numbers).

### 3.2 Lifespan Management
*   **`ml_models['model'] = joblib.load(...)`**: Loads the trained AI into memory **once** at startup. This makes every prediction instant (sub-millisecond).

---

## PART 4: `predict.py` (The Judge)
This file makes the final decision: Normal or Anomaly.

### 4.1 Scoring Logic
*   **`raw_score = model.decision_function(...)`**: Gets the raw "depth" score from the Isolation Forest.
*   **`score = float(1.0 / (1.0 + np.exp(raw_score)))`**: The **Sigmoid Formula**. This squashes the raw score into a readable 0.0 to 1.0 range.
*   **`prediction = "ANOMALY" if score >= 0.5 else "NORMAL"`**: The final threshold. If the score is above 50%, we sound the alarm.

---

## PART 5: `preprocessing.py` (The Data Cleaner)
This file ensures the data is "clean" before it hits the model.

### 5.1 Scaling
*   **`scaler = StandardScaler()`**: This is critical. It makes sure that a feature with a big range (like `request_count` 0-10,000) doesn't drown out a small feature (like `error_rate` 0-1). It centers everything around zero.

---

## 🎓 Summary for the Defense
*"Our architecture is divided into three layers: **Ingestion** (parsing raw logs), **Engineering** (extracting 12 behavioral features), and **Inference** (predicting anomalies using an Isolation Forest). By using **K-Means clustering** during training, we ensured our model is robust against a diverse range of security threats."*
