# RiskTrace Machine Learning (RiskTraceML)

This is the behavioral anomaly detection microservice for the **RiskTrace** platform. It uses an **Isolation Forest** (unsupervised learning) model to identify malicious sessions in real-time by analyzing HTTP and WAF log patterns.

## 🚀 Overview

Unlike traditional rule-based security systems, RiskTraceML focuses on **behavioral signatures**. It learns what "Normal" looks like for a specific application and flags sessions that deviate from that baseline.

### Key Features
*   **Unsupervised Learning:** No manual labeling required; adapts to new application traffic automatically.
*   **Intelligent Dataset Balancing:** Uses K-Means clustering and deduplication to keep the most diverse attack patterns while maintaining a realistic 15% anomaly ratio.
*   **FastAPI Integration:** Real-time inference service designed to work with Spring Boot and Gateway layers.
*   **Multi-App Support:** Can be trained to create unique "Normal" baselines for different client applications.

## 📊 Behavioral Feature Schema

The model is trained on **12 key features** extracted from session-aggregated logs:

| Feature | Description |
| :--- | :--- |
| `request_count` | Total requests in the session window. |
| `error_rate` | Ratio of 4xx and 5xx status codes. |
| `auth_failure_count`| Frequency of 401/403 (unauthorized/forbidden) responses. |
| `avg_response_time` | Average server response latency. |
| `unique_endpoints` | Diversity of URLs accessed. |
| `anomalous_paths` | Probes to sensitive files (e.g., `.env`, `/admin`, `.git`). |
| `request_rate` | Frequency of requests per second (DDoS detection). |
| `js_error_count` | Frontend errors reported by the tracker. |

## 🛠️ Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Pipeline Workflow

### 1. Data Aggregation
Converts raw logs from sources like NASA (normal) and ModSecurity (WAF attacks) into session-level feature vectors.
```bash
python src/ml/dataset_aggregator.py
```

### 2. Intelligent Reduction
Automatically deduplicates and clusters redundant attack data to ensure the Isolation Forest sees a diverse set of outliers.

### 3. Training & Validation
Trains the model and outputs performance metrics (Precision, Recall, F1).
```bash
python src/ml/train.py
```

### 4. REST API (Inference)
Starts the FastAPI service for real-time detection.
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

---

## 🏆 Model Performance (Latest Audit: K-Means Intelligent Reduction)
*   **Accuracy:** 92%
*   **Attack F1-Score:** 74%
*   **Attack Recall (Detection):** 75% (Catches more diverse threats)
*   **Security Strength:** 14% reduction in missed attacks (False Negatives).
*   **Algorithm:** Isolation Forest (150 Estimators, 0.15 Contamination)

---

## 🎓 Presentation Guide
For a full summary of the project architecture and how to explain it to a supervisor, see:
*   **[README_FINAL_SUMMARY.md](./README_FINAL_SUMMARY.md)**: High-level overview and results.
*   **[ML_ALGO_DEEP_DIVE.md](./ML_ALGO_DEEP_DIVE.md)**: Technical breakdown of K-Means and Isolation Forest logic.
*   **[ML_PROJECT_CODE_WALKTHROUGH.md](./ML_PROJECT_CODE_WALKTHROUGH.md)**: Line-by-line explanation of the actual code files.
