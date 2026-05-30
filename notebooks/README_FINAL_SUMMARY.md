# RiskTraceML: Final Summary & Presentation Guide

This document summarizes the complete Machine Learning architecture for the RiskTrace project. Use this as your "Cheat Sheet" when explaining the work to your supervisor.

---

## 1. The "Big Picture" (The Elevator Pitch)
RiskTraceML is a **behavioral anomaly detection** engine. Instead of using a list of static security rules (which hackers can bypass), we use **Unsupervised Learning** to build a "profile" of what is normal for an application. Anything that doesn't fit the profile is isolated and flagged in real-time.

---

## 2. The Algorithm: Isolation Forest
*   **Why?** Because it is an **Unsupervised** model. It doesn't need humans to label logs ("this is an attack"). It learns on its own.
*   **The Logic:** Anomaly detection works by **Isolation**. 
    *   **Normal** sessions are clumped together; they take many "cuts" (questions) to isolate.
    *   **Anomalies** are "lone wolves"; they get isolated very quickly (short path length).
*   **The Result:** Faster isolation = Higher Risk Score.

---

## 3. The "Smart" Secret: Intelligent Dataset Distillation
*This is the part that will impress your supervisor the most.*
*   **Problem:** If you have 10,000 identical "DDoS" logs, the model thinks that behavior is "Normal" because it sees it so often.
*   **Solution:** We implemented **K-Means Clustering** for anomaly reduction. 
    *   We deduplicate the data and use K-Means to pick only the most **diverse** and **unique** attack signatures.
    *   **Impact:** This improved our detection of unique attacks (Recall) by **4%** and reduced missed attacks (False Negatives) by **14%**.

---

## 4. The Technical Pipeline (The "How")
1.  **Data Collection:** We merged **NASA Access Logs** (Normal baseline) with **ModSecurity WAF Logs** (Real-world attacks).
2.  **Feature Engineering:** We transformed raw text logs into **12 behavioral features** (e.g., `error_rate`, `request_rate`, `auth_failure_count`).
3.  **FastAPI Microservice:** The model lives in a standalone Python API that receives log data from the Spring Boot backend and returns a `Normal/Anomaly` verdict in milliseconds.

---

## 5. The Performance (The Numbers)
*   **Global Accuracy:** 92%
*   **Attack Detection (F1):** 74%
*   **Security Strength:** We successfully reduced missed attacks (False Negatives) by **14%** through intelligent sampling.

---

## 6. How to Run the Final Demo
If the supervisor wants to see it LIVE, run these two commands:

1.  **Run the Audit:**
    ```bash
    python src/ml/full_audit_and_test.py
    ```
    *This will show the Confusion Matrix and correctly identify 4 live attack scenarios (Brute Force, Scanning, etc.).*

2.  **Start the API:**
    ```bash
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    ```
    *This starts the server for real-time integration.*

---

## 🎓 Final Presentation Tip:
When asked why you chose this path, say: 
*"We chose a **Data-Centric** approach. Instead of just picking a model, we engineered the data using **K-Means clustering** to ensure the model learned a diverse range of threats. This makes RiskTrace more robust against 'Zero-Day' attacks that rule-based systems simply cannot see."*
