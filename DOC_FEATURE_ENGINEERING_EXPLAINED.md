# Technical Code Guide: feature_engineering.py

This file is the "Brain" of the real-time system. It extracts behavioral patterns from raw logs.

---

## 1. Security Pattern Detection
```python
# Lines 29-30: The Anomaly Regex
ANOMALOUS_PATH_PATTERNS = re.compile(
    r'(?i)(/admin|\.env|\.git|\.bak|phpmyadmin|union.*select|script>|<script|eval\()'
)
```
*   **The Code**: This pre-compiled regular expression scans every URL.
*   **Logic**: 
    *   `\.env` / `\.git`: Detects hackers trying to steal configuration secrets.
    *   `union.*select`: Detects **SQL Injection** attempts.
    *   `<script`: Detects **XSS (Cross-Site Scripting)** attempts.

---

## 2. Session Aggregation Logic
```python
# Lines 115-135: Extracting time and counters
timestamps = [parse_iso_time(log.get('timestamp')) for log in logs]
duration_s = max(1, (max(timestamps) - min(timestamps)).total_seconds())

error_count = sum(1 for log in logs if int(log.get('statusCode', 200)) >= 400)
auth_failure_count = sum(1 for log in logs if int(log.get('statusCode', 200)) in [401, 403])
```
*   **`duration_s`**: Calculates the session length. If a user does 100 things in 1 second, they are likely a script (bot).
*   **`error_count`**: High error rates usually indicate a scanner trying to find broken pages.
*   **`auth_failure_count`**: This is the primary signal for **Brute Force** attacks.

---

## 3. Response Time Analytics
```python
# Lines 140-150: Performance Statistics
rts = [float(log.get('responseTime', 0)) for log in logs]
avg_rt = float(np.mean(rts)) if rts else 0.0
p95_rt = float(np.percentile(rts, 95)) if rts else 0.0
```
*   **`np.mean`**: The average speed of the server for this user.
*   **`np.percentile(..., 95)`**: The "P95" value. It tells us how slow the slowest requests were.
*   **Why?**: During a **DoS (Denial of Service)** attack, the server slows down significantly. This feature catches that delay.

---

## 4. Feature Formatting
```python
# Lines 210-230: Final Dictionary Construction
return {
    "request_count": float(len(logs)),
    "error_rate": float(error_count / len(logs)),
    "auth_failure_count": float(auth_failure_count),
    "avg_response_time_ms": avg_rt,
    "unique_endpoints": float(len(set(log.get('url', '') for log in logs))),
    # ... (other features)
}
```
*   **What it does**: This organizes all our calculations into a clean dictionary that the Machine Learning model can understand. Every value is converted to a `float` (decimal) for mathematical consistency.
