# Line-by-Line: feature_engineering.py

This file is the "Brain" of the system. It converts raw text logs into mathematical features.

---

### 1. The Regular Expression (Regex)
```python
ANOMALOUS_PATH_PATTERNS = re.compile(r'(?i)(/admin|\.env|\.git|\.bak|phpmyadmin|union.*select|script>|<script|eval\()')
```
*   **`re.compile`**: Pre-loads the search pattern to make it run faster.
*   **`(?i)`**: Tells Python to ignore uppercase/lowercase (e.g., `/ADMIN` is the same as `/admin`).
*   **`|`**: Means "OR".
*   **`.env`, `.git`**: These are sensitive configuration files hackers look for.
*   **`union.*select`**: This detects SQL Injection attacks.
*   **`<script`, `eval(`**: This detects Cross-Site Scripting (XSS) or code injection.

---

### 2. The Core Aggregator (`aggregate_session_logs`)
This function turns a list of logs into 12 features.

```python
def aggregate_session_logs(logs: List[Dict[str, Any]]) -> Dict[str, float]:
```
*   **`logs: List[Dict]`**: The input is a list of events from Spring Boot.
*   **`if not logs: return ...`**: A safety check. if there are no logs, all features are 0.

#### Step-by-Step Calculation:
```python
# Line 110: timestamps = [parse_iso_time(log.get('timestamp')) for log in logs]
```
*   We extract every timestamp from the logs and turn them into Python `datetime` objects so we can do math with them.

```python
# Line 115: duration = (max(timestamps) - min(timestamps)).total_seconds()
```
*   **`max - min`**: The time between the last request and the first request.
*   **`total_seconds()`**: Converts the time difference into a simple number (seconds).

```python
# Line 120: error_count = sum(1 for log in logs if int(log.get('statusCode', 200)) >= 400)
```
*   We count every request where the status code is 400 or higher (Errors).

```python
# Line 125: auth_fail = sum(1 for log in logs if int(log.get('statusCode', 200)) in [401, 403])
```
*   We specifically count 401 (Unauthorized) and 403 (Forbidden) to detect Brute Force or access violations.

```python
# Line 130: path_anomalies = sum(1 for log in logs if ANOMALOUS_PATH_PATTERNS.search(log.get('url', '')))
```
*   We use the Regex from Step 1 to count how many times the user tried to access a "forbidden" URL.

```python
# Line 135: unique_endpoints = len(set(log.get('url', '') for log in logs))
```
*   **`set()`**: Removes duplicates.
*   **`len()`**: Counts how many *different* pages the user visited.

```python
# Line 140: post_count = sum(1 for log in logs if log.get('method', '').upper() == 'POST')
```
*   Counts how many times the user "sent" data (POST) vs just "viewing" data (GET).

---

### 3. Formatting for Prediction
```python
def format_for_prediction(features: Dict[str, float]) -> Dict[str, float]:
```
*   This ensures the final dictionary matches exactly what the Isolation Forest expects. It fills in any missing values with 0.0 to prevent the model from crashing.
