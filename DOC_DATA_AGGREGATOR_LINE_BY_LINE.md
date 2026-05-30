# Technical Code Guide: dataset_aggregator.py

This guide combines the actual Python code blocks with their technical explanations.

---

## 1. Security Patterns & Regex
This code defines the "filters" used to identify suspicious activity in raw logs.

```python
# Line 29: Anomalous paths common in WAF hits
ANOMALOUS_PATTERNS = re.compile(r'(?i)(/admin|\.env|\.git|\.bak|phpmyadmin|union.*select|script>|<script|eval\()')
```
*   **What it does**: This is a Regular Expression (Regex). It looks for signatures of attacks like **SQL Injection** (`union select`), **Cross-Site Scripting** (`<script`), and attempts to access sensitive files (`.env`, `.git`).
*   **Why it's important**: It allows the aggregator to count "Anomalous Path Hits" as a feature for the model.

---

## 2. Session Aggregation Logic
This is where we calculate the 12 features for every user.

```python
# Lines 115-129: Feature Calculation
self.sessions.append({
    'request_count': s['request_count'],
    'error_rate': round(s['error_count'] / req_count, 3),
    'auth_failure_count': s['auth_failure_count'],
    'avg_response_time_ms': round(float(np.mean(rts)), 2) if rts else 0.0,
    'p95_response_time_ms': round(float(np.percentile(rts, 95)), 2) if rts else 0.0,
    'unique_endpoints': len(s['urls']),
    'unique_ips': 1,
    'anomalous_path_count': s['anomalous_path_count'],
    'post_ratio': round(s['post_count'] / req_count, 3),
    'js_error_count': 0,
    'request_rate': round(s['request_count'] / duration_s, 3),
    'session_duration_s': duration_s,
    'is_anomaly': s['is_attack_data']
})
```
*   **`error_rate`**: Calculates the percentage of requests that failed (status >= 400).
*   **`avg_response_time_ms`**: Uses `np.mean` to find the average latency.
*   **`request_rate`**: Calculates requests-per-second (`count / duration`). This is key for detecting DDoS or automated scripts.
*   **`is_anomaly`**: This is our "Ground Truth" label (0 for Normal, 1 for Attack).

---

## 3. Intelligent Anomaly Reduction (K-Means)
This is the advanced part where we clean the dataset.

```python
# Lines 284-296: Clustering Logic
attack_unique = attack_df.drop_duplicates(subset=feature_cols)
n_attack_target = int((target_ratio * n_normal) / (1 - target_ratio))

if len(attack_unique) > n_attack_target and n_attack_target > 0:
    kmeans = KMeans(n_clusters=n_attack_target, random_state=42, n_init='auto')
    attack_unique = attack_unique.copy()
    attack_unique['cluster'] = kmeans.fit_predict(attack_unique[feature_cols])
    
    # Keep one representative from each cluster
    attack_final = attack_unique.groupby('cluster').first().reset_index(drop=True)
```
*   **`drop_duplicates`**: Removes identical logs to prevent redundant data.
*   **`KMeans(n_clusters=...)`**: Creates "Behavioral Groups." If we have 1,000 Brute Force attacks, they all go into one group.
*   **`groupby('cluster').first()`**: This picks **one** unique example from each group. 
*   **Result**: We get a diverse dataset with many different types of attacks, instead of 10,000 identical ones.

---

## 4. Final Data Flow (Main)
```python
# Lines 323-331
process_modsecurity_logs(tracker)
process_nasa_logs(tracker)
tracker.close_all()

# Lines 347-355
df_balanced = intelligent_sample_anomalies(df, target_ratio=0.15)
df_balanced.to_csv(OUTPUT_CSV, index=False)
```
*   **`process_modsecurity_logs`**: Ingests the raw attack datasets.
*   **`process_nasa_logs`**: Ingests the normal traffic baseline.
*   **`intelligent_sample_anomalies`**: Triggers the K-Means reduction we explained in Step 3.
*   **`to_csv`**: Saves the final, balanced dataset to the file `risk_trace_training_data.csv`.
