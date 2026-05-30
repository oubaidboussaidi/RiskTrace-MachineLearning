# Line-by-Line: dataset_aggregator.py

This file is the "Distiller." It cleans and prepares the massive raw datasets (NASA & ModSecurity) for training.

---

### 1. The Session Tracker (`SessionTracker` class)
This class groups thousands of individual logs into "Sessions" based on the IP address.

```python
def log_request(self, ip, timestamp, method, url, status, is_attack_data):
```
*   **`if ip not in self.active_sessions`**: If this is a new IP, we start a new "folder" for their activity.
*   **`if timestamp - session['last_time'] > SESSION_TIMEOUT`**: If the user was quiet for more than 30 minutes, we close their old session and start a new one.

---

### 2. The Intelligent Sampling Logic (`intelligent_sample_anomalies`)
This is the most advanced part of your code. It uses K-Means to ensure your dataset is high quality.

```python
# 1. Deduplication
attack_unique = attack_df.drop_duplicates(subset=feature_cols)
```
*   **`subset=feature_cols`**: We look at the 12 behavioral features. If two sessions have identical behavior (e.g., both sent 100 requests to `/login`), we delete one. This removes "noise" from the dataset.

```python
# 2. Calculating the Target
n_attack_target = int((target_ratio * n_normal) / (1 - target_ratio))
```
*   This is math to find the 15% ratio. We keep all normal users, and we calculate exactly how many attacks we need to reach a 15%/85% split.

```python
# 3. K-Means Clustering
kmeans = KMeans(n_clusters=n_attack_target, random_state=42)
```
*   **`n_clusters=n_attack_target`**: We tell the AI to find exactly as many "groups" of attacks as we need.
*   **`random_state=42`**: Ensures the results are the same every time we run it (for consistency).

```python
# 4. Assignment
attack_unique['cluster'] = kmeans.fit_predict(attack_unique[feature_cols])
```
*   **`fit_predict`**: The AI "looks" at all attacks and assigns them a Cluster ID (e.g., "This looks like Cluster #5").

```python
# 5. Selection
attack_final = attack_unique.groupby('cluster').first().reset_index(drop=True)
```
*   **`groupby('cluster').first()`**: We take only ONE attack from every cluster.
*   **Result:** You now have a perfectly balanced dataset where every single attack is **unique** and **different** from the others.

---

### 3. The Main Execution (`main`)
```python
process_modsecurity_logs(tracker)
process_nasa_logs(tracker)
tracker.close_all()
```
*   **Phase 1:** Parses the ModSecurity (Attack) files.
*   **Phase 2:** Parses the NASA (Normal) files.
*   **Phase 3:** Triggers the K-Means reduction we explained above.
*   **Phase 4:** Saves the final table to `risk_trace_training_data.csv`.
