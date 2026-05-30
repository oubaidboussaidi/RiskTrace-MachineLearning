# Technical Deep Dive: K-Means & Isolation Forest in RiskTrace

This document provides a line-by-line technical explanation of the machine learning logic used in the RiskTraceML engine.

---

## 1. K-Means: Intelligent Anomaly Reduction
We use K-Means during the **Data Aggregation** phase to ensure our training set is diverse and not filled with 10,000 identical "DDoS" logs.

### The Logic (Intuition)
K-Means groups data points that are "close" to each other in the 12-dimensional feature space. 
*   If we have 5,000 Brute Force attacks, they will all form one "Cluster."
*   If we have 1 SQL Injection attack, it will be its own "Cluster."
*   By picking one representative from each cluster, we ensure the model learns **every type** of attack, rather than just the most frequent one.

### Line-by-Line Code Explanation (`dataset_aggregator.py`)

```python
# 1. We remove exact duplicates first to save memory
attack_unique = attack_df.drop_duplicates(subset=feature_cols)

# 2. We initialize the K-Means algorithm
# n_clusters = our 15% target (e.g., 9442 clusters)
kmeans = KMeans(n_clusters=n_attack_target, random_state=42, n_init='auto')

# 3. The 'fit_predict' step: 
# This calculates the distance between every attack and assigns it a Cluster ID
attack_unique['cluster'] = kmeans.fit_predict(attack_unique[feature_cols])

# 4. The 'Selection' step:
# We group by the Cluster ID and pick the 'first' session from each group.
# This results in exactly 9,442 DIVERSE attack patterns.
attack_final = attack_unique.groupby('cluster').first().reset_index(drop=True)
```

---

## 2. Isolation Forest: The "Tree" Logic
The Isolation Forest is the "detective" that runs during **Real-Time Inference**.

### The Logic (Intuition)
Most models try to learn what is "Normal." Isolation Forest does the opposite: it tries to **isolate** outliers.
*   **Normal points** are in the middle of a crowd. You need many "questions" (splits) to isolate them.
*   **Anomalies** are far away. You only need a few "questions" to isolate them.
*   **Short Path = Anomaly.**

### The "Binary Questions" (Tree Splits)
Each tree in our forest (we use 150 trees) asks random questions about your 12 features. Here is a conceptual "Schema" of a single path:

1.  **Question 1:** Is `error_rate` > 0.4?
    *   *No* $\rightarrow$ (Go deeper, probably normal)
    *   *Yes* $\rightarrow$ **Next Question**
2.  **Question 2:** Is `anomalous_path_count` > 2?
    *   *No* $\rightarrow$ (Go deeper)
    *   *Yes* $\rightarrow$ **Next Question**
3.  **Question 3:** Is `request_rate` > 50 req/sec?
    *   *Yes* $\rightarrow$ **ISOLATED!** (Path Length = 3). 

**Verdict:** Since it only took 3 questions to isolate this user, the **Anomaly Score** is very high (e.g., 0.85). A normal user would require 15-20 questions to be isolated.

---

## 3. The 12-Feature Behavioral Schema
These are the "answers" the model looks for in every session:

| Feature | Type | Logic for the Teacher |
| :--- | :--- | :--- |
| `request_count` | Count | Detects high-volume activity. |
| `error_rate` | Ratio | High ratio (0.8+) signals scanners or broken exploits. |
| `auth_failure_count`| Count | High count (10+) signals Brute Force attempts. |
| `avg_response_time` | float | Detects DoS attacks (server slowing down). |
| `unique_endpoints` | Count | Detects "Directory Traversals" or broad crawlers. |
| `anomalous_path_count`| Count | Direct hits on sensitive paths like `/.env` or `/admin`. |
| `post_ratio` | Ratio | Attacks often use POST for data exfiltration. |
| `request_rate` | float | Distinguishes between a fast human and a script. |
| `js_error_count` | Count | High frontend errors signal a browser-based exploit. |

---

## 🎓 Summary for the Presentation
*"By combining **K-Means** for high-quality data training and **Isolation Forest** for real-time isolation, we created a system that doesn't just look for 'bad' traffic—it looks for 'unique' traffic. This allows RiskTrace to detect 'Zero-Day' attacks that have never been seen before, simply because their behavioral footprint is too isolated from the norm."*
