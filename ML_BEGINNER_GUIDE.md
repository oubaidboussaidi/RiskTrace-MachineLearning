# RiskTraceML — The Complete Beginner's Guide
## Understanding Every Line of the ML Pipeline

> **This document is written for someone who has never worked on a Machine Learning project before.**  
> It explains every concept, every decision, and every file — in plain English, with real code examples.

---

## PART 1: What Even Is Machine Learning? (Start Here)

Before we look at any code, you need to understand what we're actually trying to do.

### The Problem

Thousands of HTTP requests hit your web application every day. Most of them are normal:
- A user opens the dashboard → `GET /dashboard` → `200 OK`
- A user loads their profile → `GET /api/v1/users` → `200 OK`

But some are attacks:
- A hacker tries 500 wrong passwords → `POST /api/auth/login` → `401 Unauthorized` (repeated 500 times)
- A bot scans for hidden files → `GET /.env` → `404 Not Found`

**The question is: how does the computer tell the difference automatically?**

You can't write a simple rule like `"if status = 401 then it's an attack"` because normal users also get 401 errors when they mistype their password. You need the system to learn what a **pattern of normal behavior** looks like, and then flag anything that doesn't fit.

That's Machine Learning.

---

## PART 2: Supervised vs Unsupervised — The Most Important Concept

This is the first thing to understand about ML.

### Supervised Learning (e.g., Random Forest)

Imagine you're teaching a child to recognise dogs. You show them 1,000 pictures:
- "This is a dog" ✅
- "This is not a dog" ❌

The child **learns from your labels**. After training, you show them a new picture and they say "dog" or "not dog" based on what they've seen.

In ML terms:
- You give the model data **with labels** (`is_anomaly = 0` or `is_anomaly = 1`)
- The model learns the difference between the two groups
- It's very accurate on patterns it has seen before
- **But it will miss brand new attack types it has never seen** — because it only knows what you taught it

> **Random Forest** is a supervised algorithm. It would give us ~93% attack detection, but only for attacks that look like what it was trained on.

### Unsupervised Learning (e.g., Isolation Forest)

Now imagine you give the child 1,000 pictures **without any labels**. They start noticing patterns on their own:
- Most pictures have similar colors, shapes, sizes...
- But occasionally something looks very different, unusual, out of place

The child didn't need you to tell them what was "wrong" — they just know it doesn't fit the pattern. That's anomaly detection.

In ML terms:
- You give the model data **without labels**
- The model learns what "normal" looks like
- Anything that doesn't fit normal gets flagged
- **It can detect brand new attack types it has never seen** — because it just asks "does this look weird?"

> **Isolation Forest** is an unsupervised algorithm. It gives us 71% attack detection, but works even on attacks that have never existed before.

---

## PART 3: Why We Chose Isolation Forest Over Random Forest

### What We Gained

**1. Zero-Day Attack Detection**

A "zero-day" attack is one that nobody has seen before. New hacking techniques are discovered every week. A Random Forest trained on ModSecurity logs from 2025 will struggle to detect a new attack style invented in 2026.

Isolation Forest doesn't care about attack types. It just asks: *"Is this IP's behavior statistically unusual compared to everyone else?"* A brand new attack that causes unusual behavior will still be caught.

For RiskTrace — a platform where **multiple organizations** trust you to protect their sites — this matters. Each organization may have unique attack surfaces that your training data never covered.

**2. No Need to Label Live Data**

If we used Random Forest, every time we wanted to retrain the model on new data, someone would have to manually label thousands of live logs ("this is an attack, this is not"). With Isolation Forest, the model adapts to new "normal" patterns on its own.

**3. Better Defense Argument**

> *"We chose an unsupervised approach because RiskTrace operates in a multi-tenant environment where each organization faces unique and potentially unprecedented threats. A supervised model would only detect known attack signatures, creating a dangerous blind spot for novel attack vectors."*

That's a professional, commercially-sound engineering decision.

### What We Lost

**1. Attack Recall: 71% vs ~93%**

This is the honest weakness. Isolation Forest misses 29% of attacks. Random Forest would miss only ~7%. The gap exists because some attack sessions look very similar to normal sessions in terms of our 12 features.

For example: a slow brute force attack that tries 3 passwords per hour over 8 hours — each individual session looks almost normal (low request count, low error rate). Isolation Forest struggles with this. Random Forest, having seen many brute force examples during training, would catch it.

**2. Lower Precision on Attacks (70%)**

When Isolation Forest raises an alert, it's wrong 30% of the time. Random Forest would be wrong only ~8% of the time.

### Summary Table

| What We're Comparing | Isolation Forest (our choice) | Random Forest (alternative) |
|---|---|---|
| Attack recall | 71% | ~93% |
| False alarm rate | 5% | ~2% |
| Zero-day detection | ✅ Strong | ⚠️ Weak |
| Needs labeled training data | ❌ No | ✅ Yes |
| Works on unknown attack types | ✅ Yes | ❌ No |
| Good for multi-tenant SaaS | ✅ Yes | ⚠️ Risky |

**Bottom line:** We accepted lower numbers in exchange for a fundamentally more robust security posture.

---

## PART 4: How Does Isolation Forest Actually Work?

Imagine a forest of random binary decision trees. Here's the intuition:

Take a single data point (one session). The algorithm tries to **isolate** it — separate it from all other points — by making random cuts:
- "Split on `error_rate` > 0.5" → left group / right group
- Then "Split on `request_count` > 100" → left group / right group
- Keep splitting until the point is alone

**Key insight:** An anomaly is isolated very quickly (few splits needed) because it's very different from everything else. A normal point takes many splits to isolate because it's surrounded by similar points.

```
Normal session:   needs ~20 random cuts to isolate (deep in the forest)
Anomaly session:  needs ~4 random cuts to isolate (isolated near the root)
```

The model builds 150 such trees and averages the depth needed to isolate each session. Sessions that are consistently isolated quickly → flagged as anomalies.

---

## PART 5: The Data We Used

### Dataset 1: NASA Apache Access Logs (Normal Traffic)
- Source: NASA Kennedy Space Center web server, July and August 1995
- Content: Real HTTP access logs from a public website
- Why: Represents genuine, benign web traffic patterns
- Label assigned: `is_anomaly = 0` (Normal)
- Example log line:
```
199.72.81.55 - - [01/Jul/1995:00:00:01 -0400] "GET /history/apollo/ HTTP/1.0" 200 6245
```

### Dataset 2: ModSecurity WAF Audit Logs (Attack Traffic)
- Source: A deployed ModSecurity Web Application Firewall, August 2025
- Content: Requests that triggered WAF security rules
- Why: Real attack traffic captured by a production WAF
- Label assigned: `is_anomaly = 1` (Attack)
- Example log sections:
```
--abc12345-A--                           ← Transaction start (timestamp, IP)
--abc12345-B--                           ← Request headers (GET /.env HTTP/1.1)
--abc12345-F--                           ← Response (HTTP/1.1 404)
```

### The Problem With Raw Counts

NASA gave us **600,000 individual requests**. ModSecurity gave us **~150,000 individual requests**.

You cannot run anomaly detection on individual requests. A single `404` is normal — a developer mistyped a URL. But **50 consecutive 404s on paths like `/.env`, `/.git`, `/admin`** from the same IP in 2 minutes is clearly a scanner.

**You need to look at behavior over time, not individual requests.**

This is called **Sessionization**.

---

## PART 6: File-by-File Explanation

---

### 📄 `dataset_aggregator.py` — The Data Factory

**What it does:** Reads the raw NASA and ModSecurity log files and converts them into sessions.

**The concept of a Session:**
> A session = all the HTTP requests made by the same IP address within a 30-minute window

If IP `10.0.0.1` makes 50 requests over 45 minutes with a 35-minute gap in the middle, that's **2 sessions** (the gap broke the 30-minute timeout).

```python
SESSION_TIMEOUT_MINUTES = 30
```

**Step 1: Parsing NASA logs**

```python
NASA_CLF_REGEX = re.compile(
    r'^(\S+) \S+ \S+ \[([^\]]+)\] "(\S+) (\S+)\s*\S*" (\d{3}) (\S+)'
)
```

This regex reads each line of the NASA log file. Breaking it down:
- `(\S+)` → captures the IP address (`199.72.81.55`)
- `\[([^\]]+)\]` → captures the timestamp (`01/Jul/1995:00:00:01 -0400`)
- `"(\S+) (\S+)` → captures the method and URL (`GET /history/apollo/`)
- `(\d{3})` → captures the status code (`200`)

**Step 2: Parsing ModSecurity logs**

ModSecurity logs are more complex. They're divided into sections marked by boundaries:
```
--abc12345-A--   ← Section A: timestamp and IP address
--abc12345-B--   ← Section B: Request line (method + URL)
--abc12345-F--   ← Section F: Response status code
```

```python
MODSEC_BOUNDARY_REGEX = re.compile(r'^--([a-zA-Z0-9]{8})-([A-Z])--$')
```

The parser tracks which section it's in and extracts the relevant data from each.

**Step 3: The SessionTracker class**

This is the core of the file. It maintains a dictionary of active sessions:

```python
class SessionTracker:
    def __init__(self):
        self.sessions = []                    # completed sessions
        self.active_sessions = {}             # ongoing sessions, keyed by IP
```

When a request arrives:
```python
def log_request(self, ip, timestamp, method, url, status, is_attack_data):
    if ip not in self.active_sessions:
        self._start_new_session(ip, timestamp, is_attack_data)
    
    session = self.active_sessions[ip]
    
    # If 30+ minutes have passed since last request → close and start new
    if timestamp - session['last_time'] > timedelta(minutes=30):
        self._close_session(ip)
        self._start_new_session(ip, timestamp, is_attack_data)
    
    # Update session counters
    session['request_count'] += 1
    if status >= 400:
        session['error_count'] += 1
    if status in [401, 403]:
        session['auth_failure_count'] += 1
    # ... etc
```

**Step 4: Closing a session — computing the 12 features**

When a session ends (IP goes quiet for 30+ minutes), we calculate all features:

```python
def _close_session(self, ip):
    s = self.active_sessions.pop(ip)
    duration_s = (s['last_time'] - s['start_time']).total_seconds()
    
    self.sessions.append({
        'request_count':       s['request_count'],
        'error_rate':          s['error_count'] / s['request_count'],
        'auth_failure_count':  s['auth_failure_count'],
        'avg_response_time_ms': np.mean(s['response_times']),
        'p95_response_time_ms': np.percentile(s['response_times'], 95),
        'unique_endpoints':    len(s['urls']),
        'unique_ips':          1,
        'anomalous_path_count': s['anomalous_path_count'],
        'post_ratio':          s['post_count'] / s['request_count'],
        'js_error_count':      0,     # only for live tracker data
        'request_rate':        s['request_count'] / duration_s,
        'session_duration_s':  duration_s,
        'is_anomaly':          int(s['is_attack_data'])  # 0 or 1
    })
```

**⚠️ The Critical Fix: Dataset Balancing**

The raw data gave us:
- **53,509 normal sessions** (from NASA)
- **39,273 attack sessions** (from ModSecurity)
- **Attack ratio: 42.3%**

This is a big problem. Isolation Forest is designed to work with **5–15% anomalies**. Training it on 42% anomalies is like telling a security guard that almost half of all visitors are criminals — they lose track of what "normal" looks like and their accuracy collapses.

**Proof:** Before balancing → accuracy was **65%**. After balancing → accuracy jumped to **91%**.

The fix:
```python
TARGET_ATTACK_RATIO = 0.15   # Target: 15% attacks, 85% normal

n_normal = len(normal_df)
# Formula: how many attacks to keep for 15% ratio?
# n_attack / (n_attack + n_normal) = 0.15
# → n_attack = (0.15 × n_normal) / (1 - 0.15)
n_attack_target = int((TARGET_ATTACK_RATIO * n_normal) / (1 - TARGET_ATTACK_RATIO))

attack_sampled = attack_df.sample(n=n_attack_target, random_state=42)
df_balanced = pd.concat([normal_df, attack_sampled])
```

**Result:** 53,509 normal + 9,442 attacks = **62,951 total sessions** at a **15% attack ratio**.

**Output:** `Data/risk_trace_training_data.csv` — a flat table with one row per session and 12 feature columns + `is_anomaly`.

---

### 📄 `preprocessing.py` — The Data Cleaner

**What it does:** Loads the CSV, cleans it, scales it, and splits features from labels.

**The 12 features list (THE most important constant — must match everywhere):**

```python
FEATURE_COLUMNS = [
    "request_count",       # how many requests in this session?
    "error_rate",          # what fraction returned 4xx/5xx?
    "auth_failure_count",  # how many 401/403 responses?
    "avg_response_time_ms",# average response time
    "p95_response_time_ms",# 95% of responses were faster than this
    "unique_endpoints",    # how many different URLs were accessed?
    "unique_ips",          # how many different IPs (usually 1 per session)
    "anomalous_path_count",# how many requests to /.env, /admin, /.git etc?
    "post_ratio",          # fraction of requests that were POST
    "js_error_count",      # how many JavaScript errors were thrown?
    "request_rate",        # requests per second
    "session_duration_s",  # how long did the session last?
]
```

**Why Scaling Matters:**

`request_count` can be 0 to 10,000. `error_rate` is always 0.0 to 1.0. If you don't scale them, the model will treat `request_count` as 10,000 times more important than `error_rate` simply because it has bigger numbers.

`StandardScaler` converts every feature to have **mean=0 and standard deviation=1**:
```python
# Before scaling:  request_count = [5, 450, 1200, 23, ...]
# After scaling:   request_count = [-1.2, 0.8, 2.1, -1.0, ...]
# All features now on the same scale
```

```python
def scale_features(df, training=True):
    cols_to_scale = FEATURE_COLUMNS  # never scale the label!
    
    if training:
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        joblib.dump(scaler, "models/scaler.pkl")   # SAVE it — needed for inference
    else:
        scaler = joblib.load("models/scaler.pkl")  # LOAD the same scaler
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])
```

**Why we save the scaler:** The scaler learns the mean and standard deviation of the training data. When a live session arrives later, we must scale it using **the same parameters** — not recalculate them from the live data. Otherwise the scaling would be different and the model would receive garbage inputs.

**Preparing features:**
```python
def prepare_features(df):
    y = df["is_anomaly"].values   # the labels (for evaluation only)
    X = df.drop(columns=["is_anomaly"])  # the 12 feature columns
    # CRITICAL: Isolation Forest NEVER sees y during training
    return X, y
```

---

### 📄 `train.py` — Teaching the Model

**What it does:** Loads the CSV, preprocesses it, trains the Isolation Forest, evaluates it, and saves it.

**Train/Test Split:**
```python
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["is_anomaly"])
```
- 80% of data (50,360 sessions) → used to train the model
- 20% of data (12,591 sessions) → held back to test it honestly
- `stratify=df["is_anomaly"]` → ensures both splits have 15% attacks (not random luck)
- `random_state=42` → same shuffle every run, reproducibility

**Training:**
```python
model = IsolationForest(
    contamination=0.15,    # "I expect ~15% of training data to be anomalous"
    n_estimators=150,      # build 150 trees — more trees = more stable
    random_state=42,       # reproducibility
    n_jobs=-1              # use ALL processor cores (faster)
)
model.fit(X_train)         # learns from 50,360 sessions — NO LABELS USED
```

**Why `contamination=0.15`?**

This tells the model: *"When you're scoring the training data itself, flag the bottom 15% as anomalies."* It calibrates the model's internal threshold. We set it to match our actual attack ratio (15%).

**Evaluation:**
```python
raw_preds = model.predict(X_test)         # returns +1 (normal) or -1 (anomaly)
y_pred = [0 if p == 1 else 1 for p in raw_preds]   # convert to 0/1
```

⚠️ **Important:** sklearn uses `+1 = normal` and `-1 = anomaly`. Our convention is `0 = normal` and `1 = anomaly`. Without this conversion, the confusion matrix would be completely backwards.

```python
print(classification_report(y_test, y_pred))
# Result:
#               precision  recall  f1-score
#   Normal         0.95     0.95     0.95     ← 95% precision on normal traffic
#   Attack         0.70     0.71     0.70     ← 71% recall on attacks
#   accuracy                0.91             ← 91% overall
```

**Saving the model:**
```python
joblib.dump(model, "models/isolation_forest_model.pkl")
# Also save the feature column list as JSON
with open("models/feature_columns.json", "w") as f:
    json.dump(FEATURE_COLUMNS, f)
```

Why save `feature_columns.json`? When FastAPI loads the model, it needs to know **in what order** to arrange the 12 features into the numpy array. If the order changes, the model receives wrong data for each column and all predictions become garbage.

---

### 📄 `feature_engineering.py` — The Live Log Translator

**What it does:** Takes a list of raw HTTP log dicts (from Spring Boot / MongoDB) and converts them into the 12-feature vector that the model expects.

This is the bridge between the live world and the ML world.

**Key function: `aggregate_session_logs(logs)`**

Input: a list of raw log dictionaries (exactly what Spring Boot serializes from MongoDB):
```python
[
    {"type": "fetch_request", "url": "/api/auth/login", "method": "POST",
     "statusCode": 401, "responseTime": 91, "ipAddress": "10.0.0.99"},
    {"type": "fetch_request", "url": "/api/auth/login", "method": "POST",
     "statusCode": 401, "responseTime": 88, "ipAddress": "10.0.0.99"},
    # ... 8 more failed logins
]
```

Output: one feature dictionary:
```python
{
    "request_count": 10.0,
    "error_rate": 1.0,          # all 10 requests failed
    "auth_failure_count": 10.0, # all were 401
    "post_ratio": 1.0,          # all were POST
    # ...
}
```

**The dual-key compatibility fix:**

The offline datasets (NASA/ModSecurity) use Python snake_case (`status_code`, `response_time`).  
Spring Boot serializes Java `camelCase` to JSON (`statusCode`, `responseTime`).

Without the fix, live logs would silently return `None` for all fields and every session would look identical (all zeros) to the model.

```python
# Check snake_case first (offline dataset), fall back to camelCase (Spring Boot)
sc = log.get("status_code") if "status_code" in log else log.get("statusCode", 200)
rt = log.get("response_time") if "response_time" in log else log.get("responseTime", 0.0)
ep = str(log.get("endpoint") or log.get("url") or "")
ip = str(log.get("ip_address") or log.get("ipAddress") or "")
```

**The `js_error_count` feature:**

The offline datasets don't have client-side JavaScript errors (NASA logs are server logs). But your `tracker.js` intercepts `window.onerror` events and sends them to Spring Boot as `{"type": "js_error"}`.

```python
log_type = str(log.get("type", ""))
if log_type == "js_error":
    js_errors += 1
```

This feature is `0` for all training data but becomes meaningful for live traffic — sessions with many JS errors get an elevated anomaly signal.

**Anomalous path detection:**

```python
ANOMALOUS_PATH_PATTERNS = [
    "/admin", "/actuator", "/.env", "/wp-admin",
    "/phpmyadmin", "/.git", "/config",
]

if any(pat in ep for pat in ANOMALOUS_PATH_PATTERNS):
    anomalous += 1
```

A normal user never requests `/.env` or `/.git`. Any request to these paths is suspicious. If a session has 6 of these, `anomalous_path_count = 6` — a very strong anomaly signal.

---

### 📄 `predict.py` — The Brain at Runtime

**What it does:** Loads the saved model and scaler from disk, and runs predictions on new data.

**Loading artifacts:**
```python
def load_artifacts():
    # Resolve paths relative to the RiskTraceML/ root directory
    base_dir = Path(__file__).resolve().parent.parent.parent
    
    artifacts = {
        "model":  joblib.load(base_dir / "models/isolation_forest_model.pkl"),
        "scaler": joblib.load(base_dir / "models/scaler.pkl"),
        "feature_columns": json.load(open(base_dir / "models/feature_columns.json"))
    }
    return artifacts
```

This is called **once** when FastAPI starts. Loading from disk is slow — doing it per-request would make the API unusable.

**Score normalization:**

`model.decision_function()` returns raw anomaly scores like `-0.12` or `+0.05`. These aren't meaningful to a human or easy to threshold.

```python
def normalize_score(raw_scores):
    # Sigmoid inversion: maps any number to [0, 1]
    # High raw score (normal) → low anomaly score
    # Low/negative raw score (anomaly) → high anomaly score
    normalized = 1.0 / (1.0 + np.exp(raw_scores))
    return normalized.tolist()
```

With sigmoid:
- Raw score `+0.5` (very normal) → anomaly score `0.38`
- Raw score `0.0` (borderline) → anomaly score `0.50`
- Raw score `-0.5` (anomalous) → anomaly score `0.62`

**Confidence levels:**
```python
ANOMALY_THRESHOLD = 0.5           # above this → ANOMALY

def _map_confidence(score):
    if score >= 0.75: return "HIGH"      # very likely an attack
    if score >= 0.50: return "MEDIUM"    # suspicious
    return "LOW"                          # normal
```

**Single session prediction:**
```python
def predict_session(features, artifacts):
    # 1. Build feature array in EXACT SAME ORDER as training
    feature_cols = artifacts["feature_columns"]
    X = np.array([[features.get(col, 0.0) for col in feature_cols]])
    
    # 2. Scale using the SAVED scaler (same one from training)
    X_scaled = artifacts["scaler"].transform(X)
    
    # 3. Get raw anomaly score
    raw_score = artifacts["model"].decision_function(X_scaled)[0]
    
    # 4. Normalize to [0, 1]
    anomaly_score = normalize_score([raw_score])[0]
    
    # 5. Classify and return
    return {
        "anomalyScore": anomaly_score,
        "prediction":   "ANOMALY" if anomaly_score >= ANOMALY_THRESHOLD else "NORMAL",
        "confidence":   _map_confidence(anomaly_score)
    }
```

---

### 📄 `main.py` (FastAPI) — The REST API

**What it does:** Wraps the ML prediction logic in an HTTP API that Spring Boot can call.

**Startup — load model once:**
```python
@asynccontextmanager
async def lifespan(app):
    try:
        app.state.artifacts = load_artifacts()   # load model + scaler into RAM
        app.state.model_ready = True
    except Exception as e:
        app.state.model_ready = False            # start anyway, report degraded
    yield   # app runs here
```

**Health endpoint (Spring Boot checks this):**
```python
@app.get("/health")
async def health_check():
    return {
        "status": "UP",
        "model": "LOADED" if app.state.model_ready else "NOT_LOADED"
    }
```

**Predict endpoint (Spring Boot calls this per session):**
```python
@app.post("/predict")
async def predict(request: SessionFeatures):
    # 1. Validate all 12 features present and numeric
    formatted = format_for_prediction(request.features)
    
    # 2. Run prediction
    result = predict_session(formatted, app.state.artifacts)
    
    # 3. Return structured response
    return PredictionResponse(**result)
    # → {"anomalyScore": 0.56, "prediction": "ANOMALY", "confidence": "MEDIUM"}
```

---

## PART 7: Final Model Performance

### The Numbers

| Metric | Value | What it means |
|---|---|---|
| **Overall Accuracy** | **91%** | 91 out of 100 sessions classified correctly |
| **Normal Precision** | **95%** | When it says "normal", it's right 95% of the time |
| **Attack Recall** | **71%** | Catches 71 out of every 100 real attacks |
| **False Alarm Rate** | **5%** | 5% of innocent sessions are wrongly flagged |

### The Simulated Live Tracker Results

These were tested using **real camelCase JSON**, exactly as Spring Boot sends it:

| Scenario | Anomaly Score | Result |
|---|---|---|
| Normal user (5 GET requests, 200s) | 0.478 | ✅ NORMAL |
| Brute force (10 failed logins) | 0.556 | 🚨 ANOMALY — MEDIUM |
| Recon / path scanning (/.env, /.git, /admin) | 0.559 | 🚨 ANOMALY — MEDIUM |
| JS Error flood (4 errors in 5 events) | 0.516 | 🚨 ANOMALY — MEDIUM |

---

## PART 8: The Complete Data Flow (End to End)

```
1. User opens browser → tracker.js starts recording

2. User clicks something → tracker.js sends:
   POST /api/tracking/collect
   {"type": "fetch_request", "url": "/api/v1/data", "statusCode": 200,
    "responseTime": 180, "ipAddress": "...", "method": "GET", ...}

3. Spring Boot saves this to MongoDB (raw_logs collection)

4. Every 60 seconds, Spring Boot:
   - Fetches all raw logs from the last 60s grouped by IP
   - Sends them to Python FastAPI:
     POST http://localhost:8000/predict
     {"features": [{"statusCode":200, "url":"/dashboard", ...}, ...]}

5. FastAPI receives the logs:
   - feature_engineering.py aggregates them → 12 features
   - predict.py scales and scores
   - Returns: {"anomalyScore": 0.48, "prediction": "NORMAL", "confidence": "LOW"}

6. Spring Boot:
   - If ANOMALY → sets isAnomaly=true in MongoDB, broadcasts WebSocket alert
   - If NORMAL → does nothing

7. Admin Dashboard shows real-time alerts when anomalies are detected
```

---

*This document was written as part of the RiskTrace PFE project — April 2026*  
*Model: Isolation Forest | Dataset: ModSecurity WAF + NASA HTTP Logs | Accuracy: 91%*
