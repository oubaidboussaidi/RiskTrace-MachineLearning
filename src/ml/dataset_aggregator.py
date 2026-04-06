import os
import re
import gzip
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "Data")
OUTPUT_CSV = os.path.join(DATA_DIR, "risk_trace_training_data.csv")

# Regex Parsers
NASA_CLF_REGEX = re.compile(
    r'^(\S+) \S+ \S+ \[([\w:/]+\s[+\-]\d{4})\] "(\S+) (\S+)\s*\S*" (\d{3}) (\S+)'
)
MODSEC_BOUNDARY_REGEX = re.compile(r'^--([a-zA-Z0-9]{8})-([A-Z])--$')
MODSEC_A_REGEX = re.compile(r'^\[([^\]]+)\]\s+\S+\s+(\S+)\s+\d+\s+\S+\s+\d+')
MODSEC_B_REGEX = re.compile(r'^([A-Z]+)\s+(\S+)\s+HTTP')
MODSEC_F_REGEX = re.compile(r'^HTTP/\S+\s+(\d{3})')

# Anomalous paths common in WAF hits
ANOMALOUS_PATTERNS = re.compile(r'(?i)(/admin|\.env|\.git|\.bak|phpmyadmin|union.*select|script>|<script|eval\()')

# Configuration
SESSION_TIMEOUT_MINUTES = 30


def parse_nasa_time(time_str: str) -> datetime:
    # Example: 01/Jul/1995:00:00:01 -0400
    try:
        return datetime.strptime(time_str[:-6], "%d/%b/%Y:%H:%M:%S")
    except Exception:
        return datetime.now()


def parse_modsec_time(time_str: str) -> datetime:
    # Example: 01/Aug/2025:10:00:00 +0000
    try:
        return datetime.strptime(time_str[:-6], "%d/%b/%Y:%H:%M:%S")
    except Exception:
        return datetime.now()


def generate_synthetic_timing(status_code: int) -> float:
    # Fake response times in ms
    if status_code >= 500: return np.random.normal(500, 150)
    if status_code >= 400: return np.random.normal(100, 20)
    return np.random.normal(200, 50)


class SessionTracker:
    def __init__(self):
        self.sessions = []
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

    def log_request(self, ip: str, timestamp: datetime, method: str, url: str, status: int, is_attack_data: bool):
        # Clean method/url
        method = method.upper()
        
        if ip not in self.active_sessions:
            self._start_new_session(ip, timestamp, is_attack_data)
        
        session = self.active_sessions[ip]
        if timestamp - session['last_time'] > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
            self._close_session(ip)
            self._start_new_session(ip, timestamp, is_attack_data)
            session = self.active_sessions[ip]

        # Update session features
        session['request_count'] += 1
        session['last_time'] = timestamp
        session['urls'].add(url)
        
        if status >= 400:
            session['error_count'] += 1
        if status in [401, 403]:
            session['auth_failure_count'] += 1
        if ANOMALOUS_PATTERNS.search(url):
            session['anomalous_path_count'] += 1
        if method == "POST":
            session['post_count'] += 1
            
        session['response_times'].append(generate_synthetic_timing(status))

    def _start_new_session(self, ip: str, timestamp: datetime, is_attack_data: bool):
        self.active_sessions[ip] = {
            'ip': ip,
            'start_time': timestamp,
            'last_time': timestamp,
            'request_count': 0,
            'error_count': 0,
            'auth_failure_count': 0,
            'urls': set(),
            'anomalous_path_count': 0,
            'post_count': 0,
            'response_times': [],
            'is_attack_data': int(is_attack_data) # Global dataset label
        }

    def _close_session(self, ip: str):
        s = self.active_sessions.pop(ip)
        duration_s = max(1, (s['last_time'] - s['start_time']).total_seconds())
        
        req_count = max(1, s['request_count'])
        rts = s['response_times']
        
        # Calculate final aggregated features
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
            'js_error_count': 0,  # App-level placeholder
            'request_rate': round(s['request_count'] / duration_s, 3),
            'session_duration_s': duration_s,
            'is_anomaly': s['is_attack_data'] # Training Ground Truth
        })

    def close_all(self):
        ips = list(self.active_sessions.keys())
        for ip in ips:
            self._close_session(ip)


def process_nasa_logs(tracker: SessionTracker):
    nasa_dir = os.path.join(DATA_DIR, "NASA_Normal_Baseline_Dataset")
    if not os.path.exists(nasa_dir):
        logging.warning("No NASA Normal Baseline directory found.")
        return
        
    nasa_files = [f for f in os.listdir(nasa_dir) if f.startswith("NASA") and f.endswith(".gz")]
    for filename in nasa_files:
        filepath = os.path.join(nasa_dir, filename)
        logging.info(f"Parsing NASA normal logs: {filepath}")
        
        count = 0
        with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = NASA_CLF_REGEX.search(line)
                if match:
                    ip, time_str, method, url, status, _ = match.groups()
                    timestamp = parse_nasa_time(time_str)
                    tracker.log_request(ip, timestamp, method, url, int(status), is_attack_data=False)
                    count += 1
                    
                if count >= 300000: # Limit normal logs to match WAF volume
                    break
        logging.info(f"Parsed {count} normal requests from {filename}.")


def process_modsecurity_logs(tracker: SessionTracker):
    modsec_dir = os.path.join(DATA_DIR, "ModSecurity_Attacks_Dataset")
    if not os.path.exists(modsec_dir):
        logging.warning("No ModSecurity Attacks directory found.")
        return

    # Find all modsec_audit.anon.log files
    log_files = []
    for root, dirs, files in os.walk(modsec_dir):
        for file in files:
            if file.endswith(".log"):
                log_files.append(os.path.join(root, file))

    logging.info(f"Found {len(log_files)} ModSecurity log files.")
    
    for filepath in log_files:
        logging.info(f"Parsing WAF attack logs: {filepath}")
        
        current_tx = None
        tx_data = {}
        count = 0
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                boundary = MODSEC_BOUNDARY_REGEX.search(line)
                if boundary:
                    tx_id, section = boundary.groups()
                    
                    if section == 'A':
                        if tx_data.get('ip') and tx_data.get('url'):
                            # Commit previous transaction
                            tracker.log_request(
                                tx_data['ip'], tx_data.get('time', datetime.now()), 
                                tx_data.get('method', 'GET'), tx_data['url'], 
                                tx_data.get('status', 403), is_attack_data=True
                            )
                            count += 1
                        current_tx = tx_id
                        tx_data = {}
                    
                    current_section = section
                    continue
                
                if current_tx and line.strip():
                    if current_section == 'A':
                        match = MODSEC_A_REGEX.search(line)
                        if match:
                            tx_data['time'] = parse_modsec_time(match.group(1))
                            tx_data['ip'] = match.group(2)
                    elif current_section == 'B':
                        if 'method' not in tx_data:
                            match = MODSEC_B_REGEX.search(line)
                            if match:
                                tx_data['method'] = match.group(1)
                                tx_data['url'] = match.group(2)
                    elif current_section == 'F' or current_section == 'H':
                        if 'status' not in tx_data:
                            match = MODSEC_F_REGEX.search(line)
                            if match:
                                tx_data['status'] = int(match.group(1))

            # Commit the very last transaction in file
            if tx_data.get('ip') and tx_data.get('url'):
                tracker.log_request(
                    tx_data['ip'], tx_data.get('time', datetime.now()), 
                    tx_data.get('method', 'GET'), tx_data['url'], 
                    tx_data.get('status', 403), is_attack_data=True
                )
                count += 1
        
        logging.info(f"Parsed {count} attack requests from {os.path.basename(filepath)}.")

def main():
    if not os.path.exists(DATA_DIR):
        logging.error(f"Data directory not found: {DATA_DIR}")
        return

    tracker = SessionTracker()
    
    logging.info("--- Phase 1: Ingesting ModSecurity WAF Attacks ---")
    process_modsecurity_logs(tracker)
    
    logging.info("--- Phase 2: Ingesting NASA Normal Traffic Baseline ---")
    process_nasa_logs(tracker)
    
    logging.info("--- Phase 3: Aggregating Sessions ---")
    tracker.close_all()
    
    if not tracker.sessions:
        logging.error("No sessions were generated. Check dataset paths.")
        return
        
    df = pd.DataFrame(tracker.sessions)
    
    logging.info(f"Raw sessions generated: {len(df)}")
    logging.info(f"  Normal sessions (raw): {len(df[df['is_anomaly']==0])}")
    logging.info(f"  Attack sessions (raw): {len(df[df['is_anomaly']==1])}")
    logging.info(f"  Attack ratio    (raw): {len(df[df['is_anomaly']==1])/len(df):.1%}")

    # ── Phase 4: Strategic Dataset Balancing ──────────────────────────────────
    # WHY: The Isolation Forest is an UNSUPERVISED anomaly detector. It is
    # mathematically designed to detect rare anomalies embedded in mostly-normal
    # data. If the training set contains ~42% attacks (as in the raw output),
    # the model cannot distinguish a clear "normal boundary" — it treats attacks
    # as part of the baseline, massively degrading accuracy (tested: 65% F1).
    #
    # A real-world enterprise environment has ~5-15% anomalous traffic at most.
    # We undersample to 15% attacks to replicate this reality and ensure the
    # Isolation Forest operates as designed.
    #
    # STRATEGY: Keep ALL normal sessions. Sample attack sessions so they
    # represent exactly TARGET_ATTACK_RATIO of the final combined dataset.
    TARGET_ATTACK_RATIO = 0.15   # 15% attacks, 85% normal — matches real SIEM data
    
    normal_df = df[df['is_anomaly'] == 0]
    attack_df = df[df['is_anomaly'] == 1]
    
    # Calculate how many attack sessions to keep: n_attack = (ratio * n_normal) / (1 - ratio)
    n_normal = len(normal_df)
    n_attack_target = int((TARGET_ATTACK_RATIO * n_normal) / (1 - TARGET_ATTACK_RATIO))
    n_attack_target = min(n_attack_target, len(attack_df))  # Can't sample more than we have
    
    attack_sampled = attack_df.sample(n=n_attack_target, random_state=42)
    df_balanced = pd.concat([normal_df, attack_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    logging.info(f"After balancing:")
    logging.info(f"  Normal sessions: {len(df_balanced[df_balanced['is_anomaly']==0])}")
    logging.info(f"  Attack sessions: {len(df_balanced[df_balanced['is_anomaly']==1])}")
    logging.info(f"  Attack ratio   : {len(df_balanced[df_balanced['is_anomaly']==1])/len(df_balanced):.1%}")

    # Save to CSV
    df_balanced.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"Success! Training dataset saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

