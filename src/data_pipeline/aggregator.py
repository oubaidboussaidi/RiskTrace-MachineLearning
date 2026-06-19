import os
import gzip
import logging
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "Data")
OUTPUT_CSV = os.path.join(DATA_DIR, "risk_trace_training_data.csv")

from .parsers import NASA_CLF_REGEX, MODSEC_BOUNDARY_REGEX, MODSEC_A_REGEX, MODSEC_B_REGEX, MODSEC_F_REGEX, parse_nasa_time, parse_modsec_time
from .session_tracker import SessionTracker
from .sampler import intelligent_sample_anomalies

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
                    
                if count >= 300000:
                    break
        logging.info(f"Parsed {count} normal requests from {filename}.")


def process_modsecurity_logs(tracker: SessionTracker):
    modsec_dir = os.path.join(DATA_DIR, "ModSecurity_Attacks_Dataset")
    if not os.path.exists(modsec_dir):
        logging.warning("No ModSecurity Attacks directory found.")
        return

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

    logging.info("--- Phase 4: Intelligent Anomaly Reduction ---")
    df_balanced = intelligent_sample_anomalies(df, target_ratio=0.15)
    
    logging.info(f"After balancing:")
    logging.info(f"  Normal sessions: {len(df_balanced[df_balanced['is_anomaly']==0])}")
    logging.info(f"  Attack sessions: {len(df_balanced[df_balanced['is_anomaly']==1])}")
    logging.info(f"  Attack ratio   : {len(df_balanced[df_balanced['is_anomaly']==1])/len(df_balanced):.1%}")

    df_balanced.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"Success! Training dataset saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
