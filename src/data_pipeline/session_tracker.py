import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
from .parsers import ANOMALOUS_PATTERNS, generate_synthetic_timing

SESSION_TIMEOUT_MINUTES = 30

class SessionTracker:
    def __init__(self):
        self.sessions = []
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

    def log_request(self, ip: str, timestamp: datetime, method: str, url: str, status: int, is_attack_data: bool):
        method = method.upper()
        
        if ip not in self.active_sessions:
            self._start_new_session(ip, timestamp, is_attack_data)
        
        session = self.active_sessions[ip]
        if timestamp - session['last_time'] > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
            self._close_session(ip)
            self._start_new_session(ip, timestamp, is_attack_data)
            session = self.active_sessions[ip]

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
            'is_attack_data': int(is_attack_data)
        }

    def _close_session(self, ip: str):
        s = self.active_sessions.pop(ip)
        duration_s = max(1, (s['last_time'] - s['start_time']).total_seconds())
        
        req_count = max(1, s['request_count'])
        rts = s['response_times']
        
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

    def close_all(self):
        ips = list(self.active_sessions.keys())
        for ip in ips:
            self._close_session(ip)
