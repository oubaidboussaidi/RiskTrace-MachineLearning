import re
from datetime import datetime
import numpy as np

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

def parse_nasa_time(time_str: str) -> datetime:
    try:
        return datetime.strptime(time_str[:-6], "%d/%b/%Y:%H:%M:%S")
    except Exception:
        return datetime.now()

def parse_modsec_time(time_str: str) -> datetime:
    try:
        return datetime.strptime(time_str[:-6], "%d/%b/%Y:%H:%M:%S")
    except Exception:
        return datetime.now()

def generate_synthetic_timing(status_code: int) -> float:
    if status_code >= 500: return np.random.normal(500, 150)
    if status_code >= 400: return np.random.normal(100, 20)
    return np.random.normal(200, 50)
