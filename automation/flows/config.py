"""
config.py

Single source of truth for connection details and pipeline constants.
Reads from environment variables where set, falls back to sensible defaults.
This is basic production hygiene: nothing here should need hunting-and-replacing
across multiple files when it changes.
"""

import os

VM_HOST = os.environ.get("VM_HOST", "https://istio-vm.syed.stream")
# If your endpoint needs auth (token, cookie, etc.), set it via environment
# variable too, e.g. VM_AUTH_TOKEN, and read it in collect.py's api_get() —
# never hardcode a real credential directly in a .py file.
VM_AUTH_TOKEN = os.environ.get("VM_AUTH_TOKEN", None)

DATA_DIR = os.environ.get("ANOMALY_DATA_DIR", "data")
RETENTION_SECONDS = int(os.environ.get("VM_RETENTION_SECONDS", 4 * 24 * 60 * 60))
WINDOW_SIZE = int(os.environ.get("ANOMALY_WINDOW_SIZE", 60))

# Feature order is fixed and used consistently across dataset.py, model.py, score.py:
# [request_rate, error_rate_5xx, error_rate_4xx, p99_latency]
METRIC_NAMES = ["rate", "error5xx", "error4xx", "p99"]
N_REAL_FEATURES = len(METRIC_NAMES)   # 4
N_TIME_FEATURES = 2                    # hour_sin, hour_cos
N_FEATURES = N_REAL_FEATURES + N_TIME_FEATURES   # 6, fed into the model
BUCKET_NAME = os.environ.get("ANOMALY_BUCKET_NAME", "istio-anomaly-detection")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "staging")
AWS_PROFILE = os.environ.get("AWS_PROFILE", "default")
