import requests
import json
import os
import time
import math
import datetime
import boto3
import glob

from prefect import flow, task
from prefect.logging import get_run_logger

from config import VM_HOST, VM_AUTH_TOKEN, DATA_DIR, RETENTION_SECONDS, METRIC_NAMES, WINDOW_SIZE
from config import BUCKET_NAME, ENVIRONMENT, AWS_PROFILE
from dotenv import load_dotenv
import os
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import torch

load_dotenv()

SERVICE_MAP_FILE = os.path.join(DATA_DIR, f"{ENVIRONMENT}/_service_map.json")
EXPECTED_INTERVAL = 60 

def download_s3_folder(bucket_name, s3_folder, local_dir=None, profile=None):
    if profile:
        s3 = boto3.Session(profile_name=profile).resource("s3")
    else:
        s3 = boto3.resource("s3")

    bucket = s3.Bucket(bucket_name)

    for obj in bucket.objects.filter(Prefix=s3_folder):
        obj_path = os.path.join(local_dir, obj.key)

        if not os.path.exists(os.path.dirname(obj_path)):
            os.makedirs(os.path.dirname(obj_path), exist_ok=True)

        if obj.key[-1] == "/" or obj.key == s3_folder:
            continue
        print("obj.key", obj.key)
        bucket.download_file(obj.key, obj_path)
        print(f"Downloaded s3://{bucket_name}/{obj.key} -> {obj_path}")


def download_from_s3(bucket_name, environment):
    s3_folder = f"{environment}/"
    try:
        download_s3_folder(bucket_name, s3_folder, local_dir=DATA_DIR)
    except Exception as e:
        print(f"Default/IRSA creds failed ({e}); retrying with local AWS profile")
        print(f"Using profile: {AWS_PROFILE}")
        download_s3_folder(bucket_name, s3_folder, local_dir=DATA_DIR, profile=AWS_PROFILE)


def download_specific_service(bucket_name, environment, services):
    """Download only the given services' files plus _service_map.json, instead of the whole <environment>/ folder.
    services: list of 'namespace/name' keys.
    S3 layout (written by collect.py): <environment>/<date>/<namespace>__<name>__<metric>.json
    and <environment>/_service_map.json"""
    for k in services:
        if "/" not in k:
            raise ValueError(f"'{k}' is not a 'namespace/name' key")
    wanted = {f"{k.split('/', 1)[0]}__{k.split('/', 1)[1]}__{m}.json" for k in services for m in METRIC_NAMES}
    map_key = f"{environment}/_service_map.json"

    def run(profile):
        if profile:
            s3 = boto3.Session(profile_name=profile).resource("s3")
        else:
            s3 = boto3.resource("s3")
        bucket = s3.Bucket(bucket_name)
        n_files, got_map = 0, False
        for obj in bucket.objects.filter(Prefix=f"{environment}/"):
            if obj.key == map_key:
                got_map = True
            elif obj.key.rsplit("/", 1)[-1] in wanted:
                n_files += 1
            else:
                continue
            obj_path = os.path.join(DATA_DIR, obj.key)
            os.makedirs(os.path.dirname(obj_path), exist_ok=True)
            bucket.download_file(obj.key, obj_path)
        return n_files, got_map

    try:
        n_files, got_map = run(None)
    except Exception as e:
        print(f"Default/IRSA creds failed ({e}); retrying with local AWS profile")
        print(f"Using profile: {AWS_PROFILE}")
        n_files, got_map = run(AWS_PROFILE)
    if not got_map:
        raise RuntimeError(f"s3://{bucket_name}/{map_key} not found")
    if n_files == 0:
        raise RuntimeError(f"No files found for {services} under s3://{bucket_name}/{environment}/. "
                           f"Use the exact 'namespace/name' key from _service_map.json.")
    print(f"Downloaded {n_files} files for {services} (and _service_map.json).")


def build_service_series(safe_name, min_length):
    per_metric = {}
    combined_metrics_name = "combined_metrics"
    os.makedirs(os.path.join(DATA_DIR, ENVIRONMENT, combined_metrics_name), exist_ok=True)
    for m in METRIC_NAMES:
       existing = {}
       directory_path = Path(f"{DATA_DIR}/{ENVIRONMENT}/")
       json_files = list(directory_path.rglob(f"{safe_name}__{m}.json"))
       for file in json_files:
            with open(file, "r", encoding="utf-8") as f:
                for t, v in json.load(f):
                    existing[float(t)] = v
       merged = sorted(existing.items())
       with open(f"{DATA_DIR}/{ENVIRONMENT}/{combined_metrics_name}/{combined_metrics_name}_{safe_name}__{m}.json", "w", encoding="utf-8") as f:
            json.dump(merged, f)


    per_metric = {m: load_metric_file(f"{DATA_DIR}/{ENVIRONMENT}/{combined_metrics_name}/{combined_metrics_name}_{safe_name}__{m}.json") for m in METRIC_NAMES}
    common_ts = sorted(set.intersection(*(set(d) for d in per_metric.values())))

    if len(common_ts) < min_length:
           return None, None

    rows = [[per_metric[m][t] for m in METRIC_NAMES] for t in common_ts]
    return np.array(rows, dtype=np.float32), common_ts


def load_metric_file(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        points = json.load(f)
    return {int(float(t) // 60) * 60: float(v) for t, v in points}




class MultiServiceMetricsDataset(Dataset):
    def __init__(self, window_size=WINDOW_SIZE, val_fraction=0.1):
        self.window_size = window_size
        self.val_fraction = val_fraction
        self.windows = []
        self.skipped_services = []
        self.norm_stats = {}   
        self.service_window_ranges = {} 

        if not os.path.exists(SERVICE_MAP_FILE):
            raise FileNotFoundError(f"{SERVICE_MAP_FILE} not found — run collect.py at least once first.")
        with open(SERVICE_MAP_FILE) as f:
            self.service_map = json.load(f)

        for key, service_id in self.service_map.items():
            ns, name = key.split("/", 1)
            safe_name = f"{ns}__{name}"
            series, timestamps = build_service_series(safe_name, min_length=window_size + 10)
            if series is None:
                self.skipped_services.append(key)
                continue
            n_total = len(series)

            split_idx = int(n_total * (1 - val_fraction))
            split_idx = max(split_idx, window_size + 1)  
            train_portion = series[:split_idx]
            mean = train_portion.mean(axis=0)
            std = np.maximum(train_portion.std(axis=0), 1e-3)
            self.norm_stats[service_id] = {"mean": mean.tolist(), "std": std.tolist()}
            series_norm = (series - mean) / std
            hours = np.array([(int(t) % 86400) / 3600 for t in timestamps])
            hour_sin = np.sin(2 * np.pi * hours / 24)
            hour_cos = np.cos(2 * np.pi * hours / 24)
            full = np.concatenate([series_norm, hour_sin[:, None], hour_cos[:, None]], axis=1)
            start_idx = len(self.windows)

            for i in range(len(full) - window_size):
                window_ts = timestamps[i: i + window_size + 1]  
                valid = True
                for j in range(len(window_ts) - 1):
                    gap = window_ts[j + 1] - window_ts[j]
                    if gap != EXPECTED_INTERVAL:
                        valid = False
                        break
                if not valid:
                    continue
                window = full[i: i + window_size]
                target = series_norm[i + window_size]
                self.windows.append((window, target, service_id))

            end_idx = len(self.windows)
            self.service_window_ranges[service_id] = (start_idx, end_idx)

    def chronological_split(self):
        train_indices, val_indices = [], []
        for service_id, (start, end) in self.service_window_ranges.items():
            n = end - start
            if n == 0:
                continue
            split_point = start + int(n * (1 - self.val_fraction))
            purge_end = max(start, split_point - self.window_size)  

            train_indices.extend(range(start, purge_end))
            val_indices.extend(range(split_point, end))
        return train_indices, val_indices

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window, target, service_id = self.windows[idx]
        return (
            torch.tensor(window, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(service_id, dtype=torch.long),
        )





if __name__ == "__main__":
    download_from_s3(BUCKET_NAME, ENVIRONMENT)
    ds = MultiServiceMetricsDataset()
    print(f"Total windows: {len(ds)}")
    print(f"Services with enough data: {len(ds.service_map) - len(ds.skipped_services)} / {len(ds.service_map)}")
    if ds.skipped_services:
        preview = ds.skipped_services[:10]
        print(f"Skipped (not enough history yet): {preview}{'...' if len(ds.skipped_services) > 10 else ''}")

    train_idx, val_idx = ds.chronological_split()
    print(f"Chronological split: {len(train_idx)} train windows, {len(val_idx)} val windows")