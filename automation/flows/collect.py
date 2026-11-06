import requests
import json
import os
import time
import math
import datetime
import boto3

from prefect import flow, task
from prefect.cache_policies import NONE
from prefect.logging import get_run_logger

from config import VM_HOST, VM_AUTH_TOKEN, DATA_DIR, RETENTION_SECONDS
from config import BUCKET_NAME, ENVIRONMENT, AWS_PROFILE
from dotenv import load_dotenv

load_dotenv()

STATE_FILE = os.path.join(DATA_DIR, f"{ENVIRONMENT}/_last_pulled.json")
SERVICE_MAP_FILE = os.path.join(DATA_DIR, f"{ENVIRONMENT}/_service_map.json")

MAX_FAILURE_RATIO = 0.2
def safe_float(v):
    f = float(v)
    if math.isnan(f) or math.isinf(f):
        return None
    return f


@task(cache_policy=NONE)
def upload_folder_to_s3(bucket_name, folder_path, s3_prefix="", profile=None):
    if profile:
        s3_client = boto3.Session(profile_name=profile).client("s3")
    else:
        s3_client = boto3.client("s3")

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, folder_path)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")
            print(f"Uploading {local_path} to s3://{bucket_name}/{s3_key}")
            s3_client.upload_file(local_path, bucket_name, s3_key)


@task(cache_policy=NONE)
def upload_to_s3(bucket_name, environment):
    folder_path = os.path.join(DATA_DIR, environment)
    try:
        upload_folder_to_s3(bucket_name, folder_path, s3_prefix=environment)
    except Exception as e:
        print(f"Default/IRSA creds failed ({e}); retrying with local AWS profile")
        print(f"Using profile: {AWS_PROFILE}")
        upload_folder_to_s3(bucket_name, folder_path, s3_prefix=environment, profile=AWS_PROFILE)


@task(cache_policy=NONE)
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
        bucket.download_file(obj.key, obj_path)
        print(f"Downloaded s3://{bucket_name}/{obj.key} -> {obj_path}")


@task(cache_policy=NONE)
def download_from_s3(bucket_name, environment, date):
    s3_folder = f"{environment}/{date}/"
    state_prefix = f"{environment}/_"   # matches _last_pulled.json and _service_map.json
    try:
        download_s3_folder(bucket_name, s3_folder, local_dir=DATA_DIR)
        download_s3_folder(bucket_name, state_prefix, local_dir=DATA_DIR)
    except Exception as e:
        print(f"Default/IRSA creds failed ({e}); retrying with local AWS profile")
        print(f"Using profile: {AWS_PROFILE}")
        download_s3_folder(bucket_name, s3_folder, local_dir=DATA_DIR, profile=AWS_PROFILE)
        download_s3_folder(bucket_name, state_prefix, local_dir=DATA_DIR, profile=AWS_PROFILE)


def _auth_headers():
    if VM_AUTH_TOKEN:
        return {"Authorization": f"Bearer {VM_AUTH_TOKEN}"}
    return {}



def api_get(path, params, max_retries=3):
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(f"{VM_HOST}{path}", params=params,
                                headers=_auth_headers(), timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"    retry {attempt + 1}/{max_retries} after error: {e} (waiting {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {max_retries} retries: {path} {params} ({last_err})")


@task(cache_policy=NONE)
def discover_all_services(min_rps=0.01, window="1h"):
    q = (f'sum(rate(istio_requests_total{{reporter="destination"}}[{window}])) '
         f'by (destination_service_name, destination_service_namespace)')
    data = api_get("/api/v1/query", {"query": q})
    results = data["data"]["result"]

    services = []
    for r in results:
        name = r["metric"].get("destination_service_name")
        ns = r["metric"].get("destination_service_namespace")
        rps = float(r["value"][1])
        if not name or not ns or rps < min_rps:
            continue
        if "ingress-nginx" in name or "kong" in name or "kgateway" in name:
            continue
        if name in ("PassthroughCluster", "InboundPassthroughCluster", "BlackHoleCluster"):
            continue

        services.append((name, ns, rps))

    services.sort(key=lambda s: -s[2])
    return services


@task(cache_policy=NONE)
def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


@task(cache_policy=NONE)
def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def append_points(fname, new_points):
    existing = {}
    if os.path.exists(fname):
        with open(fname) as f:
            for t, v in json.load(f):
                existing[float(t)] = v
    for t, v in new_points:
        existing[float(t)] = v
    merged = sorted(existing.items())
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "w") as f:
        json.dump(merged, f)
    return len(merged)


def pull_metric_range(query, start, end, step="60s"):
    data = api_get("/api/v1/query_range", {"query": query, "start": start, "end": end, "step": step})
    return data["data"]["result"]


def build_queries(name, namespace):

    base_total = (f'sum(rate(istio_requests_total{{reporter="destination", '
                  f'destination_service_name="{name}", destination_service_namespace="{namespace}"}}[5m]))')

    q_rate = (f'sum(rate(istio_requests_total{{reporter="destination", '
              f'destination_service_name="{name}", destination_service_namespace="{namespace}"}}[5m])) '
              f'by (response_code)')

    q_err5xx = (f'(sum(rate(istio_requests_total{{reporter="destination", '
                f'destination_service_name="{name}", destination_service_namespace="{namespace}", '
                f'response_code=~"5.."}}[5m])) or vector(0)) / {base_total}')

    q_err4xx = (f'(sum(rate(istio_requests_total{{reporter="destination", '
                f'destination_service_name="{name}", destination_service_namespace="{namespace}", '
                f'response_code=~"4.."}}[5m])) or vector(0)) / {base_total}')

    q_p99 = (f'histogram_quantile(0.99, sum(rate(istio_request_duration_milliseconds_bucket{{'
             f'reporter="destination", destination_service_name="{name}", '
             f'destination_service_namespace="{namespace}"}}[5m])) by (le))')

    return q_rate, q_err5xx, q_err4xx, q_p99


@task(cache_policy=NONE, task_run_name="collect-{namespace}-{name}")
def collect_service(name, namespace, last_pulled, now, date):

    key = f"{namespace}/{name}"
    if last_pulled is None:
        last_pulled = now - RETENTION_SECONDS + 300

    start = (int(max(last_pulled, now - RETENTION_SECONDS)) // 60) * 60
    end = (int(now) // 60) * 60

    if end - start < 60:
        return {"end": None, "summary": "up to date"}

    q_rate, q_err5xx, q_err4xx, q_p99 = build_queries(name, namespace)

    rate_result = pull_metric_range(q_rate, start, end)
    err5xx_result = pull_metric_range(q_err5xx, start, end)
    err4xx_result = pull_metric_range(q_err4xx, start, end)
    p99_result = pull_metric_range(q_p99, start, end)

    missing = [n for n, r in [("rate", rate_result), ("error5xx", err5xx_result),
                              ("error4xx", err4xx_result), ("p99", p99_result)] if not r]
    if missing:
        raise RuntimeError(f"{key}: query returned EMPTY for {missing}; not advancing checkpoint")
    rate = {}
    for series in rate_result:
        for t, v in series["values"]:
            f = safe_float(v)
            if f is not None:
                rate[float(t)] = rate.get(float(t), 0.0) + f

    def clean(result):
        pts = [(float(t), safe_float(v)) for t, v in result[0]["values"]]
        return [[t, v] for t, v in pts if v is not None]

    cleaned = {
        "rate": [[t, v] for t, v in sorted(rate.items())],
        "error5xx": clean(err5xx_result),
        "error4xx": clean(err4xx_result),
        "p99": clean(p99_result),
    }

    safe_name = f"{namespace}__{name}"
    base = os.path.join(DATA_DIR, ENVIRONMENT, date)
    counts = {}
    for metric in ("rate", "error5xx", "error4xx", "p99"):
        counts[metric] = append_points(os.path.join(base, f"{safe_name}__{metric}.json"), cleaned[metric])

    return {
        "end": end,
        "summary": (f"pulled (rate={counts['rate']}, 5xx={counts['error5xx']}, "
                    f"4xx={counts['error4xx']}, p99={counts['p99']} total points on disk now)"),
    }


@flow(log_prints=True)
def collect():
    logger = get_run_logger()
    os.makedirs(os.path.join(DATA_DIR, ENVIRONMENT), exist_ok=True)

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Downloading s3://{BUCKET_NAME}/{ENVIRONMENT}/{date}/ and state files -> {DATA_DIR}/{ENVIRONMENT}/")
    download_from_s3(BUCKET_NAME, ENVIRONMENT, date)

    state = load_json(STATE_FILE, {})
    service_map = load_json(SERVICE_MAP_FILE, {})
    logger.info(f"Loaded state for {len(state)} services, service map with {len(service_map)} entries")

    logger.info("Discovering active services...")
    services = discover_all_services(min_rps=0.01)
    logger.info(f"Discovered {len(services)} active services (excluding gateway/proxy layer)")

    now = int(time.time())
    failures = []
    for i, (name, ns, rps) in enumerate(services):
        key = f"{ns}/{name}"
        if key not in service_map:
            service_map[key] = len(service_map)

        prefix = f"[{i + 1}/{len(services)}] {key} ({rps:.2f} req/s)"
        try:
            result = collect_service(name, ns, state.get(key), now, date)
            if result["end"] is not None:
                state[key] = result["end"]
            print(f"{prefix}: {result['summary']}")
        except Exception as e:
            failures.append(key)
            print(f"{prefix}: FAILED: {e}")
        time.sleep(0.2)

        if i % 25 == 0:
            print("Saving state and service map")
            save_json(STATE_FILE, state)
            save_json(SERVICE_MAP_FILE, service_map)

    save_json(STATE_FILE, state)
    save_json(SERVICE_MAP_FILE, service_map)
    logger.info(f"Done. {len(service_map)} services tracked in total across all runs so far.")
    logger.info(f"Uploading {DATA_DIR}/{ENVIRONMENT}/ -> s3://{BUCKET_NAME}/{ENVIRONMENT}/")
    upload_to_s3(BUCKET_NAME, ENVIRONMENT)

    if failures:
        logger.warning(f"{len(failures)}/{len(services)} services failed: {failures[:20]}")
    if services and len(failures) / len(services) > MAX_FAILURE_RATIO:
        raise RuntimeError(f"{len(failures)}/{len(services)} services failed (> {MAX_FAILURE_RATIO:.0%})")


if __name__ == "__main__":
    collect()