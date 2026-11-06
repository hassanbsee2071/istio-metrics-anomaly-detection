"""
score.py

How surprised is the model by the latest minute of a service?

For one service it does these steps:
  1. Download the model from MLflow (the version that has the alias "production").
  2. Ask VictoriaMetrics for the last 61 minutes of the 4 metrics.
  3. Scale the numbers with the mean/std saved inside the model file (exactly like training).
  4. Give the first 60 minutes to the model. It predicts minute 61.
  5. Compare the prediction with what really happened. Bigger score = more surprised.

Usage:
    export MLFLOW_TRACKING_URI="https://mlflow.syed.stream"

    python3 score.py --service ns-01/svc-001                      # model from MLflow, alias "production"
    python3 score.py --service ns-01/svc-001 --alias candidate    # try a new version before promoting it
    python3 score.py --service ns-01/svc-001 --model-file metrics_model_services.pth   # a local file instead
    python3 score.py --all --model-name anomaly-staging-multi   # every service that model was trained on
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

from config import ENVIRONMENT, METRIC_NAMES
from model import MetricsTransformer
from collect import build_queries, pull_metric_range   # the same queries that collected the training data

STALE_AFTER_SECONDS = 5 * 60   # if the newest data is older than this, the service may have stopped reporting


# ----------------------------------------------------------------------------
# Step 1: get the model
# ----------------------------------------------------------------------------
def default_model_name(service_key):
    """The name train.py gives the model when it trains ONE service: anomaly-<environment>-<namespace>-<name>"""
    return f"anomaly-{ENVIRONMENT}-" + service_key.replace("/", "-")


def download_model_from_mlflow(model_name, alias):
    """Download model.pth of the version that has this alias, and return where it was saved."""
    import mlflow
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        sys.exit("MLFLOW_TRACKING_URI is not set. Example: export MLFLOW_TRACKING_URI=https://mlflow.syed.stream")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "3")   # by default MLflow retries for minutes when it is down
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    try:
        folder = mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{model_name}@{alias}")
    except Exception as error:
        # No fallback to a local file on purpose: you could end up scoring with an old model without knowing.
        sys.exit(f"Could not download MLflow model '{model_name}' with alias '{alias}' "
                 f"from {os.environ['MLFLOW_TRACKING_URI']}: {error}")
    path = os.path.join(folder, "model.pth")
    if not os.path.exists(path):
        sys.exit(f"model.pth is not inside MLflow model '{model_name}@{alias}'. Files found there: {os.listdir(folder)}")
    return path


def load_model(path):
    """Read the model file. It holds the weights AND the numbers needed to scale the data the same way as training."""
    checkpoint = torch.load(path, map_location="cpu")
    expected_features = list(METRIC_NAMES) + ["hour_sin", "hour_cos"]
    if "features" in checkpoint and list(checkpoint["features"]) != expected_features:
        sys.exit(f"This model was trained on the features {checkpoint['features']} but this code feeds "
                 f"{expected_features}. Use a model trained with the same features.")
    # train.py saves the model settings in the file, so an older model still loads after model.py changes
    model = MetricsTransformer(num_services=checkpoint["num_services"], **checkpoint.get("model_config", {}))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()   # switch off dropout: we are predicting, not training
    return model, checkpoint


# ----------------------------------------------------------------------------
# Step 2: get the latest minutes from VictoriaMetrics
# ----------------------------------------------------------------------------
def to_value_per_minute(result, add_up_series=False):
    """Turn a VictoriaMetrics answer into {minute: value}.

    add_up_series=True is for the rate query: it returns one series per response code (200, 404, ...)
    and we want the total. The other queries return a single series.
    NaN / Inf values (for example 0 / 0 when a service has no traffic) are skipped, like in collect.py:
    that minute simply becomes a gap.
    """
    series_to_use = result if add_up_series else result[:1]
    value_per_minute = {}
    for series in series_to_use:
        for timestamp, value in series["values"]:
            value = float(value)
            if not math.isfinite(value):
                continue
            minute = int(float(timestamp) // 60) * 60    # round DOWN to the start of the minute, like dataset.py
            value_per_minute[minute] = value_per_minute.get(minute, 0.0) + value
    return value_per_minute


def get_latest_minutes(name, namespace, how_many):
    """Return (numbers, minutes): the newest `how_many` minutes with no minute missing in between.
    numbers has one row per minute and one column per metric. Returns (None, None) if there is not enough clean data."""
    now = int(time.time())
    start = now - (how_many + 5) * 60                           # ask for a few extra minutes
    queries = dict(zip(METRIC_NAMES, build_queries(name, namespace)))   # rate, error5xx, error4xx, p99

    value_per_minute = {}
    for metric in METRIC_NAMES:
        answer = pull_metric_range(queries[metric], start, now)
        value_per_minute[metric] = to_value_per_minute(answer, add_up_series=(metric == "rate"))

    # keep only the minutes where ALL 4 metrics have a value
    minutes = set(value_per_minute[METRIC_NAMES[0]])
    for metric in METRIC_NAMES[1:]:
        minutes = minutes & set(value_per_minute[metric])
    minutes = sorted(minutes)
    if not minutes:
        return None, None

    # start at the newest minute and walk backwards until a minute is missing
    clean_run = [minutes[-1]]
    for minute in reversed(minutes[:-1]):
        if clean_run[0] - minute != 60:
            break                       # a gap: stop here, older data is not usable
        clean_run.insert(0, minute)
        if len(clean_run) == how_many:
            break
    if len(clean_run) < how_many:
        return None, None

    rows = [[value_per_minute[metric][minute] for metric in METRIC_NAMES] for minute in clean_run]
    return np.array(rows, dtype=np.float32), clean_run


# ----------------------------------------------------------------------------
# Steps 3 to 5: scale, predict, compare
# ----------------------------------------------------------------------------
def trained_services(checkpoint):
    """The services this model was really trained on. The scaling numbers (norm_stats) can exist for more services
    than that: they are calculated for every service that had data when training started."""
    if "trained_services" in checkpoint:
        return set(checkpoint["trained_services"])
    # older model files have no such list: use the services that have scaling numbers
    return {key for key, service_id in checkpoint["service_map"].items() if service_id in checkpoint["norm_stats"]}


def score_service(service_key, model, checkpoint):
    """Score the newest minute of one service. Returns a dictionary: either {"error": ...} or the result."""
    window_size = checkpoint["window_size"]                     # 60

    if service_key not in trained_services(checkpoint):
        return {"error": f"This model was not trained on '{service_key}'. "
                         f"It was trained on: {sorted(trained_services(checkpoint))}"}
    service_id = checkpoint["service_map"][service_key]

    norm_stats = checkpoint["norm_stats"]
    stats = norm_stats.get(service_id) or norm_stats.get(str(service_id))
    if stats is None:
        return {"error": f"The model was not trained on '{service_key}' (no scaling numbers saved for it)."}

    namespace, name = service_key.split("/", 1)
    try:
        numbers, minutes = get_latest_minutes(name, namespace, how_many=window_size + 1)
    except Exception as error:      # for example VictoriaMetrics is down. Report it, do not crash a whole scan.
        return {"error": f"Could not read data for '{service_key}': {error}"}
    if numbers is None:
        return {"error": f"Not enough recent data for '{service_key}': need {window_size + 1} minutes in a row "
                         f"with all 4 metrics."}

    age_seconds = int(time.time()) - minutes[-1]
    if age_seconds > STALE_AFTER_SECONDS:
        return {"error": f"STALE: the newest data for '{service_key}' is {age_seconds // 60} minutes old. "
                         f"The service may have stopped reporting."}

    # scale with the numbers from training (never recompute them here)
    mean = np.array(stats["mean"])
    std = np.array(stats["std"])
    scaled = (numbers - mean) / std

    model_input = scaled[:-1]       # the first 60 minutes
    actual_next = scaled[-1]        # minute 61: the answer the model has to guess

    # time of day for each input minute (hour_sin / hour_cos), same formula as dataset.py
    hours = np.array([(minute % 86400) / 3600 for minute in minutes[:-1]])
    hour_sin = np.sin(2 * np.pi * hours / 24)
    hour_cos = np.cos(2 * np.pi * hours / 24)
    model_input = np.concatenate([model_input, hour_sin[:, None], hour_cos[:, None]], axis=1)   # 60 rows x 6 columns

    x = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0)    # add the batch dimension -> 1 x 60 x 6
    service_id_tensor = torch.tensor([service_id], dtype=torch.long)
    with torch.no_grad():
        predicted_next = model(x, service_id_tensor).squeeze(0).numpy()

    # how wrong was the prediction? (squared difference, per metric, in scaled units)
    errors = (predicted_next - actual_next) ** 2
    score = float(errors.mean())
    if not math.isfinite(score):
        return {"error": f"The score for '{service_key}' is not a number (NaN/Inf). Check the model and the data."}

    errors_by_metric = {metric: float(errors[i]) for i, metric in enumerate(METRIC_NAMES)}
    actual_real = actual_next * std + mean          # back to real units, easier to read
    predicted_real = predicted_next * std + mean
    return {
        "service": service_key,
        "anomaly_score": score,
        "per_metric_squared_error": errors_by_metric,
        "most_surprising_metric": max(errors_by_metric, key=errors_by_metric.get),
        "actual": {metric: float(actual_real[i]) for i, metric in enumerate(METRIC_NAMES)},
        "predicted": {metric: float(predicted_real[i]) for i, metric in enumerate(METRIC_NAMES)},
        "timestamp": minutes[-1],
    }


def score_all_services(model, checkpoint):
    """Score every service the model was trained on. Returns (results sorted worst first, number of errors)."""
    results = []
    errors = 0
    for service_key in sorted(trained_services(checkpoint)):
        result = score_service(service_key, model, checkpoint)
        if "error" in result:
            errors += 1
        else:
            results.append(result)
    results.sort(key=lambda r: -r["anomaly_score"])
    return results, errors


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", help="one service, for example ns-01/svc-001")
    parser.add_argument("--all", action="store_true", help="score every service the model was trained on")
    parser.add_argument("--model-name", help="MLflow model name (default: derived from --service)")
    parser.add_argument("--alias", default="candidate", help="which MLflow version to use (default: candidate)")
    parser.add_argument("--model-file", help="use a local model.pth / .pth file instead of MLflow")
    parser.add_argument("--top", type=int, default=20, help="with --all: how many services to print")
    args = parser.parse_args()

    if not args.service and not args.all:
        sys.exit("Give --service <namespace/name> or --all")

    if args.model_file:
        model_path = args.model_file
    else:
        model_name = args.model_name
        if not model_name:
            if not args.service:
                sys.exit("With --all you must also give --model-name (for example anomaly-staging-multi)")
            model_name = default_model_name(args.service)
        print(f"Downloading model '{model_name}' (alias '{args.alias}') from MLflow ...", file=sys.stderr)
        model_path = download_model_from_mlflow(model_name, args.alias)

    model, checkpoint = load_model(model_path)

    if args.service:
        result = score_service(args.service, model, checkpoint)
        print(json.dumps(result, indent=2))
        if "error" in result:
            sys.exit(1)                 # a non-zero exit code lets a scheduler see that it failed
    else:
        results, errors = score_all_services(model, checkpoint)
        print(f"Scored {len(results)} services ({errors} could not be scored). Most surprising first:\n")
        for r in results[:args.top]:
            print(f"  {r['anomaly_score']:8.3f}  {r['service']:50s} (most surprising: {r['most_surprising_metric']})")


if __name__ == "__main__":
    main()