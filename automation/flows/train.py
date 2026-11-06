"""
train.py

Trains MetricsTransformer on the full accumulated archive, using a
chronological (not random) train/val split to avoid leakage.

--quick flag: for a FAST sanity check before committing to the full run.
It does NOT use fake or synthetic data — it's a random SAMPLE of your real,
already-collected, gap-checked, chronologically-split windows, trained for
just 1 epoch. The goal is purely "does the whole real pipeline actually work
end to end" (no crashes, loss genuinely drops) in a couple of minutes,
before spending 10-25 hours on the full real run.

Usage:
    python3 train.py            # the real, full run — all data, 20 epochs
    python3 train.py --quick    # fast sanity check — real data, small sample, 1 epoch
    python3 train.py --services "payments/payment-api" --download      # one service, data from S3

It is also a Prefect flow (deployment entrypoint: flows/train.py:train). A real run needs
MLFLOW_TRACKING_URI: it is logged to MLflow and registered as a new model version with alias "candidate".
--quick does not need MLflow and is never logged.
"""

import argparse
import inspect
import json
import math
import os
import platform
import random
import subprocess
import tempfile
from typing import Optional

import numpy as np
import torch
from prefect import flow
from torch.utils.data import DataLoader, Subset

from dataset import MultiServiceMetricsDataset, download_from_s3
from dataset import download_specific_service as download_service_files   # renamed: the flow parameter has the same name
from model import MetricsTransformer
from config import WINDOW_SIZE, METRIC_NAMES, BUCKET_NAME, ENVIRONMENT

FEATURE_VERSION = "v1"
FEATURES = list(METRIC_NAMES) + ["hour_sin", "hour_cos"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def check_mlflow():
    """Fail NOW if MLflow is unreachable, not after hours of training."""
    import mlflow
    from mlflow import MlflowClient
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "3")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    try:
        client = MlflowClient()
        client.search_experiments(max_results=1)
        client.search_registered_models(max_results=1)
    except Exception as e:
        raise RuntimeError(f"MLflow at {os.environ['MLFLOW_TRACKING_URI']} is not usable: {e}. "
                           f"Fix this first; nothing has been trained yet.") from e


def log_to_mlflow(checkpoint, history, experiment, model_name):
    """One MLflow run + one registered model version (alias 'candidate'). Nothing goes live by itself:
    you promote a version by giving it the alias 'production' in the MLflow UI."""
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment)
    keys = checkpoint["trained_services"]
    label = keys[0].replace("/", "-") if len(keys) == 1 else f"{len(keys)}-services"
    with mlflow.start_run(run_name=label) as run:
        mlflow.log_params({
            "services": ",".join(keys)[:250], "n_services_trained": len(keys), "seed": checkpoint["seed"],
            "window_size": checkpoint["window_size"], "feature_version": checkpoint["feature_version"],
            "epochs_run": len(history), "n_train_windows": checkpoint["n_train_windows"],
            "n_val_windows": checkpoint["n_val_windows"], "git_commit": checkpoint["git_commit"] or "unknown",
        })
        for h in history:
            mlflow.log_metrics({"train_loss": h["train_loss"], "val_loss": h["val_loss"]}, step=h["epoch"])
        mlflow.log_metrics({"best_val_loss": checkpoint["best_val_loss"], "best_epoch": checkpoint["best_epoch"]})
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "model"))
            torch.save(checkpoint, os.path.join(tmp, "model", "model.pth"))
            mlflow.log_artifacts(os.path.join(tmp, "model"), artifact_path="model")
        run_id, artifact_uri = run.info.run_id, run.info.artifact_uri

    if model_name is None:
        model_name = (f"anomaly-{ENVIRONMENT}-" + keys[0].replace("/", "-")) if len(keys) == 1 else f"anomaly-{ENVIRONMENT}-multi"
    client = MlflowClient()
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name)
    mv = client.create_model_version(model_name, source=f"{artifact_uri}/model", run_id=run_id,
                                     tags={"services": ",".join(keys)[:250], "feature_version": FEATURE_VERSION,
                                           "best_val_loss": f"{checkpoint['best_val_loss']:.4f}"})
    client.set_registered_model_alias(model_name, "candidate", mv.version)
    return {"run_id": run_id, "model_name": model_name, "model_version": mv.version}


@flow(name="train", log_prints=True)
def train(services: str = "", quick: bool = False, quick_train_size: int = 20000, quick_val_size: int = 2000,
          epochs: Optional[int] = None, patience: int = 5, seed: int = 42, download: bool = False,
          download_specific_service: bool = False,
          experiment: str = "istio-anomaly-detection", model_name: Optional[str] = None):
    """services: comma-separated 'namespace/name' keys; empty = all services.
    download: fetch data from s3://<bucket>/<env>/ first (needed in a fresh Prefect pod).
    download_specific_service: with download, fetch only the files of `services` instead of everything."""
    set_seed(seed)
    if not quick:
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            raise RuntimeError("MLFLOW_TRACKING_URI is not set. A real training run must be tracked in MLflow, "
                               "otherwise the model is lost when the pod ends. (--quick does not need it.)")
        check_mlflow()
    if download:
        requested = [s.strip() for s in services.split(",") if s.strip()]
        if download_specific_service and requested:
            download_service_files(BUCKET_NAME, ENVIRONMENT, requested)
        else:
            if download_specific_service:
                print("download_specific_service is set but no services were given: downloading everything.")
            download_from_s3(BUCKET_NAME, ENVIRONMENT)

    print("Loading dataset from the accumulated archive...")
    full_dataset = MultiServiceMetricsDataset(window_size=WINDOW_SIZE, val_fraction=0.1)
    n_services_total = len(full_dataset.service_map)
    n_services_usable = n_services_total - len(full_dataset.skipped_services)

    print(f"Total windows: {len(full_dataset)}")
    print(f"Services with enough data to train on: {n_services_usable} / {n_services_total}")
    if len(full_dataset) == 0:
        raise RuntimeError("No training windows: is the data downloaded? (download=True, or run dataset first)")

    train_indices, val_indices = full_dataset.chronological_split()
    print(f"Chronological split: {len(train_indices)} train windows, {len(val_indices)} val windows "
          f"(purge buffer of {WINDOW_SIZE} windows dropped at each service's boundary)")

    if services:
        wanted_keys = [s.strip() for s in services.split(",") if s.strip()]
        valid_keys = [k for k in wanted_keys if k in full_dataset.service_map]
        missing = [k for k in wanted_keys if k not in full_dataset.service_map]
        if missing:
            print(f"WARNING: not found in this archive, skipped: {missing}")
        too_short = [k for k in valid_keys if full_dataset.service_map[k] not in full_dataset.norm_stats]
        if too_short:
            print(f"WARNING: not enough history yet, skipped: {too_short}")
        valid_keys = [k for k in valid_keys if k not in too_short]
        if not valid_keys:
            raise ValueError(f"None of {wanted_keys} can be trained: not in _service_map.json (use the exact "
                             f"'namespace/name' key) or not enough history yet.")
        wanted_ids = set(full_dataset.service_map[k] for k in valid_keys)

        train_indices = [i for i in train_indices if full_dataset.windows[i][2] in wanted_ids]
        val_indices = [i for i in val_indices if full_dataset.windows[i][2] in wanted_ids]
        num_epochs = epochs or 20   # FULL epoch count — this is a real, small-scope model

        print(f"\n--services mode: training on ALL real windows from {valid_keys}")
        print(f"  ({len(train_indices)} real train windows, {len(val_indices)} real val windows, "
              f"{num_epochs} full epochs)")
        print("This IS a real, evaluable model for these specific services — run score.py")
        print("against it once training finishes.\n")

    elif quick:
        # RANDOM SAMPLE across ALL services — real data, but too thin per-service
        # to be meaningfully evaluated. This is a BUG CHECK ONLY, not a testable model.
        random.seed(0)   # reproducible sample, so re-running --quick is comparable
        train_indices = random.sample(train_indices, min(quick_train_size, len(train_indices)))
        val_indices = random.sample(val_indices, min(quick_val_size, len(val_indices)))
        num_epochs = epochs or 1
        print(f"\n--quick mode: sampled down to {len(train_indices)} real train windows, "
              f"{len(val_indices)} real val windows, {num_epochs} epoch(s).")
        print("This is a BUG CHECK ONLY — confirms the pipeline runs without crashing.")
        print("It does NOT produce a model worth running score.py against (too little")
        print("data per service). Use --services instead for a real, testable result.\n")
    else:
        num_epochs = epochs or 20

    if not train_indices or not val_indices:
        raise RuntimeError("No train or validation windows for this selection: collect more data first.")

    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)


    """
    For example, Batch 1 contains 32 examples:

    Object:      What it contains
    window:      The 32 historical metric windows
    target:      The 32 corresponding prediction targets
    service_id:  The 32 corresponding service IDs
    """    

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # scale progress-print frequency to THIS run's actual size — aim for
    # roughly 5-10 progress lines per epoch, whether that's a 119-batch
    # --services run or a 57,121-batch full run
    print_every = max(1, len(train_loader) // 8)

    model = MetricsTransformer(num_services=n_services_total)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    best_val, best_epoch, best_state, bad_epochs = float("inf"), 0, None, 0
    history = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_num, (window, target, service_id) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = model(window, service_id)
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_num % print_every == 0:
                print(f"  epoch {epoch + 1}, batch {batch_num}/{len(train_loader)}, "
                      f"running_loss={total_loss / (batch_num + 1):.4f}", flush=True)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for window, target, service_id in val_loader:
                pred = model(window, service_id)
                val_loss += torch.nn.functional.mse_loss(pred, target).item()

        avg_train = total_loss / max(len(train_loader), 1)
        avg_val = val_loss / max(len(val_loader), 1)
        if not math.isfinite(avg_val) and best_state is None:
            raise RuntimeError(f"Validation loss is {avg_val} after epoch {epoch + 1}. Usually a NaN/Inf value "
                               f"in the archive files for this service. Not saving a model.")
        history.append({"epoch": epoch + 1, "train_loss": avg_train, "val_loss": avg_val})
        marker = ""
        if avg_val < best_val:
            best_val, best_epoch, bad_epochs = avg_val, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = "  <- best so far"
        else:
            bad_epochs += 1
        print(f"Epoch {epoch + 1:2d}/{num_epochs}: train_loss={avg_train:.4f}  val_loss={avg_val:.4f}{marker}")
        if bad_epochs >= patience:
            print(f"Early stop: validation loss has not improved for {patience} epochs.")
            break

    model.load_state_dict(best_state)
    print(f"Keeping epoch {best_epoch} (val_loss={best_val:.4f}), not the last epoch.")

    if services:
        out_path = "metrics_model_services.pth"
    elif quick:
        out_path = "metrics_model_quick.pth"
    else:
        out_path = "metrics_model.pth"

    trained_services = valid_keys if services else [k for k, sid in full_dataset.service_map.items()
                                                    if sid in full_dataset.norm_stats]
    # same keys as before (score.py reads them) plus what is needed to reproduce / audit the model
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_services": n_services_total,
        "service_map": full_dataset.service_map,
        "norm_stats": full_dataset.norm_stats,
        "window_size": WINDOW_SIZE,
        "feature_version": FEATURE_VERSION,
        "features": FEATURES,
        "trained_services": trained_services,
        "model_config": {k: p.default for k, p in inspect.signature(MetricsTransformer.__init__).parameters.items()
                         if p.default is not inspect.Parameter.empty},
        "seed": seed, "best_epoch": best_epoch, "best_val_loss": best_val,
        "n_train_windows": len(train_indices), "n_val_windows": len(val_indices),
        "git_commit": git_commit(), "torch": str(torch.__version__), "python": str(platform.python_version()),
    }
    torch.save(checkpoint, out_path)
    print(f"\nSaved {out_path}")
    if quick:
        print("This was a quick sanity-check model — once you trust the pipeline, "
              "run `python3 train.py` (no --quick) for the real, full training run.")
        return {"saved": out_path}

    info = log_to_mlflow(checkpoint, history, experiment, model_name)
    print(f"MLflow: {info['model_name']} version {info['model_version']} (alias 'candidate'), run {info['run_id']}")
    return {"saved": out_path, "best_epoch": best_epoch, "best_val_loss": best_val, "mlflow": info}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                         help="Fast sanity check: real data, random sample, 1 epoch. "
                              "Use this before committing to the full multi-hour run.")
    parser.add_argument("--quick-train-size", type=int, default=20000,
                         help="How many real training windows to sample in --quick mode.")
    parser.add_argument("--quick-val-size", type=int, default=2000,
                         help="How many real validation windows to sample in --quick mode.")
    parser.add_argument("--epochs", type=int, default=None,
                         help="Override the number of epochs (default: 1 for --quick, 20 for full).")
    parser.add_argument("--services", type=str, default="",
                         help="Comma-separated real service keys (e.g. 'payments/payment-api'). "
                              "Trains on ALL of just these services' real windows, for the FULL epoch count.")
    parser.add_argument("--patience", type=int, default=5, help="stop after this many epochs without improvement")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true", help="download data from S3 first")
    parser.add_argument("--download-specific-service", action="store_true",
                         help="with --download: fetch only the files of --services, not everything")
    parser.add_argument("--experiment", default="istio-anomaly-detection")
    parser.add_argument("--model-name", default=None, help="MLflow registered model name (default: derived)")
    args = parser.parse_args()
    train(services=args.services, quick=args.quick, quick_train_size=args.quick_train_size,
          quick_val_size=args.quick_val_size, epochs=args.epochs, patience=args.patience, seed=args.seed,
          download=args.download, download_specific_service=args.download_specific_service,
          experiment=args.experiment, model_name=args.model_name)


if __name__ == "__main__":
    main()