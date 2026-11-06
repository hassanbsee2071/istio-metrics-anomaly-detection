# Istio Metrics Anomaly Detection

A small transformer model that learns the normal traffic pattern of every service behind
an Istio mesh — request rate, 5xx rate, 4xx rate, and p99 latency — and flags the minutes
that don't look like anything it has seen before.

Instead of static thresholds ("page me if p99 > 500ms"), the model is trained to *predict*
the next minute of a service's metrics from its last 60 minutes. When the real value is far
from the prediction, that minute gets a high **anomaly score** — a `rate` that stays flat is
boring no matter how low it is, but a drop that breaks the usual pattern is not.

> **Status:** a personal / learning project (see
> [`create_llm_from_scratch`](..), the parent folder it lives in), run against one real
> Istio + VictoriaMetrics environment. It works, but it's not a maintained product — see
> [Limitations](#limitations--honest-caveats) before relying on it for anything.


## How it works

```
VictoriaMetrics (Istio metrics)
        │  PromQL queries, every N hours
        ▼
   collect.py  ──► per-service, per-metric JSON files ──► S3 (durable archive)
        │                                                     │
        ▼                                                     ▼
  dataset.py  (merges each service's history into          download for
   fixed-size sliding windows, normalized per service)      training pods
        │
        ▼
   train.py  ──► MetricsTransformer (model.py) ──► checkpoint (.pth) ──► MLflow
                  ONE shared model, many services                     (versioned,
                  via a learned per-service embedding                  "candidate"/
                                                                        "production")
        │
        ▼
   score.py  ──► pulls the latest 61 minutes, predicts minute 61,
                  compares it to what really happened ──► anomaly_score
```

- **One global model, not one model per service.** `MetricsTransformer` is a small
  decoder-only transformer (2 layers, 2 heads, 32-dim embeddings — intentionally tiny) that
  takes a 60-minute window of `[rate, error_5xx, error_4xx, p99, hour_sin, hour_cos]` and
  predicts minute 61's `[rate, error_5xx, error_4xx, p99]`. A learned per-service embedding
  is concatenated into the input so the one model can specialize its predictions per
  service while still sharing general traffic-pattern knowledge (daily/weekly seasonality,
  what a typical error spike looks like) across all of them.
- **Anomaly score = prediction error.** `score.py` compares the model's prediction for
  "now" against what actually happened, in normalized units. No labeled anomalies are
  needed anywhere in this pipeline — it's self-supervised next-step prediction.
- **Data lives in S3, not just on disk.** `collect.py` is designed to run on a schedule
  (every 6 hours by default) and append new points to a durable, append-only JSON archive
  per service/metric, uploaded to S3 after each run. `dataset.py` and `train.py` download
  that archive to train on the full accumulated history, not just whatever happens to be on
  the local disk.
- **MLflow is the handoff point between training and scoring.** `train.py` always creates a
  new model version tagged `candidate` — nothing goes live automatically. You look at the
  run in MLflow and promote a version to the `production` alias yourself; `score.py`
  defaults to reading whatever has that alias.

## Repository layout

```
istio-metrics-anomaly-detection/
├── automation/
│   ├── flows/
│   │   ├── config.py     # env-var based config: VM host, data dir, feature list, etc.
│   │   ├── collect.py    # Prefect flow: PromQL → per-service JSON files → S3
│   │   ├── dataset.py    # merges per-service history into normalized sliding windows
│   │   ├── model.py      # MetricsTransformer (the model architecture)
│   │   ├── train.py      # Prefect flow: trains + logs to MLflow
│   │   ├── score.py      # CLI: anomaly score for one service or all trained services
│   │   └── hello.py      # minimal Prefect flow, used to sanity-check a new deployment
│   ├── prefect.yaml       # Prefect deployment definitions (collect + train)
│   └── requirements.txt
├── sample-data/
│   └── staging/           # ~5 weeks of real, anonymized per-service metrics (see below)
├── mlflow/                 # Kubernetes manifests to self-host MLflow
├── prefect/                 # Kubernetes manifests to self-host Prefect server + worker
└── README.md
```

`sample-data/` is anonymized output from a real `collect.py` run: namespace and service
names are replaced with `ns-NN` / `svc-NNN` throughout (including inside
`_service_map.json`), so you can run the whole pipeline — `dataset.py`, `train.py`,
`score.py` against a local model file — without access to any real cluster. It does **not**
include anything that lets you reconstruct real service or namespace names.

## Quickstart — try it with the bundled sample data

This gets you a trained model and a scored anomaly, without touching VictoriaMetrics, S3,
Prefect, or MLflow.

```bash
git clone https://github.com/hassanbsee2071/istio-metrics-anomaly-detection.git
cd istio-metrics-anomaly-detection/automation
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Point the pipeline at the bundled sample data instead of a live cluster
export ANOMALY_DATA_DIR="$(pwd)/../sample-data"
export ENVIRONMENT=staging

cd flows
python3 train.py --quick
```

`--quick` is a fast sanity check (1 epoch, a random sample of windows) that confirms the
whole pipeline runs end to end in a couple of minutes. It is **not** a usable model — see
the warning it prints. To get something you can actually evaluate, train on one or two
specific services for the full epoch count:

```bash
python3 train.py --services "ns-01/svc-001"
```

This needs `MLFLOW_TRACKING_URI` set (training always logs to MLflow — see
[Scheduling it with Prefect](#scheduling-it-with-prefect) for how to get one running
locally in a few minutes if you don't already have one). Once it finishes, score the
service it just trained on:

```bash
python3 score.py --service ns-01/svc-001 --alias candidate \
  --model-name anomaly-staging-ns-01-svc-001
```

> Note: `score.py` calls VictoriaMetrics for the *latest* data, so it needs `VM_HOST` to
> point at a real, currently-running VictoriaMetrics instance — the bundled sample data is a
> fixed historical window and can't be used to simulate "the last 61 minutes, right now."
> Training against the sample data works fully offline; scoring does not.

## Running it against your own cluster

Everything is configured through environment variables, read in `automation/flows/config.py`
(an `.env` file in `automation/flows/` is also picked up automatically via `python-dotenv`):

| Variable | Default | What it's for |
|---|---|---|
| `VM_HOST` | *(none — set your own)* | Your VictoriaMetrics/Prometheus-compatible query endpoint, e.g. `https://your-vm-host` |
| `VM_AUTH_TOKEN` | unset | Bearer token, if your endpoint needs auth |
| `ANOMALY_DATA_DIR` | `data` | Local working directory for pulled metrics |
| `VM_RETENTION_SECONDS` | `345600` (4 days) | How far back `collect.py` will look for new data on a fresh checkpoint |
| `ANOMALY_WINDOW_SIZE` | `60` | Minutes of history the model sees before predicting the next minute |
| `ANOMALY_BUCKET_NAME` | `istio-anomaly-detection` | S3 bucket used as the durable archive |
| `ENVIRONMENT` | `staging` | Namespacing prefix inside the S3 bucket and local data dir |
| `AWS_PROFILE` | `default` | Fallback AWS credentials profile if IAM role credentials (IRSA) aren't available |
| `MLFLOW_TRACKING_URI` | *(none — set your own)* | Required for `train.py` (non-`--quick`) and `score.py` |

Requirements:
- A VictoriaMetrics (or compatible Prometheus-remote-read) instance scraping Istio's
  `istio_requests_total` and `istio_request_duration_milliseconds_bucket` metrics with the
  standard `destination_service_name` / `destination_service_namespace` / `response_code`
  labels — this is the default shape Istio's own telemetry produces, so most meshes need no
  extra configuration.
- An S3 bucket, and either an IAM role (IRSA, if running in EKS) or a local AWS profile with
  read/write access to it.
- Python 3.10+.

Then, one-off:

```bash
cd automation/flows
python3 collect.py          # discovers active services, pulls new metrics, uploads to S3
python3 train.py --download # downloads the full S3 archive first, then trains on everything
python3 score.py --all --model-name anomaly-<environment>-multi
```

`collect.py` auto-discovers every service currently receiving traffic (via a live PromQL
query) — we don't need to list services by hand. It deliberately excludes the
ingress/gateway layer itself (`ingress-nginx`, `kong`, `kgateway`, and Istio's internal
passthrough/blackhole clusters), since those aren't "services" in the sense this model
cares about.

## Scheduling it with Prefect

The flows in `automation/flows/` (`collect`, `train`, `hello`) are also deployable as
[Prefect](https://www.prefect.io/) deployments — `automation/prefect.yaml` defines a
`collect` deployment on a 6-hour schedule and a manually-triggered `train` deployment.

The quickest way to get a Prefect server + MLflow running locally for development:

```bash
pip install prefect mlflow
prefect server start &            # http://127.0.0.1:4200
mlflow server --host 0.0.0.0 --port 5000 &   # http://127.0.0.1:5000
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

To actually deploy the scheduled flows (rather than run them ad hoc), **fork this repo
first** — `prefect.yaml`'s `pull` step clones from a specific GitHub URL at deploy time, so
point it at your own fork before running `prefect deploy`:

```yaml
pull:
  - prefect.deployments.steps.git_clone:
      repository: git@github.com:<your-github-username>/istio-metrics-anomaly-detection.git
      branch: main
```

```bash
cd automation
prefect deploy --all
```

A Prefect worker then needs to be running against the `default` work pool to actually pick
up and execute the scheduled runs (`prefect worker start --pool default`, or see the
Kubernetes worker deployment below).

## Optional: the full Kubernetes deployment (Prefect + MLflow)

`mlflow/` and `prefect/` contain the plain Kubernetes manifests (exported from their
respective Helm charts) this project's own instance runs on: a self-hosted MLflow server
backed by Postgres with artifacts on a PVC, and a self-hosted Prefect server + worker that
clones this repo over SSH to execute scheduled flows, routed through
[Gateway API](https://gateway-api.sigs.k8s.io/) `HTTPRoute`s (for a
[kgateway](https://kgateway.dev/) internal gateway named `kgateway-gw-internal` — swap the
`parentRefs` in `mlflow/httproute.yaml` / `prefect/httproute.yaml` for whatever ingress
controller or gateway you actually run).

This is genuinely optional — everything above runs perfectly well with `mlflow server`/
`prefect server start` on a laptop, or any managed MLflow/Prefect Cloud offering. Only reach
for these manifests if you want the same always-on, scheduled setup this project itself
uses.

**Before applying any of this, see the [checklist below](#️-before-you-publish-or-fork-this)
— several of these manifests reference a personal hostname and ship template
credentials that must be replaced with your own**, specifically:

1. Create your own Postgres credentials and update `mlflow/postgress-secret.yaml` and
   `prefect/postgres-secret.yaml` / `prefect/prefect-secret.yaml` accordingly (or let your
   Postgres Helm chart generate them, rather than reusing the values committed here).
2. Generate a deploy key for your fork and populate
   `prefect/personal-github-ssh-key-secret.yaml` from it, as instructed in the comment at
   the top of that file — never commit the actual private key material.
3. Set `eks.amazonaws.com/role-arn` in `prefect/prefect-worker-sa.yaml` to an IAM role you
   control (it ships as `arn:aws:iam::YOUR_AWS_ACCOUNT_ID:role/prefect-worker`, a
   placeholder) if the worker needs AWS access (e.g. to read/write the S3 bucket above).
4. Replace every `*.syed.stream` hostname (`mlflow/httproute.yaml`,
   `mlflow/deployment.yaml`'s `--allowed-hosts`, `prefect/httproute.yaml`, and
   `prefect/prefect-server.yaml`'s `PREFECT_UI_API_URL`) with your own domain.

## Data format reference

Under `<ANOMALY_DATA_DIR>/<ENVIRONMENT>/`:

- `<date>/<namespace>__<name>__<metric>.json` — one file per service per metric per day,
  written by `collect.py`. Each file is a JSON array of `[unix_timestamp, value]` pairs,
  one per minute, e.g.:
  ```json
  [[1791210120.0, 0.4], [1791210180.0, 0.4], [1791210240.0, 0.38]]
  ```
  `metric` is one of `rate` (requests/sec), `error5xx` / `error4xx` (fraction of requests,
  0–1), or `p99` (milliseconds).
- `_service_map.json` — `{"namespace/name": integer_id}`, stable across runs; the
  integer id is what the model's per-service embedding actually indexes on.
- `_last_pulled.json` — `{"namespace/name": last_pulled_unix_timestamp}`, the checkpoint
  `collect.py` resumes from on its next run so it never re-queries data it already has.
- `combined_metrics/` — built by `dataset.py` at training time: one merged, deduplicated
  time series per service/metric across every date-folder found on disk. Safe to delete;
  it's regenerated from the daily files every time.

## Model and feature details

- **Features (fixed order, enforced by `score.py` at load time):** `rate`, `error5xx`,
  `error4xx`, `p99`, `hour_sin`, `hour_cos`. The two time-of-day features let the model learn
  daily seasonality (e.g. "traffic is always low at 3am") without needing day-of-week or
  calendar features, which would require much more history to learn reliably.
- **Normalization is per-service, computed on the training split only.** Each service's
  mean/std for the 4 real metrics is computed from its own *training* portion of the data
  (never the validation portion, to avoid leakage) and saved inside the model checkpoint, so
  `score.py` always scales live data the exact same way training did.
- **Chronological, purged train/val split.** `dataset.py` splits each service's windows by
  time (not randomly) and drops a `window_size`-sized buffer at the boundary, so no
  validation window shares timestamps with a training window — a random split would leak
  future information into training.
- **A window is only used if every minute inside it is contiguous** (exactly 60 seconds
  apart, start to end). Any gap in the underlying data — a missed scrape, a restart, a
  `collect.py` outage — causes that window to be silently dropped rather than training on
  an artificially "smoothed over" sequence.
- **Model size is intentionally small** (32-dim embeddings, 2 transformer blocks, 2 attention
  heads) — this is forecasting 4 numbers from 60 timesteps per service, not language
  modeling; a much larger model would overfit long before it generalized.

## Limitations / caveats

- **This detects "does the next minute look like what I'd expect," not "is this good or
  bad."** A sudden, real, intentional spike in legitimate traffic will score as anomalous
  exactly like a real incident would — the model has no notion of *why* something looks
  different, only *that* it does.
- **`--quick` training mode produces a bug-check model, not an evaluable one** — it trains on
  a random sample across every service for a single epoch, specifically to confirm the
  pipeline runs without crashing. Don't score against a `--quick` checkpoint and draw
  conclusions from the numbers.
- **A service needs enough contiguous history to be trained on at all** (`window_size + 10`
  minutes of gap-free data, by default 70 minutes' worth, accumulated over however many
  `collect.py` runs it takes) — a brand-new or very low-traffic service will be silently
  skipped (`dataset.py` reports this as `skipped_services`) until it has enough history.
- **One shared model, trained at a point in time.** It doesn't retrain itself as traffic
  patterns drift; that's a deliberate manual step (re-run `train.py`, review the new
  `candidate` version in MLflow, re-promote to `production` yourself) rather than an
  automatic online-learning loop.
- **No alerting is built in.** `score.py` prints a number; wiring that into PagerDuty, Slack,
  or anywhere else is left to you.
- This was built and tuned against one real environment's traffic patterns and has not been
  validated against meshes with very different characteristics (e.g. extremely bursty
  batch/cron-driven traffic, or services with sub-second request lifetimes at very high
  cardinality).

