# ML Engineering

## What

The tooling and practices that turn ML code into a working, maintainable production system: version control, containerization, serving APIs, testing, CI/CD, distributed data processing, and databases. Where [`../ml-system-design/README.md`](../ml-system-design/README.md) covers *what pieces a production ML system has*, this section covers *the concrete engineering tools you build those pieces with*.

Two topics here have their own full note plus a ported, genericized code example (see `code/`):

- [`deployment-systemd-daemons.md`](./deployment-systemd-daemons.md) — running a long-lived process as a managed OS service, graceful shutdown on SIGTERM, and how that connects to Kubernetes pod termination.
- [`environment-setup-colab-kaggle.md`](./environment-setup-colab-kaggle.md) — reproducible-*enough*-for-experimentation environment setup, contrasted with production-grade pinned/containerized reproducibility.

Everything else below is a cram-level index paragraph per topic.

---

## Git

Branching strategies for ML work the same as for regular software (trunk-based, GitHub flow, or `main`/`develop` + feature branches) — pick the lightest one the team can keep consistent. The wrinkle specific to ML: **data and model artifacts don't fit well in git** — git is built for diffable text, not multi-GB binary datasets/checkpoints, and a huge binary blob committed directly bloats the repo forever (git never forgets history). Two standard fixes: **Git LFS** (Large File Storage — stores a pointer in git, the actual blob in separate LFS storage) for occasional large files, or **DVC** (Data Version Control — versions datasets/models alongside git commits, backed by S3/GCS/etc., purpose-built for ML pipelines and integrates with experiment tracking). Common interview questions: rebase vs. merge (rebase = rewrite commit history onto a new base, linear but destructive to shared branches; merge = preserve history, adds a merge commit, safe for shared branches), how to resolve a merge conflict, when to squash commits.

## Docker

Containerizing an ML service solves two problems at once: **dependency isolation** (the notorious "works on my machine" — a specific numpy/CUDA/torch version combo that's fragile to reproduce manually) and **reproducibility across dev/prod** (the exact same image runs identically on a laptop and in the cluster — no environment drift). Minimal structure for a Python ML service:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "serve.py"]
```

Base image -> copy + install pinned deps first (maximizes Docker layer-cache hits so code-only changes don't reinstall everything) -> copy application code -> `CMD` to launch. This is the image that then gets deployed as a systemd-managed process (see [`deployment-systemd-daemons.md`](./deployment-systemd-daemons.md)) or, more commonly at scale, orchestrated by Kubernetes.

## APIs / FastAPI

FastAPI is the dominant framework for ML serving endpoints because it gives you **async support** (handle concurrent requests without blocking on I/O-bound work, e.g. a downstream feature-store lookup), **automatic OpenAPI docs** (interactive `/docs` UI generated from your endpoint signatures, no separate documentation effort), and **Pydantic validation** (request/response schemas are enforced automatically — a malformed request is rejected before it ever reaches your model code). Minimal prediction endpoint:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictRequest(BaseModel):
    features: list[float]

class PredictResponse(BaseModel):
    prediction: float

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    prediction = model.predict([req.features])[0]
    return PredictResponse(prediction=prediction)
```

## Testing

Testing ML code differs from testing regular application code because of **non-determinism** (a model's output can legitimately vary run-to-run) and **data-dependent behavior** (correctness depends on the data, not just the code). The practical split: **test the pipeline mechanics** deterministically — a feature transform on a small fixed fixture always produces the exact expected output, a data-loading function handles nulls/edge-cases correctly, a preprocessing step is idempotent — using fixed seeds and small, hand-crafted fixtures so these tests are fast and 100% reproducible. Keep that separate from **evaluating model quality**, which is inherently statistical (a metric on a held-out set, checked against a threshold with some tolerance, not asserted for exact equality) and belongs in a different kind of check (a training/eval report, not a pass/fail unit test).

## CI/CD

A CI pipeline for an ML repo typically checks: linting/formatting, unit tests (the pipeline-mechanics kind described above), and optionally a **smoke-test training run** on a tiny sample of data (a few rows, one epoch) just to confirm the training code path still executes without crashing — not to validate model quality. CD for ML has a wrinkle regular software CD doesn't: you're often deploying **two independent artifacts that can each change separately** — the serving *code* (versioned in git, deployed like any other service) and the *model artifact* (versioned separately, e.g. in a model registry) — so a deployment might ship new code with an old model, or a new model with unchanged code, and the pipeline needs to handle both cases (and roll each back independently).

## Experiment tracking / MLflow

Briefly: the "what did I try and how well did it do" side of reproducibility, versus git's "what code produced this" side. Full coverage — tools (MLflow, Weights & Biases, DVC) and what metadata to track — lives in [`../ml-system-design/README.md#model-versioning--experiment-tracking`](../ml-system-design/README.md#model-versioning--experiment-tracking) to avoid duplicating it here.

## Spark / PySpark / distributed computing

Needed once data no longer fits comfortably in memory on a single machine (pandas' practical ceiling is roughly "fits in RAM"; Spark scales horizontally across a cluster). Core concepts: **DataFrame** (Spark's structured, distributed table abstraction — a superset replacement for the older, lower-level **RDD**); **lazy evaluation** (transformations like `.filter()`/`.select()` build up a query plan but don't execute until an *action* like `.collect()`/`.write()` is called, letting Spark optimize the whole plan before running it); **partitioning** (data is split across the cluster into partitions, and how well-partitioned your data is directly drives parallelism and shuffle cost). Illustrative snippet:

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("features").getOrCreate()
df = spark.read.parquet("s3://bucket/events/")

feature_df = (
    df.filter(F.col("event_type") == "purchase")
      .groupBy("user_id")
      .agg(F.count("*").alias("purchase_count"),
           F.sum("amount").alias("total_spend"))
)
feature_df.write.mode("overwrite").parquet("s3://bucket/features/user_purchase_stats/")
```

Nothing above actually runs until `.write()` (the action) is called — the `filter`/`groupBy`/`agg` chain is lazily planned first.

## Databases

The practical DS/ML angle: most feature engineering starts with **reading from a data warehouse** (Snowflake, BigQuery, Redshift) via SQL — see [`../sql/`](../sql/) (joins, window functions, CTEs) for the query side of this, which is directly reusable here. **ORMs vs. raw SQL**: an ORM (SQLAlchemy, Django ORM) is convenient for application CRUD code and keeps queries Pythonic/composable, but for the large aggregate/analytical queries typical of ML feature pipelines, raw SQL (or a query builder) is usually clearer and lets the database's own optimizer do more work — most ML pipelines lean toward raw SQL or a dataframe library (pandas/Spark) reading query results, rather than an ORM. **Connection pooling**: opening a new DB connection per request/job is expensive; a connection pool (maintained by the DB client library, e.g. SQLAlchemy's pool, or a dedicated pooler like PgBouncer) reuses a fixed set of open connections across requests, which matters once an inference service is issuing per-request feature lookups against an online store/DB.

## Deployment

Tying the whole section together — the typical path an ML system takes from prototype to production:

1. **Notebook**: exploratory, unpinned, interactive — fine for prototyping, not for anything else (see [`environment-setup-colab-kaggle.md`](./environment-setup-colab-kaggle.md)).
2. **Script**: the notebook's logic extracted into a plain `.py` script/module — testable, versionable in git, runnable non-interactively.
3. **Containerized service**: the script wrapped in a FastAPI (or similar) server, packaged into a Docker image with pinned dependencies — reproducible across machines (see Docker above).
4. **Orchestrated deployment**: the container run and managed at scale by an orchestrator — conceptually, **Kubernetes** schedules containers onto a cluster, restarts them on failure, load-balances traffic across replicas, and (as covered in [`deployment-systemd-daemons.md`](./deployment-systemd-daemons.md)) sends SIGTERM before SIGKILL on shutdown — the same signal-handling discipline a systemd-managed process needs, one layer up the stack. See [`../ml-system-design/README.md#model-serving`](../ml-system-design/README.md#model-serving) for canary/shadow/blue-green rollout strategies at this stage.

## See also

- [`../ml-system-design/README.md`](../ml-system-design/README.md) — the system-design view this tooling supports
- [`../sql/`](../sql/) — query skills for the data-warehouse side of feature pipelines
- [`../python/`](../python/) — language-level fundamentals (typing, decorators, context managers) used throughout the code here
