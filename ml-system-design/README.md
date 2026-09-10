# ML System Design

## What

ML system design interviews test whether you can take a vague product goal ("recommend products", "detect fraud") and design the *full* pipeline around a model — not just the model itself: how data flows in, how features get computed consistently between training and serving, how the model gets deployed and rolled out safely, and how you know it's still working a month later. This note is a single cram sheet covering the pieces that recur across almost every ML system design question.

## Why

Unlike a pure ML/stats interview, this round is graded on breadth and tradeoff reasoning, not depth on any one algorithm. Interviewers are checking whether you think about the system end-to-end: data -> features -> training -> serving -> monitoring -> retraining, as a loop, not a one-shot pipeline.

---

## Data pipeline

- **Batch ETL**: extract-transform-load run on a schedule (hourly/daily), reading from a data warehouse/lake, writing back transformed tables. Simple, easy to reason about, but data is stale between runs.
- **Streaming ingestion**: events (Kafka, Kinesis, Pub/Sub) processed as they arrive, enabling near-real-time features/labels. More complex (exactly-once semantics, windowing, out-of-order events) but necessary when freshness matters (fraud, real-time bidding).
- **Data validation at ingestion**: schema checks, null-rate checks, range/type checks, cardinality checks — run *before* data enters the pipeline (e.g. Great Expectations, TFX Data Validation) so a broken upstream source fails loudly instead of silently poisoning downstream features/models.
- **Schema evolution**: upstream producers change fields over time (new column, renamed field, type change). Handle with a schema registry, backward/forward-compatible schema changes (e.g. only add optional fields, never repurpose an existing field's meaning), and versioned schemas so old and new data can coexist during a migration.

## Feature pipeline

- **Feature store**: a system that computes features once and serves them consistently to both training and serving, avoiding training/serving skew from two independent implementations of the same logic.
  - **Offline store**: large-scale, historical, used for generating training datasets (e.g. a data warehouse table, Parquet on S3).
  - **Online store**: low-latency key-value lookup (e.g. Redis, DynamoDB) used to fetch current feature values at inference time within a few milliseconds.
- **Point-in-time correctness**: when building a training set, a feature value must reflect what was known *at the time the label was generated* — not a later, "future" value. Example: a "user's total purchases" feature must use the count as of the moment of the labeled event, not the count as of today, or the model implicitly leaks the outcome (this is a specific, sneaky form of **data leakage** and one of the most common ways offline training accuracy is inflated relative to what the model actually achieves in production).
- **Feature versioning**: feature definitions change over time (a bug fix, a new aggregation window). Version them so a model trained against v1 of a feature can be reproduced/debugged/rolled back independently of v2's rollout, and so online/offline stores agree on which definition is currently "live."

## Training pipeline

- **Offline training jobs**: run on a schedule or triggered, typically on a fixed, versioned snapshot of training data — not live-queried each time, for reproducibility.
- **Data versioning**: pin the exact dataset (e.g. via DVC or a warehouse table snapshot/partition) used for a given model, so "which data trained this model" is always answerable.
- **Reproducibility**: fix random seeds (numpy/torch/sklearn), pin dependency versions (`requirements.txt`/lockfile or a Docker image — see [`../ml-engineering/README.md`](../ml-engineering/README.md#docker)), and log the exact code version (git commit hash) alongside the run.
- **Retraining triggers**:
  - **Scheduled**: retrain weekly/monthly regardless of performance — simple, predictable cost, but wasteful if nothing changed (or too slow if something did).
  - **Drift-triggered**: retrain when monitoring detects meaningful data or concept drift (see below) — more efficient, but needs reliable drift detection to avoid false-alarm retrains.

## Batch vs. online inference

| | Batch inference | Online inference |
|---|---|---|
| When predictions are made | Precomputed on a schedule, stored, served on read | Computed per-request, in real time |
| Latency requirement | None — minutes/hours is fine | Low (tens to low-hundreds of ms) |
| Infra cost/complexity | Cheap — just a scheduled job | Needs a low-latency serving stack (see below) |
| Freshness | Only as fresh as the last batch run | Always reflects the current request's inputs |
| Example use cases | Nightly churn-risk scoring for all users; monthly credit-risk re-scoring; precomputed "customers who might like X" email recommendations | Fraud detection on a live transaction; search ranking for a live query; a chatbot's next-turn response; real-time ad bidding |

Rule of thumb: use batch whenever the product can tolerate stale predictions (saves infra cost and complexity); use online only when the prediction genuinely depends on information only available at request time, or the product needs a fresh answer to *this* request right now.

## Model serving

- **REST/gRPC endpoints**: REST (JSON over HTTP) is simpler and more universally compatible; gRPC (protobuf over HTTP/2) is faster and strongly typed, common for internal service-to-service calls at higher throughput.
- **Model server frameworks**: purpose-built servers handle batching, versioning, and hardware acceleration so you don't hand-roll a serving loop — **TorchServe** (PyTorch), **TensorFlow Serving** (TF), **Triton Inference Server** (NVIDIA, multi-framework, GPU-optimized), **BentoML** (framework-agnostic, packages model + code + deps as a deployable "bento").
- **Containerized deployment**: package the model server + model artifact + dependencies into a Docker image so it deploys identically across environments — see [`../ml-engineering/README.md`](../ml-engineering/README.md#docker) for the general pattern, and [`../ml-engineering/deployment-systemd-daemons.md`](../ml-engineering/deployment-systemd-daemons.md) for how a long-running server process should handle shutdown signals within that container.
- **Safe rollout strategies** for a new model version:
  - **Canary**: route a small % of live traffic to the new model, watch metrics, ramp up gradually if healthy.
  - **Shadow**: send the new model a copy of live traffic *without* using its output — compare its predictions against the current model offline, zero user-facing risk.
  - **Blue-green**: run old ("blue") and new ("green") versions fully in parallel behind a router; switch all traffic at once, keep blue ready for instant rollback.

## Caching

- **Prediction caching**: cache the model's output for a given input (or input hash) so repeat requests skip inference entirely — effective when the same inputs recur often (e.g. popular product recommendations) and predictions don't need to be instantaneous-fresh.
- **Feature caching**: cache expensive-to-compute features (the online feature store itself is effectively this) to avoid recomputing them per request.
- **Cache invalidation**: the classic hard problem — if underlying data changes (user behavior, item catalog), a stale cached prediction/feature becomes actively wrong. Mitigate with a TTL (accept some staleness), explicit invalidation on known write events, or versioned cache keys tied to the input's known freshness.

## Monitoring

- **Serving health**: latency (p50/p95/p99), throughput (QPS), error rate — standard service SLOs, cheap to measure, available immediately.
- **Prediction distribution monitoring**: track the distribution of the model's *outputs* over time (mean predicted probability, class balance) — a sudden shift often signals an upstream problem (broken feature pipeline, drifted input data) even before ground truth is available.
- **Ground-truth-delayed evaluation**: for many products the true label arrives long after the prediction (e.g. "did the user churn in 30 days," "was the loan repaid in a year") — you can't compute accuracy/AUC in real time. Interim signals: monitor input/feature distributions (proxy for data drift), monitor prediction distributions, use any available leading indicators/proxy labels, and backfill true accuracy metrics once labels do arrive to catch degradation retroactively.
- **Alerting thresholds**: set thresholds on the above (e.g. p99 latency > 200ms, predicted-positive rate deviates >X% from a rolling baseline) with enough margin to avoid alert fatigue, but tight enough to catch real regressions before they compound.

## Drift

- **Data drift (covariate shift)**: the distribution of the model's *input features* changes, while the relationship between features and label stays the same. Example: an e-commerce model trained pre-holiday-season sees a shift in typical basket size in December — same underlying behavior pattern, different input distribution.
- **Concept drift**: the relationship between features and the label itself changes — the same input now implies a different outcome. Example: a fraud model's features look the same, but fraudsters change tactics, so what used to predict "not fraud" now co-occurs with actual fraud.
- **Detection**: statistical tests comparing a reference (training-time) distribution against a live window of recent data per feature — population stability index (PSI), KL/JS divergence, or a two-sample hypothesis test (e.g. Kolmogorov-Smirnov for continuous features, chi-squared for categorical). See [`../statistics/hypothesis-testing-pvalue.md`](../statistics/hypothesis-testing-pvalue.md) for the underlying test mechanics and [`../statistics/ab-testing.md`](../statistics/ab-testing.md) for the broader experiment-comparison framing these tests share.
- **What to do when drift is detected**: alert first (don't auto-retrain blindly — confirm it's not a pipeline bug); if the drift is confirmed and meaningful, retrain on recent data; if a bad rollout is the actual cause rather than real-world drift, roll back to the previous model version instead of retraining.

## Model versioning & experiment tracking

You need **both**, and they solve different problems: experiment tracking answers "what did I try and how well did it do," model versioning answers "what is currently deployed and can I get back to it."

- **Tools**: MLflow (tracking + model registry), Weights & Biases (tracking, especially strong for deep learning experiment visualization), DVC (data + model artifact versioning, git-friendly).
- **Metadata to track per run**: hyperparameters, evaluation metrics, the exact data version/snapshot used, and the code version (git commit hash) — so any past result is fully reproducible and any two runs are comparable on equal footing.
- See [`../ml-engineering/README.md`](../ml-engineering/README.md#experiment-tracking--mlflow) for the tooling angle on this from the engineering side.

## Scalability & distributed systems basics

- **Horizontal vs. vertical scaling**: horizontal = add more machines/replicas behind a load balancer; vertical = get a bigger machine. Serving infra strongly favors horizontal scaling — it's cheaper at the margin, has no single hardware ceiling, and improves fault tolerance (one instance dying doesn't take the service down).
- **Load balancing**: distributes incoming requests across replicas (round-robin, least-connections, etc.) — a prerequisite for horizontal scaling to actually help.
- **Stateless services scale easier**: if a serving instance holds no per-user session state, any replica can handle any request, so you can add/remove replicas freely and a load balancer can route arbitrarily. Statefulness (e.g. an in-memory per-user cache on one instance) forces sticky routing and complicates scaling — push state out to a shared store (the online feature store, a cache layer) instead.
- **Distributed training, one sentence each**:
  - **Data parallelism**: split the *data* across workers, each holding a full copy of the model, and synchronize gradients — the default choice, used when the model fits on one device but the dataset/batch doesn't need to.
  - **Model parallelism**: split the *model itself* across devices (different layers/parameters on different GPUs) — used when the model is too large to fit on a single device, at the cost of more complex inter-device communication.

---

## Worked mini-example: "Design a product recommendation system"

Walking through the pieces above end-to-end, in interview order:

1. **Clarify requirements**: personalized recommendations on a homepage; latency budget ~100ms; freshness matters somewhat but not real-time-critical.
2. **Data pipeline**: batch ETL nightly ingests purchase/click/view events from the warehouse into cleaned event tables; validate schema/nulls at ingestion so a broken tracking event doesn't silently corrupt downstream features.
3. **Feature pipeline**: offline store computes user-level features (recent categories browsed, purchase history) and item-level features (popularity, category); point-in-time-correct joins when building the training set so a user's "purchase count" reflects the state *at* each historical recommendation event, not today's count; online store serves the current feature values at request time.
4. **Training pipeline**: retrain weekly (scheduled) on the latest snapshot, tracked in MLflow with hyperparameters/metrics/data version/git commit logged; candidate model evaluated offline against a held-out time-based split (never a random split, to avoid leaking future purchase info backward).
5. **Serving**: batch-precompute top-N recommendations nightly for most users (cheap, latency is a non-issue); for users active *right now*, an online model re-ranks the precomputed candidates with fresh session features — a hybrid of batch and online inference, which is extremely common in practice.
6. **Rollout**: new model version shadow-tested against live traffic first, then canaried to 5% of users while watching click-through rate, then ramped to 100% if healthy.
7. **Monitoring**: track serving latency/QPS, the distribution of recommended-item categories (a proxy signal before conversion data arrives), and — once available days later — actual click-through/purchase-conversion rate as the ground-truth-delayed metric.
8. **Drift**: watch for data drift in browsing-feature distributions (e.g. a new product category launch shifts input distributions) and concept drift in what predicts a purchase (e.g. a change in seasonal buying behavior); on detected drift, alert and retrain on recent data rather than auto-rolling-back a healthy model.

The same skeleton — data -> features -> training -> serving -> monitoring -> drift/retraining loop — applies near-verbatim to "design a fraud detection system," swapping: streaming ingestion instead of nightly batch (fraud needs near-real-time), purely online inference (a transaction must be scored before it completes), concept drift becoming the dominant concern (fraud tactics actively adapt to evade the current model), and heavier weight on low-latency serving and canary rollout (a bad fraud model directly costs money per minute it's live).

## Common interview questions

1. Walk me through designing [a recommender / a fraud detector / a churn predictor] end to end.
2. How do you avoid training/serving skew?
3. What's the difference between data drift and concept drift, and how would you detect each?
4. How do you monitor a model's quality when the true label arrives weeks later?
5. Batch or online inference for this use case, and why?
6. How would you safely roll out a new model version to production?
7. Why do you need both an offline and an online feature store?
8. What causes data leakage in a feature pipeline, and how does point-in-time correctness prevent it?

## Common mistakes

- Designing only the model, ignoring data/feature/serving/monitoring — the actual point of this interview format.
- Missing point-in-time correctness, silently leaking future information into training features.
- Assuming online inference is always needed "to be safe" — regurgitating architecture without ever asking about the actual latency/freshness requirement.
- Conflating data drift and concept drift, or claiming you can't monitor quality at all before ground truth arrives (interim proxies exist).
- Forgetting rollback/canary/shadow strategies and describing a rollout as "just deploy the new model."
- Treating monitoring as an afterthought instead of a first-class part of the design.

## See also

- [`../ml-engineering/README.md`](../ml-engineering/README.md) — the engineering/tooling side (Docker, CI/CD, Git, testing) that implements this design
- [`../ml-engineering/deployment-systemd-daemons.md`](../ml-engineering/deployment-systemd-daemons.md) — process-level mechanics of running a long-lived serving process
- [`../statistics/hypothesis-testing-pvalue.md`](../statistics/hypothesis-testing-pvalue.md), [`../statistics/ab-testing.md`](../statistics/ab-testing.md) — statistical tests underlying drift detection
