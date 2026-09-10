# Hyperparameter Optimization

## What is it?

Systematically searching over a model's hyperparameters (values set before
training, not learned from data — e.g. `alpha`, `max_depth`, `learning_rate`,
`n_estimators`) to find a combination that generalizes well, as measured via
cross-validation.

## Why?

Hyperparameters control the bias-variance tradeoff and training dynamics of
a model (see [Bias-Variance Tradeoff](model-evaluation/bias-variance-tradeoff.md)),
and good values are rarely obvious in advance — they depend on the dataset.
Manual tuning doesn't scale past a couple of parameters; systematic search
does.

## How does it work?

### Grid search

Enumerate every combination of hyperparameter values from a fixed grid, and
evaluate all of them via cross-validation.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {"max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.3]}
search = GridSearchCV(model, param_grid, cv=5, scoring="neg_log_loss")
```

**Cost grows exponentially** with the number of hyperparameters — a grid
over 5 parameters with 5 values each is `5⁵ = 3125` fits (times `cv` folds).
Exhaustive, but only practical for a small number of parameters/values.

### Random search

Sample a fixed number of random combinations from specified distributions
instead of trying every grid point.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

param_dist = {"learning_rate": loguniform(1e-3, 1e-1), "max_depth": [3, 5, 7, 9]}
search = RandomizedSearchCV(model, param_dist, n_iter=50, cv=5, scoring="neg_log_loss")
```

**Why random search often beats grid search in high dimensions:** most
hyperparameters don't matter equally — typically only a few dimensions
actually affect performance meaningfully for a given problem. A grid spends
its fixed budget spreading points evenly across *every* dimension, including
unimportant ones, so it only tests a handful of distinct values along the
dimensions that actually matter. Random search, by contrast, samples each
dimension independently and continuously, so with the same budget it
explores far more distinct values along every axis — including the
important ones — rather than wasting evaluations repeating the same values
of an unimportant parameter. This effect (from Bergstra & Bengio, 2012) gets
more pronounced as the number of hyperparameters grows.

**Why random search is theoretically justified — a coverage-probability
argument.** Say the top 5% of the hyperparameter space (by score) counts as
"good enough" — you don't need the single global optimum, just a config in
that region. Sample `n` configurations independently and uniformly at
random. For any one sample, the probability it lands *outside* the top-5%
region is `1 - 0.05 = 0.95`. Assuming independence, the probability that
**none** of the `n` samples lands in the top 5% is:

```
P(none in top 5%) = (1 - 0.05)^n
```

So the probability that **at least one** sample lands in the top 5% is its
complement:

```
P(at least one in top 5%) = 1 - (1 - 0.05)^n
```

Solve for the `n` that gets this to at least 95% confidence:

```
1 - (1 - 0.05)^n >= 0.95
(0.95)^n <= 0.05
n * ln(0.95) <= ln(0.05)
n >= ln(0.05) / ln(0.95)   (inequality flips — dividing by a negative number)
n >= 58.4  ->  n = 59
```

So **~59 random samples give a ≥95% chance that at least one of them lands
in the top 5% of the search space** — regardless of how many dimensions the
space has, and without knowing anything about *where* that region is. This
is the concrete answer to "why does random search work theoretically": it
doesn't need to cover the space, it needs enough independent draws that
missing the good region on every single one becomes unlikely — and that
count grows only logarithmically as you demand a smaller target region or
higher confidence (`n >= ln(1 - confidence) / ln(1 - region_size)`).

### Bayesian optimization

Instead of sampling blindly (grid) or uniformly at random, build a
probabilistic model of the objective function ("hyperparameters →
validation score") from the trials run so far, and use it to decide which
point to evaluate *next* — without ever differentiating the objective
(there's no closed form for it anyway; every evaluation is a full
train-and-validate run, which is exactly what makes trials expensive enough
to be worth being smart about).

It has two core components:

1. **A surrogate model** — a probabilistic model that approximates the true
   objective given the (hyperparameters, score) pairs observed so far. A
   **Gaussian Process** is the classic choice: at any point `x` it predicts
   both a mean score `μ(x)` and an uncertainty `σ(x)`, where `σ(x)` shrinks
   near points that have already been evaluated and stays wide in
   unexplored regions.
2. **An acquisition function** `a(x)` — decides, using the surrogate's
   current mean/uncertainty, which point to try next. It has to balance:
   - **exploitation** — favor points where the surrogate's mean `μ(x)` is
     already good, versus
   - **exploration** — favor points where the surrogate's uncertainty
     `σ(x)` is large, because an unexplored region might hide something
     better than anything seen so far.

   A simple, common acquisition function is **Upper Confidence Bound
   (UCB)**:

   ```
   a(x) = μ(x) + β·σ(x)
   ```

   `μ(x)` is the exploit term, `β·σ(x)` is the explore term, and `β` is a
   tunable knob for how much to weight exploration. Other standard choices
   are **Expected Improvement (EI)** and **Probability of Improvement
   (PI)**, which instead ask "how much (or how likely) is this point to
   beat the best score observed so far."

**The algorithm loop.** Let `S_t` be the set of observations gathered so
far, `S_t = {(x_1, f(x_1)), ..., (x_t, f(x_t))}`:

1. At iteration `t+1`, pick the next point by maximizing the acquisition
   function over the search space, conditioned on what's been observed so
   far: `x_{t+1} = argmax_{x in X} a(x | S_t)`.
2. Evaluate the true objective at that point — `f(x_{t+1})` — i.e. actually
   train and validate a model with those hyperparameters.
3. Update the observed set: `S_{t+1} = S_t ∪ {(x_{t+1}, f(x_{t+1}))}`.
4. Update (refit) the surrogate model on the enlarged `S_{t+1}` and repeat.

Bayesian methods typically need far fewer trials than grid/random search to
reach a comparable or better optimum, because each trial is chosen using
information from every previous trial rather than independently — the cost
is the (usually small, relative to training a model) overhead of fitting
and maximizing the acquisition function over the surrogate at each step.

### TPE (Tree-structured Parzen Estimator)

Optuna's default sampler, and the algorithm most people are actually
running when they say "Bayesian hyperparameter optimization" in practice.
Classic Bayesian optimization (above) models `P(score | hyperparameters)`
directly through the surrogate. **TPE inverts this**: rather than modeling
how the score depends on the hyperparameters, it models how the
*hyperparameters themselves* are distributed, separately, conditioned on
whether the outcome was good or bad.

How it works:

1. Build the search space as a **tree of parameters** — this is what lets
   it naturally handle conditional/hierarchical spaces (see [Conditional /
   hierarchical search spaces](#conditional--hierarchical-search-spaces)
   below): a branch of the tree is simply never visited for trials where
   its parent parameter took a different value.
2. Generate `n` sampled trials, and split the observed trials into a "good"
   group and a "bad" group by a fixed quantile cutoff on their scores — the
   top ~20% by score become the "good" group, the rest become the "bad"
   group.
3. Fit a density estimate over each group separately: `l(x) = P(x | good)`
   and `g(x) = P(x | bad)`.
4. Score candidate points by the **ratio `l(x) / g(x)`** — a point is
   promising if it's much more likely under the "good" density than under
   the "bad" one — and sample the next trial from where this ratio is high.

This density-ratio formulation (`l(x)/g(x)`, in place of modeling
`P(score | x)` directly) is why TPE scales better than a classic
Gaussian-Process-based Bayesian optimizer to high-dimensional and
conditional search spaces: it never needs to fit one joint surrogate over
the whole (possibly branching) space — just two independent, per-parameter
density estimates.

### Conditional / hierarchical search spaces

A **conditional hyperparameter** only makes sense (or only has an effect)
given a particular value of another hyperparameter. Grid and random search
struggle to express this cleanly — a plain grid/dict-of-lists format has no
way to say "only vary `bagging_temperature` when `bootstrap_type ==
'Bayesian'`"; you'd either waste trials repeating the same value of a
parameter that does nothing, or write awkward manual grid-splitting logic.

Optuna handles this naturally because the search space is built **inside the
objective function**, trial by trial, rather than declared upfront as a
static grid — so later `suggest_*` calls can depend on earlier ones within
the same trial. A real example, from a CatBoost hyperparameter search
(`bagging_temperature` only controls Bayesian bootstrap — it's meaningless
for Bernoulli or MVS bootstrap):

```python
def objective(trial):
    bootstrap_type = trial.suggest_categorical(
        "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
    )
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "depth": trial.suggest_int("depth", 4, 10),
        "bootstrap_type": bootstrap_type,
    }
    # Conditional: only sample bagging_temperature for Bayesian bootstrap.
    if bootstrap_type == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    return model.get_best_score()["validation"]["Logloss"]

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
```

Full runnable version:
[`trees-ensembles/code/hp_boosting_optuna.py`](trees-ensembles/code/hp_boosting_optuna.py)
(also demonstrates early stopping picking the boosting-round count instead
of tuning `n_estimators` directly, and a simple 2-model averaging ensemble).

### Population Based Training (PBT)

Everything above — grid search, random search, classic Bayesian
optimization, TPE — shares one assumption: each trial is a **complete,
independent training run**, scored only once it finishes. That's a fine
assumption for a boosted tree (train from scratch in minutes) but wasteful
for **deep learning**, where a single run can take hours or days and
hyperparameters like learning rate are often more useful to *change over
the course of training* than to fix upfront.

PBT trains a **population** of models in parallel, all training
simultaneously, and periodically has each population member:

- **exploit** — if it's underperforming relative to the rest of the
  population, copy the weights *and* hyperparameters of a better-performing
  member, and
- **explore** — perturb the copied hyperparameters (e.g. multiply the
  learning rate by a random factor) before continuing to train.

This lets hyperparameters evolve *during* training instead of requiring a
full independent run per configuration — it's effectively doing
scheduling and search at the same time. It's also parallelizable and can
warm-start from / reuse the partial results of earlier runs rather than
discarding every unfinished trial the way early-stopped grid/random/TPE
trials are discarded.

**Main practical limitation:** PBT needs many parallel workers to be
efficient — roughly **20 to 90 concurrent workers** in practice — because
the exploit/explore mechanism only has something worth copying from when
there's enough population diversity training at the same time. That makes
it a poor fit outside well-resourced multi-GPU/multi-node setups: a single
machine or a small cluster doesn't have enough parallelism for the
population dynamics to pay off. Contrast with Optuna/TPE, which need only
as many workers as you're willing to run trials on concurrently and work
fine even with `n_jobs=1`.

## When to use / when not

**Grid search:** few hyperparameters (1-2), small discrete value sets, you
want exhaustive guarantees over that grid, or you need reproducible,
easily-explained results.

**Random search:** more hyperparameters, limited compute budget, no strong
prior about which values matter — a solid, simple default that beats grid
search per unit of compute in most realistic settings.

**Bayesian optimization (Optuna/Hyperopt, TPE by default):** expensive-to-train
models (large boosted trees, neural nets) where every trial's cost matters, many
hyperparameters, and/or you have conditional structure in the search space.
The overhead of the Bayesian model itself only pays off when each trial is
expensive enough that saving trials matters more than the sampler's own
compute.

**Population Based Training:** deep learning specifically, when you can run
many workers in parallel (roughly 20-90) and want hyperparameters like
learning rate to adapt over the course of training rather than stay fixed
per trial. Not worth setting up for cheap-to-train models (boosted trees,
linear models) or when you can't get that many parallel workers — Optuna/TPE
is the better default there.

**Avoid extensive hyperparameter search when:** the dataset is tiny (search
will just overfit the validation folds — see
[Cross-Validation](model-evaluation/cross-validation.md#nested-cv) on nested
CV to get an honest estimate afterward), or the model is already
well-regularized by defaults and the marginal gain isn't worth the compute.

## Common interview questions

- Why does random search often outperform grid search for the same budget?
- Derive: with `n` random samples, what's the probability at least one
  lands in the top 5% of the search space, and how large does `n` need to
  be for 95% confidence? (`1 - (1-0.05)^n >= 0.95` -> `n >= 59`.)
- What are the two core components of Bayesian optimization, and what does
  the acquisition function balance?
- What does Optuna's TPE sampler actually model, and how does that differ
  from classic Gaussian-Process Bayesian optimization?
- Why does PBT need so many parallel workers, and why doesn't that matter
  for Optuna/TPE?
- Give an example of a conditional hyperparameter and explain why a static
  grid can't express it well.
- How would you avoid overfitting your hyperparameters to the validation
  set? (Nested CV, or a separate final holdout never touched during search.)
- What's the tradeoff between `n_iter` in random search and search quality?
- Why might grid search still be preferable in some cases despite its
  exponential cost? (Reproducibility, interpretability of "we tried exactly
  these values," small parameter count.)
- How does early stopping interact with hyperparameter search? (Removes the
  need to tune `n_estimators`/epoch count directly — let the model pick its
  own stopping point per trial.)

## Common mistakes

- Using the same CV split both to select hyperparameters and to report final
  performance — optimistically biased (see
  [Data Leakage](model-evaluation/data-leakage.md) and nested CV in
  [Cross-Validation](model-evaluation/cross-validation.md)).
- Setting a grid too coarse (missing the real optimum) or too fine (wasting
  compute) without a log-scale — most learning-rate-like parameters should
  be searched on a log scale (`loguniform`), not linear.
- Sampling a conditional hyperparameter unconditionally (e.g. always
  sampling `bagging_temperature` regardless of `bootstrap_type`) — wastes
  search budget on trials where the value has no effect.
- Not fixing a `random_state`/seed, making the search non-reproducible.
- Tuning hyperparameters against a metric that doesn't match the actual
  deployment objective (e.g. optimizing log loss when the business cares
  about a specific precision/recall tradeoff — see
  [Classification Metrics](model-evaluation/classification-metrics.md)).

## Example

See [`trees-ensembles/code/hp_boosting_optuna.py`](trees-ensembles/code/hp_boosting_optuna.py)
for a full LightGBM + CatBoost binary-classification search with a
conditional CatBoost search space, ported and fixed from the original
coursework script.

See also:
[Cross-Validation](model-evaluation/cross-validation.md),
[Bias-Variance Tradeoff](model-evaluation/bias-variance-tradeoff.md),
[Hyperparameter Tuning (Deep Learning)](../deep-learning/hyperparameter-tuning.md).
