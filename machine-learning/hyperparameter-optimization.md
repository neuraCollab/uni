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

### Bayesian optimization (Optuna, Hyperopt)

Instead of sampling blindly (grid) or uniformly at random, build a
probabilistic model of "hyperparameters → validation score" from trials
run so far, and use it to pick the *next* point expected to be most
informative (balancing exploring uncertain regions against exploiting
regions known to score well). Optuna's default sampler is **TPE**
(Tree-structured Parzen Estimator), which models `P(hyperparameters | good
score)` vs. `P(hyperparameters | bad score)` and samples from the region
where the "good" density is relatively high.

Bayesian methods typically need far fewer trials than grid/random search to
reach a comparable or better optimum, because each trial is chosen using
information from all previous trials rather than independently.

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

## When to use / when not

**Grid search:** few hyperparameters (1-2), small discrete value sets, you
want exhaustive guarantees over that grid, or you need reproducible,
easily-explained results.

**Random search:** more hyperparameters, limited compute budget, no strong
prior about which values matter — a solid, simple default that beats grid
search per unit of compute in most realistic settings.

**Bayesian optimization (Optuna/Hyperopt):** expensive-to-train models
(large boosted trees, neural nets) where every trial's cost matters, many
hyperparameters, and/or you have conditional structure in the search space.
The overhead of the Bayesian model itself only pays off when each trial is
expensive enough that saving trials matters more than the sampler's own
compute.

**Avoid extensive hyperparameter search when:** the dataset is tiny (search
will just overfit the validation folds — see
[Cross-Validation](model-evaluation/cross-validation.md#nested-cv) on nested
CV to get an honest estimate afterward), or the model is already
well-regularized by defaults and the marginal gain isn't worth the compute.

## Common interview questions

- Why does random search often outperform grid search for the same budget?
- What does Optuna's TPE sampler actually model?
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
