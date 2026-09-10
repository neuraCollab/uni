# Bias-Variance Tradeoff

## What is it?

A framework for understanding a model's generalization error as the sum of
three components — **bias**, **variance**, and irreducible **noise** — and
the fact that, for a fixed amount of data, reducing bias (making the model
more flexible) tends to increase variance, and vice versa.

## Why?

It's the conceptual backbone behind almost every "why is my model
underperforming, and what do I do about it" diagnosis: is the model too
simple to capture the pattern (high bias / underfitting), or too sensitive
to the specific training sample (high variance / overfitting)?

## How does it work?

### Underfitting vs. overfitting

- **Underfitting (high bias)** — the model is too simple to capture the
  true relationship. Both training and validation error are high and close
  together. Example: fitting a straight line to clearly curved data.
- **Overfitting (high variance)** — the model is flexible enough to fit
  noise in the training set, not just the signal. Training error is low but
  validation error is much higher. Example: a very deep, unpruned decision
  tree that memorizes individual training rows.
- **Good fit** — training and validation error are both low and reasonably
  close to each other.

### The decomposition

For a model's expected squared error at a point, one standard decomposition is:

```
Expected Error = Bias² + Variance + Irreducible Noise
```

- **Bias** — error from the model's own simplifying assumptions being wrong
  (e.g. assuming a linear relationship when the truth is nonlinear). High
  bias means the model systematically misses the true pattern *even with
  infinite training data*.
- **Variance** — how much the model's predictions would change if it were
  retrained on a different sample from the same distribution. High variance
  means the model is overly sensitive to the particular noise/sample it
  happened to see.
- **Irreducible noise** — inherent randomness in the data-generating process
  that no model can capture, setting a floor on achievable error regardless
  of model choice.

Intuition: a very simple model (e.g. predict the mean for everyone) has zero
variance (it doesn't change no matter what sample you train it on) but high
bias (it ignores real signal). A very flexible model (e.g. a 1-nearest-
neighbor classifier, or an unregularized high-degree polynomial) has low
bias (can fit almost anything) but high variance (a different training
sample gives a very different fitted function).

### What affects each

| Lever | Effect on bias | Effect on variance |
|---|---|---|
| More model complexity (deeper tree, higher polynomial degree, more neurons) | ↓ decreases | ↑ increases |
| More regularization (higher `alpha`/`C` penalty, pruning, dropout) | ↑ increases | ↓ decreases |
| More training data | ~unchanged | ↓ decreases (more data → less sensitivity to any one sample) |
| Fewer, more informative features / better feature engineering | ↓ decreases | can go either way |
| Ensembling (bagging) | ~unchanged | ↓ decreases (averaging independent models cancels variance) |
| Ensembling (boosting) | ↓ decreases | can increase if not regularized |

Regularization (L1/L2 penalties, tree depth limits, dropout, early stopping)
is the standard tool for trading a bit of bias for a lot of variance
reduction when a model is overfitting — see
[Regularization](../linear-models/regularization.md) for the linear-model
mechanics (Ridge/Lasso/ElasticNet) of exactly this tradeoff.

### Learning curves as a diagnostic

Plot training-set score and validation-set score as a function of training
set size:

- **High bias signature**: both curves converge to a similarly *low* score,
  close together, and adding more data doesn't help much — the model has
  run out of capacity to improve, not out of data. Fix: use a more flexible
  model, add features, reduce regularization.
- **High variance signature**: a large, persistent gap between a high
  training score and a lower validation score, where the gap *narrows* as
  training size grows. Fix: get more data, increase regularization, reduce
  model complexity, or use bagging.

```python
from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring="neg_root_mean_squared_error"
)
# Plot train_scores.mean(axis=1) and val_scores.mean(axis=1) against train_sizes.
```

## When to use / when not

This isn't a technique you "apply" so much as a lens for diagnosing model
behavior — always worth checking whenever a model's validation performance
isn't what you'd hope, before reaching for more data, more features, or a
different algorithm blindly.

## Common interview questions

- Define bias and variance in your own words, with an example model that's
  high in each.
- Why does more training data reduce variance but not bias?
- How does regularization strength (`alpha`) move a model along the
  bias-variance tradeoff?
- What does a learning curve look like for an overfitting model? For an
  underfitting one?
- Why does bagging (e.g. random forest) reduce variance without much
  affecting bias, while a single deep decision tree has high variance?
- If training error is low and validation error is high, what's your
  hypothesis and what would you try first?
- If both training and validation error are high and similar, what would you
  try? (More capacity, better features, less regularization — NOT more data.)

## Common mistakes

- Assuming "more data always helps" — it helps variance, not bias; an
  underfitting model won't improve much from more rows of the same features.
- Diagnosing overfitting from a single train/test split instead of a
  learning curve or cross-validated scores (see
  [Cross-Validation](cross-validation.md)) — a single split's gap can be
  noise.
- Adding regularization to fix what's actually an underfitting (high-bias)
  problem, making it worse.
- Confusing "high variance between CV folds' scores" (an estimation-noise
  issue, e.g. from a too-small dataset) with "high model variance" (the
  bias-variance-tradeoff sense) — related but distinct concepts.
- Chasing training-set performance instead of validation performance when
  tuning complexity — training error decreases monotonically with
  complexity and will always look better; it's not the number that matters.

## Example

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import numpy as np

# Sweep regularization strength: small alpha -> low bias/high variance,
# large alpha -> high bias/low variance. Look for the alpha minimizing
# cross-validated error, not training error.
for alpha in [0.001, 0.1, 1, 10, 100]:
    scores = cross_val_score(Ridge(alpha=alpha), X, y, cv=5, scoring="neg_mean_squared_error")
    print(f"alpha={alpha:>7}: CV RMSE = {np.sqrt(-scores.mean()):.3f}")
```

See also: [Regularization](../linear-models/regularization.md),
[Cross-Validation](cross-validation.md),
[Decision Trees](../trees-ensembles/decision-trees.md) (a textbook
high-variance model) and
[random-forest.md](../trees-ensembles/random-forest.md) (bagging as a
variance-reduction technique).
