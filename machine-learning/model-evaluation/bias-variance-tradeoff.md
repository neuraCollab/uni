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

### The formal decomposition

Let `y(x, ε) = f(x) + ε` be the true (noisy) data-generating process at
point `x` — `f(x)` is the true deterministic target function and `ε` is
irreducible label noise. Let `a(x, X)` be a model trained on training set
`X` and evaluated at point `x`. The expected squared error, averaged over
both the randomness in the training set `X` and the label noise `ε`, is:

```
Q(a) = E_x E_{X,ε} [ y(x, ε) - a(x, X) ]²
```

This decomposes exactly into three additive terms:

```
Q(a) = E_x[ bias_x(a)² ] + E_x[ Var_x(a(x, X)) ] + σ²
```

where:

```
bias_x(a(x, X)) = f(x) - E_X[ a(x, X) ]

Var_x[a(x, X)] = E_X[ (a(x, X) - E_X[a(x, X)])² ]

σ² = E_x E_ε[ (y(x, ε) - f(x))² ]
```

In plain language:

- **`bias_x`** — the gap between the true function `f(x)` and the *average*
  prediction the model would make at `x` if you retrained it over and over
  on fresh training sets drawn from the same distribution. This is
  systematic error: it doesn't go away no matter how many times you
  retrain, because it comes from the model family itself being unable to
  represent `f(x)` (e.g. a linear model trying to fit a curve). More
  training data doesn't fix it.
- **`Var_x`** — how much `a(x, X)` itself swings around its own average as
  `X` varies. This is sensitivity to *which particular training set you
  happened to draw* — not a property of the model family being wrong, but
  of the model being unstable given finite, resampled data. More training
  data shrinks this (less sensitivity to any one sample); more model
  flexibility (deeper trees, fewer constraints) grows it.
- **`σ²`** — the variance of the label noise `ε` itself, independent of any
  model or training set. This is the error floor: even the true function
  `f(x)` itself, predicted perfectly, still misses `y(x, ε)` by `ε` on
  average. No amount of modeling, data, or tuning reduces this term — it's
  a property of the problem, not the model.

Squaring and summing these three (rather than, say, adding `bias` and
`variance` directly) is exactly why the informal version of this framework
is usually written `Bias² + Variance + Noise` — the derivation above is
where that square on `bias` actually comes from.

### Where bagging and boosting sit in this decomposition

This decomposition is also the precise reason bagging and boosting behave
oppositely on the bias/variance split (see
[Random Forest](../trees-ensembles/random-forest.md) and
[Gradient Boosting](../trees-ensembles/gradient-boosting-catboost-lgbm.md)
for the full derivations):

- **Bagging** (Random Forest) averages `k` base models trained on bootstrap
  resamples. Averaging is linear, so it doesn't shift `E_X[a(x,X)]` —
  `bias_x` is untouched. But averaging *does* shrink `Var_x[a(x,X)]`,
  toward a `1/k` factor under the (approximate) assumption that the base
  models don't correlate. Net effect: **the variance term shrinks, the
  bias term is unchanged**, `σ²` is untouched (it's a property of the data,
  not reachable by any model). This is why bagging almost never hurts and
  reliably helps a high-variance base learner like an unpruned tree, but
  can't fix a base learner that's systematically wrong.
- **Boosting** (gradient boosting/CatBoost/LightGBM/XGBoost) trains each
  new base learner explicitly to reduce the *remaining* error of the
  current ensemble (fit to the anti-gradient of the loss). By construction
  this directly attacks the systematic part of the error each round — it
  **reduces the bias term**. But because rounds are added greedily and
  depend on each other, boosting can also inflate variance if left
  unchecked (too many rounds, too little shrinkage/regularization start
  fitting noise in the residuals) — which is why boosting needs early
  stopping and careful regularization in a way bagging doesn't.

This is the clean answer to the classic interview framing "why does
bagging help with overfitting, but too much boosting can hurt": they're
acting on different terms of the same decomposition — bagging trades
nothing for a variance reduction, boosting trades a (controllable) variance
increase for a bias reduction.

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
