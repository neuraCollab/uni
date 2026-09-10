# Quantile Regression

## What is it?

A regression method that predicts a specific **quantile** of the target's conditional distribution (e.g. the 10th, 50th, 90th percentile) instead of the conditional **mean**, which is what OLS/Ridge/Lasso predict.

## Why?

Sometimes you don't want a single point estimate — you want an **interval**, or you specifically care about the distribution's tails rather than its center. Examples:
- **Demand forecasting:** predicting the 90th percentile of demand tells you how much inventory to stock to avoid running out 90% of the time — very different from predicting the average demand.
- **Risk-sensitive predictions:** e.g. predicting a low quantile of expected returns for a conservative estimate.
- Building a **prediction interval** by fitting two models, e.g. quantile 0.1 and quantile 0.9, gives you an 80% interval directly, without assuming symmetric/Gaussian errors (which mean+std intervals implicitly do).

## How does it work?

Standard regression minimizes squared error, whose minimizer is the conditional **mean**. Quantile regression instead minimizes the **pinball loss** (a.k.a. quantile loss), whose minimizer is the conditional quantile `q`:

```
L_q(y, y_hat) = (y - y_hat) * q          if y >= y_hat
              = (y_hat - y) * (1 - q)    if y < y_hat
```

Equivalently: `L_q(y, y_hat) = max( q * (y - y_hat), (q - 1) * (y - y_hat) )`.

- `q = 0.5` → the pinball loss becomes (half of) absolute error, and its minimizer is the **median** — this recovers median regression, which is already more outlier-robust than OLS.
- `q` close to 1 → heavily penalizes under-prediction (predicting too low) more than over-prediction → pushes the fit toward the upper tail of the distribution.
- `q` close to 0 → the opposite, pushes toward the lower tail.

The loss is **asymmetric** by design — that asymmetry is exactly what makes the minimizer land on a quantile other than the mean/median.

## When to use / when not to use

**Use when:** you need an interval, not just a point estimate; the target's conditional distribution is skewed or heteroscedastic (variance changes with the input, so a single mean+constant-std interval would be wrong); business decisions depend specifically on a tail outcome (e.g. worst-case, best-case).

**Avoid when:** you only need the mean and the target is roughly symmetric — plain OLS/Ridge is simpler, faster, and better understood. Also note: fitting several independent quantiles (e.g. 0.1, 0.5, 0.9) does not guarantee they won't **cross** each other (the 0.1 model could predict a higher value than the 0.5 model, which is a known limitation) unless a monotonicity constraint is applied.

## Common interview questions

- What loss does quantile regression minimize, and why does it produce a quantile instead of a mean?
- Why is `q=0.5` equivalent to minimizing absolute error?
- How would you build a prediction interval using quantile regression?
- What's "quantile crossing," and why can it happen with independently-fit quantile models?
- Why is quantile regression more robust to outliers than OLS? *(Bounded influence of large residuals, similar in spirit to Huber/robust regression — see [Robust Regression](robust-regression.md).)*

## Common mistakes

- Fitting two quantile models independently and assuming the interval they form is automatically well-calibrated/non-crossing.
- Confusing quantile regression's `alpha` (L1 regularization strength in sklearn's `QuantileRegressor`) with the quantile level `quantile` parameter — they're unrelated.
- Using `solver='highs'` default without checking sklearn version support; older sklearn versions use a different default LP solver, which can be much slower on larger datasets.

## Example

```python
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=500, n_features=3, noise=20, random_state=0)
# Make the noise heteroscedastic: variance grows with the first feature
y = y + X[:, 0] * np.random.default_rng(0).normal(0, 15, size=len(y))

models = {q: QuantileRegressor(quantile=q, alpha=0.0, solver="highs") for q in [0.1, 0.5, 0.9]}
for q, model in models.items():
    model.fit(X, y)
    print(f"quantile={q}: coef={np.round(model.coef_, 3)}, intercept={model.intercept_:.3f}")

# 80% prediction interval for a new point
x_new = X[:1]
lower = models[0.1].predict(x_new)[0]
upper = models[0.9].predict(x_new)[0]
median = models[0.5].predict(x_new)[0]
print(f"Predicted median: {median:.2f}, 80% interval: [{lower:.2f}, {upper:.2f}]")
```

*(The archived source had no runnable quantile-regression code at all — just a link — so this example is written from scratch.)*

## Related notes

- [Robust Regression](robust-regression.md) — both de-emphasize squared-error sensitivity to outliers.
- [Bayesian Regression](bayesian-regression.md) — another way to get predictive uncertainty, via a full posterior instead of separate quantile fits.
