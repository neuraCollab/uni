# Regression Metrics

## What is it?

Metrics for scoring a regression model's numeric predictions against true
continuous targets: MSE/RMSE, R², MAE, the relative-error family
(MAPE/SMAPE/WAPE), RMSLE, and simple threshold-based operational metrics.

## Why?

Different metrics answer different questions — absolute vs. relative error,
outlier sensitivity, unit interpretability, behavior near zero. Picking the
wrong one can hide the exact failure mode you care about. MSE, for example,
can bury the fact that 90% of your predictions are essentially perfect and
10% are catastrophic inside one average dominated by the squared outliers —
which is either exactly the point (you want outliers to move the score) or
exactly the trap (a few bad predictions make an otherwise-good model look
much worse than it is), depending on what you're optimizing for.

## How does it work?

### MSE — Mean Squared Error

$$MSE = \frac{1}{N} \sum_i (y_i - f(x_i))^2$$

Squaring means large errors are penalized disproportionately more than
small ones — MSE is **highly sensitive to outliers**. $\sup(MSE) = +\infty$, so
a raw MSE number carries no intrinsic "good vs. bad" scale on its own — you
need to compare it against the variance of $y$ (exactly what R² does below)
or against a baseline model's MSE.

### RMSE — Root Mean Squared Error

$$RMSE = \sqrt{MSE}$$

Same outlier sensitivity as MSE, but expressed back in the **original units
of the target** (dollars, seconds, requests/sec, ...) — much easier to
sanity-check ("off by \$340 on average, roughly") than raw MSE. Usually the
number to report to a non-technical stakeholder.

### R² — Coefficient of Determination

$$R^2 = 1 - \frac{\sum_i (y_i - f(x_i))^2}{\sum_i (y_i - \bar{y})^2}$$

where $\bar{y}$ is the mean of the true targets. R² measures **the fraction
of variance in $y$ explained by the model**, relative to the trivial
baseline of always predicting $\bar{y}$:

- $R^2 = 1$ → perfect predictions.
- $R^2 = 0$ → the model is exactly as good as predicting the mean every
  time.
- $R^2 < 0$ → the model is *worse* than predicting the mean — a real
  possibility (an overfit model evaluated out-of-sample, or a badly
  misspecified model), and a useful smell test precisely because
  accuracy-style metrics can never go negative.

R² is the standard way to give MSE a "good vs. bad" scale, since raw MSE
alone is unbounded above and has no natural reference point.

### MAE — Mean Absolute Error

$$MAE = \frac{1}{N} \sum_i |y_i - f(x_i)|$$

Same units as the target, like RMSE, but **less sensitive to outliers**
than MSE/RMSE, since errors aren't squared — a single huge miss moves MAE
only linearly, not quadratically. Prefer MAE over RMSE when outliers are
expected and you don't want a handful of bad predictions dominating the
score; prefer RMSE when large errors are disproportionately costly and you
*want* them penalized harder.

### Relative-error metrics: MAPE, SMAPE, WAPE

Absolute-error metrics (MAE, RMSE) don't account for scale — being off by
10 units is a rounding error when $y \sim 100{,}000$ and a disaster when
$y \sim 12$. Relative-error metrics normalize by the target's magnitude, which
also makes them comparable across series of very different scale (e.g.
averaging error across products with wildly different demand volumes).

**MAPE** — Mean Absolute Percentage Error:

$$MAPE = \frac{1}{N} \sum_i \frac{|y_i - f(x_i)|}{|y_i|}$$

Intuitive to report ("on average, X% off"), but **explodes/undefined as
$y_i \to 0$**, and asymmetric — it penalizes over-forecasting
($f(x_i) > y_i$) more harshly than under-forecasting in percentage terms,
because the denominator is always the *true* value.

**SMAPE** — "symmetric" MAPE:

$$SMAPE = \frac{1}{N} \sum_i \frac{2|y_i - f(x_i)|}{y_i + f(x_i)}$$

Bounded (unlike plain MAPE) and offers some protection against the
near-zero-$y_i$ blow-up, since the predicted value also sits in the
denominator, not just the true one. It isn't a complete fix, though — SMAPE
is still unstable when $y_i$ and $f(x_i)$ are *both* near zero, and despite
the name it isn't fully symmetric between over- and under-prediction of the
same absolute size. Reach for it over MAPE when some targets are near zero
but never negative.

**WAPE** — Weighted (a.k.a. Aggregate) MAPE:

$$WAPE = \frac{\sum_i |y_i - f(x_i)|}{\sum_i |y_i|}$$

Instead of averaging per-item percentage errors, WAPE sums all the absolute
errors and all the actuals separately, then divides once. That makes it far
more robust when individual $y_i$ can be zero or near-zero — a single
near-zero-demand period no longer blows up the metric the way it would
inside a per-item MAPE average. The classic use case is demand forecasting
with strong seasonality, where some periods legitimately have near-zero
true demand. Smaller WAPE is better.

### RMSLE — Root Mean Squared Log Error

$$RMSLE = \sqrt{\frac{1}{N} \sum_i \left(\log(y_i + c) - \log(f(x_i) + c)\right)^2}$$

where $c$ is a small positive constant (commonly $1$) added for numerical
stability, so the metric stays defined at $y_i = 0$ (requires
$y_i, f(x_i) \geq -1 + \varepsilon$ for $\log(\cdot + c)$ to be defined). Taking the log
before squaring means RMSLE:

- Cares about **relative**, not absolute, error — under-predicting 100 as
  90 and under-predicting 100,000 as 90,000 contribute similarly to RMSLE,
  unlike RMSE, where the second error dominates completely.
- Penalizes **under-prediction more than over-prediction** — useful when
  under-forecasting is the more expensive mistake (e.g. under-stocking
  inventory, under-provisioning capacity).
- Fits naturally when the target spans several orders of magnitude (view
  counts, sales volume, income).

### Fraction of predictions beyond an error threshold

A simple, highly interpretable operational metric — what fraction of
predictions missed by more than some tolerance $\alpha$ that the business
actually cares about:

$$\frac{1}{N} \sum_i \mathbb{1}\left[ |y_i - f(x_i)| > \alpha \right]$$

e.g. "92% of ETA predictions are within 5 minutes of the actual arrival
time." Very easy to explain to a non-technical stakeholder, but it throws
away magnitude information above/below the threshold — pair it with
MAE/RMSE rather than using it alone.

## Which metric for which situation

| Situation | Recommended metric(s) | Avoid |
|---|---|---|
| Outliers present, shouldn't dominate the score | MAE, WAPE | MSE / RMSE |
| Outliers present, *should* be penalized hard | MSE / RMSE | MAE |
| Need a bounded "good vs. bad" scale, not just raw units | R² | raw MSE alone |
| Need scale-dependent, human-readable units (dollars, minutes) | RMSE, MAE | MAPE / SMAPE (unitless) |
| Need relative/percentage error across differently-scaled series | MAPE (only if $y$ never near 0), SMAPE, WAPE | MAE / RMSE alone |
| Target can be zero or near-zero (e.g. intermittent demand) | WAPE, SMAPE | MAPE |
| Target spans orders of magnitude; relative error matters; under-prediction is worse | RMSLE | RMSE |
| Need a simple stakeholder-facing operational number | fraction of predictions within threshold $\alpha$ | raw error metrics alone |

## Common interview questions

- Why is MSE sensitive to outliers while MAE isn't? Walk through the math.
- What does R² = 0 mean? Can R² be negative, and how?
- Why does RMSE share units with the target but MSE doesn't?
- When would you prefer MAPE vs. SMAPE vs. WAPE?
- What specifically breaks about MAPE when the true value is near zero?
- Why take logs before squaring in RMSLE — what does that buy you, and what
  does it cost you?
- How would you explain a regression model's quality to a non-technical
  stakeholder?

## Common mistakes

- Using MAPE on a target that can be zero or near-zero without checking
  first — it explodes or is literally undefined.
- Reporting MSE alone with no reference scale (R², or a baseline
  comparison) to say whether it's actually "good."
- Using RMSE when outliers are expected and shouldn't dominate the score —
  MAE is usually the better call there.
- Forgetting RMSLE's asymmetry (it punishes under-prediction harder) when
  that's the opposite of what the business actually needs penalized.
- Comparing RMSE/MAE values across differently-scaled targets or datasets
  as if they were comparable — they aren't, without normalizing
  (MAPE/SMAPE/WAPE/R² instead).

## Example

```python
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)

y_pred = model.predict(X_val)

mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5  # or sklearn's root_mean_squared_error in newer versions
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
mape = mean_absolute_percentage_error(y_val, y_pred)
```

## Related notes

- [Classification Metrics](classification-metrics.md) — the
  classification-side analogue (log-loss, ROC-AUC/PR-AUC).
- [Probability Calibration](calibration.md) — Brier Score is literally MSE
  applied to predicted probabilities instead of a continuous target.
- [Bias-Variance Tradeoff](bias-variance-tradeoff.md)
- [Regularization](../linear-models/regularization.md) — the models these
  metrics are typically used to evaluate and tune.
