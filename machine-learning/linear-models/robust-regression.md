# Robust Regression: Theil-Sen, Huber, RANSAC

## What is it?

A family of regression methods designed to stay accurate when the data contains **outliers** — points that violate the assumed noise model and would otherwise dominate an OLS fit.

## Why?

**Robustness** = an algorithm's ability to keep working well under non-ideal conditions:
1. noise and outliers,
2. distribution shift,
3. small perturbations in input/parameters.

Plain OLS is *not* robust: it minimizes squared error, so a single far-away outlier contributes a huge squared term to the loss and can pull the entire fitted line toward it. A handful of bad sensor readings, data-entry errors, or corrupted rows can silently wreck an otherwise-good model — and standard $R^2$/MSE on the same corrupted data won't necessarily reveal it as clearly as a robust vs. non-robust comparison would.

## How do the three methods work?

- **Huber Regression** — uses the **Huber loss**: quadratic (like squared error) for small residuals, but **linear** (like absolute error) beyond a threshold `epsilon`. Small errors are still penalized smoothly (good gradient behavior near the optimum); large errors (outliers) are penalized linearly instead of quadratically, so they can't dominate the loss. `HuberRegressor` in sklearn.
- **Theil-Sen Estimator** — computes the **median of the slopes** between all pairs of points (a generalization of the median to regression). Since it's based on the median rather than a mean, it tolerates a substantial fraction of outliers without their values pulling the estimate — very high breakdown point, but expensive ($O(n^2)$ pairs) for large `n`.
- **RANSAC (Random Sample Consensus)** — repeatedly: (1) fit a model on a small random subset of points, (2) count how many *other* points agree with it within a tolerance (the "inliers"), (3) keep the model with the most inlier support. Highly resistant to **large, gross outliers** (even if outliers are the majority in some setups), but requires tuning a hyperparameter (the inlier-distance threshold, and implicitly the expected inlier fraction), and the resulting fit can be less smooth/stable across re-runs than Huber or Theil-Sen because it depends on random sampling.

## Comparison

| Method | Handles | Sensitivity | Cost | Notes |
|---|---|---|---|---|
| OLS | no outliers assumed | very sensitive to any outlier | cheap | baseline |
| Huber | small-to-moderate contamination | robust for outliers beyond `epsilon` | cheap-moderate | smooth loss, easy to optimize |
| Theil-Sen | moderate outlier fraction | robust (median-based) | expensive at scale ($O(n^2)$) | good for small/medium `n` |
| RANSAC | large/gross outliers, even majority contamination | most robust against big outliers | needs threshold tuning | fit can be less smooth, stochastic |

**Practical guidance from experimentation:** OLS is extremely sensitive to any anomaly; Theil-Sen and Huber both work well under mild-to-moderate violations; RANSAC is the most robust against large outliers but needs its inlier-fraction hyperparameter set correctly, and can be less stable/smooth.

## When to use / when not to use

**Use when:** you suspect (or know) the data has outliers/label noise/sensor errors and can't clean them out beforehand; when a single anomalous point noticeably shifts your OLS fit under cross-validation.

**Avoid when:** data is genuinely clean — robust methods add computational cost and, for Huber, an extra hyperparameter (`epsilon`) with no benefit if there's nothing to be robust against.

## Common interview questions

- Why is OLS not robust to outliers, mechanically (in terms of the loss function)?
- How does Huber loss interpolate between L2 and L1 loss?
- Why does Theil-Sen use pairwise slope medians instead of a mean?
- How does RANSAC decide which points are inliers?
- Which of the three would you pick if you expect >50% of points to be outliers? *(RANSAC, generally — Theil-Sen's breakdown point is bounded well below 50%, and Huber assumes the majority of residuals are "normal.")*

## Common mistakes

- Applying robust regression as a substitute for actually inspecting/cleaning the data — it should complement, not replace, EDA.
- Picking RANSAC's inlier threshold arbitrarily — it needs to reflect the expected noise scale of *good* points, not be a generic default.
- Assuming robust regressors also fix violations like non-linearity or heteroscedasticity — they specifically target outliers, not every OLS assumption.

## Related notes

- [Regularization](regularization.md) — a different failure mode (multicollinearity/overfitting) with a different fix.
- [Quantile Regression](quantile-regression.md) — pinball loss is also less sensitive to outliers than squared error, though its main purpose is interval prediction rather than robustness per se.
