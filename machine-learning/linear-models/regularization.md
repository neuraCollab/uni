# Regularization in Linear Models: OLS, Ridge, Lasso, Elastic Net, LARS, OMP

## Baseline: Ordinary Least Squares (OLS)

**What is it?** Fit `y = Xw + b` by minimizing the sum of squared residuals:

```
L(w) = ||y - Xw||^2
```

Closed-form solution via the **normal equations**:

```
w = (XᵀX)^(-1) Xᵀy
```

**Assumptions** (violate them and OLS degrades, though it still "runs"):
- **Linearity** — the true relationship between features and target is linear (in the parameters).
- **No/low multicollinearity** — features aren't strongly correlated with each other. If they are, `XᵀX` becomes ill-conditioned (near-singular), so `(XᵀX)^-1` blows up → huge, unstable coefficients that swing wildly with small data changes.
- **Homoscedasticity** — residual variance is constant across the range of predictions (not fanning out).
- **Errors uncorrelated / no autocorrelation** — important for time series.
- (For valid inference, also normally-distributed errors — not required just to fit a point estimate.)

**When OLS fails in practice:** `n_features > n_samples`, near-duplicate/collinear features, or noisy high-dimensional data → overfitting, unstable coefficients. This is exactly the gap regularization closes.

**Non-Negative Least Squares (NNLS):** OLS with the constraint `w ≥ 0` (`LinearRegression(positive=True)` in sklearn). Useful whenever negative weights are physically meaningless — e.g. spectral unmixing (concentrations can't be negative), image reconstruction, blending forecasts. Same normal-equations problem, solved with a constrained (active-set / NNLS) solver instead of a closed form.

## Why regularize?

Diagnosing overfitting: compare train vs. test error (or R²). If train error is low but test error is much higher, the model has memorized noise instead of learning signal — regularization is one of the standard fixes (see [Bias-Variance Tradeoff](../model-evaluation/bias-variance-tradeoff.md)). A well-fit model has train and test error close together.

All the methods below minimize `loss + penalty(w)`:

```
L(w) = ||y - Xw||^2 + penalty(w)
```

## Ridge Regression (L2)

**Penalty:** `alpha * ||w||_2^2` (sum of squared coefficients).

**Why:** Shrinks all coefficients toward zero but rarely to exactly zero. Directly counteracts the `XᵀX` ill-conditioning problem: Ridge's closed form is

```
w = (XᵀX + alpha·I)^(-1) Xᵀy
```

Adding `alpha·I` to the diagonal makes the matrix always invertible, which is precisely why Ridge is a numerically stable fallback when features are collinear. Large/unstable coefficient blow-ups from correlated features get "deflated."

**Connection to PCA/SVD:** `XᵀX` is (up to scaling) the feature covariance matrix once data is centered — the same matrix PCA eigendecomposes. Ridge can be viewed as shrinking coefficients more aggressively along the low-variance principal-component directions (where `XᵀX` is close to singular) and less along high-variance directions.

**When to use:** many correlated/redundant features, you want to keep all features (no sparsity), you mainly want variance reduction.

**When not to:** you need automatic feature selection.

- `Ridge(alpha=...)`, tune with `RidgeCV(alphas=...)` (efficient leave-one-out CV built in).
- `RidgeClassifier` — solves the classification problem as a regression on ±1-encoded targets (least-squares fit + sign/argmax decision), which makes it noticeably faster to train than `LogisticRegression`, at the cost of not producing calibrated probabilities. `RidgeClassifierCV` adds built-in alpha search.

## Lasso Regression (L1)

**Penalty:** `alpha * ||w||_1` (sum of absolute coefficients).

**Why:** The L1 penalty has corners at zero in weight space, so the optimum frequently lands exactly on an axis — many coefficients become **exactly 0**. Lasso performs implicit **feature selection** as a side effect of regularization.

**Low alpha:** weak regularization, almost all coefficients nonzero (model "trusts" every feature).
**High alpha:** strong regularization, more coefficients pushed to exactly zero → sparser, simpler model, less overfitting, but risk of underfitting if alpha is too large.

**Tuning alpha:**
- `LassoCV(cv=5)` — cross-validated search over an alpha path; look for the alpha minimizing CV MSE (`model.mse_path_`, `model.alpha_`).
- `LassoLarsIC(criterion='aic' | 'bic')` — instead of CV, pick alpha via an **information criterion** on the training fit alone (no folds needed → much faster). AIC and BIC usually pick nearly the same alpha as CV; **AIC is generally preferred** — it's more stable and cheaper to compute (`O(2·df)` vs. BIC's `O(log N · df)` penalty term, though both are cheap — the real win over CV is skipping the refit-per-fold cost entirely). Requires `n_samples > n_features` to be well-defined.
  - Intuition for **degrees of freedom (df)**: the number of independent quantities that were free to be estimated (roughly, the number of nonzero coefficients along the Lasso path). More df → the model can fit the training data more closely → the information criterion penalizes it more, to guard against overfitting.
- `LassoLarsCV` — CV but computed along the LARS path, typically faster than plain `LassoCV` on small/medium datasets.

**Lasso ↔ SVM regularization equivalence:** scikit-learn's Lasso `alpha` and an SVM's `C` parametrize the same tradeoff (loss vs. penalty) in inverted form:

```
alpha = 1 / C
# or, accounting for how sklearn scales the Lasso objective by n_samples:
alpha = 1 / (n_samples * C)
```

Useful to know in interviews: a small `alpha` (weak Lasso penalty) corresponds to a **large** `C` (SVM cares much more about fitting the data, weak regularization), and vice versa.

**Multi-task Lasso:** `MultiTaskLasso` / `MultiTaskLassoCV` fit several regression targets jointly while **sharing the sparsity pattern** across tasks — if a feature is irrelevant, it's zeroed out for *all* targets at once (an L2-over-tasks, L1-over-features group penalty), rather than each task selecting features independently. Useful when tasks are related and you expect the same subset of features to matter for each of them.

## Elastic Net (L1 + L2)

**Penalty:** a convex mix of both:

```
alpha * [ l1_ratio * ||w||_1 + (1 - l1_ratio) * 0.5 * ||w||_2^2 ]
```

`l1_ratio` (0 to 1) controls the mix: `l1_ratio=1` → pure Lasso, `l1_ratio=0` → pure Ridge.

**Why it exists:** Lasso has two known weaknesses—
1. With strongly correlated features, Lasso tends to arbitrarily pick *one* of them and zero out the rest, instead of spreading weight across the group.
2. When `n_features > n_samples`, Lasso saturates at selecting at most `n_samples` nonzero coefficients.

Elastic Net's L2 component stabilizes selection among correlated features (they tend to get similar, nonzero coefficients — a "grouping effect") while the L1 component still gives sparsity, fixing both issues.

**When to use:** correlated features where you still want feature selection; a safer general-purpose default than pure Lasso when you're unsure how correlated your features are.

- `ElasticNet(alpha=..., l1_ratio=...)`, tuned via `ElasticNetCV`.
- `MultiTaskElasticNetCV` for the multi-output version (same joint-sparsity idea as `MultiTaskLasso`).

## Sparse / Greedy Methods: LARS and OMP

### LARS (Least Angle Regression)

An efficient algorithm for computing the **entire Lasso regularization path** (all alphas at once) in roughly the cost of one OLS fit — very useful for high-dimensional, moderate-sample data.

**How it works (intuition):** Start with all coefficients at 0. Find the feature most correlated with the residual and start moving its coefficient in that direction. As soon as another feature becomes *equally* correlated with the (shrinking) residual, instead of committing fully to the first feature, LARS moves in the direction **equiangular** between the two (an angle-bisecting direction) — hence "least angle." Repeat, adding one feature to the active set at a time, until all features are active or the full OLS solution is reached.

The result is a full piecewise-linear coefficient path — exactly what you need for efficient cross-validation over alpha.

**When to use:** high-dimensional (`p >> n`) data, when you want the whole Lasso path cheaply. **When not to:** very noisy data — being greedy on correlations makes LARS somewhat more sensitive to noise than coordinate-descent Lasso.

### Orthogonal Matching Pursuit (OMP)

Also a greedy sparse-approximation method, but with an explicit hard sparsity constraint rather than an L1 penalty. Two equivalent formulations:

| Formulation | What's fixed | What's minimized |
|---|---|---|
| `n_nonzero_coefs` | number of nonzero coefficients (k) | reconstruction error |
| `tol` | error tolerance | number of nonzero coefficients |

**How it works:** greedily pick the feature most correlated with the current residual, add it to the active set, **refit least squares exactly on the active set** (this "orthogonal" refit/re-projection step is what distinguishes OMP from plain matching pursuit), update the residual, repeat until the stopping criterion (`k` features reached, or residual norm ≤ `tol`) is hit.

**When to use:** you know (or want to directly control) exactly how many features the final model should use — e.g. hard interpretability or compute budgets. Common in signal processing / compressed sensing.

## Common interview questions

- Why does Ridge (L2) shrink coefficients but rarely to zero, while Lasso (L1) produces exact zeros? *(Geometric argument: L1's diamond-shaped constraint region has corners on the axes; the L2 ball is smooth, so the least-squares contours almost never touch it exactly on an axis.)*
- Why is Ridge more numerically stable than OLS under multicollinearity?
- What does `alpha` control, and what happens as `alpha → 0` / `alpha → ∞` for Ridge and Lasso?
- When would you prefer Elastic Net over Lasso?
- What's the LARS algorithm doing geometrically, and why is it efficient for computing the whole path?
- Difference between the two OMP formulations (fixed-k vs. tolerance-based)?
- How is Lasso's `alpha` related to an SVM's `C`?
- What are the normal-equation assumptions, and what breaks when `n_features > n_samples`?

## Common mistakes

- Not scaling features before Ridge/Lasso/ElasticNet — the penalty is applied uniformly to raw coefficient magnitudes, so unscaled features get unfairly penalized relative to their natural scale. Always `StandardScaler` first.
- Reading Lasso zero-coefficients as "these features are useless" — they may just be **correlated with** a feature that got picked instead (Lasso arbitrarily favors one of a correlated group).
- Using `LassoLarsIC` when `n_samples <= n_features` — AIC/BIC formulas assume `n > p`.
- Forgetting that `RidgeClassifier` doesn't give you calibrated `predict_proba`-style probabilities the way `LogisticRegression` does (it uses a least-squares decision rule, not a proper probabilistic model — see [Logistic Regression](logistic-regression.md)).

## Example

See [`code/regularization_pipeline.py`](code/regularization_pipeline.py) for a side-by-side comparison of OLS/Ridge/Lasso/ElasticNet, and the per-method scripts: [`code/ridge.py`](code/ridge.py), [`code/lasso.py`](code/lasso.py), [`code/elasticnet.py`](code/elasticnet.py), [`code/lars_omp.py`](code/lars_omp.py).

## Related notes

- [Bayesian Regression](bayesian-regression.md) — regularization as a prior, rather than a fixed penalty.
- [Robust Regression](robust-regression.md) — handling outliers, a different failure mode than multicollinearity.
- [Bias-Variance Tradeoff](../model-evaluation/bias-variance-tradeoff.md)
- [Kernel Ridge Regression](../kernel-methods/kernel-ridge-regression.md) — Ridge's kernelized, nonlinear cousin.
