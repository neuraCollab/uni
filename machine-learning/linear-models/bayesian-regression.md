# Bayesian Regression

## What is it?

A regression method that treats the model's weights `w` as random variables with a **prior distribution**, then updates that belief using the training data (via Bayes' rule) to get a **posterior distribution** over `w` — instead of a single point estimate. The regularization strength is no longer a hyperparameter you have to tune externally: it falls out of the data itself.

## Why?

- Ridge/Lasso require you to pick `alpha` (by CV, AIC/BIC, etc.). Bayesian regression **estimates the equivalent of `alpha` automatically** from the data, as part of fitting.
- It gives you a **predictive uncertainty** (a standard deviation / credible interval per prediction), not just a point prediction — useful whenever "how confident is the model?" matters (e.g. active learning, risk-sensitive decisions).
- It naturally handles small/ill-posed datasets by letting the prior regularize the estimate.

## How does it work?

1. **Prior:** assume a distribution over the weights before seeing data, typically a zero-mean Gaussian: `w ~ N(0, lambda^-1 * I)`. `lambda` is the *precision* (inverse variance) of the prior — high `lambda` means "I strongly believe weights are near 0" (strong regularization); low `lambda` means a wide, weak prior.
2. **Likelihood:** assume Gaussian observation noise: `y | X, w ~ N(Xw, alpha^-1 * I)`, `alpha` = noise precision.
3. **Posterior:** combine prior and likelihood via Bayes' rule. For a Gaussian prior + Gaussian likelihood (conjugate case), the posterior over `w` is *also Gaussian*, with a closed-form mean and covariance. The **posterior mean** is the point prediction; the **posterior covariance** gives you the uncertainty band.
4. **Uninformative (flat/wide) priors:** used when you have no prior belief about the weights — a very wide/flat Gaussian. In the limit of an infinitely flat prior, the Bayesian posterior mean converges to the **MLE / plain OLS solution** — Bayesian regression subsumes OLS as a special case.
5. **Hyperparameters `alpha` and `lambda` aren't fixed by hand** — `BayesianRidge` estimates them from the data by maximizing the **marginal likelihood** (also called "evidence"): the likelihood of the data with the weights integrated out, `p(y | X, alpha, lambda) = ∫ p(y | X, w, alpha) p(w | lambda) dw`. This is different from plain MLE, which would maximize `p(y | X, w)` over `w` directly (and can overfit); maximizing the *marginal* likelihood instead automatically penalizes model complexity (an Occam's-razor effect), which is exactly why it can pick `alpha`/`lambda` without a separate validation set.

**MLE vs. marginal likelihood, briefly:**
- MLE: `argmax_w p(y | X, w)` — point-estimates `w` directly, no built-in complexity penalty.
- Marginal (type-II ML / "evidence"): `argmax_{alpha, lambda} p(y | X, alpha, lambda)` — averages over all plausible `w` weighted by the prior, so hyperparameters that made `w` overconfidently large get penalized (the corresponding predictions become uncertain/spread out, worsening the marginal likelihood).

## BayesianRidge vs. ARD Regression

Both are Bayesian linear models with a Gaussian prior on `w`, but they differ in how much structure that prior has:

| | `BayesianRidge` | `ARDRegression` |
|---|---|---|
| Prior precision | **one shared** `lambda` for all weights | **one `lambda_i` per weight** (Automatic Relevance Determination) |
| Effect | uniform shrinkage, like Ridge | per-feature shrinkage — irrelevant features can get `lambda_i -> infinity`, forcing `w_i -> 0` |
| Sparsity | rarely exactly sparse | **more sparse** — behaves more like Bayesian Lasso in practice |
| Cost | cheaper, more stable | more expensive, can be less stable with many features |

**Interview framing:** ARD is to BayesianRidge roughly what Lasso is to Ridge — same Bayesian machinery, but a more flexible (per-weight) prior lets it prune irrelevant features more aggressively.

## When to use / when not to

**Use when:** small-to-medium datasets, you want automatic regularization-strength selection, you need predictive uncertainty, or you're fitting a flexible basis (e.g. polynomial features) where overfitting risk is high without tuning.

**Avoid when:** you have a very large dataset (marginal-likelihood optimization and posterior covariance computation don't scale as well as plain Ridge/Lasso via coordinate descent or SGD), or you only need point predictions and speed matters more than uncertainty.

## Common interview questions

- What's the difference between a point estimate (OLS) and a Bayesian posterior?
- How does an "uninformative prior" relate back to OLS?
- What's being maximized when fitting `BayesianRidge` — and how is it different from plain MLE?
- Why does ARD tend to produce sparser weights than BayesianRidge?
- What do `alpha_` and `lambda_` mean after fitting a `BayesianRidge`?

## Common mistakes

- Confusing the *prior* precision (`lambda`, regularization strength) with the *noise* precision (`alpha`) — they play different roles in the model.
- Assuming Bayesian regression always gives sparse solutions — plain `BayesianRidge` does not (it shrinks smoothly like Ridge); only ARD's per-weight prior tends toward sparsity.
- Treating the predictive `std` as calibrated for any data distribution — it's only as good as the Gaussian-noise assumption underlying the model.

## Example

See [`code/bayesian_reg.py`](code/bayesian_reg.py) — fits `BayesianRidge` and `ARDRegression` to a noisy sine curve using a degree-3 polynomial basis, plots the predictive mean plus uncertainty band, and compares coefficient sparsity between the two.

## Related notes

- [Regularization](regularization.md) — the non-Bayesian, fixed-`alpha` view of the same shrinkage idea.
- [Generalized Linear Models](generalized-linear-models.md) — a different way of moving beyond squared-error loss.
