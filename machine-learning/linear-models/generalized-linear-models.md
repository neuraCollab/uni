# Generalized Linear Models (GLM)

## What is it?

A generalization of linear regression that allows the target `y` to follow **any distribution from the exponential dispersion family** (Gaussian, Bernoulli/Binomial, Poisson, Gamma, Tweedie, ...) instead of assuming Gaussian noise, and replaces squared-error loss with a distribution-appropriate loss called **deviance**.

## Why?

Plain linear regression (squared loss) implicitly assumes:
- target noise is Gaussian (symmetric, constant variance),
- the target can take any real value.

This is a bad fit whenever the target is:
- a **count** (0, 1, 2, ... — Poisson: e.g. number of claims, clicks, events per period),
- **binary/proportion** (Bernoulli/Binomial: e.g. click or not),
- strictly **positive, right-skewed, multiplicative-ish noise** (Gamma: e.g. insurance claim amounts, wait times).

For these targets, squared-error loss either allows nonsensical predictions (negative counts) or fits poorly because the true noise variance changes with the mean (heteroscedasticity that squared loss doesn't account for).

## How does it work?

**Deviance** replaces `(y - y_hat)^2` as the thing being minimized. For a distribution in the exponential dispersion family:

```
D(y, mu) = 2 * [ ll(y; y) - ll(y; mu) ]
```

where `ll(y; theta)` is the log-likelihood, `mu = y` is the **saturated model** (a hypothetical model that fits each observation exactly), and `mu` is the model's actual prediction. Deviance measures how far the actual model's log-likelihood is from the best-possible (saturated) log-likelihood — the smaller, the better the fit.

The **unit deviance** `d(y, mu)` is one observation's contribution: `D = sum_i d(y_i, mu_i)`.

| Family | Unit deviance `d(y, mu)` | Use case |
|---|---|---|
| Gaussian | `(y - mu)^2` | reduces to ordinary squared loss |
| Bernoulli (y in {0,1}) | `-2[y*ln(mu) + (1-y)*ln(1-mu)]` | binary classification (this is log-loss) |
| Poisson | `2[y*ln(y/mu) - (y - mu)]` | non-negative integer counts |
| Gamma | `2[(y-mu)/mu - ln(y/mu)]` | positive, right-skewed continuous values |

GLM fitting minimizes:

```
(1 / 2n) * sum_i d(y_i, y_hat_i)  +  (alpha / 2) * ||w||_2^2
```

i.e. average unit deviance plus an (optional) L2 penalty — equivalent to maximizing a regularized likelihood under the chosen distribution.

A **link function** `g` connects the linear predictor to the mean: `g(mu) = Xw`. The canonical choice for Poisson/Gamma is the **log link** (`mu = exp(Xw)`), which automatically keeps predictions positive — solving the "negative predicted count" problem outright.

## When to use / when not to use

**Use when:**
- Target is a count → `PoissonRegressor`.
- Target is a positive, skewed continuous value → `GammaRegressor`.
- You want one family that interpolates between these → `TweedieRegressor(power=...)` (`power=0`→Gaussian, `1`→Poisson, `2`→Gamma, `1<power<2`→compound Poisson-Gamma, common for insurance claim severity with many exact zeros).
- You care about getting the *variance structure* right, not just the mean.

**Avoid when:**
- Target is genuinely continuous and roughly symmetric around the prediction — plain OLS/Ridge/Lasso (Gaussian deviance = squared error) is simpler and just as correct.
- You need probabilistic classification with more standard tooling — `LogisticRegression` (Bernoulli GLM) is the standard entry point rather than manually configuring a GLM class, though conceptually it *is* a GLM.

## Common interview questions

- Why not just use OLS on count data?
- What is deviance, and how does it generalize squared error?
- What's the "saturated model," and why is it the reference point for deviance?
- What does a log link function accomplish here?
- How does Poisson regression differ from treating the counts as continuous and running OLS?
- What is Tweedie regression's `power` parameter doing?

## Common mistakes

- Using plain linear regression on count data and getting negative predicted counts — a hard signal that a log-link GLM (Poisson) is a better fit.
- Confusing deviance with residual sum of squares in general — they're only equal for the Gaussian family.
- Forgetting `power` selection for `TweedieRegressor` needs domain knowledge or a scan — it's not automatically tuned.

## Example

See [`code/glm.py`](code/glm.py) — a real worked Poisson regression example on simulated count data (the archived source only had a 4-line `TweedieRegressor` stub with no actual training/evaluation, so this note replaces it with a full fit + deviance comparison against a naive OLS baseline).

## Related notes

- [Logistic Regression](logistic-regression.md) — the Bernoulli-family GLM, treated on its own given how central it is to interviews.
- [Regularization](regularization.md) — the L2 penalty term used here is the same Ridge penalty.
