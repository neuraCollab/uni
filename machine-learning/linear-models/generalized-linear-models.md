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

## Where GLM comes from: the exponential family and canonical link

The section above covers the loss-function side of GLM (deviance, per-family unit deviance, the saturated model). This section covers where the framework itself comes from — why exponential-family distributions specifically, and how the "canonical" link functions (identity, logit, log) are derived rather than chosen by convention.

### The three components, precisely

A GLM is built from exactly three pieces:

1. **Random component.** Assume the target, conditional on the input, follows *some* distribution from the exponential family:

   ```
   y | x  ~  ExponentialFamily(θ)
   ```

   This isn't an arbitrary restriction — the exponential family is specifically the class of *maximum-entropy* distributions under fixed-moment constraints (Koopman-Pitman-Darmois theorem; see [Entropy and KL Divergence](../probabilistic-ml/entropy-and-kl-divergence.md) for the full statement and derivation). If all you're willing to assume about your noise is a handful of moments, the exponential family is the least-additional-assumption, most "honest" choice consistent with that — which is why GLM restricts to it rather than allowing arbitrary distributions.

2. **Linear predictor.** A linear combination of the features, with unrestricted range over the reals:

   ```
   η = <x, ω>
   ```

3. **Link function.** A function `g` that connects the linear predictor `η` to the distribution's mean `μ = E[y]`:

   ```
   g(μ) = η = <x, ω>     ⟺     μ = g^{-1}(η)
   ```

   The link function's job is purely to translate the unrestricted range of `η` (all of `ℝ`) into whatever valid range `μ` must actually live in — positive for Poisson counts, `(0,1)` for a Bernoulli probability, and so on.

### Canonical parameterization

Rewrite the exponential family in its **canonical (natural) parameterization**, isolating a single scalar parameter `θ` that determines the location of the distribution:

```
p(y | θ, φ) = exp( (y·θ - a(θ)) / φ + b(y, φ) )
```

- **`θ`** — the natural/canonical parameter: it's what controls "where the distribution is centered" (the analogue of a mean parameter). This is the piece that will eventually get tied to `x` and `ω`.
- **`φ`** — a dispersion/scale parameter (e.g. variance-like). Often fixed in advance (e.g. `φ = 1`, as for Bernoulli/Poisson).
- **`a(θ)`** — a function specific to each distribution in the family, chosen so the density normalizes to integrate to 1.
- **`b(y, φ)`** — does not depend on `θ`; needed for normalization, but doesn't affect how the density depends on `θ`.

**The mean as a function of θ.** By the moment-generating structure of the exponential family (a consequence of the Koopman-Pitman-Darmois result — see [Entropy and KL Divergence](../probabilistic-ml/entropy-and-kl-divergence.md)):

```
μ = E[y] = φ · E[u₁(y)] = φ · ∂/∂θ ( a(θ) / φ ) = a'(θ)
```

i.e. the mean is simply the derivative of the normalizing function `a` with respect to the natural parameter `θ`.

### Deriving the canonical link function

Now introduce the dependence on `x` by setting the natural parameter equal to the linear predictor — the simplest, most direct way to let the features influence the distribution:

```
θ = <x, ω>
```

GLM defines the link function `g` by requiring it to connect `x, ω` to the mean:

```
g( E[y|x] ) = <x, ω>     ⟹     E[y|x] = g^{-1}(<x, ω>)
```

But we also know, from the canonical-parameterization derivation above, that `E[y] = a'(θ) = a'(<x, ω>)`. Combining these two expressions for `E[y|x]` pins down `g` uniquely:

```
g = (a')^{-1}
```

i.e. `g(μ) = θ` where `μ = a'(θ)`. This `g` is called the **canonical link function**.

**Why the canonical link is the default (but not mandatory).** Using `θ = <x, ω>` directly — i.e. letting the linear predictor *be* the natural parameter — is the simplest possible way to hook the features into the exponential family, and it has real practical payoffs: the resulting log-likelihood is concave in `ω`, giving clean convergence guarantees for Newton's method / IRLS (iteratively reweighted least squares), and it yields clean sufficient statistics. But it is a choice, not a requirement — any exponential-family random component can be paired with *any* valid link function, canonical or not. A standard example: **probit regression** pairs a Bernoulli random component with the inverse-Gaussian-CDF link (`Φ^{-1}`) instead of the canonical logit link — a non-canonical but perfectly valid GLM.

**Canonical links for the common cases:**

| Family | Canonical link `g` | `μ = g^{-1}(η)` | Model |
|---|---|---|---|
| Gaussian | identity: `g(μ) = μ` | `μ = η` | Ordinary linear regression — GLM's simplest case |
| Bernoulli | logit: `g(μ) = ln(μ/(1-μ))` | `μ = σ(η) = 1/(1+e^{-η})` | Logistic regression |
| Poisson | log: `g(μ) = ln(μ)` | `μ = e^η` | Poisson regression (see above) |

(See [Logistic Regression](logistic-regression.md) for the Bernoulli case in depth, and the deviance table above for how the loss side of each of these looks.)

## Related notes

- [Logistic Regression](logistic-regression.md) — the Bernoulli-family GLM, treated on its own given how central it is to interviews.
- [Regularization](regularization.md) — the L2 penalty term used here is the same Ridge penalty.
- [Entropy and KL Divergence](../probabilistic-ml/entropy-and-kl-divergence.md) — the exponential-family and Koopman-Pitman-Darmois background this section builds on.
- [MLE and Loss Functions](../probabilistic-ml/mle-and-loss-functions.md) — the general MLE-as-loss-function framework GLM is a structured special case of.
