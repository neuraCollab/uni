# Maximum Likelihood Estimation and the Origin of Loss Functions

## What is it?

A reframing of supervised learning: instead of picking a loss function by engineering intuition ("squared error feels reasonable"), you pick a **noise distribution** for how the target deviates from the model's prediction, and the "correct" loss function — negative log-likelihood — falls out automatically. Every common loss (MSE, MAE, log-loss/cross-entropy, Huber, ...) is the negative log-likelihood of *some* noise distribution. Choosing a loss is choosing a distributional assumption, whether you say so explicitly or not.

## Why?

Two ways to arrive at the same model:

- **Engineering view:** pick a loss function `L(y, f(x))` directly, at a fixed `x`, and minimize its average over the training set. The loss is chosen by taste, convention, or empirical performance.
- **Probabilistic view:** model the target as signal plus noise,

```
y = f_ω(x) + ε
```

where `ε` is an additive noise term — reinterpreting "the model's prediction error" as "irreducible randomness in the data-generating process." Once you commit to a distribution for `ε` (e.g. `ε ~ N(0, σ²)`), that noise distribution induces a conditional distribution over the target given `x` and the parameters `ω`:

```
p(y | x, ω)
```

For fixed `x` and `ω`, `y - f_ω(x) = ε`, so `p(y | x, ω) = p_ε(y - f_ω(x))` — the conditional density of `y` is just the noise density, shifted to be centered at the model's prediction.

The two views are equivalent: choosing a noise distribution for `ε` is the probabilistic-side decision that corresponds exactly to the engineering-side decision of choosing a loss function. This note derives that correspondence precisely, because "what does MSE assume about your data?" is a question worth being able to answer rigorously, not just gesture at.

## How does it work?

### Maximum Likelihood Estimation (MLE)

**Goal:** find the parameters `ω_MLE` such that the model `p(y | x, ω)` assigns the highest possible probability (density) to the data actually observed. Treat the joint density of the observed targets, viewed as a function of `ω` with the data fixed, as the **likelihood function**.

Assuming the training examples are i.i.d. given `x_i, ω`:

```
ω_MLE = argmax_ω p(y | X, ω)
       = argmax_ω  ∏_{i=1}^N p(y_i | x_i, ω)          (independence → product)
       = argmax_ω  ∏_{i=1}^N p_ε(y_i - f_ω(x_i))       (substitute the noise density)
```

**Log-likelihood.** Products of many small probabilities are numerically unpleasant and analytically awkward to differentiate (product rule blows up). Since `log` is strictly monotonic increasing, it preserves the location of the maximum, so:

```
l(y | X, ω) = log ∏_i p_ε(y_i - f_ω(x_i)) = sum_{i=1}^N log p_ε(y_i - f_ω(x_i))

ω_MLE = argmax_ω l(y | X, ω)
```

**Negative log-likelihood (NLL) as a loss.** Optimization tooling is conventionally built around *minimization*. Flipping the sign turns the maximization into an equivalent minimization — literally just reflecting the objective vertically, so the old peak becomes a valley:

```
ω_MLE = argmax_ω l(y | X, ω)  ⟺  argmin_ω  sum_{i=1}^N [ -log p_ε(y_i - f_ω(x_i)) ]
```

That sum of `-log p_ε(residual)` terms *is* the loss function. This is the general recipe: **loss(y, ŷ) = -log p_ε(y - ŷ)**, for whatever noise density `p_ε` you assumed. Everything below is just plugging in specific `p_ε`.

### The key derivation: Gaussian noise ⟺ MSE

Assume `ε ~ N(0, σ²)`, i.e.

```
p_ε(r) = 1/√(2πσ²) · exp(-r² / (2σ²))
```

Take the log:

```
log p_ε(r) = -r² / (2σ²) - log(√(2πσ²))
           = -r² / (2σ²) + const          (const doesn't depend on ω)
```

So the NLL objective becomes:

```
sum_i [ -log p_ε(y_i - f_ω(x_i)) ] = sum_i [ (y_i - f_ω(x_i))² / (2σ²) ] + N·const
                                    = (1/(2σ²)) · sum_i (y_i - f_ω(x_i))²  + const
```

`1/(2σ²)` and the additive constant don't depend on `ω`, so they don't affect the `argmin`. What's left is exactly:

```
argmin_ω  sum_i (y_i - f_ω(x_i))²
```

**— ordinary least squares / MSE minimization.** This is not an analogy or a loose parallel: assuming additive Gaussian noise and running MLE is *algebraically identical* to minimizing sum of squared residuals. This is the rigorous answer to "why do we use MSE, and what does it silently assume about your data?" — it assumes residuals are i.i.d. Gaussian around zero. If that assumption is wrong (e.g. your errors are heavy-tailed, or asymmetric), MSE is no longer the loss implied by MLE, and using it anyway means training against a mismatched noise model.

### Laplace noise ⟺ MAE

Assume `ε` follows a Laplace (double-exponential) distribution instead:

```
p_ε(r) = 1/(2b) · exp(-|r| / b)
```

Take the log:

```
log p_ε(r) = -|r| / b - log(2b) = -|r|/b + const
```

NLL objective:

```
sum_i [ -log p_ε(y_i - f_ω(x_i)) ] = (1/b) · sum_i |y_i - f_ω(x_i)| + const

argmin_ω  sum_i |y_i - f_ω(x_i)|
```

**— exactly MAE (mean absolute error) minimization.** So the choice between MSE and MAE is, underneath, a choice between assuming Gaussian vs. Laplace noise.

**Why this explains MAE's outlier-robustness:** the Laplace distribution has heavier tails than the Gaussian (its density decays as `exp(-|r|/b)` — linearly in `|r|` inside the exponent — instead of `exp(-r²/(2σ²))`, quadratically in `r`). A heavier-tailed noise model assigns much higher probability to large residuals being "just noise," rather than treating them as astronomically unlikely the way the Gaussian does. That's precisely why MAE punishes large residuals less severely than MSE (linear penalty vs. quadratic penalty), and why MAE-trained models are less dragged around by outliers: the implicit noise model MAE assumes considers outliers unsurprising, while MSE's implicit Gaussian model considers them almost impossible and therefore weights fitting them very heavily. The general pattern: **the outlier-robustness of a loss is a direct readout of how heavy-tailed its implied noise distribution is.** (This is also the same reasoning behind Huber loss — quadratic near zero, linear in the tails — as an MLE-consistent compromise between a Gaussian's center and a Laplace's tails.)

### Connection to classification: Bernoulli noise ⟺ log-loss / cross-entropy

The same recipe applies to classification once you write down the right conditional distribution. For binary targets `y ∈ {0, 1}` with model output `p = σ(f_ω(x)) ∈ (0, 1)` (e.g. sigmoid of a linear score), model `y | x, ω` as Bernoulli with parameter `p`:

```
p(y | x, ω) = p^y · (1 - p)^(1 - y)
```

Log-likelihood for one example:

```
log p(y | x, ω) = y·log(p) + (1-y)·log(1-p)
```

Negative log-likelihood, summed over the dataset:

```
sum_i [ -( y_i·log(p_i) + (1 - y_i)·log(1 - p_i) ) ]
```

— exactly **binary cross-entropy / log-loss**, the standard classification loss (see [Logistic Regression](../linear-models/logistic-regression.md) for the sigmoid mechanics, and [Classification Metrics](../model-evaluation/classification-metrics.md) for how log-loss is used and evaluated in practice). This file is the "why" companion to that "what": log-loss isn't an arbitrary penalty for wrong-confident predictions, it is the MLE-consistent loss under an assumed Bernoulli distribution of the label given the model's predicted probability, exactly the way MSE is the MLE-consistent loss under assumed Gaussian noise. The deeper reason Bernoulli (and Gaussian) are the "natural" distributions to reach for here — rather than arbitrary choices — is the max-entropy justification in the exponential family; see [Entropy and KL Divergence](entropy-and-kl-divergence.md).

## Practical takeaway

Choosing a loss function is *implicitly* choosing an assumed noise/error distribution for your targets, whether or not you ever write that distribution down:

| Loss | Implied noise distribution | Behavior |
|---|---|---|
| MSE | Gaussian, `N(0, σ²)` | Penalizes large residuals quadratically → sensitive to outliers |
| MAE | Laplace | Penalizes large residuals linearly → robust to outliers (heavier tails) |
| Log-loss / binary cross-entropy | Bernoulli | Standard classification loss, penalizes confident wrong predictions heavily |
| Huber | Gaussian near 0, Laplace in the tails | Compromise: quadratic near the mode, linear in the tails |

When picking (or justifying) a loss function, the honest interview-level question to ask is: "what distribution does this loss implicitly assume for the residuals, and is that assumption plausible for this data?" — e.g. if you know your target has occasional large outliers, MAE (or Huber) is the MLE-consistent choice, not a hack bolted on to "reduce outlier sensitivity."

## Common interview questions

- Derive the equivalence between MLE under Gaussian noise and OLS/MSE minimization.
- Why does assuming Laplace-distributed noise lead to MAE instead of MSE?
- What does minimizing log-loss assume about your label-generating process?
- If your residuals are known to be heavy-tailed, what loss function does MLE suggest, and why?
- Why do we take the log of the likelihood before optimizing, and why does that not change the argmax?
- Why is NLL minimization equivalent to likelihood maximization — what's actually happening when you negate the objective?

## Common mistakes

- Treating "loss function choice" as a purely engineering/empirical decision, unable to state what it assumes statistically.
- Forgetting that the equivalence between MLE-Gaussian and MSE is exact (not approximate) — the extra constants (`1/(2σ²)`, normalizing terms) genuinely don't affect the `argmin_ω`, but do matter if you need the actual likelihood value (e.g. for model comparison via AIC/BIC, or for a calibrated `σ` estimate).
- Assuming cross-entropy is only justifiable "because it's convex and penalizes confident wrong answers" — that's a true and useful property, but the deeper justification is the Bernoulli MLE derivation above.
- Conflating MAE's robustness with "it's just less sensitive to big numbers" — the rigorous reason is the heavier tail of the implied Laplace distribution relative to the Gaussian.

## Related notes

- [Entropy and KL Divergence](entropy-and-kl-divergence.md) — the information-theoretic view of why cross-entropy is the natural classification loss, and the exponential-family background for *why* Gaussian/Bernoulli/etc. are principled choices of noise distribution in the first place.
- [Generalized Linear Models](../linear-models/generalized-linear-models.md) — generalizes this whole idea: any exponential-family noise distribution plus a link function gives you an MLE-consistent model and loss (deviance).
- [Logistic Regression](../linear-models/logistic-regression.md) — the sigmoid + log-loss mechanics referenced above.
- [Classification Metrics](../model-evaluation/classification-metrics.md) — how log-loss and related metrics are used to evaluate classifiers in practice.
- [Regularization](../linear-models/regularization.md) — MSE/MAE as base losses before penalty terms are added.
