# Maximum Likelihood Estimation and the Origin of Loss Functions

## What is it?

A reframing of supervised learning: instead of picking a loss function by engineering intuition ("squared error feels reasonable"), you pick a **noise distribution** for how the target deviates from the model's prediction, and the "correct" loss function — negative log-likelihood — falls out automatically. Every common loss (MSE, MAE, log-loss/cross-entropy, Huber, ...) is the negative log-likelihood of *some* noise distribution. Choosing a loss is choosing a distributional assumption, whether you say so explicitly or not.

## Why?

Two ways to arrive at the same model:

- **Engineering view:** pick a loss function $L(y, f(x))$ directly, at a fixed $x$, and minimize its average over the training set. The loss is chosen by taste, convention, or empirical performance.
- **Probabilistic view:** model the target as signal plus noise,

$$y = f_\omega(x) + \varepsilon$$

where $\varepsilon$ is an additive noise term — reinterpreting "the model's prediction error" as "irreducible randomness in the data-generating process." Once you commit to a distribution for $\varepsilon$ (e.g. $\varepsilon \sim \mathcal{N}(0, \sigma^2)$), that noise distribution induces a conditional distribution over the target given $x$ and the parameters $\omega$:

$$p(y \mid x, \omega)$$

For fixed $x$ and $\omega$, $y - f_\omega(x) = \varepsilon$, so $p(y \mid x, \omega) = p_\varepsilon(y - f_\omega(x))$ — the conditional density of $y$ is just the noise density, shifted to be centered at the model's prediction.

The two views are equivalent: choosing a noise distribution for $\varepsilon$ is the probabilistic-side decision that corresponds exactly to the engineering-side decision of choosing a loss function. This note derives that correspondence precisely, because "what does MSE assume about your data?" is a question worth being able to answer rigorously, not just gesture at.

## How does it work?

### Maximum Likelihood Estimation (MLE)

**Goal:** find the parameters $\omega_{\text{MLE}}$ such that the model $p(y \mid x, \omega)$ assigns the highest possible probability (density) to the data actually observed. Treat the joint density of the observed targets, viewed as a function of $\omega$ with the data fixed, as the **likelihood function**.

Assuming the training examples are i.i.d. given $x_i, \omega$:

$$
\begin{aligned}
\omega_{\text{MLE}} &= \arg\max_\omega\, p(y \mid X, \omega) \\
&= \arg\max_\omega \prod_{i=1}^N p(y_i \mid x_i, \omega) && \text{(independence → product)} \\
&= \arg\max_\omega \prod_{i=1}^N p_\varepsilon(y_i - f_\omega(x_i)) && \text{(substitute the noise density)}
\end{aligned}
$$

**Log-likelihood.** Products of many small probabilities are numerically unpleasant and analytically awkward to differentiate (product rule blows up). Since $\log$ is strictly monotonic increasing, it preserves the location of the maximum, so:

$$
\begin{aligned}
\ell(y \mid X, \omega) &= \log \prod_i p_\varepsilon(y_i - f_\omega(x_i)) = \sum_{i=1}^N \log p_\varepsilon(y_i - f_\omega(x_i)) \\
\omega_{\text{MLE}} &= \arg\max_\omega\, \ell(y \mid X, \omega)
\end{aligned}
$$

**Negative log-likelihood (NLL) as a loss.** Optimization tooling is conventionally built around *minimization*. Flipping the sign turns the maximization into an equivalent minimization — literally just reflecting the objective vertically, so the old peak becomes a valley:

$$\omega_{\text{MLE}} = \arg\max_\omega\, \ell(y \mid X, \omega) \quad \Longleftrightarrow \quad \arg\min_\omega \sum_{i=1}^N \left[ -\log p_\varepsilon(y_i - f_\omega(x_i)) \right]$$

That sum of $-\log p_\varepsilon(\text{residual})$ terms *is* the loss function. This is the general recipe: **$\text{loss}(y, \hat{y}) = -\log p_\varepsilon(y - \hat{y})$**, for whatever noise density $p_\varepsilon$ you assumed. Everything below is just plugging in specific $p_\varepsilon$.

### The key derivation: Gaussian noise ⟺ MSE

Assume $\varepsilon \sim \mathcal{N}(0, \sigma^2)$, i.e.

$$p_\varepsilon(r) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{r^2}{2\sigma^2}\right)$$

Take the log:

$$
\begin{aligned}
\log p_\varepsilon(r) &= -\frac{r^2}{2\sigma^2} - \log\left(\sqrt{2\pi\sigma^2}\right) \\
&= -\frac{r^2}{2\sigma^2} + \text{const} && \text{(const doesn't depend on } \omega \text{)}
\end{aligned}
$$

So the NLL objective becomes:

$$
\begin{aligned}
\sum_i \left[ -\log p_\varepsilon(y_i - f_\omega(x_i)) \right] &= \sum_i \left[ \frac{(y_i - f_\omega(x_i))^2}{2\sigma^2} \right] + N \cdot \text{const} \\
&= \frac{1}{2\sigma^2} \sum_i (y_i - f_\omega(x_i))^2 + \text{const}
\end{aligned}
$$

$\frac{1}{2\sigma^2}$ and the additive constant don't depend on $\omega$, so they don't affect the $\arg\min$. What's left is exactly:

$$\arg\min_\omega \sum_i (y_i - f_\omega(x_i))^2$$

**— ordinary least squares / MSE minimization.** This is not an analogy or a loose parallel: assuming additive Gaussian noise and running MLE is *algebraically identical* to minimizing sum of squared residuals. This is the rigorous answer to "why do we use MSE, and what does it silently assume about your data?" — it assumes residuals are i.i.d. Gaussian around zero. If that assumption is wrong (e.g. your errors are heavy-tailed, or asymmetric), MSE is no longer the loss implied by MLE, and using it anyway means training against a mismatched noise model.

### Laplace noise ⟺ MAE

Assume $\varepsilon$ follows a Laplace (double-exponential) distribution instead:

$$p_\varepsilon(r) = \frac{1}{2b} \exp\left(-\frac{|r|}{b}\right)$$

Take the log:

$$\log p_\varepsilon(r) = -\frac{|r|}{b} - \log(2b) = -\frac{|r|}{b} + \text{const}$$

NLL objective:

$$\sum_i \left[ -\log p_\varepsilon(y_i - f_\omega(x_i)) \right] = \frac{1}{b} \sum_i |y_i - f_\omega(x_i)| + \text{const}$$

$$\arg\min_\omega \sum_i |y_i - f_\omega(x_i)|$$

**— exactly MAE (mean absolute error) minimization.** So the choice between MSE and MAE is, underneath, a choice between assuming Gaussian vs. Laplace noise.

**Why this explains MAE's outlier-robustness:** the Laplace distribution has heavier tails than the Gaussian (its density decays as $\exp(-|r|/b)$ — linearly in $|r|$ inside the exponent — instead of $\exp(-r^2/(2\sigma^2))$, quadratically in $r$). A heavier-tailed noise model assigns much higher probability to large residuals being "just noise," rather than treating them as astronomically unlikely the way the Gaussian does. That's precisely why MAE punishes large residuals less severely than MSE (linear penalty vs. quadratic penalty), and why MAE-trained models are less dragged around by outliers: the implicit noise model MAE assumes considers outliers unsurprising, while MSE's implicit Gaussian model considers them almost impossible and therefore weights fitting them very heavily. The general pattern: **the outlier-robustness of a loss is a direct readout of how heavy-tailed its implied noise distribution is.** (This is also the same reasoning behind Huber loss — quadratic near zero, linear in the tails — as an MLE-consistent compromise between a Gaussian's center and a Laplace's tails.)

### Connection to classification: Bernoulli noise ⟺ log-loss / cross-entropy

The same recipe applies to classification once you write down the right conditional distribution. For binary targets $y \in \{0, 1\}$ with model output $p = \sigma(f_\omega(x)) \in (0, 1)$ (e.g. sigmoid of a linear score), model $y \mid x, \omega$ as Bernoulli with parameter $p$:

$$p(y \mid x, \omega) = p^y (1 - p)^{1 - y}$$

Log-likelihood for one example:

$$\log p(y \mid x, \omega) = y \log(p) + (1-y)\log(1-p)$$

Negative log-likelihood, summed over the dataset:

$$\sum_i \left[ -\left( y_i \log(p_i) + (1 - y_i)\log(1 - p_i) \right) \right]$$

— exactly **binary cross-entropy / log-loss**, the standard classification loss (see [Logistic Regression](../linear-models/logistic-regression.md) for the sigmoid mechanics, and [Classification Metrics](../model-evaluation/classification-metrics.md) for how log-loss is used and evaluated in practice). This file is the "why" companion to that "what": log-loss isn't an arbitrary penalty for wrong-confident predictions, it is the MLE-consistent loss under an assumed Bernoulli distribution of the label given the model's predicted probability, exactly the way MSE is the MLE-consistent loss under assumed Gaussian noise. The deeper reason Bernoulli (and Gaussian) are the "natural" distributions to reach for here — rather than arbitrary choices — is the max-entropy justification in the exponential family; see [Entropy and KL Divergence](entropy-and-kl-divergence.md).

## Practical takeaway

Choosing a loss function is *implicitly* choosing an assumed noise/error distribution for your targets, whether or not you ever write that distribution down:

| Loss | Implied noise distribution | Behavior |
|---|---|---|
| MSE | Gaussian, $\mathcal{N}(0, \sigma^2)$ | Penalizes large residuals quadratically → sensitive to outliers |
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
- Forgetting that the equivalence between MLE-Gaussian and MSE is exact (not approximate) — the extra constants ($\frac{1}{2\sigma^2}$, normalizing terms) genuinely don't affect the $\arg\min_\omega$, but do matter if you need the actual likelihood value (e.g. for model comparison via AIC/BIC, or for a calibrated $\sigma$ estimate).
- Assuming cross-entropy is only justifiable "because it's convex and penalizes confident wrong answers" — that's a true and useful property, but the deeper justification is the Bernoulli MLE derivation above.
- Conflating MAE's robustness with "it's just less sensitive to big numbers" — the rigorous reason is the heavier tail of the implied Laplace distribution relative to the Gaussian.

## Related notes

- [Entropy and KL Divergence](entropy-and-kl-divergence.md) — the information-theoretic view of why cross-entropy is the natural classification loss, and the exponential-family background for *why* Gaussian/Bernoulli/etc. are principled choices of noise distribution in the first place.
- [Generalized Linear Models](../linear-models/generalized-linear-models.md) — generalizes this whole idea: any exponential-family noise distribution plus a link function gives you an MLE-consistent model and loss (deviance).
- [Logistic Regression](../linear-models/logistic-regression.md) — the sigmoid + log-loss mechanics referenced above.
- [Classification Metrics](../model-evaluation/classification-metrics.md) — how log-loss and related metrics are used to evaluate classifiers in practice.
- [Regularization](../linear-models/regularization.md) — MSE/MAE as base losses before penalty terms are added.
