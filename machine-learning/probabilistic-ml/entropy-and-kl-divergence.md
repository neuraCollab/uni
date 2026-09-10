# Entropy, KL Divergence, and the Exponential Family

## What is it?

The information-theoretic toolkit that (a) quantifies how "hard to predict" a distribution is, (b) measures how different two distributions are in a way directly tied to classification loss functions, and (c) identifies the exponential family of distributions as the principled, "least-assumption" choice whenever you know only a few facts (moments) about your data — which is exactly the justification GLMs rely on. This note builds the machinery; [Generalized Linear Models](../linear-models/generalized-linear-models.md) is where it gets applied.

## Shannon entropy

**Definition (discrete):**

$$H(p) = -\sum_x p(x) \log p(x)$$

**Definition (continuous / differential entropy):**

$$H(p) = -\int p(x) \log p(x)\, dx$$

Entropy measures "how much knowledge" a distribution lacks, in two equivalent readings:

- **By meaning:** how hard it is to predict the value of a random variable drawn from $p$. A distribution concentrated on one outcome has entropy 0 (perfectly predictable); a uniform distribution over many outcomes has maximal entropy (least predictable).
- **By coding/complexity:** the average number of bits (if $\log$ is base 2) or nats (if natural log) you need to spend, on average, to transmit/encode a sample's value using an optimally-designed code for $p$. Low-entropy distributions compress well (few bits needed on average, because outcomes are concentrated); high-entropy distributions don't.

## KL divergence

**Definition:**

$$KL(p \| q) = \int p(x) \log\left(\frac{p(x)}{q(x)}\right) dx$$

(discrete case: replace the integral with a sum).

**Cross-entropy decomposition.** Split the log of a ratio into a difference of logs:

$$
\begin{aligned}
KL(p \| q) &= \int p(x) \log p(x)\, dx - \int p(x) \log q(x)\, dx \\
&= \left( -\int p(x) \log q(x)\, dx \right) - \left( -\int p(x) \log p(x)\, dx \right) \\
&= H(p, q) - H(p)
\end{aligned}
$$

where $H(p, q) = -\int p(x) \log q(x)\, dx$ is the **cross-entropy** of $q$ relative to $p$, and $H(p) = -\int p(x) \log p(x)\, dx$ is the (ordinary) entropy of $p$. So:

$$KL(p \| q) = \text{cross-entropy}(p, q) - \text{entropy}(p)$$

**Operational interpretation:** $KL(p\|q)$ is the extra average code length you pay if you design your encoding scheme assuming the data comes from $q$, when the data is actually distributed as $p$. $H(p)$ is the best possible (optimal) average code length, achieved only if you correctly encode for $p$; $H(p,q)$ is the actual average code length you get using a $q$-optimized code on $p$-distributed data. The gap between them is pure waste caused by using the wrong distribution — which is exactly what KL divergence quantifies. Loosely, "KL divergence is a kind of distance between distributions" — see below for why that phrase needs a caveat.

**Key properties:**

1. **$KL(p \| q) \geq 0$ for all $p, q$, with equality iff $p = q$ (almost everywhere).** This is provable via Jensen's inequality / Gibbs' inequality (concavity of $\log$ applied to $\mathbb{E}_p[\log(q(x)/p(x))]$); the full proof isn't reproduced here, but the intuition matches the coding interpretation above — you can never do *better* than the optimal code by using the wrong distribution, only worse or equal.
2. **$KL(p \| q) \neq KL(q \| p)$ in general — it is asymmetric.** Because of this (and because it doesn't satisfy the triangle inequality), KL divergence is a *divergence*, not a true metric/distance in the mathematical sense, even though it behaves like a "distance" informally (non-negative, zero iff equal).

### Why this matters for ML: cross-entropy loss *is* KL minimization

In classification, let $p$ be the true (empirical) label distribution and $q = p_{\text{model}}$ be the model's predicted distribution. Since the entropy $H(p)$ of the true label distribution is a fixed number — it doesn't depend on the model's parameters at all — minimizing cross-entropy $H(p, q)$ with respect to the model is *exactly the same optimization problem* as minimizing $KL(p \| q)$:

$$\arg\min_\theta H(p, q_\theta) = \arg\min_\theta \left[ H(p, q_\theta) - H(p) \right] = \arg\min_\theta KL(p \| q_\theta)$$

(subtracting the constant $H(p)$ doesn't change the argmin). This is the rigorous justification for why cross-entropy is the natural classification loss: minimizing it drives the model's predicted distribution $q$ as close as possible, in the KL sense, to the true label distribution $p$. It's the same underlying idea as the MLE-derivation in [MLE and Loss Functions](mle-and-loss-functions.md) — that file shows cross-entropy falls out of Bernoulli MLE; this is the matching information-theoretic view: cross-entropy minimization = KL-divergence minimization = likelihood maximization, three descriptions of the same optimization.

## Exponential family of distributions

**General form.** A distribution is a member of the exponential family if its density/mass function can be written as:

$$p(x \mid \theta) = \frac{1}{h(\theta)} g(x) \exp\left( \theta^\top u(x) \right)$$

- **$\theta$** — vector of real-valued *natural (canonical) parameters*: the values that pick out a specific distribution within the family.
- **$u(x)$** — vector of *sufficient statistics*: functions of the data whose expectations are what $\theta$ actually controls.
- **$g(x)$** — the *base measure*: depends on $x$ but not $\theta$.
- **$h(\theta)$** — the *normalizing constant* (partition function), chosen so the density integrates/sums to 1:

$$h(\theta) = \int g(x) \exp\left(\theta^\top u(x)\right) dx \quad \text{(sum, in the discrete case)}$$

This form covers most of the "named" distributions you already know — Gaussian (with known or unknown variance), Bernoulli, Binomial, Poisson, Gamma, Exponential, Beta, Dirichlet, and more — each recovered with the right choice of $\theta$, $u(x)$, $g(x)$, $h(\theta)$.

**Worked example — Bernoulli.** For $y \in \{0,1\}$ with success probability $\mu$:

$$p(y\mid\mu) = \mu^y (1-\mu)^{1-y} = \exp\left( y \log\left(\frac{\mu}{1-\mu}\right) + \log(1-\mu) \right)$$

Matching to the general form: $u(y) = y$, $\theta = \log\left(\frac{\mu}{1-\mu}\right)$ (the log-odds / logit — this is exactly the canonical parameter that shows up as the canonical link for logistic regression, see [Generalized Linear Models](../linear-models/generalized-linear-models.md)), $g(y) = 1$, and $-\log h(\theta) = \log(1-\mu)$.

## Koopman-Pitman-Darmois theorem

**Statement.** Let $p(x) = \frac{1}{h(\theta)} \exp(\theta^\top u(x))$ be an exponential-family distribution ($\theta$ a vector of length $n$), and suppose $\mathbb{E}[u_i(x)] = a_i$ for some fixed constants $a_1, \ldots, a_n$. Then:

- Among **all** distributions on the same support satisfying the same moment constraints $\mathbb{E}[u_i(x)] = a_i$ for $i = 1, \ldots, n$, the exponential-family distribution $p(x)$ has **maximum entropy**.
- It is **unique** with this property: any other distribution satisfying the same constraints and achieving the same maximum entropy must coincide with $p(x)$ almost everywhere (i.e. everywhere except possibly on a measure-zero set).

**Why this matters practically.** This is the rigorous justification for reaching specifically for exponential-family distributions when modeling noise or a target's conditional distribution in a GLM (see [Generalized Linear Models](../linear-models/generalized-linear-models.md)) — it's not an arbitrary mathematical convenience. If all you actually know (or are willing to assume) about your target's distribution is a handful of moments — e.g. "the mean is $\mu$" or "the mean and variance are $\mu$, $\sigma^2$" — then the **maximum-entropy distribution consistent with that knowledge** is the most honest, least-additional-assumption choice: it doesn't smuggle in any structure beyond what you actually asserted. And Koopman-Pitman-Darmois says that maximum-entropy distribution is automatically a member of the exponential family, with the constrained moments $u(x)$ appearing directly as its sufficient statistics. Concretely: if you only know a distribution's mean and variance, Gaussian is the max-entropy choice; if you only know a rate/count is non-negative with a fixed mean, Poisson (or Exponential, for continuous waiting times) is the max-entropy choice. This is the theoretical backbone of why GLMs restrict the "random component" to exponential-family distributions rather than allowing arbitrary distributions — it's the largest, best-justified class you get almost for free once you fix what you're willing to assume.

## Common interview questions

- What does entropy measure, and give both the "unpredictability" and "coding" interpretations.
- Derive the cross-entropy/entropy decomposition of KL divergence.
- Why is KL divergence not a true distance metric?
- Why is $KL(p\|q) \geq 0$ always true (sketch the Jensen's-inequality argument)?
- Show that minimizing cross-entropy loss in classification is equivalent to minimizing KL divergence to the true label distribution.
- State the Koopman-Pitman-Darmois theorem and explain why it justifies using exponential-family distributions in GLMs.
- Give the exponential-family form of the Bernoulli distribution and identify $\theta$, $u(x)$, $h(\theta)$.

## Common mistakes

- Calling KL divergence a "distance" without the caveat that it's asymmetric and fails the triangle inequality.
- Forgetting *which* direction of KL is used in ML: $KL(p_{\text{true}} \| p_{\text{model}})$, not $KL(p_{\text{model}} \| p_{\text{true}})$ — these are genuinely different objectives (the reversed direction is used in some variational-inference settings, but it's a different optimization with different behavior, e.g. mode-seeking vs. mass-covering).
- Treating "exponential family" as an arbitrary textbook definition rather than understanding it as *the* maximum-entropy family under moment constraints — that's the fact that makes it the natural choice for GLMs rather than an ad hoc restriction.
- Confusing entropy (a property of one distribution) with cross-entropy or KL divergence (properties of a pair of distributions).

## Related notes

- [MLE and Loss Functions](mle-and-loss-functions.md) — the likelihood-based derivation of the same losses this file reaches via information theory (Gaussian MLE ⟺ MSE, Bernoulli MLE ⟺ cross-entropy).
- [Generalized Linear Models](../linear-models/generalized-linear-models.md) — where the exponential family and Koopman-Pitman-Darmois get applied to derive the canonical link function.
- [Logistic Regression](../linear-models/logistic-regression.md) — Bernoulli as a concrete exponential-family member.
- [Classification Metrics](../model-evaluation/classification-metrics.md) — where cross-entropy/log-loss is used in practice.
