# Bayesian Inference: Priors, Conjugacy, MAP, and Model Selection

## The frequentist-vs-Bayesian shift

Classical ("frequentist") estimation treats a model parameter $\theta$ as a
single fixed but unknown true value, and looks for the one point estimate
that best explains the data (e.g. maximum likelihood). The Bayesian
approach makes a fundamental philosophical move: instead of hunting for one
"correct" value of $\theta$, treat $\theta$ itself as a **random variable**
and maintain a full **distribution** over which values are plausible.

- **Prior**, $p(\theta)$ (also written $p(\omega)$, or $p(\theta, b)$ when the
  parameter set includes multiple pieces like weights and a bias): your
  belief about $\theta$ *before* seeing any data.
- **Posterior**, $p(\theta \mid \text{data})$: your updated belief about $\theta$
  *after* seeing data — which values are plausible, and which aren't, given
  what was observed.

Rather than compute one "correct" $\theta$, you build the whole distribution
$p(\theta \mid \text{data})$, and can answer questions like "how confident are we?" or
"what's the range of plausible predictions?" — not just "what's the single
best guess?"

## Bayes' rule for parameters

$$p(\theta \mid \text{data}) \propto \underbrace{p(\text{data} \mid \theta)}_{\text{likelihood}} \, \underbrace{p(\theta)}_{\text{prior}}$$

(The proportionality hides the normalizer $p(\text{data}) = \int p(\text{data}\mid\theta)
p(\theta)\, d\theta$, which doesn't depend on $\theta$ and is often the hard
part to compute — see Bayesian model selection below.)

## Conjugate priors

A prior $p(\theta)$ is **conjugate** to a likelihood $p(\text{data}\mid\theta)$ if the
resulting posterior $p(\theta\mid\text{data})$ is in the *same family* of
distributions as the prior. This matters practically because it turns
Bayesian updating into simple closed-form arithmetic on the family's own
parameters, instead of an intractable integral.

Conjugate priors exist systematically whenever the likelihood belongs to
the **exponential family** (see
[Entropy, KL Divergence, and the Exponential Family](entropy-and-kl-divergence.md#exponential-family-of-distributions)
for the general exponential-family form and why it's the natural class of
distributions to reason about here). For a likelihood of exponential-family
form, you can construct a matching prior with the same functional
signature — $p(\theta) \propto \frac{1}{h(\theta)} \exp(\eta^\top \theta)$ —
by a specific recipe tied to the likelihood's sufficient statistics. This
prior is guaranteed to combine with the likelihood to produce a posterior
in that same family — a conjugate pair. Classic examples: Beta prior +
Binomial likelihood -> Beta posterior; Gamma prior + Poisson likelihood ->
Gamma posterior; Gaussian prior + Gaussian likelihood (known variance) ->
Gaussian posterior; Dirichlet prior + Categorical/Multinomial likelihood ->
Dirichlet posterior (this last one is exactly what Laplace smoothing in
[Naive Bayes](naive-bayes.md) is doing under the hood — it's the MAP
estimate of a Dirichlet-Categorical conjugate pair).

## Sequential (online) Bayesian updating

Because the posterior only depends on the prior and the likelihood of the
data actually seen, Bayesian updating composes naturally: today's posterior
becomes tomorrow's prior when new data arrives.

$$p(\omega \mid \{x_i, y_i\}_{i=1}^M) = \frac{p(\{y_i\}_{i=N+1}^M \mid \{x_i\}_{i=N+1}^M, \omega) \cdot p(\omega \mid \{x_i, y_i\}_{i=1}^N)}{p(\{y_i\}_{i=N+1}^M \mid \{x_i\}_{i=N+1}^M)}$$

i.e. the posterior after seeing all $M$ points equals: take the posterior
after the first $N$ points as your new *prior*, then apply Bayes' rule
again using only the likelihood of the *new* batch ($N+1$ through $M$).

An important consistency property falls out of this: the final posterior is
the **same** regardless of whether you update on the whole dataset at once,
or incrementally in arbitrarily many batches, in any order — the math is
associative. This is what makes Bayesian methods naturally suited to
online/streaming learning: you never need to re-derive the posterior from
scratch when new data shows up, and batching data differently can't change
the final answer.

## MAP estimation

Maintaining a full posterior distribution is often more than you need (or
can afford to compute/store). **Maximum A Posteriori (MAP)** estimation
collapses the posterior back down to a single point estimate — the most
probable parameter value under the posterior:

$$\theta_{\text{MAP}} = \arg\max_\theta p(\theta \mid y) = \arg\max_\theta\, p(y \mid \theta) \cdot p(\theta)$$

This is a direct generalization of maximum likelihood estimation: **MAP
with a flat (uninformative) prior is exactly MLE** — if $p(\theta)$ is
constant, maximizing $p(y\mid\theta)p(\theta)$ reduces to maximizing
$p(y\mid\theta)$ alone.

### The regularization connection (a favorite interview insight)

Taking $-\log$ of the MAP objective turns the product into a sum, and turns
$\arg\max$ into $\arg\min$:

$$\theta_{\text{MAP}} = \arg\min_\theta \left[ \underbrace{-\log p(y\mid\theta)}_{\text{data term}} \underbrace{- \log p(\theta)}_{\text{prior term}} \right]$$

- If the prior $p(\theta)$ is **Gaussian** (zero-mean, isotropic), $-\log
  p(\theta)$ is proportional to $\|\theta\|_2^2$ — so MAP under a Gaussian
  prior is **exactly** equivalent to **L2 / Ridge regularization**, with
  the prior's variance controlling the effective regularization strength.
- If the prior $p(\theta)$ is **Laplace** (zero-mean), $-\log p(\theta)$ is
  proportional to $\|\theta\|_1$ — so MAP under a Laplace prior is
  **exactly** equivalent to **L1 / Lasso regularization**.

This is the cleanest bridge between the Bayesian framework and classical
penalized regression: regularization penalties aren't just ad-hoc
"shrink the weights" tricks, they're MAP estimation under an implicit prior
belief about parameter magnitudes. See
[Regularization](../linear-models/regularization.md) for the classical
(non-Bayesian) treatment of Ridge/Lasso, and
[Bayesian Regression](../linear-models/bayesian-regression.md) for a full
Bayesian regression model built on this idea.

## Bayesian model selection

The same machinery extends from choosing parameters within a model to
choosing *between models*. Let $J$ be a family of candidate models indexed
by a discrete $j$. The posterior probability of model $j$ given the data is

$$p(j \mid y, X) = \frac{p(y \mid X, j) \cdot p(j)}{\sum_{j' \in J} p(j', y \mid X)}$$

Pick the model with the highest posterior probability as the best. If every
model is treated as equally likely a priori ($p(j)$ uniform over $J$), this
reduces to maximizing what's called the **evidence** or **marginal
likelihood**:

$$p_j(y \mid X) = \int p_j(y \mid X, \omega) \, p_j(\omega) \, d\omega$$

Crucially, this integrates the model parameters $\omega$ **out** entirely —
it is not "plug in the best point estimate $\hat{\omega}$ and evaluate the
likelihood," it's "average the likelihood over every possible $\omega$,
weighted by how plausible that $\omega$ is under the prior." This integral
is often intractable in closed form, so in practice it's approximated —
e.g. via a Taylor expansion of the log-likelihood around its mode (the
**Laplace approximation**).

## BIC: a tractable approximation

The **Bayesian Information Criterion (BIC)** is the standard tractable
stand-in for full evidence-based model comparison — a "penalized
likelihood" score you compute directly from a fitted model, no integral
required:

$$\text{BIC} = D \log(N) - 2 \log\left( p(y \mid X, \hat{\omega}) \right)$$

- $D$ — number of model parameters. The $D \log(N)$ term is the
  **complexity penalty**: more parameters costs more, and the penalty grows
  with the log of the sample size $N$.
- $-2 \log p(y\mid X, \hat{\omega})$ — the **goodness-of-fit** term, evaluated at
  the fitted (MLE) parameters $\hat{\omega}$: a better fit lowers this term.

**Lower BIC is better** — you're trading off fit quality against model
complexity, and BIC formalizes exactly how much fit improvement is "worth"
one extra parameter as $N$ grows.

**BIC vs. AIC:** the Akaike Information Criterion is a close cousin,
$\text{AIC} = 2D - 2\log p(y\mid X,\hat{\omega})$ — same goodness-of-fit term, but a
complexity penalty of $2D$ instead of $D\log(N)$. Since $\log(N) > 2$
whenever $N > 7$, BIC penalizes extra parameters more harshly than AIC for
essentially any realistic dataset size, and the gap grows as $N$ grows — so
BIC systematically favors simpler models more than AIC does, especially on
large datasets. (AIC is derived from an information-theoretic/predictive
argument; BIC is derived as a large-$N$ approximation to the Bayesian
evidence — different motivations that happen to produce similarly-shaped
penalized-likelihood formulas.)

## See also

- [Generative Classification: GDA, QDA, LDA](generative-classification-gda-lda.md)
  and [Naive Bayes](naive-bayes.md) — MLE-based generative classifiers;
  everything here generalizes their point estimates to full posteriors.
- [Entropy, KL Divergence, and the Exponential Family](entropy-and-kl-divergence.md) —
  background on the exponential family that conjugate priors are built on.
- [Regularization](../linear-models/regularization.md) — the classical view
  of the Ridge/Lasso penalties that MAP estimation re-derives from priors.
- [EM Algorithm](em-algorithm.md) — a different (non-Bayesian) way of
  handling unobserved/latent structure via point estimates and expectation.
