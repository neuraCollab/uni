# Naive Bayes

## The conditional independence assumption

Naive Bayes is a **generative** classifier (see
[Generative Classification: GDA, QDA, LDA](generative-classification-gda-lda.md)
for the discriminative-vs-generative background and the Bayes'-rule
argmax derivation it shares). Its distinguishing move is how it models the
class-conditional density $P(x \mid y)$ for a feature vector
$x = (x^1, x^2, \ldots, x^d)$:

$$
\begin{aligned}
P(x \mid y) &= P(x^1, x^2, \ldots, x^d \mid y) \\
&= P(x^1 \mid y) \cdot P(x^2 \mid y) \cdots P(x^d \mid y)
\end{aligned}
$$

That is: **given the class label, the features are assumed mutually
independent of each other.** This is the "naive" part — it's a strong
assumption that is almost always literally false (features are usually at
least somewhat correlated even within a class), but the classifier still
tends to work well in practice. The reason: classification only needs the
*argmax* over classes to come out right, not the actual posterior
probabilities to be numerically accurate or well-calibrated. Even when the
independence assumption distorts the magnitude of $P(x \mid y)$ for each class,
it often distorts all classes' scores in a similar enough way that the
*ranking* between classes — and hence the predicted label — stays correct.

### The resulting classification rule

Plugging the independence factorization into the generic generative-model
argmax rule gives:

$$a(x) = \arg\max_{y \in \mathcal{Y}} P(y) \cdot P(x^1 \mid y) \cdot P(x^2 \mid y) \cdots P(x^d \mid y)$$

Each $P(x^k \mid y)$ is a simple 1-D distribution estimated independently per
feature per class — trivial to fit even with limited data, which is a big
part of Naive Bayes's practical appeal.

## Laplace smoothing

**The zero-probability problem:** if some feature value $x_j$ never occurs
together with a given class in the training set, the raw empirical estimate
$P(X=x_j) = \#\{X=x_j\}/N$ is exactly $0$ for that (class, value) combination.
Since the classification rule *multiplies* the per-feature likelihoods
together, a single zero factor forces the **entire product to zero** for
that class — no matter how strongly every other feature points to it. In
other words, one unseen feature value can override all the other evidence.

**The fix:** add a pseudo-count $\alpha$ to every observed count before
normalizing:

$$\hat{P}(X = x_j) = \frac{\#\{X = x_j\} + \alpha}{N + m \alpha}$$

where $m$ is the number of distinct values $X$ can take, and $\alpha$ is a
smoothing hyperparameter ($\alpha = 1$ is the classic "add-one" / Laplace
smoothing; $\alpha < 1$ is sometimes called Lidstone smoothing).

**How $\alpha$ trades off empirical frequency vs. a uniform prior:**
- $\alpha \to 0$: the $+\alpha$ terms vanish, and this recovers the raw MLE
  frequency $\#\{X=x_j\}/N$ exactly — fully "trusting" the observed data, zero
  counts included.
- $\alpha$ large: the additive terms dominate both numerator and
  denominator, and $\hat{P}(X=x_j) \to 1/m$ for every value $x_j$ — pulling
  the estimate toward a **uniform distribution** over the $m$ possible
  values, regardless of what was actually observed.

So $\alpha$ is effectively a dial between "trust the data completely" (small
$\alpha$, risk of zero-probability overrides) and "trust a uniform prior"
(large $\alpha$, risk of washing out real signal in the data). It's exactly
the same idea as a Bayesian prior pulling a maximum-likelihood estimate
toward a default belief — see
[Bayesian Inference](bayesian-inference.md) for the general version of this
(Laplace smoothing is in fact the MAP estimate of a categorical/multinomial
parameter under a Dirichlet prior).

## Continuous features: kernel density estimation (Parzen windows)

Laplace smoothing assumes $X$ takes one of finitely many discrete values.
For a **continuous** feature, there's no finite set of values to count —
you need a density estimate instead. One standard non-parametric approach
(rather than assuming a parametric family like Gaussian, see below) is the
**Parzen window** / **kernel density estimator (KDE)**.

Simplest version — a box (uniform) kernel of half-width $h$: count how many
training points $x_j$ fall within $h$ of the query point $a$, and normalize:

$$\hat{p}(a) = \frac{1}{2h} \sum_j \mathbb{1}[ a - h < x_j < a + h ]$$

More generally, replace the hard indicator with a smooth **kernel function**
$K_h$:

$$\hat{p}(a) = \frac{1}{2h} \sum_j K_h(x_j - a)$$

Each training point contributes a small "bump" of density around itself
(shaped by $K_h$, scaled by the bandwidth $h$), and the estimate at any
point $a$ is the sum of all these bumps. This gives Naive Bayes a way to
estimate $P(x^k \mid y)$ for a continuous feature $x^k$ without committing to a
specific parametric shape.

**The parametric alternative** (also common, and what `GaussianNB` in
scikit-learn does): just assume each continuous feature is Gaussian within
each class, and estimate a mean/variance per (feature, class) pair via
maximum likelihood — much cheaper than KDE, and a fine choice when the
per-feature, per-class distribution really does look roughly bell-shaped.

## When to use Naive Bayes — and when not to

**Good fit:**
- **Text classification / spam filtering** — the classic use case. Features
  are typically word counts or TF-IDF weights over a huge, sparse
  vocabulary; the independence assumption ("word A appearing is independent
  of word B appearing, given the topic") is clearly false but empirically
  causes little harm, and the huge feature dimensionality is exactly where
  Naive Bayes's cheap per-feature estimation shines.
- Very fast to train (closed-form per-feature counting/estimation, no
  iterative optimization), works fine with relatively little data, and
  makes a strong, cheap baseline to beat before reaching for something more
  expensive.

**Where it fails:**
- **Strongly correlated / redundant features.** If two features are
  near-duplicates of each other (or one is derived from the other), Naive
  Bayes effectively counts their (near-identical) evidence *twice* in the
  product, over-weighting whatever they agree on relative to the
  independent features. This is the independence assumption actively
  hurting rather than just being harmlessly wrong — e.g. including both a
  raw feature and a scaled copy of it, or two features that are both proxies
  for the same underlying signal.

## See also

- [Generative Classification: GDA, QDA, LDA](generative-classification-gda-lda.md) —
  the Gaussian alternative to independence assumptions for $P(x \mid y)$.
- [Bayesian Inference](bayesian-inference.md) — the general Bayesian
  machinery (priors, MAP) that Laplace smoothing is a special case of.
