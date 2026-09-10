# Generative Classification: GDA, QDA, LDA

## Discriminative vs. generative models

Two fundamentally different ways to approach classification:

- **Discriminative models** directly estimate `P(y|x)` — the conditional
  distribution of the label given the features. They never model how `x`
  itself is distributed; they just learn a decision function that maps `x`
  to a label (or a probability over labels). Example: **logistic
  regression**.
- **Generative models** estimate the *joint* distribution `P(x,y)`, factored
  as `P(x,y) = P(x|y) · P(y)`. They learn what data from each class *looks
  like* (`P(x|y)`, the class-conditional density) and how common each class
  is (`P(y)`, the prior), then derive `P(y|x)` via Bayes' rule when a
  prediction is needed. Examples: **Naive Bayes, LDA, QDA**.

## Bayes' rule for classification

Given the joint factorization `P(x,y) = P(x|y)P(y)`, Bayes' rule gives the
posterior over labels:

```
P(y|x) = P(x,y) / P(x) = P(y)·P(x|y) / sum_{y' in Y} P(y')·P(x|y')
```

The denominator `sum_{y'} P(y')P(x|y')` is just `P(x)` — a normalizing
constant that doesn't depend on `y`. For **classification** we only need the
label that maximizes the posterior, so we can drop it entirely:

```
a(x) = argmax_{y in Y} P(y|x)
     = argmax_{y in Y}  P(y)P(x|y) / sum_{y'} P(y')P(x|y')
     = argmax_{y in Y}  P(y)P(x|y)
```

This is why generative classifiers only need to model `P(y)` and `P(x|y)` —
never the full normalized posterior. Estimating the prior `P(y)` is trivial:
it's just the empirical class frequency,

```
P(y) = #{i : y_i = y} / N
```

Everything interesting is in how you choose to model `P(x|y)`. Gaussian
Discriminant Analysis (QDA/LDA below) assumes it's Gaussian; Naive Bayes
(see [Naive Bayes](naive-bayes.md)) assumes conditional feature
independence instead.

## Gaussian Discriminant Analysis / QDA

**Assumption:** the features of every class `y` follow a multivariate
Gaussian, with a mean and covariance specific to that class:

```
p(x|y) = 1 / ((2*pi)^(n/2) * |Sigma_y|^(1/2)) * exp( -1/2 * (x - mu_y)^T * Sigma_y^(-1) * (x - mu_y) )
```

where `n` is the feature dimension, `mu_y` is the class-`y` mean vector, and
`Sigma_y` is the class-`y` covariance matrix. Because it allows an
independent covariance matrix per class, this model is called **Gaussian
Discriminant Analysis (GDA)**, or more specifically **Quadratic
Discriminant Analysis (QDA)** — the "quadratic" naming will be justified by
the decision-boundary derivation below.

### Maximum-likelihood parameter estimates

The likelihood of the whole dataset under this model (including the class
priors) is

```
L(P(Y), mu, Sigma) = prod_{i=1}^N  p(x_i | y_i, mu_{y_i}, Sigma_{y_i}) * P(y_i)
```

Maximizing this factorizes cleanly per class (each class's data only
touches that class's parameters), and the maximizer is exactly the sample
mean and sample covariance restricted to that class:

```
mu_y    = ( sum_i x_i * 1[y_i = y] ) / ( sum_i 1[y_i = y] )

Sigma_y = ( sum_i (x_i - mu_y)(x_i - mu_y)^T * 1[y_i = y] ) / ( sum_i 1[y_i = y] )
```

i.e. for each class, just compute the empirical mean and empirical
covariance of the points belonging to that class. No iterative optimization
needed — this is a closed-form MLE.

### Decision boundary: why it's quadratic

The boundary between classes `i` and `j` is the set of points where the
model is indifferent between them: `P(y_i|x) = P(y_j|x)`. Using the argmax
simplification above, this is equivalent to

```
P(x|y_i) * P(y_i) = P(x|y_j) * P(y_j)
```

Taking logs of both sides and moving everything to one side:

```
log P(x|y_i) + log P(y_i) - log P(x|y_j) - log P(y_j) = 0
```

Substituting the Gaussian log-density for each class,

```
-1/2 * (x - mu_yi)^T * Sigma_yi^(-1) * (x - mu_yi) - log((2*pi)^(n/2) * |Sigma_yi|^(1/2)) + log P(y_i)
  - [ -1/2 * (x - mu_yj)^T * Sigma_yj^(-1) * (x - mu_yj) - log((2*pi)^(n/2) * |Sigma_yj|^(1/2)) + log P(y_j) ]
  = 0
```

Expand the quadratic forms `(x - mu)^T Sigma^(-1) (x - mu)`: each one
contains a term `x^T Sigma^(-1) x` that is quadratic in `x`, plus a term
linear in `x`, plus a constant. Because `Sigma_yi` and `Sigma_yj` are
different matrices, the two quadratic terms `x^T Sigma_yi^(-1) x` and
`x^T Sigma_yj^(-1) x` do **not** cancel when you subtract — leaving an
equation that is quadratic in `x`. The boundary this traces out is a
general conic section (ellipse, parabola, hyperbola, or a pair of lines,
depending on the specific `Sigma`'s) — hence **Quadratic Discriminant
Analysis**.

## Linear Discriminant Analysis (LDA)

**Additional assumption:** all classes share the *same* covariance matrix,
`Sigma_y = Sigma` for every class `y` (only the means differ between
classes).

Look again at the quadratic term that appeared in the boundary equation
above: with class-specific covariances it was `x^T(Sigma_yi^(-1) -
Sigma_yj^(-1))x`. If `Sigma_yi = Sigma_yj = Sigma`, this term becomes
`x^T(Sigma^(-1) - Sigma^(-1))x = 0` — **it cancels exactly**. What's left in
the boundary equation is only linear-in-`x` and constant terms, so the
decision boundary between any two classes is a **hyperplane** — hence
**Linear Discriminant Analysis**.

### Parameter estimation: shared covariance

LDA still estimates a separate mean per class exactly as QDA does, but
pools *all* classes together to estimate one shared covariance matrix
(using each point's own class mean, but accumulating the outer products
across the whole dataset and normalizing by the total `N`, not per-class
counts):

```
Sigma_hat = (1/N) * sum_{i=1}^N (x_i - mu_hat_{y_i})(x_i - mu_hat_{y_i})^T
```

This is the key operational difference from QDA at training time: QDA
computes `K` separate `d x d` covariance matrices (one per class), LDA
computes exactly one.

## QDA vs. LDA — comparison

| | QDA | LDA |
|---|---|---|
| Covariance assumption | Separate `Sigma_y` per class | Single shared `Sigma` across all classes |
| Parameters for covariance | `O(K * d^2)` (`K` full `d x d` matrices) | `O(d^2)` (one full `d x d` matrix) |
| Data needed | More — each `Sigma_y` fit on only that class's data | Less — `Sigma` is fit on the whole dataset, pooled across classes |
| Decision boundary | Quadratic (curved: ellipse/parabola/hyperbola) | Linear (hyperplane) |
| Flexibility | Higher — can capture different spreads/orientations per class | Lower — assumes all classes have the same "shape" |
| Robustness with small data | Lower — more parameters, more variance in the estimate | Higher — pooling data across classes stabilizes the covariance estimate |
| Related to | — | Fisher's Linear Discriminant (same linear boundary, derived from a different — non-probabilistic — objective) |

**Rule of thumb:** if you have ample data per class and suspect the classes
genuinely have different spreads/orientations, QDA's extra flexibility pays
off. If data is limited (especially per-class) or you want a simpler, more
regularized model less prone to overfitting the covariance structure, LDA
is usually the safer default.

## See also

- [Naive Bayes](naive-bayes.md) — a different, non-Gaussian generative
  assumption (conditional feature independence) for `P(x|y)`.
- [Bayesian Inference](bayesian-inference.md) — the Bayesian view of
  parameter estimation more broadly (MLE here is the "flat prior" special
  case).
- [LDA vs. PCA](../dimensionality-reduction/lda-vs-pca.md) — LDA used as a
  *dimensionality-reduction* technique (via the between/within-class
  scatter-ratio objective) rather than derived from Bayes' rule; this file
  covers the probabilistic derivation that framing builds on.
