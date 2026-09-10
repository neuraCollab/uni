# The EM Algorithm and Mixture Models

## Motivation: from hard to soft clustering

Standard k-means assigns every point to exactly one cluster — a **hard**
assignment. But real data is often ambiguous: a point near the boundary
between two clusters plausibly belongs to *both*, to different degrees.
Models with **latent (hidden) variables** generalize k-means-style
clustering to **soft** assignment: instead of "object `i` belongs to
cluster `k`," you get "object `i` belongs `70%` to cluster `k` and `30%` to
cluster `k'`." The EM algorithm is, informally, "k-means plus soft
distribution over classes."

## Mixture models

A **mixture of distributions** models the overall density as a weighted
sum of component densities:

```
p(x) = sum_{k=1}^K  pi_k * p_k(x)

subject to:  sum_{k=1}^K pi_k = 1,   pi_k >= 0
```

- `K` — number of mixture components.
- `pi_k` — the mixing weight / prior probability of component `k` (how
  common that component is overall).
- `p_k(x)` — the density (continuous case) or probability mass function
  (discrete case) of component `k`.

The canonical example is the **Gaussian Mixture Model (GMM)**: each
`p_k(x)` is a multivariate Gaussian with its own mean `mu_k` and covariance
`Sigma_k`. Fitting a GMM means estimating all of `{pi_k, mu_k, Sigma_k}`
for `k = 1..K` from data — and, implicitly, recovering a soft cluster
assignment for every point.

## Why direct maximum likelihood is hard

If you knew which component generated each point (a latent/hidden variable
`Z`, one per data point), maximizing the **complete-data** log-likelihood
`log p(X, Z | theta)` would decompose into simple, independent per-component
problems — same as the per-class MLE in
[Generative Classification: GDA, QDA, LDA](generative-classification-gda-lda.md).
The problem is that `Z` is **unobserved**. Maximizing the actual
(incomplete-data) log-likelihood requires marginalizing `Z` out first:

```
log p(X | theta) = log sum_Z p(X, Z | theta)
```

The `log` of a **sum** over the unknown `Z` doesn't decompose into a nice
sum of logs the way `log` of a **product** would. So there is no simple
closed form for the joint maximizer over both `theta` (component
parameters) and the hidden assignments `Z` simultaneously — you'd have to
jointly search a combinatorially large space of assignments and a
continuous parameter space at once.

## The EM algorithm: alternating optimization

**EM (Expectation-Maximization)** sidesteps the joint optimization by
alternating between two easier subproblems: given a current estimate of the
parameters, figure out the (soft) hidden assignments; given the hidden
assignments, re-estimate the parameters.

A first, naive way to do this would be to alternate **point estimates**:

```
1.1:  Z*      = argmax_Z      p(Z | X, theta_old)   =  argmax_Z p(X, Z | theta_old)
1.2:  theta_new = argmax_theta p(X, Z* | theta)
```

...but committing to a single hardest-guess `Z*` at each step throws away
uncertainty and tends to get stuck. EM instead keeps the **full posterior
distribution** over the hidden variable at each step, rather than
collapsing it to a point — this is what makes it "soft."

### E-step

Given the current parameter estimate `theta_old`, compute the posterior
distribution over the hidden variables, `p(Z | X, theta_old)` — these are
the **responsibilities**: for each data point, how much of it "belongs" to
each mixture component under the current parameters. Then form the
**expected complete-data log-likelihood**, averaging `log p(X,Z|theta)`
over that posterior:

```
Q(theta, theta_old) = E_{Z ~ p(Z|X,theta_old)} [ log p(X, Z | theta) ]
                     = sum_Z  p(Z | X, theta_old) * log p(X, Z | theta)
```

`Q` is a function of the *free* variable `theta` (the parameters you're
about to re-optimize), with `theta_old` and the responsibilities held
fixed as constants inside it. Computing `Q` — i.e., computing the current
responsibilities — is the entirety of the E-step.

### M-step

Maximize `Q(theta, theta_old)` over `theta` to get the next parameter
estimate:

```
theta_new = argmax_theta  Q(theta, theta_old)
          = argmax_theta  sum_Z  p(Z | X, theta_old) * log p(X, Z | theta)
```

Because `Q` involves `log p(X,Z|theta)` — the complete-data log-likelihood,
as if `Z` *were* observed (just weighted by how likely each `Z` value is)
— this maximization is typically as easy as ordinary MLE: for a GMM, the
M-step update for each component's mean/covariance/weight looks just like
the weighted version of the per-class MLE formulas from GDA, weighted by
the current responsibilities instead of hard 0/1 class membership.

### Repeat

E and M steps **alternate**: compute responsibilities under the current
parameters (E), re-fit parameters under those responsibilities (M), repeat
until the parameters (or the log-likelihood) stop changing appreciably.

## Why EM works: the monotonic-improvement guarantee

Each full EM iteration is **guaranteed to never decrease** the true
(incomplete-data) log-likelihood `log p(X|theta)`, even though what's
actually being maximized at each step is the surrogate `Q`, not
`log p(X|theta)` directly.

**Intuition (Jensen's-inequality mechanism, without the full proof):** `Q`
can be shown to be a lower bound on the true log-likelihood that is
**tight** at `theta = theta_old` — the bound touches the true objective
exactly at the point you expanded it around. So any `theta_new` that
increases `Q` above `Q(theta_old, theta_old)` must also increase the true
log-likelihood `log p(X|theta)` above `log p(X|theta_old)`, because the true
curve lies at-or-above the bound everywhere and coincides with it at
`theta_old`. This is exactly what licenses optimizing the easier surrogate
`Q` in place of the hard-to-touch true objective at each step.

## Practical caveats

- **Local optima only.** EM is a coordinate-ascent-style procedure; it
  converges to *a* local maximum of the log-likelihood, not necessarily the
  global one. Different runs can converge to different (and sometimes
  much worse) solutions.
- **Initialization sensitivity.** Because of the above, where you start
  matters a lot. Standard fixes: multiple random restarts (keep the run
  with the best final log-likelihood), or a smarter initialization such as
  k-means++ to seed the initial component parameters/assignments before
  running EM.
- **k-means as a limiting case of EM.** k-means is, in fact, a special
  case of EM on a Gaussian Mixture Model where every component is
  constrained to an equal, isotropic covariance that is taken to shrink
  toward zero — in that limit, the soft responsibilities collapse to hard
  0/1 assignments (nearest-centroid), and the M-step reduces exactly to
  recomputing cluster means. A nice one-line way to connect the two ideas
  in an interview.

## See also

- [Generative Classification: GDA, QDA, LDA](generative-classification-gda-lda.md) —
  the fully-observed (no latent variable) Gaussian MLE that the M-step
  generalizes.
- [Bayesian Inference](bayesian-inference.md) — a complementary way of
  handling uncertainty (full posteriors over parameters) rather than over
  hidden discrete assignments.
- [Clustering Overview](../clustering/overview.md) — k-means and other hard
  clustering algorithms that GMM/EM soft-clustering generalizes.
