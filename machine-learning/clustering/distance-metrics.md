# Distance Metrics for Clustering

See [`code/distances.py`](code/distances.py) for the implementations behind
this note — every clusterer in this project ([Overview](overview.md)) picks
its distance function via `BaseClusterer`'s `metric` parameter, which routes
through `get_distance()` here.

## What is it?

A function `d(a, b)` measuring dissimilarity between two points, used both to
decide cluster assignment ("which centroid/cluster is closest") and to
evaluate cluster quality ([Clustering Evaluation Metrics](clustering-evaluation-metrics.md)).
The choice of metric changes what "similar" even means, so it's as important
a modeling decision as the clustering algorithm itself.

## The metrics

**Euclidean (`euclidean_distance`)** — straight-line (L2) distance:
$\sqrt{\sum_i (a_i - b_i)^2}$. The default, intuitive notion of distance; assumes
all dimensions are commensurable (same units/scale) and combines them
isotropically (no dimension is treated as "more important").

**Squared Euclidean (`squared_euclidean_distance`)** — Euclidean without the
final `sqrt`. Same ordering of distances as Euclidean (monotonic transform),
so it produces identical clustering assignments, but is cheaper to compute
and is what k-means' objective (within-cluster sum of squares) actually
minimizes internally.

**Manhattan (`manhattan_distance`)** — L1 / "taxicab" distance:
$\sum_i |a_i - b_i|$. Less sensitive to large deviations in a single dimension
than Euclidean (no squaring), so it's more robust when some dimensions
occasionally have big outlying differences. Common in high-dimensional or
grid-like/count data.

**Chebyshev (`chebyshev_distance`)** — L∞ distance: $\max_i |a_i - b_i|$, the
single largest per-coordinate difference. Useful when the "worst" dimension
alone should determine dissimilarity (e.g. tolerance/threshold-style
problems — two points are "different" if *any one* attribute differs a lot,
regardless of the others).

**Minkowski (`minkowski_distance`)** — the generalization:
$\left(\sum_i |a_i - b_i|^p\right)^{1/p}$. $p=1$ → Manhattan, $p=2$ → Euclidean, $p \to \infty$ →
Chebyshev. Tuning $p$ interpolates between "sum up all differences fairly"
(low $p$) and "only the worst dimension matters" (high $p$).

**Pearson correlation distance (`pearson_distance = 1 - pearson_correlation`)**
— measures *shape/pattern* similarity, not magnitude. Two vectors that are
perfectly linearly related ($b = a \cdot c + d$ for any positive $c$) get
distance 0, even if their absolute values are wildly different scales. This
matters when you care about **trend, not level** — e.g. clustering time
series by whether they rise and fall together (co-movement), clustering gene
expression profiles by pattern across conditions, or clustering
survey respondents by response *pattern* rather than how high/low they
scored overall. Euclidean distance would treat two series with the same
shape but different baseline levels as very far apart; Pearson distance
treats them as identical.

## Choosing a metric for clustering specifically — scaling implications

**Euclidean (and Minkowski generally) is scale-sensitive**: a feature
measured in thousands (income) will dominate a feature measured in single
digits (age) purely because of units, not because it's actually more
informative. **Always scale features first** (see
[Scaling & Categorical Encoding](../preprocessing/scaling-categorical.md))
before using any of Euclidean/Manhattan/Chebyshev/Minkowski for clustering —
otherwise the clustering is implicitly weighting features by their raw
numeric range.

Pearson correlation distance is the exception: because it's computed on each
vector standardized internally (correlation is scale- and shift-invariant),
it doesn't require pre-scaling for that reason — but it's answering a
different question (shape similarity) than the $L_p$ family (magnitude
similarity), so the choice should be driven by what "similar" should mean for
the problem, not just convenience.

| Metric | Scale-sensitive? | Good for |
|---|---|---|
| Euclidean | Yes — scale first | General-purpose, roughly isotropic features |
| Manhattan | Yes — scale first | Robustness to single-dimension outliers, high-dim/count data |
| Chebyshev | Yes — scale first | "Worst dimension decides" problems |
| Minkowski (p) | Yes — scale first | Tunable interpolation between the above |
| Pearson | No (correlation is scale/shift invariant) | Shape/trend similarity regardless of magnitude |

## Common interview questions

- What does the Minkowski `p` parameter control, and what do $p=1$, $p=2$,
  $p\to\infty$ reduce to?
- Why must you scale features before Euclidean-distance clustering, but not
  before Pearson-distance clustering?
- When would you prefer Manhattan over Euclidean distance?
- Give an example where Pearson distance and Euclidean distance would rank
  the "most similar" pair of points differently.
- Why does squared Euclidean give the same clustering as Euclidean?

## Common mistakes

- Running k-means/CURE/FOREL/ISODATA with Euclidean distance on unscaled
  features (e.g. mixing a 0–1 feature with a 0–100000 feature).
- Using Euclidean distance when the real question is about *shape* (e.g.
  comparing time series trends) instead of Pearson distance.
- Assuming Chebyshev and Manhattan give similar results — they optimize for
  opposite things (worst-dimension vs. sum-of-all-dimensions).

## Example

```python
from distances import get_distance

euclidean = get_distance("euclidean")
manhattan = get_distance("manhattan")
pearson = get_distance("pearson")

a, b = [1, 2, 3], [4, 5, 6]
euclidean(a, b)   # sqrt(3*9) ~= 5.196
manhattan(a, b)   # 9
pearson(a, b)     # 0.0 -- perfectly correlated shapes (b = a + 3)
```

## See also

- [Clustering Overview](overview.md)
- [k-means, Hierarchical, CURE, FOREL, ISODATA](kmeans-hierarchical-cure-forel-isodata.md)
- [Scaling & Categorical Encoding](../preprocessing/scaling-categorical.md)
- [kNN](../knn.md) — the other major algorithm family where distance-metric
  choice and feature scaling matter just as much.
