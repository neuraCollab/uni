# Clustering: Overview

## What is it?

Unsupervised grouping of points into clusters such that points in the same
cluster are more similar to each other than to points in other clusters — no
ground-truth labels are used to fit the model (though they're sometimes
available afterward, purely to *evaluate* the clustering — see
[Clustering Evaluation Metrics](clustering-evaluation-metrics.md)).

All the clusterers in this project ([`code/`](code/)) share a common
interface via [`code/base.py`](code/base.py)'s `BaseClusterer` — a
scikit-learn-flavored `fit`/`predict`/`fit_predict` ABC, plus a shared
`distance_func` built from a metric name (`'euclidean' | 'manhattan' |
'chebyshev' | 'minkowski' | 'pearson' | 'squared_euclidean'`, see
[Distance Metrics](distance-metrics.md)) so every subclass gets pluggable
distance handling for free.

## Why?

Real data often has no labels, or labels are expensive to obtain (manual
annotation). Clustering finds structure anyway — customer segments, anomaly
groups, document topics — and is often step one before a supervised model
even exists (e.g. clustering to generate candidate labels, or to understand
the data before deciding what to predict).

## The baseline everyone should know first: k-means

Not part of this project's ported code, but the algorithm every clustering
interview question is implicitly compared against:

**Lloyd's algorithm:**
1. Pick `k` initial centroids (random points, or `k-means++` for
   smarter spread-out seeding).
2. **Assign**: each point joins the nearest centroid (Euclidean distance).
3. **Update**: recompute each centroid as the mean of its assigned points.
4. Repeat 2–3 until assignments stop changing (or a max-iteration cap).

**Choosing k:**
- **Elbow method** — plot within-cluster sum of squares (inertia) vs. `k`;
  pick the `k` where the curve's improvement starts flattening.
- **Silhouette score** — pick the `k` that maximizes average silhouette (see
  [Clustering Evaluation Metrics](clustering-evaluation-metrics.md)).

**Sensitivities (k-means' well-known weaknesses, each of which motivates one
of the algorithms below):**
- **Initialization** — bad starting centroids can converge to a poor local
  optimum; `k-means++` and multiple random restarts (`n_init`) mitigate this.
- **Outliers** — a single far-away point drags its centroid's mean toward it,
  distorting the whole cluster.
- **Non-spherical clusters** — k-means implicitly assumes roughly round,
  similarly-sized, similarly-dense clusters (it partitions space into convex
  Voronoi regions around centroids), so it fails on elongated, nested, or
  very differently-sized/-dense clusters.
- **Fixed k** — you must decide the number of clusters up front.

## How the ported algorithms differ from k-means (and each other)

See [k-means/Hierarchical/CURE/FOREL/ISODATA in detail](kmeans-hierarchical-cure-forel-isodata.md)
for step-by-step mechanics; the short version:

- **CURE** ([`code/cure.py`](code/cure.py)) — represents each cluster by
  several scattered representative points (shrunk toward the centroid)
  instead of a single centroid, which lets it trace non-spherical cluster
  shapes and is more robust to outliers than a pure centroid-based method —
  the exact case k-means struggles with.
- **FOREL** ([`code/forel.py`](code/forel.py)) — a density/radius-driven
  method: pick a seed point, grow a ball of radius `r` around it, recenter on
  the ball's mean, repeat until the whole dataset is covered. Unlike k-means,
  you don't choose the number of clusters — you choose the radius, and the
  cluster count falls out of the data.
- **ISODATA** ([`code/isodata.py`](code/isodata.py)) — a k-means variant that
  adjusts the cluster count *during* fitting: after each assignment/update
  pass, clusters that collapse to too few points are dropped and centers that
  end up too close together are merged (the classic algorithm also *splits*
  clusters with excessive internal spread — this port implements only the
  discard/merge half, see the note in the algorithm-detail page). Useful when
  you have a rough guess at `k` but suspect the true cluster count differs.
- **Hierarchical/agglomerative** ([`code/hierarchical.py`](code/hierarchical.py))
  — bottom-up merging: start with every point as its own cluster and
  repeatedly merge the two closest clusters (by a **linkage** rule — single,
  complete, or average) until `n_clusters` remain. Produces a full dendrogram
  of nested clusterings "for free," and doesn't force a spherical-cluster
  assumption the way k-means does (though single linkage in particular has
  its own quirks — see the detail page).

## When to use clustering / when not to

**Use** when you need to discover structure with no labels, for
segmentation/exploration, as a preprocessing step (e.g. cluster then build a
model per cluster), or for anomaly detection (points far from any cluster).

**Avoid** treating clustering output as ground truth — it's exploratory by
nature; different algorithms/hyperparameters can produce meaningfully
different partitions of the same data, and there's rarely a single "correct"
answer without domain validation.

## Common interview questions

- Walk through Lloyd's algorithm for k-means.
- How do you choose `k`? (Elbow, silhouette.)
- Why does k-means struggle with non-spherical clusters, and what would you
  use instead?
- What's the practical difference between CURE and k-means?
- How does FOREL decide the number of clusters, and how does that compare to
  k-means/ISODATA?
- Single vs. complete vs. average linkage — what does each optimize for, and
  what's each one's failure mode?

## Common mistakes

- Running k-means (or any Euclidean-distance clusterer) on unscaled features
  — see [Distance Metrics](distance-metrics.md).
- Assuming clustering "found the truth" instead of validating with internal
  metrics (and external ones, if labels happen to be available) — see
  [Clustering Evaluation Metrics](clustering-evaluation-metrics.md).
- Picking k-means by default without checking whether clusters are plausibly
  spherical/similarly sized first.

## See also

- [k-means, Hierarchical, CURE, FOREL, ISODATA — algorithm detail](kmeans-hierarchical-cure-forel-isodata.md)
- [Distance Metrics](distance-metrics.md)
- [Clustering Evaluation Metrics](clustering-evaluation-metrics.md)
