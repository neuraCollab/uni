# k-means, Hierarchical, CURE, FOREL, ISODATA — Algorithm Detail

See [Clustering Overview](overview.md) first for the big picture and how
these compare conceptually. This note walks through each ported algorithm's
actual mechanics, complexity, and a "when to pick this over k-means" note.

## k-means (baseline, not in this repo's code)

**Procedure:** initialize `k` centroids → assign each point to its nearest
centroid → recompute centroids as cluster means → repeat until convergence.

**Complexity:** `O(n * k * d * iters)` (`n` points, `k` clusters, `d`
dimensions) — cheap, which is exactly why it's the default first thing to
try.

**Pick this when:** clusters are plausibly round/similarly sized, `k` is
known or easy to estimate, and you want something fast on large data.

---

## CURE ([`code/cure.py`](code/cure.py))

**Procedure (as implemented):**
1. Start with every point as its own singleton cluster.
2. Each cluster stores up to `num_reps` **representative points**: seed with
   the cluster's first point, then repeatedly add the point farthest from the
   representatives chosen so far (scatter them across the cluster's shape),
   then **shrink all representatives toward the cluster centroid** by
   `compression_rate` (0 = stay at the extremities, 1 = collapse onto the
   centroid — at 1 CURE degenerates into a centroid-only method).
3. Repeatedly find the two clusters whose representative points are closest
   (minimum distance between any pair of representatives across the two
   clusters — single-linkage-style) and merge them, recomputing the merged
   cluster's representatives.
4. Stop when `n_clusters` remain.

**Time complexity:** naive `O(n^2)` per merge search over all cluster pairs,
`O(n)` merges → roughly `O(n^3)` for this from-scratch implementation (the
original CURE paper uses k-d trees and a heap to get this down to
`O(n^2 log n)`; not implemented here).

**What makes it choose the clusters it does:** because representative points
capture a cluster's *shape*, not just its center, clusters merge based on how
close their boundaries/extremities are — this is what lets CURE trace
elongated or irregular shapes that a single centroid can't represent, and
what makes it more robust to a few outlier points (an outlier only pulls the
representatives near it, not the whole cluster's single center).

**Bug found and fixed during porting:** the representative-selection loop
(`_compute_representative_points`) originally computed
`distance_func(self.p, rep)` — passing the Minkowski exponent `self.p` (a
scalar the class never even set) as if it were a point. That's not a subtle
correctness issue, it's an `AttributeError` waiting to happen: it raises the
first time a cluster grows past `num_reps` points, i.e. after a few merges,
so a naive smoke test on tiny input could miss it entirely. Fixed to do the
actual textbook farthest-first traversal (compare each remaining candidate
point to its nearest already-chosen representative, keep the one that's
farthest). Worth remembering as an interview point itself: a line that
*looks* plausible (`distance_func(x, y)` with two args) can still reference
an attribute that was never defined — always check what `self.p` actually is
before trusting it, and prefer a test with enough points to exercise every
branch over one that only hits the early-return path.

**Pick CURE over k-means when:** clusters are non-spherical, sizes vary
a lot, or a few outliers would otherwise distort centroid-based clustering.

---

## FOREL ([`code/forel.py`](code/forel.py))

**Procedure:**
1. Pick any unclustered point as a tentative cluster center.
2. Collect every unclustered point within `radius` of it.
3. Recompute the center as the mean of that collected ball's points (in the
   classic algorithm this "recompute center → re-collect points within
   radius" step loops to convergence per cluster before finalizing it; **this
   port does a single growth step** — grab once, take the mean, done. That's
   a legitimate one-pass simplification, not a bug, but don't assume this
   implementation iterates a cluster to a fixed point internally).
4. Remove that cluster's points from the pool; repeat from a new unclustered
   seed point until none remain.

**Time complexity:** `O(n^2)` worst case (each of up to `n` cluster-growth
steps scans the remaining pool).

**What makes it choose the clusters it does:** the **radius** is the
hyperparameter, not the cluster count — a small radius produces many tight
clusters, a large radius produces few, sprawling ones. The number of clusters
is emergent from the data and radius, not fixed up front.

**Pick FOREL over k-means when:** you don't know (or don't want to commit to)
the number of clusters in advance, but you do have a sense of what
"neighborhood size" should count as one cluster.

---

## ISODATA ([`code/isodata.py`](code/isodata.py))

**Procedure (as implemented — see note below on what's omitted):**
1. Initialize `k_initial` centers randomly.
2. **Assignment step** (identical to k-means): each point joins its nearest
   center.
3. **Update step, with a size floor**: recompute each cluster's center as the
   mean of its points, but **discard** any cluster with fewer than
   `min_points_per_cluster` points instead of keeping a degenerate center.
4. **Merge step**: any pair of centers within `merge_threshold` distance is
   averaged into one.
5. Repeat 2–4 up to `max_iterations` (or until `max_clusters` is reached).

**Note on what's implemented:** the classical ISODATA algorithm also
**splits** any cluster whose internal spread (standard deviation along its
largest dimension) exceeds `sigma_threshold`, into two. This port accepts
`sigma_threshold` for API compatibility but does **not** implement the split
step — only the discard-small / merge-close halves of the loop run. So this
implementation can shrink `k` (via discard/merge) but won't grow it (via
split) beyond `k_initial`, unlike full ISODATA.

**Time complexity:** same order as k-means per iteration,
`O(n * k * d)`, times `max_iterations`, plus `O(k^2)` per merge check.

**What makes it choose the clusters it does:** it's k-means's assignment/
update loop with two guardrails layered on top — degenerate (too-small)
clusters get pruned instead of persisting, and redundant (too-close)
clusters get consolidated. The final `k` is whatever survives those checks,
not the `k_initial` you asked for.

**Pick ISODATA over k-means when:** you have a rough guess at `k` but
suspect it's off, and want the algorithm to self-correct for spurious tiny
clusters or accidentally duplicated ones — while being aware this specific
implementation only shrinks, never splits, `k`.

---

## Hierarchical / Agglomerative ([`code/hierarchical.py`](code/hierarchical.py))

**Procedure:**
1. Compute the full pairwise distance matrix ([`pairwise_distance_matrix`](code/distances.py)).
2. Start with every point as its own cluster.
3. Repeatedly merge the two closest clusters under the chosen **linkage**,
   until `n_clusters` remain:
   - **single**: `min` distance between any pair of points across the two
     clusters. Can trace elongated/non-convex shapes but is prone to
     "chaining" — a thin bridge of noise points can link two otherwise
     distinct clusters into one.
   - **complete**: `max` distance between any pair. Favors compact,
     similarly-sized clusters; sensitive to outliers (one far point inflates
     the max for the whole cluster pair).
   - **average**: mean pairwise distance. A middle ground between the two.
4. Labels come from whatever partition survives at `n_clusters`.

**Time complexity:** this is a naive implementation — `O(n^2)` for the
distance matrix, and each of `O(n)` merges re-scans all remaining cluster
pairs, so overall roughly `O(n^3)`. `scipy.cluster.hierarchy.linkage`/
`fcluster` implement the same idea with a much faster algorithm
(`O(n^2 log n)` or better depending on linkage) — use those for real
datasets; this from-scratch version is a small-demo-dataset teaching
implementation.

**Consolidation note:** the source project also had a separate
`single_linkage.py` that built a distance matrix and delegated to
`sklearn.cluster.AgglomerativeClustering(linkage='single', metric=
'precomputed')`. That's functionally the same case already covered here by
`HierarchicalClusterer(linkage="single")`, just via sklearn instead of from
scratch, so it wasn't ported as a separate file — use `linkage="single"` to
get that behavior.

**What makes it choose the clusters it does:** the whole merge sequence forms
a dendrogram; cutting it at different heights gives different `n_clusters`
values "for free" without refitting — a benefit centroid-based methods don't
offer directly.

**Pick hierarchical over k-means when:** you want the dendrogram itself (to
inspect nested structure or decide `k` visually by where to cut), your
similarity notion isn't naturally centroid-based, or the dataset is small
enough that the extra compute is a non-issue.

## Common interview questions

- Walk through CURE's representative-point mechanism and why it helps with
  non-spherical clusters.
- How does FOREL decide when a cluster is "done" growing, and what
  determines the final cluster count?
- What's the difference between what full ISODATA does and what this
  implementation does?
- Single vs. complete vs. average linkage — failure modes of each?
- Why is the naive `O(n^3)`-ish hierarchical implementation impractical for
  large `n`, and what would you use instead?

## Common mistakes

- Assuming ISODATA in this repo can grow past `k_initial` — it can only
  shrink/merge here, not split.
- Assuming FOREL iterates each cluster to convergence internally — this port
  does one growth step per cluster.
- Using single linkage on noisy data and being surprised by "chained"
  clusters that don't look visually separate.
- Forgetting these are demo-scale (`O(n^2)`–`O(n^3)`) implementations — swap
  to scipy/sklearn's optimized routines for anything beyond a few thousand
  points.

## See also

- [Clustering Overview](overview.md)
- [Distance Metrics](distance-metrics.md)
- [Clustering Evaluation Metrics](clustering-evaluation-metrics.md)
