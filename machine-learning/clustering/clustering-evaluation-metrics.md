# Clustering Evaluation Metrics

See [`code/metrics.py`](code/metrics.py) for the implementations. Clustering
evaluation splits into two families depending on whether ground-truth labels
are available.

## External metrics — need ground-truth labels

All four implemented here are built the same way: treat every **pair** of
points and ask whether the two labelings (true vs. predicted) agree on
whether that pair belongs together — this gives a pair-counting confusion
matrix (TP/FP/FN/TN over *pairs*, not over individual points).

- **TP**: pair is together in both true and predicted labels.
- **FP**: pair is together in predicted labels but not true labels.
- **FN**: pair is together in true labels but not predicted labels.
- **TN**: pair is apart in both.

**Rand Index** (`rand_index`) — $\dfrac{TP + TN}{TP + FP + FN + TN}$: fraction
of all pairs the two labelings agree on. Simple and intuitive, but **not
corrected for chance** — two random labelings of many small clusters will
still score a deceptively high Rand Index, because most random pairs of
points land in different clusters by chance alone (that agreement inflates
TN). Use sklearn's `adjusted_rand_score` when chance-correction matters.

**Jaccard Index** (`jaccard_index`) — $\dfrac{TP}{TP + FP + FN}$: like Rand but
**ignores true negatives** entirely. Harsher whenever most pairs are
"different cluster" by chance (many small clusters among many points), since
it doesn't let a clustering get credit just for correctly keeping unrelated
points apart.

**Fowlkes-Mallows Index** (`fowlkes_mallows_index`) —
$\dfrac{TP}{\sqrt{(TP+FP)(TP+FN)}}$: the geometric mean of pairwise precision and
recall. Less sensitive to cluster-size imbalance than Rand Index.

**Phi index / pairwise Matthews correlation** (`phi_index`) —
$\dfrac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$, ranging **-1 to 1**.
Unlike the other three, it uses all four confusion cells in a
chance-corrected-ish way and can go **negative** for a labeling that actively
disagrees with the ground truth more than random chance would — the other
three metrics are bounded at 0 and can't express "worse than random."

## Internal metrics — no ground truth needed

**Compactness** (`compactness`) — average intra-cluster pairwise distance,
summed across clusters. Lower is better (tighter clusters). Not normalized —
it scales with cluster count/size, so only compare compactness across runs
on the *same* dataset and `k`, never across datasets.

**Separation** (`separation`) — minimum distance between any two cluster
centroids. Higher is better (well-separated clusters).

**Why pair them:** either alone can be gamed. One giant cluster is trivially
"compact" (small average intra-cluster distance relative to the whole
dataset's spread) but has no separation to measure at all (only one
centroid); many tiny clusters can be maximally separated but each
individually meaningless. A good clustering needs to be compact *and*
separated simultaneously.

## Silhouette score (not in this codebase, but the metric interviewers ask
about most)

For each point $i$: let $a(i)$ = mean distance to other points in its own
cluster (intra-cluster distance — like a per-point compactness), and $b(i)$
= mean distance to points in the *nearest other* cluster (nearest-neighboring
inter-cluster distance). Then:

$$\text{silhouette}(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Ranges -1 to 1: close to 1 means the point is well inside its own cluster and
far from the next-nearest one; close to 0 means it sits near a cluster
boundary; negative means it's probably in the wrong cluster. The **average
silhouette score** across all points is the single most common internal
metric used in practice (and to pick `k` — see
[Clustering Overview](overview.md)) because, unlike raw compactness/
separation, it's normalized per point and comparable across different `k`
values on the same dataset.

## When you have ground truth vs. when you don't

**Ground truth available**: rare in real unsupervised settings — if you had
labels, you'd often just run a supervised model instead. It does come up:
validating a clustering against a small labeled sample, comparing a
clustering algorithm to a known partition (e.g. species labels in a
classic benchmark dataset), or checking cluster stability by comparing
runs against each other (treating one run's labels as "ground truth" for
another). In those cases, use Rand/Jaccard/Fowlkes-Mallows/Phi as above —
prefer Fowlkes-Mallows or Phi over raw Rand Index when cluster sizes are
imbalanced, since Rand Index's TN term can dominate and hide poor
clustering.

**No ground truth (the common case)**: use internal metrics — compactness +
separation together, or silhouette score as the standard single-number
summary. None of these tell you the clustering is "correct" in any
objective sense, only that it's internally self-consistent (tight,
separated clusters) — always sanity-check against domain knowledge too.

## Common interview questions

- How is the Rand Index computed, and why is "adjusted" Rand Index usually
  preferred over the raw version?
- Why does Jaccard Index ignore true negatives, and when does that matter?
- What does a negative Phi index / Matthews correlation mean?
- Explain silhouette score for a single point, in words.
- Why should you look at compactness and separation together rather than
  either alone?
- You don't have ground truth labels — how do you decide if your clustering
  is any good?

## Common mistakes

- Comparing raw Rand Index scores across differently-imbalanced labelings
  and concluding one clustering is much better, when the difference is
  mostly driven by TN inflation.
- Using compactness alone to pick the number of clusters — it monotonically
  improves as `k` increases (more, smaller clusters are always "tighter"),
  so it will always favor $k = n$. Pair it with separation, or use
  silhouette/elbow instead.
- Treating external metrics as available by default — in most real
  unsupervised projects you won't have ground-truth labels at all.

## Example

```python
from metrics import rand_index, jaccard_index, fowlkes_mallows_index, phi_index, compactness, separation

# External (only if you have ground truth):
rand_index(y_true, y_pred)
fowlkes_mallows_index(y_true, y_pred)

# Internal (always available):
compactness(X, labels)
separation(X, labels)
```

## See also

- [Clustering Overview](overview.md)
- [Distance Metrics](distance-metrics.md)
- [k-means, Hierarchical, CURE, FOREL, ISODATA](kmeans-hierarchical-cure-forel-isodata.md)
