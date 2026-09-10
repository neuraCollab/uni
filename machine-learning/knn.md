# k-Nearest Neighbors (kNN)

## What is it?

A non-parametric, instance-based model: to predict for a new point, find its
`k` closest points in the training set (by some [distance metric](clustering/distance-metrics.md))
and combine their labels/values. No explicit training phase learns
parameters — the "model" is just the stored training data.

## How does it work?

- **Classification**: majority vote among the `k` nearest neighbors' labels.
- **Regression**: average (or weighted average) of the `k` nearest
  neighbors' target values.
- **Distance-weighted variants**: instead of every neighbor voting equally,
  weight each neighbor's vote/value by `1/distance` (or similar) so closer
  neighbors count more — reduces the impact of a tie-breaking far-away
  neighbor pulled into the `k` set only because `k` was set slightly too
  large.

## Choosing k — the bias-variance tradeoff

- **Small k (e.g. k=1)**: prediction depends on a single nearby point — very
  flexible, low bias, but **high variance**: sensitive to noise, a single
  mislabeled or outlying neighbor flips the prediction. Decision boundary is
  jagged, tends to overfit.
- **Large k**: prediction averages over many neighbors — smoother, more
  stable, **high bias**: can wash out genuine local structure and
  underfit, eventually converging toward always predicting the global
  majority class/mean as `k → n`.

This is the bias-variance tradeoff (see
[Bias-Variance Tradeoff](model-evaluation/bias-variance-tradeoff.md)),
expressed directly through a single hyperparameter — a clean example
interviewers like to probe. Pick `k` via cross-validated accuracy/error over
a range of odd values (odd to avoid ties in binary classification).

## Curse of dimensionality

kNN's core assumption — that "nearby" points are meaningfully more similar
than "far" points — degrades as dimensionality grows. In high dimensions,
distances between points **concentrate**: the ratio between the nearest and
farthest neighbor's distance tends toward 1, so "nearest" stops being a
meaningfully distinctive property (most points become roughly equidistant
from a given query point). Practically: kNN needs exponentially more data to
maintain the same neighbor density as dimensions increase, and beyond a
moderate number of dimensions (rule-of-thumb: tens, not hundreds, without
dimensionality reduction) it tends to perform poorly compared to models that
don't rely on raw distance (trees, linear models) or compared to first
reducing dimensionality (PCA, feature selection — see
[Feature Engineering](preprocessing/feature-engineering.md)).

## Computational cost

- **"Training"**: essentially free — just store the data (optionally build
  an index like a k-d tree or ball tree for faster lookup).
- **Prediction**: expensive. Naively, each query compares against all `n`
  training points — `O(n * d)` per prediction (`d` = dimensions). Spatial
  index structures (k-d tree, ball tree) can bring this down to roughly
  `O(log n)` in low dimensions, but those structures themselves degrade
  toward brute-force search in high dimensions (another curse-of-
  dimensionality symptom).

This is the **opposite** cost profile from trees or linear models, which do
expensive work upfront (fitting) but then predict in `O(depth)` or `O(d)` —
near-instant per query. kNN defers all the work to inference time, which
matters a lot for latency-sensitive production serving with large training
sets.

## When kNN is a reasonable baseline vs. when it fails

**Reasonable baseline when**: the dataset is small-to-moderate, low-to-moderate
dimensional, features are meaningfully scaled/comparable, and you want a
quick, interpretable ("nearest similar examples") first model with almost no
tuning beyond `k` and the distance metric.

**Fails when**: high-dimensional sparse data (curse of dimensionality erodes
distance meaningfulness — text/TF-IDF-style features are a classic bad
case unless dimensionality-reduced first); very large datasets needing fast
inference (kNN's prediction cost scales with training set size, unlike a
fitted linear model or tree); features aren't scaled (a large-range feature
dominates the distance calculation — see
[Scaling & Categorical Encoding](preprocessing/scaling-categorical.md), the
exact same requirement as Euclidean-distance clustering, see
[Distance Metrics](clustering/distance-metrics.md)).

## Common interview questions

- Walk through how kNN classification/regression makes a prediction.
- Why does k=1 have low bias but high variance? What about large k?
- What's the curse of dimensionality, and specifically why does it hurt
  kNN more than, say, a linear model?
- Why does kNN have "no training cost" but high prediction cost — how does
  that compare to a decision tree or linear regression?
- Why must features be scaled before using kNN?
- How would you speed up kNN inference on a large dataset? (Spatial indexes,
  approximate nearest neighbor methods, dimensionality reduction.)

## Common mistakes

- Running kNN on unscaled features (a feature with a larger numeric range
  silently dominates distance).
- Using an even `k` for binary classification (ties).
- Assuming kNN "doesn't need training" means it's cheap at prediction time
  too — it's the opposite tradeoff from most models.
- Applying kNN directly to high-dimensional sparse data without
  dimensionality reduction or feature selection first.
- Picking `k` by eye instead of by cross-validation.

## Example

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ("scaler", StandardScaler()),  # required -- kNN is distance-based
    ("knn", KNeighborsClassifier(weights="distance")),
])
search = GridSearchCV(pipe, {"knn__n_neighbors": [3, 5, 7, 9, 11, 15]}, cv=5)
search.fit(X_train, y_train)
```

## See also

- [Distance Metrics](clustering/distance-metrics.md)
- [Scaling & Categorical Encoding](preprocessing/scaling-categorical.md)
- [Bias-Variance Tradeoff](model-evaluation/bias-variance-tradeoff.md)
- [Imbalanced Data](imbalanced-data.md) — SMOTE is conceptually related to
  kNN (interpolates between nearest minority-class neighbors).
