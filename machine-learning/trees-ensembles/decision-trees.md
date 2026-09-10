# Decision Trees

## What is it?

A supervised model that recursively splits the feature space into axis-aligned
regions, arranged as a binary tree. Each internal node asks a single yes/no
question about one feature ("is `age < 30`?"); each leaf holds a prediction —
a class label (majority vote) for classification, or a mean value for
regression.

## Why?

Trees are the building block of almost every strong tabular-data model
(random forests, gradient boosting). On their own they're valued for being
**interpretable** (you can read the decision path) and for needing **no
feature scaling** — splits only compare a feature to a threshold, so
monotonic transforms of a feature don't change anything.

## How does it work?

**Growing the tree** — at each node, try every feature and every candidate
threshold, and pick the split that most improves a purity criterion:

- **Gini impurity** (classification): `1 - Σ p_i²` over classes in the node.
  0 when the node is pure, higher when classes are mixed.
- **Entropy** (classification): `-Σ p_i log2(p_i)`. Similar shape to Gini,
  slightly more expensive to compute, splits chosen are usually near-identical
  in practice.
- **Information gain** = impurity(parent) − weighted-average impurity(children).
  The split maximizing information gain wins.
- **Variance reduction / MSE** (regression): pick the split that most reduces
  the sum of squared errors within child nodes.

Recurse on each child until a stopping condition is hit (max depth, minimum
samples per leaf/split, or no split improves purity).

### Impurity criteria — the exact math

A node `Xm` is split into a left/right child `Xl`, `Xr`. The split quality
(how much a candidate split improves purity) is:

```
Q(Xm, split) = H(Xm) - (|Xl| / |Xm|) * H(Xl) - (|Xr| / |Xm|) * H(Xr)
```

i.e. impurity of the parent minus the size-weighted impurity of the children.
The algorithm searches over features and thresholds for the split that
maximizes `Q`. `H(Xm)` itself is defined differently for regression and
classification:

**Regression — variance of the targets in the node.** Predicting a constant
`c` for every point in `Xm`, the MSE-minimizing constant is the node mean
`ȳ_m`, so plugging it back in gives:

```
H(Xm) = (1 / |Xm|) * Σ_{(xi, yi) ∈ Xm} (yi - ȳ_m)²,   ȳ_m = (1/|Xm|) * Σ yi
```

This is exactly the variance-reduction / MSE criterion already listed above,
just written out with the node mean made explicit.

**Classification — everything is a function of `p_k`.** Let `p_k` be the
fraction of class-`k` points in the current node:

```
p_k = (1 / |Xm|) * Σ_{(xi, yi) ∈ Xm} [yi = k]
```

- **Misclassification error**: predict the majority class, so the error rate
  is `1 - p_k*` where `p_k* = max_k p_k`:

  ```
  H(Xm) = 1 - p_k*
  ```

- **Entropy**: fit a categorical distribution `c_1, ..., c_K` (`Σ c_k = 1`) to
  the node by maximum likelihood — minimize the average negative
  log-likelihood of the labels under `c`:

  ```
  H(Xm) = min_{Σ c_k = 1}  -(1/|Xm|) * Σ_{(xi,yi)∈Xm} Σ_k [yi=k] * log(c_k)
  ```

  The minimizer is `c_k = p_k` (the empirical class frequencies), which gives
  the classical Shannon entropy:

  ```
  H(Xm) = -Σ_{k=1}^K p_k * log(p_k)
  ```

- **Gini criterion**: instead of a log-likelihood objective, treat each
  class indicator `[yi = k]` as a regression target and fit a constant `c_k`
  per class by least squares (again `Σ c_k = 1`):

  ```
  H(Xm) = min_{Σ c_k = 1}  (1/|Xm|) * Σ_{(xi,yi)∈Xm} Σ_k (c_k - [yi=k])²
  ```

  The minimizer is again `c_k = p_k`, which gives:

  ```
  H(Xm) = Σ_{k=1}^K p_k * (1 - p_k)   (= 1 - Σ p_k²)
  ```

  matching the Gini formula above. So Gini is literally the squared-error
  relaxation of the same fitting problem entropy solves with a log-loss
  relaxation — both are smooth surrogates for the 0/1 misclassification
  objective, fit via a "predict the class probabilities in this node"
  sub-problem.

**Why entropy/Gini instead of raw misclassification error?** All three are
zero for a pure node and maximal for a 50/50 node, but misclassification
error is piecewise-linear in `p_k` and often *flat* between two candidate
splits that both keep the majority class the same — it can't tell a split
that pushes the minority class from 40% to 30% apart from one that pushes it
from 40% to 10%, even though the second is clearly better progress. Entropy
and Gini are strictly concave in `p_k`, so they're strictly sensitive to any
change in the class-probability mix, which makes them decrease monotonically
as a split gets purer and gives the tree-growing search a usable gradient
signal to optimize instead of a flat one. In practice Gini and entropy pick
almost identical splits; misclassification error is mostly used as the
*final reported metric*, not as the *split-selection criterion*.

**Overfitting and pruning** — an unconstrained tree grows until every leaf is
pure (often one training point per leaf), which memorizes noise. Controls:

- `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes` —
  stop growing early ("pre-pruning").
- **Cost-complexity pruning** ("post-pruning") — grow the full tree, then cut
  back subtrees whose removal doesn't hurt a penalized error metric
  (`ccp_alpha` in scikit-learn). Usually generalizes better than pre-pruning
  alone because it can see the full tree before deciding what's noise.

**Pros**
- Interpretable (can literally draw and read the decision path).
- No feature scaling needed.
- Captures non-linear relationships and feature interactions natively.
- Handles mixed numeric/categorical features (with appropriate encoding) and
  is robust to monotonic outliers in a single feature.

**Cons**
- **High variance / instability** — a small change in training data can
  produce a very different tree (different top split cascades down).
  This is exactly why bagging (→ [random-forest.md](random-forest.md)) helps so much.
- Prone to overfitting if not regularized.
- Axis-aligned splits only — a diagonal decision boundary needs many steps to
  approximate.
- Biased toward features with many possible split points (see
  [random-forest.md](random-forest.md) for the feature-importance version of this caveat).

## When to use / when NOT to

**Use** when you want a quick, interpretable baseline, when the audience
needs to see *why* a prediction was made (regulated domains, debugging), or
as the base learner inside an ensemble.

**Avoid** a single tree in production when predictive accuracy matters more
than interpretability — almost always use it inside an ensemble
(random forest, gradient boosting) instead. A lone tree is rarely the best
final model.

## Common interview questions

- How does a decision tree decide where to split? (Gini/entropy/information
  gain for classification, variance reduction for regression.)
- Why don't decision trees need feature scaling?
- What causes a decision tree to overfit, and how do you prevent it?
- Gini vs. entropy — does it matter in practice? (Rarely; Gini is cheaper,
  splits chosen are usually very similar.)
- Why is a single decision tree considered high-variance / unstable?
- What is cost-complexity pruning?
- How would a decision tree handle a categorical feature with many levels?
  (One-hot can dilute splits across many low-information branches; some
  implementations, e.g. LightGBM/CatBoost, split categoricals natively —
  see [gradient-boosting-catboost-lgbm.md](gradient-boosting-catboost-lgbm.md).)

## Common mistakes

- Growing a tree with no depth/leaf limit and being surprised it overfits.
- Assuming a decision tree needs scaled/normalized features (it doesn't).
- Reading feature importance from a single tree as if it were stable —
  it's noisy; average over many trees (random forest) for a trustworthy signal.
- Using accuracy alone to judge a tree on imbalanced data — see
  [../imbalanced-data.md](../imbalanced-data.md).

## Example

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

clf = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_leaf=20,
    random_state=42,
)
clf.fit(X_train, y_train)
plot_tree(clf, feature_names=X_train.columns, filled=True)
```

See also: [random-forest.md](random-forest.md) (bagging many trees to fix the
variance problem) and [gradient-boosting-catboost-lgbm.md](gradient-boosting-catboost-lgbm.md)
(building trees sequentially to fix each other's errors).
