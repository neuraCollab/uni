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
