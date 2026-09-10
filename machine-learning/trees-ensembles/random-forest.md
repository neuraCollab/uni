# Random Forest

See [Decision Trees](decision-trees.md) first — this note assumes you know how
a single tree splits and why it overfits. Random Forest is the fix for that
tree's biggest weakness: high variance.

## What is it?

An ensemble of many decision trees, each trained on a randomized view of the
data, whose predictions are averaged (regression) or majority-voted
(classification). Two sources of randomness, both required:

1. **Bagging** (Bootstrap AGGregatING) — each tree trains on a bootstrap
   sample (sampled with replacement, same size as the original dataset, so
   ~63.2% of rows appear at least once per tree).
2. **Random feature subsampling** — at each split, only a random subset of
   features (`max_features`, commonly $\sqrt{n_{\text{features}}}$ for classification,
   $n_{\text{features}}/3$ for regression) is considered as split candidates.

## Why both matter — decorrelating trees

Averaging reduces variance only if the things being averaged are not too
correlated. If every tree in the ensemble always splits on the same dominant
feature first, bagging alone still produces highly correlated trees (all
rediscover the same structure from slightly different samples), and
averaging correlated trees barely reduces variance versus a single tree.

Random feature subsampling forces different trees to discover different
splits — a tree that can't see the dominant feature at a given node is forced
to use a weaker-but-informative one instead. This decorrelates the trees, and
averaging $N$ decorrelated, unbiased estimators reduces variance roughly by a
factor related to their average pairwise correlation, not by $1/N$ alone.
Bagging supplies the "many different training sets" half; feature
subsampling supplies the "and don't let them agree on structure" half.

## The formal bagging argument (why averaging reduces variance)

Bagging (bootstrap aggregation) trains $k$ base algorithms $b_i(x, X_i)$ on
$k$ bootstrap resamples $X_1, \ldots, X_k$ of the training set and averages
them:

$$a(x) = \frac{1}{k}\left(b_1(x, X_1) + b_2(x, X_2) + \ldots + b_k(x, X_k)\right)$$

Two claims follow directly from this construction (see
[bias-variance-tradeoff.md](../model-evaluation/bias-variance-tradeoff.md)
for the full formal decomposition these refer to):

- **Bias is unchanged.** Expectation is linear, so $\mathbb{E}[a(x, X)]$ is just the
  average of the $k$ base models' expectations. Averaging identically
  distributed estimators doesn't shift their expected value — if each tree
  is (say) slightly biased downward on average, the average of many such
  trees is biased downward by the same amount. Bagging never fixes a model
  that's systematically wrong.
- **Variance shrinks — if and only if the base models don't correlate.**
  For $k$ i.i.d. random variables with variance $\sigma^2$ each, the variance of
  their average is $\sigma^2/k$ — this is the ideal case bagging is chasing.
  Bagging's implicit assumption is that $\text{cov}(b_i(x, X_i), b_j(x, X_j)) \approx 0$
  for $i \neq j$. In the ideal (zero-correlation) case you get the full $1/k$
  variance reduction; the more correlated the base learners are, the less
  variance reduction you actually get, because averaging correlated
  variables can't cancel their shared error component.

**In practice, trees trained on bootstrap resamples of the *same* dataset
are not independent** — they overlap heavily (each bootstrap sample still
contains ~63% of the same rows) and, left alone, will tend to rediscover the
same dominant splits near the root. That's a direct violation of bagging's
"don't correlate" assumption, and it's exactly why Random Forest adds
**random feature subsampling on top of bagging**: restricting each split to
a random subset of $n < N$ features forces different trees down different
structural paths, lowering $\text{cov}(b_i, b_j)$ and pushing the realized variance
reduction closer to the ideal $1/k$. Bagging alone gives you *some*
decorrelation (different resamples); feature subsampling gives you a lot
more (different candidate splits), which is why RF reliably outperforms
plain bagged trees.

**The tradeoff this creates:**

- **Tree depth** — deeper trees fit the training data more closely, which
  lowers each individual tree's bias but raises its variance (a classic
  single-tree bias-variance tradeoff, see [Decision Trees](decision-trees.md)).
  RF typically lets trees grow deep/unpruned, because...
- **...averaging over many trees is what controls variance instead of
  pruning each tree.** RF intentionally uses high-variance, low-bias base
  learners (deep trees) and leans on ensemble averaging to bring variance
  down, rather than constraining each tree individually the way you would a
  standalone tree.
- **`n_features` per split (`max_features`)** — fewer candidate features per
  split means *less correlation between trees* (more decorrelation, closer
  to the ideal $1/k$), at the cost of each individual split being weaker
  (a tree might be forced to use a less-informative feature), which can
  raise the bias of each tree slightly. More candidate features per split
  means more correlation between trees (closer to plain bagging) but
  stronger individual splits. `max_features` is the knob that trades off
  "how decorrelated are the trees" against "how good is each tree on its
  own."
- **Number of trees (`n_estimators`)** — more trees only keeps helping
  while they're not too correlated; past a point extra trees mostly average
  out sampling noise without reducing variance further, so validation error
  vs. `n_estimators` plateaus rather than degrading (unlike boosting rounds).
  In practice: plot validation error against `n_estimators` and pick the
  point where the curve flattens, then weigh any further tiny gains against
  training/inference cost — there's rarely a reason to keep adding trees
  once the curve is flat.

## Out-of-bag (OOB) error

Because each tree only sees ~63% of rows (bootstrap sampling), the remaining
~37% ("out-of-bag" for that tree) were never used to train it. For each
training point, average the predictions of only the trees that didn't see it
— this gives an unbiased-ish estimate of test error **without needing a
separate validation split** (`oob_score=True` in scikit-learn). Convenient
when data is scarce, but it's not a full substitute for proper CV when you
need to compare across very different pipelines/preprocessing choices.

## Feature importance — and its caveat

**Mean Decrease in Impurity (MDI, the default `.feature_importances_`)**:
average, over all trees, of the impurity reduction each feature causes when
it's used to split, weighted by the number of samples reaching that node.

**Caveat: biased toward high-cardinality / continuous features.** A feature
with many unique values (a continuous number, or a categorical with many
levels) offers more candidate split points, so it has more chances to appear
to reduce impurity — even if it's no more informative than a coarser feature.
This inflates its importance score artificially.

**Fix: permutation importance.** After fitting, shuffle one feature's values
(breaking its relationship with the target) and measure how much the model's
score drops on held-out data. Repeat per feature. This measures actual
predictive contribution rather than "how often did this feature get a chance
to split," and is model-agnostic (works for any fitted model, not just
trees). More expensive to compute (requires re-scoring per feature) but far
more trustworthy — always prefer it over MDI when importance rankings drive a
real decision (feature selection, stakeholder explanations).

## When RF beats a single tree

Almost always. Averaging many trees trades a bit of bias (each individual
tree is still a shallow-ish, imperfect model) for a large reduction in
variance, and in practice this trade is favorable on nearly all tabular
problems — RF is rarely worse than a single tuned tree and usually
substantially better on unseen data.

## When boosting beats Random Forest

Gradient boosting (see
[Gradient Boosting / CatBoost / LightGBM](gradient-boosting-catboost-lgbm.md))
usually achieves better accuracy on tabular data than RF, because it fits
trees sequentially to correct the ensemble's current errors rather than
averaging independent, identically-distributed trees — it can drive bias down
further, not just variance. The cost: boosting trees depend on each other,
so training is inherently sequential (RF's trees are trivially
parallelizable), and boosting has more hyperparameters that interact
(learning rate, tree count, depth, regularization) and is more prone to
overfitting if not tuned/early-stopped carefully. RF is the safer, lower-effort
default; boosting is the higher-ceiling option when you can afford to tune it.

## Common interview questions

- Why does bagging alone not fully decorrelate trees, and how does random
  feature subsampling help?
- What is OOB error and why is it "almost free"?
- Why is MDI feature importance biased, and what's the fix?
- Random Forest vs. a single decision tree — what specifically improves?
- Random Forest vs. gradient boosting — when would you pick each?
- Does Random Forest need feature scaling? (No — same reason as a single
  tree: splits only compare a feature to a threshold.)
- What happens to RF performance as you increase the number of trees? (Error
  decreases and plateaus — more trees essentially never hurts test
  performance, unlike boosting where too many rounds can overfit.)

## Common mistakes

- Treating `n_estimators` as a tunable-for-overfitting hyperparameter the way
  you would boosting rounds — more trees in RF reduces variance and plateaus,
  it doesn't cause overfitting the way more boosting rounds can.
- Using default MDI feature importance for a decision that matters and being
  misled by a high-cardinality feature that looks artificially important.
- Setting `max_features` to include all features at every split — this
  removes the decorrelation effect and makes RF behave closer to plain
  bagging of similar trees.
- Forgetting that OOB error is still computed on training data (just
  data each individual tree didn't see) — it's a good sanity check but a true
  held-out test set is still the gold standard for final evaluation.

## Example

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

rf = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    oob_score=True,
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train, y_train)
print("OOB score:", rf.oob_score_)

# Prefer permutation importance over rf.feature_importances_ for anything
# high-stakes -- it isn't biased toward high-cardinality features.
result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
```

See also: [Gradient Boosting](gradient-boosting-catboost-lgbm.md) (sequential
trees that reduce bias instead of averaging independent ones) and
[Stacking & Blending](stacking-blending.md) (combining different model
*types* via a meta-model, rather than many trees of the same type via
voting/averaging).
