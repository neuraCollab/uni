# Stacking & Blending

See [Random Forest](random-forest.md) and
[Gradient Boosting: CatBoost / LightGBM](gradient-boosting-catboost-lgbm.md)
first — those combine *many trees* into one ensemble. Stacking and blending
are a level up: they combine *different model types* (kNN, linear
regression, a tree, a boosted model, ...) into one ensemble via a learned
meta-model, rather than by simple averaging/voting.

## What is it?

**Stacking** trains several diverse base models, then trains a **meta-model**
whose inputs are the base models' *predictions* (not the original features)
and whose target is the original label. The key subtlety is how the
meta-model's training data is generated without leaking information: use
K-fold cross-validation — for each fold, train the base models on the other
`K-1` folds and predict on the held-out fold. Stitching these held-out
predictions back together gives, for every training row, a prediction from
a base model that *never saw that row during training* — exactly the
out-of-fold discipline that avoids the meta-model learning to trust an
overfit base model's training-set predictions. Those out-of-fold
predictions become the feature matrix for the meta-model.

**Blending** is a simpler, cheaper variant: instead of K-fold, hold out a
single validation split up front. Train the base models on the remaining
(non-holdout) data, generate their predictions on the holdout split, and
fit the meta-model on those holdout predictions. Faster (one split instead
of `K` training rounds per base model) and simpler to implement, but it
uses less data for both the base models (they never see the holdout rows)
and the meta-model (it only trains on one split's worth of predictions),
so it has higher variance than stacking's K-fold version — the meta-model's
quality now depends heavily on which particular holdout split you drew.

## Why?

A single strong model has one characteristic error pattern — the specific
inputs and regions of feature space it systematically gets wrong. Different
model families (a distance-based kNN, a linear model, a tree ensemble) tend
to get *different* things wrong, because they encode different inductive
biases. A meta-model that learns *when to trust which base model* can do
better than any single base model or a fixed unweighted average, by
implicitly learning per-region weights instead of one global blend ratio.

## How does it work?

**Stacking, step by step:**

1. Choose a diverse set of base models — diversity in error pattern matters
   more than any single model's raw strength (a mediocre model with
   different failure modes can still add value to the ensemble).
2. K-fold cross-validate: for each fold, train every base model on the
   other `K-1` folds, predict on the held-out fold. After `K` rounds, every
   training row has an out-of-fold prediction from every base model.
3. Stack these predictions into a new feature matrix — one column per base
   model (optionally alongside the original features, "feature-weighted
   linear stacking") — and train a meta-model (often something simple:
   logistic/linear regression, so it just learns to weight/combine the base
   predictions rather than re-discovering complex patterns).
4. At inference time: run all base models on the new point (each base model
   is refit on the *full* training set for this final step, not on folds),
   feed their predictions into the trained meta-model, get the final
   prediction.

**Blending, step by step:**

1. Split training data into `train` and `holdout` once.
2. Train base models on `train` only.
3. Generate base-model predictions on `holdout`.
4. Fit the meta-model on `holdout` predictions vs. the true holdout labels.
5. At inference: same as stacking — run base models (refit on all available
   data, or just `train`, depending on how much you want to match the
   training conditions the meta-model saw), feed into the meta-model.

## When to use / when NOT to

**Use** stacking/blending in settings where squeezing out the last bit of
accuracy is worth the complexity — Kaggle/competition leaderboards are the
classic case, where combining a boosted tree, a neural net, and a linear
model via a meta-model routinely beats any single one of them, and
inference cost/maintainability aren't judged.

**Avoid** in most production systems: you now have `N` base models plus a
meta-model to version, monitor, retrain, and serve, multiplying latency and
operational surface area for what's usually a small accuracy gain over a
well-tuned single model (or a plain averaging ensemble, which needs no
meta-model training at all). Reach for it only when the accuracy gain is
demonstrably worth that ongoing cost — e.g. a high-value ranking/pricing
model where a fraction of a percent is worth real money and the team can
support the extra complexity.

## Common interview questions

- Stacking vs. blending — what's the practical difference, and why does
  stacking usually generalize better?
- Why must the meta-model be trained on out-of-fold (or holdout)
  predictions rather than each base model's in-sample predictions? (In-sample
  predictions from an overfit base model look artificially good, so the
  meta-model would learn to over-trust that model's own noise.)
- Why does base-model diversity matter more than base-model individual
  accuracy?
- Stacking vs. a simple averaging/voting ensemble — when is the extra
  meta-model complexity worth it?
- How would you decide whether stacking's accuracy gain justifies its
  production cost?

## Common mistakes

- Generating the meta-model's training features from each base model's
  in-sample (non-out-of-fold) predictions — this is a data leakage bug,
  structurally the same mistake as fitting a preprocessor before a CV split
  (see [Data Leakage](../model-evaluation/data-leakage.md)): the meta-model
  ends up trusting whichever base model overfits the training data hardest.
- Stacking near-identical base models (e.g. several boosted-tree variants
  with similar hyperparameters) — with correlated errors there's little for
  the meta-model to exploit, and most of the complexity buys nothing.
- Reaching for stacking as a default "make it better" move in production
  instead of first tuning a single strong model — the marginal accuracy
  gain rarely justifies the added serving/maintenance cost outside
  competition settings.
- Forgetting to refit base models on the full training set (stacking) or
  the full `train` split (blending) before final deployment, and instead
  shipping the fold-restricted models used only to generate meta-features.

## Example

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# sklearn's StackingClassifier handles the K-fold out-of-fold prediction
# generation internally (via `cv=`), so the meta-model never sees in-sample
# base-model predictions.
stack = StackingClassifier(
    estimators=[
        ("knn", KNeighborsClassifier(n_neighbors=15)),
        ("tree", DecisionTreeClassifier(max_depth=5, random_state=42)),
    ],
    final_estimator=LogisticRegression(),
    cv=5,
)
stack.fit(X_train, y_train)
```

See also: [Random Forest](random-forest.md) (ensembling many trees of the
*same* type via bagging) and
[Gradient Boosting](gradient-boosting-catboost-lgbm.md) (ensembling many
trees of the same type sequentially) — stacking/blending is the
model-agnostic generalization of "combine several predictors into one,"
applied across model families rather than within one.
