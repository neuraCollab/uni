# Gradient Boosting: CatBoost & LightGBM (vs. XGBoost)

See [Decision Trees](decision-trees.md) and [Random Forest](random-forest.md)
first. This note covers how boosted trees differ from bagged trees, and what
CatBoost and LightGBM specifically do differently from each other (and from
XGBoost).

## Bagging vs. boosting

| | Bagging (Random Forest) | Boosting (GBM/CatBoost/LightGBM/XGBoost) |
|---|---|---|
| Trees trained | independently, in parallel | sequentially, each depending on the last |
| Each tree targets | the original target, on a bootstrap sample | the *errors* of the current ensemble |
| Reduces | mainly variance | mainly bias (and variance, if regularized) |
| Overfitting risk | low, plateaus with more trees | higher — more rounds *can* overfit, needs early stopping |
| Parallelizable | fully | not across boosting rounds (within-round split-finding is parallel) |

## Gradient boosting core idea

Build an additive ensemble `F(x) = f_1(x) + f_2(x) + ... + f_M(x)` one tree at
a time. At step `m`, instead of refitting the target, fit a new tree `f_m` to
the **negative gradient of the loss function** with respect to the current
ensemble's predictions (the "pseudo-residuals") — for squared-error loss this
gradient is literally `y - F_{m-1}(x)`, i.e. the plain residual, which is why
"fit each tree to the previous tree's errors" is the common shorthand. For
other losses (logloss, etc.) it's the gradient of that loss, scaled by a
learning rate (`shrinkage`) so no single tree dominates the ensemble.

### The derivation: functional gradient descent

Each new base algorithm is trained to reduce the *remaining* error of
everything trained before it — which is what drives down bias, in contrast
to bagging's independent, identically-targeted trees (see
[bias-variance-tradeoff.md](../model-evaluation/bias-variance-tradeoff.md)).

**Step 1 — start from squared-error regression, where the target is
obvious.** With `L(y, b(x)) = (y - b(x))²`, fit base learners one at a time:

```
b1(x) = argmin_{b∈B} L(yi, b(xi))
a1(x) = b1(x)

s_i = yi - Σ_{j=1}^{k} b_j(xi) = yi - a_k(xi)     # residual after k rounds

b_{k+1}(x) = argmin_{b∈B} L(s_i, b(xi))            # fit next tree to the residual
a_{k+1}(x) = a_k(x) + b_{k+1}(x)
```

At each step, the current ensemble `a_k(x)` under-predicts or over-predicts
by `s_i = yi - a_k(xi)`, and the next base learner is trained to predict
exactly that leftover residual, closing the gap. This is the "first
approximation": it works, but it's phrased in terms of *residuals*, which
only makes sense for squared-error loss.

**Step 2 — generalize: move in the direction of the anti-gradient.** The
residual `yi - a_k(xi)` is not a special quantity in general — it's what the
negative gradient of squared-error loss happens to equal. The correct
generalization to *any* differentiable loss is to fit each new tree to the
**anti-gradient (negative gradient) of the loss with respect to the current
prediction**, evaluated pointwise at each training example:

```
g_i^k = ∂L(yi, z) / ∂z  |  z = a_k(xi)

b_{k+1}(x) = argmin_{b∈B} L(-g_i^k, b(xi))          # fit tree to -g_i^k
a_{k+1}(x) = a_k(x) + b_{k+1}(x)
```

For squared-error loss `L = (y - z)²`, `∂L/∂z = -2(y - z)`, so
`-g_i^k = 2(yi - a_k(xi))` — proportional to the plain residual, recovering
Step 1 exactly. That's the point: **residual-fitting is the MSE special
case of anti-gradient fitting**, not a separate technique. Framing boosting
as functional gradient descent (take a step, in function space, in the
direction that most decreases the loss) is what lets it plug in *any*
differentiable loss — logloss for classification, quantile loss, or a
ranking loss — while "fit the residual" only ever made sense for squared
error.

**Ranking as evidence this generalizes: Pair Logit.** Gradient boosting for
learning-to-rank fits trees to the anti-gradient of a *pairwise* loss
instead of a pointwise one. One common pairwise loss (Pair Logit) is:

```
Pair Logit = -(1/|pairs|) * Σ_{(p,n)∈pairs}  log( 1 / (1 + e^(-(a_p - a_n))) )
```

where `a_p`, `a_n` are the current model's predictions for a relevant (`p`)
and less-relevant (`n`) document/item in a labeled pair. This is a logistic
loss over the *difference* of two predictions rather than over a single
target, and gradient boosting still applies unchanged — compute its
gradient w.r.t. each prediction, fit trees to the anti-gradient, exactly as
above. This is the clearest evidence that gradient boosting is a general
optimization scheme, not a regression-specific trick.

### Training the base tree at each step

Given the per-example anti-gradients `-g_i^k` (some texts denote this
`h_i = -g_i^k`), building the actual base learner at round `k` is a
two-step recipe:

1. Compute the anti-gradient of the loss function at every training point,
   evaluated at the current ensemble's predictions:
   `h_i = -g_i^k = -∂L(yi, z)/∂z |_{z = a_k(xi)}`.
2. On the training set `(x_i, h_i)`, fit a regression tree that minimizes
   whichever *evaluation function* is chosen for the tree itself (typically
   squared error over the `h_i` targets, regardless of what the outer loss
   `L` is) — i.e. the base learner is always a plain regression tree, even
   when the overall task is classification or ranking, because it's
   predicting a continuous gradient value, not a class label.

## CatBoost: ordered boosting + native categorical handling

**Ordered boosting** — standard gradient boosting computes each point's
pseudo-residual using a model that was (indirectly) trained using that same
point's target, which leaks a little target information into its own
gradient estimate ("prediction shift" / a subtle target leakage). CatBoost
instead maintains several models trained on different random permutations of
the data and, for each point, computes its gradient using only a model that
was fit on points *preceding* it in that permutation's ordering — closer to
how the model would behave on truly unseen data. This reduces overfitting,
especially on small/medium datasets.

**Native categorical handling** — CatBoost converts categorical features into
numeric statistics (an ordered form of target encoding, computed similarly
"using only preceding points" to avoid leakage) internally, plus it can
combine categorical features into new composite features automatically. This
means you often don't need to one-hot/label-encode categoricals yourself —
pass column indices via `cat_features` and CatBoost handles it, which matters
a lot on datasets with many categorical columns (less preprocessing code,
and no dimensionality blow-up from one-hot encoding high-cardinality
columns).

## LightGBM: leaf-wise growth + histogram binning

**Leaf-wise (best-first) growth** — instead of growing a tree level-by-level
(every leaf at depth `d` splits before any leaf at depth `d+1` does, which is
what XGBoost does by default), LightGBM always splits whichever *leaf*
currently gives the largest loss reduction (best score), regardless of its
depth or which branch it's on. The main growth-limiting hyperparameter is
therefore `num_leaves` (max leaves in the whole tree), not depth. Because
splits are chosen purely by "best score wins" with no per-level symmetry
constraint, the resulting tree can be — and usually is — asymmetric: one
branch might end up with far more leaves than a sibling branch, since a
leaf-wise grower keeps drilling into whichever region still has the most
loss to recover. This converges to lower loss with fewer splits (faster,
often more accurate), but produces deeper, more unbalanced trees that
overfit more easily, especially on small datasets — `max_depth` /
`num_leaves` need tighter control than with level-wise growth.

**Histogram-based binning** — continuous features are bucketed into a fixed
number of discrete bins (e.g. 255) before split-finding, so split search
scans over bins instead of every unique value. Much faster and more
memory-efficient than exact greedy split-finding, at the cost of some split
precision (usually negligible).

## XGBoost: level-wise (depth-wise) growth

XGBoost's default grower builds the tree **level by level**, expanding every
node at the current depth before moving to the next depth, until it hits
`max_depth`. Because every branch is extended in lockstep rather than
racing toward whichever leaf currently looks best, the result is a more
symmetric, more balanced binary tree (assuming no other constraints prune
individual branches early) — and, because it doesn't greedily chase the
single best-looking leaf every round, it's generally **less prone to
overfitting** than leaf-wise growth for the same number of leaf splits.
XGBoost does also offer a `grow_policy="lossguide"` mode that mimics
LightGBM's leaf-wise strategy, but level-wise/depth-wise is the classic,
default XGBoost behavior this comparison refers to.

**LightGBM vs. XGBoost, summarized:** LightGBM's leaf-wise + histogram-first
design is generally faster on large datasets and competitive or better in
accuracy, but is more prone to overfitting small datasets without careful
regularization (`min_child_samples`, `num_leaves`,
`feature_fraction`/`bagging_fraction` subsampling — all visible in the
Optuna search space below). XGBoost's level-wise growth is more
conservative and typically needs less anti-overfitting tuning out of the
box, at some cost in training speed and a slightly higher chance of wasting
splits on branches that weren't worth expanding.

## CatBoost: oblivious (symmetric) trees

CatBoost's base learners are **oblivious decision trees**: every node *at a
given depth* uses the exact same split — the same feature and the same
threshold (e.g. "is `x_5 >= 6.5`?" at every node of level 3), not just the
same feature. This is a much stronger symmetry constraint than XGBoost's
level-wise growth (which balances tree *shape* but still lets each node
pick its own best feature/threshold). Growing a full oblivious tree of
depth `d` this way is equivalent to evaluating `d` shared yes/no predicates
and looking up the resulting `2^d`-way leaf — which makes prediction very
fast (just `d` comparisons, no tree traversal) and acts as an implicit
regularizer, since a single split predicate has to work reasonably well
across the *entire* level rather than being locally optimized for one
node's subset of data. A side effect: because the split is shared, some of
the `2^d` leaf combinations may end up with no training examples routed
into them at all — those subtrees/leaves simply see zero training points.
Combined with **ordered boosting** (described above, which fixes the
target-leakage issue from computing a point's own gradient using a model
that indirectly saw that point's label), this is what makes CatBoost
comparatively resistant to overfitting on small/medium tabular data.

## Early stopping

Because boosting keeps reducing training error indefinitely, the number of
rounds is a critical hyperparameter. Rather than tuning `n_estimators`
directly, hold out a validation set and stop adding trees once validation
loss hasn't improved for `N` rounds (`lgb.early_stopping(50)`, CatBoost's
`od_type="Iter", od_wait=50`). This picks a near-optimal round count
automatically and is the standard way to prevent boosting from overfitting.

## Feature importance: gain vs. split-count vs. SHAP

- **Split count (weight)** — how many times a feature was used to split,
  across all trees. Cheap, but biased the same way as decision-tree
  importance toward features with many candidate thresholds.
- **Gain** — total loss reduction attributed to a feature across all its
  splits. Usually more meaningful than split count, but still an
  average/aggregate that can hide sign and interaction effects (a feature can
  have high gain while pushing predictions in opposite directions for
  different subgroups).
- **SHAP values (modern standard, not used in this repo's code but worth
  knowing)** — a game-theoretic per-prediction attribution: for each
  individual prediction, SHAP assigns each feature a signed contribution that
  sums to `prediction - baseline`. Averaging `|SHAP|` across all predictions
  gives a global importance ranking that's more consistent and interaction-aware
  than gain/split-count, and per-prediction SHAP plots let you explain *why*
  one specific prediction was made — the standard tool when explainability
  matters (both LightGBM and CatBoost have built-in fast SHAP support via
  TreeSHAP).

## What the ported code demonstrates

[`code/hp_boosting_optuna.py`](code/hp_boosting_optuna.py) — binary
classification with LightGBM and CatBoost, each hyperparameter-tuned by
Optuna, then combined into a simple unweighted 2-model averaging ensemble.
Two points worth knowing for interviews:

- **Conditional search spaces.** Optuna's `suggest_*` calls build the search
  space *inside* the objective function, so a parameter can depend on another
  one sampled earlier in the same trial. Here `bagging_temperature` is only
  sampled when CatBoost's `bootstrap_type == "Bayesian"` (Bernoulli/MVS
  bootstrap don't use it) — a plain grid/random search over a fixed parameter
  set can't express that without wasting trials on invalid combinations.
- **Ensembling two differently-biased models is a cheap variance reducer** —
  averaging LightGBM's (leaf-wise, histogram) and CatBoost's (ordered
  boosting) predictions smooths out each model's individual quirks, similar
  in spirit to why Random Forest averages many trees, just at the model
  level instead of the tree level.
- A porting bug worth remembering as a lesson: the original code cast the
  averaged *probability* straight to `int` (truncating almost everything to
  0, since an average rarely reaches exactly 1.0) instead of thresholding at
  0.5 first. Always sanity-check a value's range immediately before a type
  cast.

[`code/h2_lgbm_pipeline.py`](code/h2_lgbm_pipeline.py) — a LightGBM
regression pipeline (log-target house-price prediction) tuned with
`RandomizedSearchCV`. Two real bugs were found and fixed while porting this
file, both good "what not to do" examples:

1. **Data leakage** — the original fit the `ColumnTransformer` preprocessor
   (median-impute + scale numeric, impute + one-hot categorical) on the
   *full* training set **before** the train/validation split and before
   cross-validation, so the scaler's mean/std and the imputer's median were
   computed using rows that later ended up in the held-out fold. Fixed by
   wrapping preprocessor + model in one `sklearn.Pipeline`, so
   `cross_val_score`/`RandomizedSearchCV` refit the preprocessor from scratch
   on each fold's training rows only. This is the standard fix any time you
   see a `preprocessor.fit(...)` call sitting before a CV loop or split — see
   [Scaling & Categorical Encoding](../preprocessing/scaling-categorical.md).
2. **Invalid stratification** — the original used `StratifiedKFold` on a
   continuous regression target (log-transformed sale price), which only
   makes sense for categorical labels. Fixed by switching to plain `KFold`.

## When to use / when not to

**Use** gradient boosting when predictive accuracy on tabular data is the
priority and you can afford to tune it (learning rate, tree count via early
stopping, depth/leaves, regularization) — it's usually the strongest
off-the-shelf tabular model. Reach for **CatBoost** specifically when you have
many categorical features and want to avoid manual encoding; reach for
**LightGBM** when you have large data and need fast training, and are willing
to watch for overfitting on smaller subsets.

**Avoid** boosting when you need a quick baseline with minimal tuning
(Random Forest is more forgiving), when training must be trivially
parallelizable across independent trees, or when interpretability without
extra tooling (SHAP) matters more than squeezing out accuracy.

## Common interview questions

- How does gradient boosting differ from bagging conceptually?
- What does a boosted tree actually fit at each step? (The negative gradient
  of the loss w.r.t. current predictions — plain residuals for squared error.)
- What is ordered boosting in CatBoost and what problem does it solve?
- Leaf-wise vs. level-wise tree growth — tradeoffs?
- Why does LightGBM need tighter regularization on small datasets?
- How does early stopping replace manually tuning `n_estimators`?
- Gain vs. split-count vs. SHAP for feature importance — which would you
  trust for a stakeholder-facing explanation, and why?
- Why is a `ColumnTransformer.fit()` call before a train/test split or CV
  loop a data leakage bug?

## Common mistakes

- Fitting preprocessing (scalers/imputers/encoders) on the full dataset
  before splitting — see the `h2_lgbm_pipeline.py` leakage bug above.
- Using `StratifiedKFold` on a continuous target.
- Treating boosting round count as free — always pair a large
  `n_estimators`/`iterations` with early stopping, not a fixed guess.
- Casting an averaged probability straight to `int` without thresholding
  first (truncates almost everything to 0).
- Reading raw gain/split-count importance as causal or stable across retrains
  without cross-checking against permutation importance or SHAP.

## Example

See [`code/hp_boosting_optuna.py`](code/hp_boosting_optuna.py) (LightGBM +
CatBoost + Optuna + simple ensembling) and
[`code/h2_lgbm_pipeline.py`](code/h2_lgbm_pipeline.py) (leakage-safe LightGBM
regression pipeline with `RandomizedSearchCV`).

See also: [Random Forest](random-forest.md) for the bagging counterpart, and
[Imbalanced Data](../imbalanced-data.md) for handling a skewed target class
like the `Depression` binary target in `hp_boosting_optuna.py`.
