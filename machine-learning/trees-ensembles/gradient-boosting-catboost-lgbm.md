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
currently gives the largest loss reduction, regardless of its depth. This
converges to lower loss with fewer splits (faster, often more accurate), but
can produce deeper, more unbalanced trees that overfit more easily on small
datasets — `max_depth` / `num_leaves` need tighter control than with
level-wise growth.

**Histogram-based binning** — continuous features are bucketed into a fixed
number of discrete bins (e.g. 255) before split-finding, so split search
scans over bins instead of every unique value. Much faster and more
memory-efficient than exact greedy split-finding, at the cost of some split
precision (usually negligible).

**LightGBM vs. XGBoost (for context, not in this repo's code):** XGBoost
defaults to level-wise growth with exact or approximate histogram
split-finding; LightGBM's leaf-wise + histogram-first design is generally
faster on large datasets and competitive or better in accuracy, but is more
prone to overfitting small datasets without careful regularization
(`min_child_samples`, `num_leaves`, `feature_fraction`/`bagging_fraction`
subsampling — all visible in the Optuna search space below).

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
