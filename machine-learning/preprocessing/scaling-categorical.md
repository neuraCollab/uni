# Scaling & Categorical Encoding

## What is it?

Two related preprocessing steps that most models need before they can
consume raw tabular data well: **scaling** transforms numeric features onto
comparable ranges, and **encoding** turns categorical (non-numeric) features
into a numeric representation. Both [`hp_boosting_optuna.py`](../trees-ensembles/code/hp_boosting_optuna.py)
and [`h2_lgbm_pipeline.py`](../trees-ensembles/code/h2_lgbm_pipeline.py) build
a `ColumnTransformer` that does exactly this split (median-impute + scale for
numeric, constant-impute + one-hot for categorical) — see
[Gradient Boosting / CatBoost / LightGBM](../trees-ensembles/gradient-boosting-catboost-lgbm.md)
for that pipeline's full write-up, including a data-leakage bug that came from
fitting this exact kind of transformer in the wrong place.

## Scaling: StandardScaler vs. MinMaxScaler vs. RobustScaler

| Scaler | Transform | Sensitive to outliers? | When to use |
|---|---|---|---|
| `StandardScaler` | $\dfrac{x - \text{mean}}{\text{std}}$ → mean 0, unit variance | Yes — mean/std are both pulled by outliers | Default choice for roughly-normal data, most linear models/neural nets |
| `MinMaxScaler` | $\dfrac{x - \min}{\max - \min}$ → range [0, 1] | Very — a single extreme value compresses the rest of the range | Bounded-input models (e.g. some neural net activations), when you need a known fixed range |
| `RobustScaler` | $\dfrac{x - \text{median}}{\text{IQR}}$ | No — uses median and interquartile range, both outlier-resistant | Data with real outliers you don't want to remove but don't want dominating the scale |

**When scaling matters:**
- **Distance-based models** — kNN (see [kNN](../knn.md)), k-means and other
  Euclidean-distance clustering (see
  [Distance Metrics](../clustering/distance-metrics.md)), SVMs — an unscaled
  feature with a larger numeric range dominates the distance calculation
  purely due to units, not actual importance.
- **Gradient-based models** — linear/logistic regression (especially with
  regularization — see [Regularization](../linear-models/regularization.md),
  which penalizes raw coefficient magnitude, so unscaled features get
  unfairly penalized relative to their natural scale), neural networks —
  unscaled features distort the loss surface and slow/destabilize gradient
  descent.

**When scaling doesn't matter:** tree-based models (decision trees, random
forest, gradient boosting) are **scale-invariant** — a split only compares a
feature to a threshold, so any monotonic transform of that feature (scaling
included) doesn't change which splits get chosen. See
[Decision Trees](../trees-ensembles/decision-trees.md).

## Categorical encoding

**One-hot encoding** — one binary column per category. Safe default; no
implied ordering between categories. Downsides: blows up dimensionality with
high-cardinality columns, and can dilute a tree's split candidates across
many near-identical low-information branches (see
[Decision Trees](../trees-ensembles/decision-trees.md)'s note on this).

**Ordinal encoding** — one integer per category (`0, 1, 2, ...`). Only
appropriate when categories have a genuine order (`low < medium < high`) —
using it on unordered categories silently invents a fake ordinal
relationship the model may pick up on (e.g. treating `red=0, green=1,
blue=2` as if blue is somehow "more" than red).

**Target/mean encoding** — replace each category with a statistic of the
target computed within that category (e.g. mean target value for rows with
that category). Handles high-cardinality categoricals without
dimensionality blow-up, and often outperforms one-hot on tree models. The
serious risk: **leakage**. If a category's encoding is computed using the
*same* rows it will then be applied to (including the row's own target
value baked into its own encoding), the model effectively sees the target
leaking through the feature — inflated validation performance that
collapses at deployment. The fix is the same discipline as any preprocessing
step: compute target encodings using proper cross-validation (encode each
fold using statistics from the *other* folds only, or use an out-of-fold /
leave-one-out scheme), exactly as you'd fit a `ColumnTransformer` only on
training folds — see
[Data Leakage](../model-evaluation/data-leakage.md) and the leakage bug fixed
in [`h2_lgbm_pipeline.py`](../trees-ensembles/code/h2_lgbm_pipeline.py).

**High-cardinality categoricals (hundreds/thousands of levels)** — one-hot
becomes impractical (too many columns, mostly zeros). Options:
- **Hashing trick** — hash category values into a fixed number of buckets,
  trading a small amount of collision noise for bounded dimensionality; no
  fitted vocabulary needed, works in streaming/online settings.
- **Embeddings** — learn a low-dimensional dense vector per category (common
  in neural nets); captures similarity between categories but needs enough
  data per category to learn well, and isn't directly usable by classic
  sklearn estimators without extra tooling.
- Target/mean encoding (above) is also a common practical answer here, again
  with the leakage caveat front and center.

## When to use / when not to

**Use** scaling whenever the downstream model is distance- or gradient-based;
use one-hot as the safe default categorical encoding, moving to target
encoding only when cardinality is high and you can enforce proper CV
discipline, or hashing/embeddings when cardinality is extreme.

**Skip** scaling for tree-based/boosted models — it's wasted effort and
changes nothing about their predictions. Skip target encoding entirely if
you can't guarantee the CV-safe computation — a leaky target encoding is
worse than a boring one-hot encoding that at least can't cheat.

## Common interview questions

- Why do tree-based models not need feature scaling, but linear
  models/kNN/SVMs do?
- StandardScaler vs. RobustScaler — when would you pick RobustScaler?
- What's wrong with using ordinal encoding on an unordered categorical
  feature?
- How can target encoding leak information, and how do you prevent it?
- How would you handle a categorical feature with 50,000 unique values?
- Why must a scaler be fit only on the training fold, never on the full
  dataset before splitting?

## Common mistakes

- Scaling before splitting into train/test (or before CV folds) — the
  same class of leakage as fitting a `ColumnTransformer` on the full
  dataset before splitting; see
  [`h2_lgbm_pipeline.py`](../trees-ensembles/code/h2_lgbm_pipeline.py)'s
  fixed bug and [Data Leakage](../model-evaluation/data-leakage.md).
- Applying `StandardScaler` blindly on data with heavy outliers instead of
  reaching for `RobustScaler`.
- One-hot encoding a very high-cardinality column without considering
  target encoding, hashing, or dropping rare categories into an "other"
  bucket first.
- Computing target encoding on the whole training set once, then using it
  inside cross-validation — this leaks target information across folds even
  if train/test itself was split correctly.
- Forgetting `handle_unknown="ignore"` (one-hot) or an equivalent fallback
  for categories seen at inference time but not during training.

## Example

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("num", num_pipe, num_feats),
    ("cat", cat_pipe, cat_feats),
])
# Wrap in a Pipeline with the model so preprocessing is refit per CV fold --
# see gradient-boosting-catboost-lgbm.md for what goes wrong otherwise.
```

## See also

- [Missing Values / Imputation](missing-values-imputation.md)
- [Feature Engineering](feature-engineering.md)
- [Regularization](../linear-models/regularization.md)
- [Data Leakage](../model-evaluation/data-leakage.md)
