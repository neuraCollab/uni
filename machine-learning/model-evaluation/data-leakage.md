# Data Leakage

## What is it?

Any situation where information that wouldn't be available at real
prediction time leaks into model training or evaluation, making offline
metrics look better than the model will actually perform in production.
Leakage is one of the most common — and most silently damaging — bugs in ML
pipelines, because the model still "runs" and gives plausible-looking
numbers; it just lies about how good it is.

## Why it matters

A leaky pipeline can show great cross-validation or test scores and then
fail badly in production, because the leaked information (validation-set
statistics, a future value, a duplicate row) simply won't exist when the
model has to make a prediction on genuinely new data.

## Spot the leakage: a real example

[`../trees-ensembles/code/h2_lgbm_pipeline.py`](../trees-ensembles/code/h2_lgbm_pipeline.py)
is a fixed version of a house-price regression pipeline that originally had
two real bugs (see that file's module docstring for the full fix — this is
a condensed walkthrough).

### Bug 1: preprocessor fit on the full training set before the split

The original code did this:

```python
# BUGGY
preprocessor.fit(train.drop(columns=["Id", "SalePrice"]))   # fit on ALL rows
train_set, val_set = train_test_split(train, test_size=0.2, random_state=42)
# ... later: preprocessor.transform(train_set), preprocessor.transform(val_set)
```

**Why it leaks:** the `StandardScaler`'s mean/std and the `SimpleImputer`'s
median were computed using rows that later ended up in the *validation*
split. The model's inputs during "validation" were partly derived from
statistics of the validation set itself — the validation fold is no longer a
clean stand-in for genuinely unseen data. In cross-validation this is even
more damaging: every fold's preprocessing statistics would be contaminated
by every other fold, so CV scores overestimate real generalization on *every*
fold, not just one split.

**The fix:** wrap the preprocessor and the model together in one
`sklearn.Pipeline`, and only ever call `.fit()` on the pipeline with training
data:

```python
# FIXED
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", base_model),
])

train_set, val_set = train_test_split(train, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)  # preprocessor is fit on X_train ONLY
preds = pipeline.predict(X_val)  # X_val is only ever *transformed*, never fit on
```

The same `Pipeline` object, passed straight into `cross_val_score` or
`GridSearchCV`/`RandomizedSearchCV`, automatically refits the preprocessor
from scratch on each fold's training portion and only transforms that fold's
held-out portion — this is the standard, general-purpose fix any time you
see a `preprocessor.fit(...)` call sitting *before* a split or a CV loop.

### Bug 2: `StratifiedKFold` on a regression target

```python
# BUGGY
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # SalePrice is continuous!
```

Not leakage in the strict "information from the future" sense, but the same
family of bug: a validity mismatch between the CV strategy and the target
type produces meaningless folds and an unreliable performance estimate. See
[Cross-Validation](cross-validation.md#stratified-k-fold) for the full
explanation and fix (`KFold` for regression, or quantile-bucket-then-stratify
if you need balanced folds for a skewed continuous target).

## Other common leakage types

**Target leakage** — a feature that wouldn't actually be available (or
wouldn't yet have its final value) at prediction time gets included in
training. Classic examples: including a "cancellation reason" field when
predicting whether an order will be cancelled (it's only populated *after*
cancellation), or including a post-treatment lab result when predicting
whether a patient needed treatment. The tell-tale sign is a feature with
suspiciously high importance/correlation with the target — always ask "would
I actually have this value at the moment I need to make the prediction?"

**Feature engineering computed on the full dataset before splitting** — the
same pattern as Bug 1 above, but in feature-engineering code rather than a
`ColumnTransformer`: e.g., computing a target-encoded category mean, a
global normalization constant, a PCA transform, or a frequency count using
the *entire* dataset (train + val + test) before splitting. Any statistic
derived from data should be derived from the training fold only, then
applied (not re-fit) to validation/test.

**Duplicate (or near-duplicate) rows split across train and test** — if the
same or a near-identical row appears in both the training and test sets
(common with scraped/augmented data, or multiple snapshots of the same
entity), the model can effectively "memorize" that row during training and
then trivially "predict" it during evaluation. Check for and deduplicate
before splitting, and be careful that augmented/derived rows of the same
underlying entity don't get split across the boundary (group by entity ID
using `GroupKFold` when relevant).

## When to use / when not

This isn't a technique to selectively apply — it's a checklist to run against
every pipeline. Always check: (1) is every fitted transformer fit only on
the current training fold? (2) does every feature only use information
available at prediction time? (3) are there duplicate/near-duplicate rows
crossing the train/test boundary? (4) does the CV strategy match the target
type and any grouping/temporal structure in the data?

## Common interview questions

- Give an example of target leakage you'd watch for in a specific domain
  (fraud, churn, medical).
- Why does fitting a scaler on the full dataset before splitting inflate
  validation performance?
- How does wrapping preprocessing in a `Pipeline` prevent leakage during
  cross-validation specifically?
- What's the difference between leakage and just having a small/unrepresentative
  test set?
- How would you check whether a feature is leaking the target?
- Why is time-series data especially leakage-prone? (See
  [Cross-Validation](cross-validation.md#time-series-split-walk-forward-validation) —
  shuffling breaks the temporal boundary.)
- If you inherited a pipeline with a suspiciously high test AUC (e.g. 0.999
  on a hard problem), what would you check first?

## Common mistakes

- Calling `.fit()` or `.fit_transform()` on a preprocessor using the full
  dataset before doing `train_test_split` or setting up CV.
- Passing `StratifiedKFold` (or anything stratification-based) a continuous
  target.
- Engineering features (target encoding, aggregates, ratios) using
  statistics from the whole dataset instead of the training fold only.
- Not deduplicating rows, or not grouping by entity, before splitting.
- Evaluating a "final" model on the same validation set that was used
  repeatedly to pick hyperparameters (a slower, "leak by iteration" form of
  the same problem — see [Cross-Validation](cross-validation.md#nested-cv)
  for the nested-CV fix).

## Example

See [`../trees-ensembles/code/h2_lgbm_pipeline.py`](../trees-ensembles/code/h2_lgbm_pipeline.py)
for the complete before/after of both bugs described above, in a real
LightGBM regression pipeline with `RandomizedSearchCV`.

```python
# General pattern: always fit preprocessing INSIDE a Pipeline, never before the split.
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score

pipeline = Pipeline([
    ("preprocessor", preprocessor),   # scaler / imputer / encoder
    ("model", model),
])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)                       # preprocessor fit on X_train only
scores = cross_val_score(pipeline, X_train, y_train)  # re-fit per fold, no leakage
```

See also: [Cross-Validation](cross-validation.md),
[Classification Metrics](classification-metrics.md).
