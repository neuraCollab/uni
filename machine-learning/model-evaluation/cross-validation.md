# Cross-Validation

## What is it?

A family of techniques for estimating how a model will perform on unseen
data by repeatedly splitting the available data into train/validation
folds, fitting on each training fold, and evaluating on the held-out
portion — instead of relying on a single train/test split (which is a noisy
estimate, especially on small datasets).

## Why?

A single held-out split gives one sample of "how well does this
generalize" — its result depends on which rows happened to land in the
validation set. Cross-validation averages over several such splits, giving a
lower-variance, more trustworthy estimate of generalization error, and lets
you use nearly all the data for both training and evaluation across folds.

## How does it work?

### Hold-out (train/test split)

The simplest form of validation: split the data once into a training set
and a test set (a common default is 80% train / 20% test), fit on the
training set, score once on the test set.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)
```

`shuffle=True` is the **default** because most tabular datasets are treated
as i.i.d. rows with no meaningful order — shuffling before splitting avoids
handing the test set a biased slice (e.g. if the data happens to be sorted
by label, by time, or by some other column). **Turn it off
(`shuffle=False`) only for time-ordered data.** For a time series,
shuffling before splitting lets rows from *after* the test period land in
the training set — the model ends up training on the future and being
evaluated on the past it has effectively already seen, which produces an
optimistic, leaky estimate. Use a fixed chronological cutoff instead (or
[`TimeSeriesSplit`](#time-series-split-walk-forward-validation) below).

In practice a three-way split (`train` / `val` / `test`) is the more useful
default once you're tuning anything at all:

- **`train`** — fit the model's parameters.
- **`val`** — the "practice test," used repeatedly while developing: to
  select hyperparameters, compare architectures/preprocessing choices, and
  decide when to stop. You're allowed to look at this score as many times
  as you like.
- **`test`** — touched exactly once, at the very end, to report the number
  you'll actually trust as an estimate of real-world generalization.
  Because no decision was ever made based on it, it isn't biased by the
  tuning process the way the val score is.

Tuning against `val` and then *also* reporting `val` as your final
performance number gives an optimistically biased estimate — see [Data
Leakage](data-leakage.md) and [Nested CV](#nested-cv) below for the
cross-validation version of the same problem.

### K-Fold

Split the data into `k` equal-ish chunks ("folds"). For each of the `k`
folds in turn, train on the other `k-1` folds and evaluate on that one; average
the `k` scores. `k=5` or `k=10` are standard defaults — more folds means less
bias in the estimate (more data per training fold) but higher variance and
compute cost per fold.

```python
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring="neg_root_mean_squared_error")
```

#### What do you do with the K trained models?

Running K-Fold leaves you with `k` trained models and `k` scores — but
usually you need a single final artifact (or a single final prediction).
There are three options here, and they answer different questions:

1. **Pick the best-scoring hyperparameter config, then retrain one model on
   the full dataset.** This is the standard move when CV is being used for
   **hyperparameter tuning**: the `k` folds exist purely to get a
   low-variance *estimate* of how well a given configuration generalizes —
   not to produce `k` models you intend to keep. Once you know which
   configuration scored best on average, discard the `k` fold-restricted
   models and refit that exact configuration on 100% of the training data;
   more data almost always beats keeping a model that only saw `(k-1)/k` of
   it. This is what `GridSearchCV`/`RandomizedSearchCV` do automatically
   under `refit=True`.
2. **Average the predictions of all `k` models** (or otherwise ensemble
   them). This treats CV as a mini-bagging technique: each fold's model saw
   a different slice of the training data, so their errors are partially
   decorrelated, and averaging reduces prediction variance the same way
   bagging does. This is the more common choice in Kaggle-style pipelines,
   where the deliverable is *predictions on a fixed test set* rather than a
   single reusable model object, and the small ensembling bump is worth
   paying `k`x the inference cost for.
3. **Just keep one of the fold models** (e.g. the first one). Occasionally
   used as a cheap shortcut when a full refit is expensive and fold-to-fold
   variance is known to be small — but it throws away the other `k-1`
   folds' worth of training signal and is rarely the right default; treat
   it as a fallback, not a strategy.

Rule of thumb: **tuning hyperparameters → retrain one final model on all the
data (option 1). Optimizing leaderboard/production predictions directly →
average or ensemble the fold models (option 2).**

### Stratified K-Fold

Same idea, but each fold is constructed to preserve the **overall class
proportions** of `y` — important when classes are imbalanced, so no fold
ends up with, say, zero examples of a rare class purely by chance.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**`StratifiedKFold` is for classification targets only.** It stratifies by
preserving *class label* proportions, which requires `y` to actually be a
finite set of categories. Passing it a continuous regression target either
errors out or (worse) silently treats every distinct float value as its own
"class," producing near-meaningless folds — each "class" has ~1 member, so
"preserving class proportions" is vacuous and the split degenerates. This is
a real bug documented in
[`../trees-ensembles/code/h2_lgbm_pipeline.py`](../trees-ensembles/code/h2_lgbm_pipeline.py)
(see [Data Leakage](data-leakage.md) for the full story) — the original code
used `StratifiedKFold` to cross-validate a log-transformed `SalePrice`
regression target; the fix was switching to plain `KFold`. If you actually
need CV folds balanced on a skewed regression target, bucket the target into
quantile bins first and stratify on the bin label, not on the raw values.

### Leave-One-Out (LOO)

The extreme case of K-Fold where `k = n`: train on all but one sample,
evaluate on that one, repeat for every sample. Nearly unbiased (each training
fold uses almost all the data) but very high variance between folds and
`O(n)` model fits — only practical for small datasets, or algorithms with a
cheap closed-form leave-one-out update (e.g. `RidgeCV`'s efficient built-in
LOO).

### Time-series split (walk-forward validation)

Never shuffle time-ordered data for CV — a random K-Fold split lets the
model "see the future" (train on data from after the validation window) and
produces an optimistic, unrealistic estimate. Instead, walk forward: each
fold trains on all data *up to* some point in time and validates on the
chunk immediately after it, and the training window only grows (or slides)
forward with each fold.

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    ...  # train_idx always precedes val_idx in time
```

This is one of the most common CV traps in interviews and in practice: any
time the rows have a temporal order and future values could leak information
about (or be trivially correlated with) past ones, shuffling before
splitting silently inflates your validation score.

### Nested CV

Used when you're both **tuning hyperparameters** and **estimating
generalization performance** — using the same CV loop for both gives an
optimistically biased performance estimate, because the hyperparameters were
chosen specifically to do well on that validation data.

Nested CV fixes this with two loops:
- **Inner loop**: for each outer-training fold, run a full
  CV-based hyperparameter search (e.g. `GridSearchCV`) to pick the best
  hyperparameters using only that fold's data.
- **Outer loop**: evaluate the inner loop's best model on the outer
  fold's held-out data, which the hyperparameter search never saw.

The outer loop's averaged score is an honest estimate of "if I tune
hyperparameters this way on similar future data, how well will the result
generalize" — it does *not* give you a single final hyperparameter setting
to deploy (refit the search on the full dataset separately for that).

```python
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold

inner_cv = KFold(n_splits=3, shuffle=True, random_state=0)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)

search = GridSearchCV(model, param_grid, cv=inner_cv, scoring="accuracy")
nested_scores = cross_val_score(search, X, y, cv=outer_cv)
print("Honest generalization estimate:", nested_scores.mean())
```

## When to use / when not

A plain hold-out split is fine for large datasets and quick iteration where
a single val score is enough to compare candidates cheaply. Use
K-Fold/Stratified K-Fold as the default for i.i.d. tabular data when the
dataset is small enough that a single split's variance would be misleading.
Use LOO
only on small datasets or when a cheap closed-form exists. Use time-series
split whenever rows are temporally ordered — never plain shuffled K-Fold.
Use nested CV whenever you need a defensible, non-overoptimistic estimate of
tuned-model performance (e.g. reporting results in a paper/production
readiness review); skip it (plain CV/single holdout is fine) for quick
iteration where a slightly optimistic estimate doesn't matter yet.

## Common interview questions

- Why does a single train/test split give a noisier estimate than K-Fold CV?
- Why is `shuffle=True` the default for `train_test_split`/`KFold`, and when
  would you explicitly turn it off?
- After running K-Fold CV, do you keep all `k` models, retrain one on
  everything, or average their predictions? What determines which?
- When would you use Stratified K-Fold instead of plain K-Fold?
- What goes wrong if you use `StratifiedKFold` on a regression target?
- Why can't you shuffle time-series data before cross-validating?
- What is nested CV solving that plain `GridSearchCV` + `cross_val_score`
  doesn't?
- Why is LOO nearly unbiased but high-variance?
- How many models do you train in total for `k`-fold CV combined with a grid
  search over `m` hyperparameter combinations? (`k * m`, or `k_outer *
  k_inner * m` for nested CV.)

## Common mistakes

- Fitting preprocessing (scalers, imputers, target encoders) on the full
  dataset before cross-validating — see [Data Leakage](data-leakage.md) for
  the fix (wrap everything in a `Pipeline`).
- Using `StratifiedKFold` on a continuous target.
- Shuffling time-ordered data before splitting.
- Using the same CV loop to both select hyperparameters and report the final
  performance number — optimistically biased; use nested CV or a held-out
  test set that was never touched during tuning.
- Forgetting `shuffle=True` (or a `random_state`) when it's actually needed —
  e.g. `KFold` without shuffling on data that's sorted by an unrelated
  column can produce folds that aren't representative.

## Example

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression()),
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=skf, scoring="f1_macro")
print(f"F1 (macro): {scores.mean():.3f} +/- {scores.std():.3f}")
```

See also: [Data Leakage](data-leakage.md),
[Bias-Variance Tradeoff](bias-variance-tradeoff.md),
[Hyperparameter Optimization](../hyperparameter-optimization.md).
