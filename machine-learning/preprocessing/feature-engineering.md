# Feature Engineering

## What is it?

Creating, transforming, or selecting input features to make a model's job
easier — as distinct from imputation ([Missing Values](missing-values-imputation.md))
and encoding/scaling ([Scaling & Categorical Encoding](scaling-categorical.md)),
which prepare existing features; feature engineering is about deriving *new*
signal or removing noise.

## Why?

**Domain-informed features often beat automated ones.** A model can only
combine the features you give it in the ways its architecture allows (a
linear model can't discover a ratio between two columns on its own; even a
tree needs enough data/splits to approximate one). A single well-chosen
domain feature (e.g. "debt-to-income ratio" instead of separate debt and
income columns) can outperform hundreds of automatically generated
polynomial/interaction features, because it encodes prior knowledge about
what actually drives the target, and needs far less data to be learned
reliably.

## Common transforms

- **Log / Box-Cox for skewed distributions** — a right-skewed feature (income,
  prices, counts) compresses its long tail and makes it more symmetric,
  which helps linear models (whose assumptions favor roughly normal
  residuals) and distance-based methods (a few huge values no longer
  dominate). `log1p` handles zeros; Box-Cox (`scipy.stats.boxcox`) finds an
  optimal power transform but requires strictly positive values.
- **Binning / discretization** — converting a continuous feature into
  categorical buckets (e.g. age → age groups). Useful when the
  relationship with the target is non-linear/non-monotonic and you want a
  linear model to capture it, or for interpretability (business rules often
  think in buckets). Costs: loses information, and boundary choice is
  itself a modeling decision.
- **Interaction terms** — explicit products/ratios of two features
  (`feature_a * feature_b`, or a ratio) when you suspect their *combined*
  effect matters beyond each one's individual (additive) effect. Linear
  models need these spelled out explicitly; tree-based models can
  approximate interactions natively (a split on A followed by a split on B)
  but still often benefit from an explicit interaction feature when data is
  limited.
- **Datetime decomposition** — a raw timestamp is nearly useless to most
  models directly; break it into day-of-week, hour, is-weekend,
  is-holiday, month, or cyclical encodings (`sin`/`cos` of hour-of-day so
  23:00 and 00:00 are recognized as close together, not far apart). Captures
  seasonality/periodicity patterns a raw timestamp integer can't express.

## Feature selection basics

Once you have candidate features (original + engineered), selection trims
down to what's actually useful — reduces overfitting risk, training time,
and improves interpretability.

- **Filter methods** — score each feature independently of any model
  (correlation with target, mutual information, chi-squared for
  categoricals) and keep the top-scoring ones. Fast, model-agnostic, but
  ignores feature interactions and redundancy between features.
- **Wrapper methods** — evaluate feature subsets by actually training a
  model on each candidate subset and checking validation performance.
  Recursive Feature Elimination (RFE) is the classic example: fit a model,
  drop the least important feature(s), refit, repeat. Captures interactions
  the model actually uses, but expensive (many refits) and tied to whichever
  model you wrap.
- **Embedded methods** — feature selection happens as a side effect of
  training a single model. L1-regularized linear models (Lasso) are the
  classic example — the L1 penalty drives many coefficients to exactly zero
  as part of fitting, effectively selecting features "for free" as part of
  the optimization; see [Regularization](../linear-models/regularization.md)
  for the mechanism. Tree-based feature importance (with the permutation-
  importance caveat, see [Random Forest](../trees-ensembles/random-forest.md))
  is another embedded-style signal, though it's usually used to *rank*
  rather than to hard-select.

A real-world example of wrapper-method feature selection in this repo: the
clustering project's original coursework also included from-scratch
add/delete/stepwise wrapper selection (greedily adding or removing features
based on a model-evaluation criterion) — that code wasn't ported into this
knowledge base's scope, but it's the textbook wrapper-method pattern above,
applied concretely.

## When to use / when not to

**Use** domain feature engineering whenever you (or a subject-matter expert)
understand the mechanism behind the target — it's usually the highest
return-on-effort step in a modeling project. Use automated interaction/
polynomial features cautiously and mainly for linear models; tree ensembles
get less benefit since they can approximate interactions on their own.

**Avoid** over-engineering features faster than you can validate them — every
new feature is a chance to introduce leakage (e.g. a feature computed using
future information, or using the target itself) or to blow up
dimensionality without added signal. Always re-check for leakage (see
[Data Leakage](../model-evaluation/data-leakage.md)) whenever a new feature
is derived from anything other than the raw, already-available-at-prediction-time
inputs.

## Common interview questions

- Give an example of a domain-informed feature that would help a specific
  problem (e.g. e-commerce, credit risk).
- Why would you log-transform a skewed feature before fitting a linear
  model, but not before fitting a tree-based model?
- Filter vs. wrapper vs. embedded feature selection — tradeoffs of each?
- Why encode time-of-day cyclically instead of as a raw integer 0-23?
- How can feature engineering introduce data leakage?
- When do interaction terms matter more — linear models or tree ensembles?

## Common mistakes

- Deriving a feature using information that wouldn't actually be available
  at prediction time (a common, subtle leakage source).
- Blindly generating polynomial/interaction features for a tree-based model
  that could approximate them anyway, inflating dimensionality with little
  benefit.
- Binning a continuous feature too coarsely and throwing away real signal.
- Selecting features using the full dataset (including validation/test rows)
  instead of only the training fold — the same leakage discipline as
  scaling/encoding, see [Scaling & Categorical Encoding](scaling-categorical.md).

## See also

- [Missing Values / Imputation](missing-values-imputation.md)
- [Scaling & Categorical Encoding](scaling-categorical.md)
- [Regularization](../linear-models/regularization.md)
- [Random Forest](../trees-ensembles/random-forest.md) (feature importance)
- [Data Leakage](../model-evaluation/data-leakage.md)
