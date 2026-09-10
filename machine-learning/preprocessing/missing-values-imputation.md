# Missing Values: Imputation Strategies

See [`code/imputation_methods.py`](code/imputation_methods.py) (ten
strategies behind one `fill_missing()` dispatch function) and
[`code/evaluation.py`](code/evaluation.py) (a synthetic-missingness
benchmark harness) for the implementations behind this note.

## What is it?

Filling in missing values (`NaN`) in a dataset so downstream models — most of
which can't handle `NaN` natively — can be trained on it. The "right" way to
fill depends heavily on *why* the data is missing, not just what looks
statistically convenient.

## Why the missingness mechanism matters (the standard framing)

Any "how do you handle missing data" interview question expects this
three-way distinction:

- **MCAR (Missing Completely At Random)** — the fact that a value is missing
  has nothing to do with any variable, observed or not (e.g. a sensor
  randomly drops readings due to unrelated hardware noise). Dropping rows or
  using simple statistics introduces no systematic bias here — you lose
  precision, not accuracy.
- **MAR (Missing At Random)** — missingness depends on *other observed*
  variables, but not on the missing value itself (e.g. older survey
  respondents are less likely to answer an income question, but conditional
  on age, whether income is missing doesn't depend on the income value
  itself). Model-based imputation that uses the observed variables (regression
  imputation, MICE) can recover this correctly; simple mean/median imputation
  ignores the relationship and introduces bias.
- **MNAR (Missing Not At Random)** — missingness depends on the missing
  value itself (e.g. people with very high incomes are less likely to
  disclose them, or a lab test is only ordered because the doctor already
  suspects an abnormal result). No amount of imputation using only the
  observed data can fully fix this — you need external information, a
  missingness indicator as its own feature, or domain-specific modeling of
  the missingness process itself.

**Why "just drop rows" is often wrong:** `dropna()` (`method="drop_rows"`)
is only unbiased under MCAR. Under MAR/MNAR, the rows that get dropped are
systematically different from the rows that remain (e.g. dropping all rows
missing income silently removes disproportionately many high earners under
MNAR) — you don't just lose sample size, you bias every downstream
estimate. It also wastes any *other* complete columns in the dropped rows.

## The ten strategies (from `imputation_methods.py`)

| Method | What it does | Best suited for |
|---|---|---|
| `drop_rows` | Listwise deletion | MCAR only, and only when missingness is rare |
| `pairwise_deletion` | Not imputation — returns the pairwise-complete correlation matrix | When you only need correlations/covariance, not a filled dataset |
| `mean` / `median` | Fill numeric columns with a single global statistic | Quick baseline, MCAR, low missingness; median is safer under skew/outliers |
| `mode` | Fill with the most frequent value | Categorical columns, or numeric columns that are effectively discrete |
| `hot_deck` | Forward-fill then back-fill (nearest observed value along row order) | Data with meaningful row order (e.g. time-ordered/sorted records) where adjacent rows are plausibly similar |
| `group_mean` | Fill with the mean within the same group (`group_col`) | MAR where a categorical grouping variable explains the missingness/level (e.g. impute salary by department mean) |
| `ffill` | Forward-fill only | Time series where the last known value is the best available guess |
| `linear_regression` | Predict the missing column from other columns via OLS | MAR, when the target column is linearly related to other observed features |
| `stochastic_regression` | Same as above, plus injected residual noise | Same as linear regression, but when you need to preserve the target's original variance (plain regression imputation systematically shrinks variance — every imputed point lands exactly on the regression line) |
| `spline` | Fit a univariate spline over row order per numeric column | Smoothly-varying numeric sequences (e.g. sparse time series) where linear interpolation is too crude |
| `iterative` (MICE) | `IterativeImputer` — models each column as a function of all others, iterated to convergence | MAR with multiple interdependent columns; the most statistically principled option here, at the highest compute cost |

**On stochastic regression's noise injection**
(`imputation_methods.py:106-112`): plain regression imputation places every
imputed value exactly on the fitted regression line, which shrinks the
target column's variance and can distort downstream analyses that depend on
it (e.g. confidence intervals, correlations). Adding residual noise sampled
from the training residuals' spread (`np.random.normal(0, std(residuals),
...)`) restores realistic variance to the imputed values.

**On MICE/`iterative`**: this is the closest thing to a "safe default" among
the ten when relationships between columns matter and you have compute
budget to spare — it iteratively re-imputes each column conditioned on
current estimates of all the others, which handles MAR patterns across
multiple correlated columns better than any single-column method above.

## How `evaluation.py`'s benchmark works

You almost never have ground truth for real missing values, so you can't
directly measure an imputation method's error on your actual missing data.
`evaluate_imputation_methods()` works around this with a standard trick:

1. Start from a **fully-observed** subset of the data (rows/columns with no
   missing values already).
2. **Inject synthetic missingness**: randomly null out a chosen percentage of
   (row, column) cells, remembering the true values that were there.
3. **Impute** with each candidate method.
4. **Score**: mean relative error between imputed and true values on the
   masked cells, plus how far each column's summary statistics (mean, std,
   quantiles) drift from the true, un-masked distribution.
5. Repeat across several missingness percentages (3/5/10/20/30%) and several
   random runs per percentage, then report the best-scoring method at each
   level.

This is a good way to **A/B test imputation strategies before deploying
one**: rather than guessing which of the ten methods suits your dataset,
run this benchmark on your own (temporarily-complete) data to see which
method actually reconstructs values best at realistic missingness rates —
and note that the best method can change as the missingness percentage
increases (a robust method at 5% missing isn't guaranteed to stay best at
30%).

## When to use / when not to

**Use** model-based imputation (regression/MICE) when columns are genuinely
related and missingness is plausibly MAR; use simple mean/median/mode as a
fast baseline under MCAR or low missingness; use `ffill`/hot-deck/spline
specifically for ordered/time-series data.

**Avoid** any imputation method under a genuine MNAR mechanism without also
adding a "was this value missing" indicator feature — imputation alone
can't recover information the data doesn't contain.

## Common interview questions

- Explain MCAR vs. MAR vs. MNAR with an example of each.
- Why is dropping rows with missing values not always safe?
- Why does stochastic regression imputation add noise instead of using the
  regression prediction directly?
- How would you decide which imputation method to use on a new dataset?
  (→ the injected-missingness benchmark approach.)
- What's the difference between single imputation (mean/regression) and
  multiple imputation (MICE), and why does MICE tend to give more honest
  uncertainty estimates?
- When would `pairwise_deletion` be preferable to imputing a full dataset?

## Common mistakes

- Defaulting to mean/median imputation without checking whether missingness
  correlates with anything else (i.e., without at least considering MAR).
- Using plain regression imputation and not noticing it artificially
  shrinks variance in the imputed column.
- Evaluating an imputation method's quality only by eyeballing the filled
  data, instead of a systematic injected-missingness benchmark.
- Imputing before splitting into train/test (or before CV folds) — same
  leakage failure mode as fitting a scaler on the full dataset, see
  [Scaling & Categorical Encoding](scaling-categorical.md).

## Example

```python
from imputation_methods import fill_missing

filled = fill_missing(df, method="iterative")
filled = fill_missing(df, method="group_mean", group_col="department")
filled = fill_missing(df, method="stochastic_regression",
                       target_col="income", feature_cols=["age", "education_years"])
```

See [`code/evaluation.py`](code/evaluation.py) for the full benchmark harness.

## See also

- [Scaling & Categorical Encoding](scaling-categorical.md)
- [Feature Engineering](feature-engineering.md)
