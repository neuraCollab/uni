# Classification Metrics

## What is it?

The set of metrics used to judge a classifier's predictions against ground
truth: accuracy, precision, recall, F1, the confusion matrix, and
threshold-independent ranking metrics (ROC-AUC, PR-AUC).

## Why?

Accuracy alone is easy to compute but easy to misread — a single number
can't tell you *what kind* of mistakes a model makes, and it can be
dangerously misleading on imbalanced data (see below). Precision/recall/F1
break performance down per class and per error type.

## How does it work?

### Confusion matrix

For binary classification (positive class = the class of interest):

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

### Precision

*Of everything the model called positive, how much actually was?*

```
Precision = TP / (TP + FP)
```

High precision → few false alarms. Matters when a false positive is costly
(e.g. flagging a legitimate transaction as fraud, or a healthy patient as
sick).

### Recall (a.k.a. sensitivity, true positive rate)

*Of everything that actually was positive, how much did the model catch?*

```
Recall = TP / (TP + FN)
```

High recall → few missed positives. Matters when a false negative is costly
(e.g. missing an actual fraud case, or an actual disease).

Precision and recall trade off against each other as you move the
classification threshold — this is exactly what the ROC and PR curves trace
out (see below).

### F1-score

The harmonic mean of precision and recall — penalizes a large gap between
the two more than a simple average would:

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

Use F1 when you want a single number balancing both error types and don't
have a strong reason to weight one more than the other. `F_beta` generalizes
this to weight recall `beta` times as much as precision.

### Support

Simply the number of true instances of each class in the evaluation set —
not a "score," just the denominator context needed to judge whether a
per-class metric is based on 5 examples or 5,000.

### Worked example

```
              precision    recall  f1-score   support

           0       0.96      0.98      0.97       108
           1       0.98      0.96      0.97        92

    accuracy                           0.97       200
   macro avg       0.97      0.97      0.97       200
weighted avg       0.97      0.97      0.97       200
```

- Class 0: of everything predicted class 0, 96% actually was (precision);
  of everything actually class 0, 98% was caught (recall).
- Class 1: 98% precision, 96% recall — the mirror image, as is typical in
  binary classification (raising one class's threshold-implied recall tends
  to lower the other class's precision).
- Overall accuracy: `(TP_0 + TP_1) / 200 = 0.97`.

### Macro vs. weighted vs. micro averaging

When you have more than one class (or want one number from a binary
report), how you combine per-class metrics matters:

- **Macro average** — unweighted mean across classes. Every class counts
  equally regardless of how many examples it has. Best when you care about
  performance on *rare* classes just as much as common ones (macro F1 will
  visibly tank if the model ignores a minority class, even if that class is
  a small fraction of the data).
- **Weighted average** — mean weighted by each class's support. Dominated by
  whichever class has the most examples; can look good even if a minority
  class is doing poorly, because that class barely moves the number.
- **Micro average** — pool all TP/FP/FN across classes first, *then* compute
  precision/recall/F1 once. For multiclass single-label classification,
  micro-averaged precision = recall = F1 = accuracy. More meaningful in
  multi-label settings or when you want to weight by *individual
  predictions* rather than by class.

**Which to report:** macro if minority-class performance matters (fraud,
disease detection, rare defect classes); weighted if you just want an
overall picture proportional to real-world class frequency; micro when doing
multi-label classification.

**Formal definitions** (`K` = number of classes): for **micro-averaging**,
first average (equivalently, sum) the per-class confusion-matrix counts —
`TP_micro = (1/K) * sum_k TP_k`, and likewise for `FP`/`FN`/`TN` — then
compute precision/recall/F1 *once* from those pooled counts. For
**macro-averaging**, compute precision/recall/F1 separately for each class
first, then average the `K` resulting scores. The two differ exactly in
*when* the averaging happens — before or after the ratio is computed — which
is why macro tanks on a poorly-served minority class while micro barely
notices it (a small class contributes only a small slice of the pooled
`TP`/`FP`/`FN` totals).

### When accuracy is misleading

On an imbalanced dataset, a model that always predicts the majority class
gets high accuracy while being useless. If 99% of transactions are
legitimate, predicting "legitimate" for everything scores 99% accuracy while
catching zero fraud. See [Imbalanced Data](../imbalanced-data.md) for
resampling/weighting strategies to address this at the modeling level;
metrics-wise, the fix is to look at recall/precision/F1 (and ROC/PR curves)
per class instead of relying on the single accuracy number.

### ROC-AUC vs. PR-AUC

Both summarize a classifier's ranking quality *across all thresholds*
(rather than at one fixed threshold like precision/recall/F1 above).

- **ROC curve**: true positive rate (recall) vs. false positive rate
  (`FP / (FP + TN)`) as the threshold varies. **AUC** = probability a
  randomly chosen positive is ranked above a randomly chosen negative.
- **PR curve**: precision vs. recall as the threshold varies. **PR-AUC**
  (a.k.a. average precision) summarizes that curve.

**Prefer PR-AUC over ROC-AUC on imbalanced data.** ROC's false-positive-rate
term is normalized by the (huge) number of true negatives, so ROC-AUC can
look deceptively good even when precision is terrible — a fixed number of
false positives barely moves FPR when true negatives are abundant, but can
devastate precision when true positives are scarce. PR-AUC, built from
precision directly, is far more sensitive to exactly this failure mode and
is the standard recommendation for rare-positive-class problems (fraud,
disease screening, anomaly detection).

A related reason PR-AUC is preferred under imbalance: a random classifier's
**PR-AUC baseline equals the fraction of positives in the dataset** (e.g.
0.01 if 1% of objects are positive), which visibly reflects how hard the
problem is. A random classifier's **ROC-AUC baseline is always 0.5**,
regardless of class balance — so 0.5 tells you nothing about how skewed the
data is, while a low PR-AUC baseline is itself informative.

#### AUC-ROC as a pairwise ranking probability

There's a more precise, threshold-free definition than "the area under the
ROC curve": **AUC-ROC equals the fraction of (positive, negative) pairs that
the model ranks correctly** — equivalently, the probability that a randomly
chosen positive example receives a higher score than a randomly chosen
negative example:

```
AUC = sum_i sum_j I[y_i < y_j] * I'[a_i < a_j]
      -----------------------------------------
             sum_i sum_j I[y_i < y_j]

I[y_i < y_j]  = 1 if y_i < y_j, else 0       # y in {0, 1}; counts (negative, positive) label pairs
I'[a_i < a_j] = 1    if a_i < a_j
              = 0.5  if a_i == a_j
              = 0    if a_i > a_j
```

where `a_i` is the model's score on object `i`, `y_i` its true label, and
`q` the number of test objects. This is exactly the statistic behind the
Mann-Whitney U test — AUC-ROC is a **rank statistic**, which is why it is
invariant to any monotonic transformation of the scores (in particular,
calibrating a model — see [Probability Calibration](calibration.md) — never
changes its AUC-ROC, since calibration only reshapes the score axis, not the
ranking).

**Gini coefficient**, sometimes reported alongside AUC-ROC (common in credit
scoring):

```
Gini = 2 * AUC_ROC - 1
```

#### Computing AUC from discrete points: the trapezoidal rule

In practice a ROC or PR curve is a finite set of points (one per distinct
threshold), not a continuous function, so "area under the curve" is computed
by summing the trapezoids between consecutive points. For two adjacent
points `(r_{k-1}, p_{k-1})` and `(r_k, p_k)` on a precision-recall curve, the
line segment between them is:

```
p(r) = p_{k-1} + (p_k - p_{k-1}) / (r_k - r_{k-1}) * (r - r_{k-1})
```

Integrating that line and summing over all `m` segments gives the
trapezoidal-rule AUC:

```
AUC = integral_0^1 p(r) dr  ~=  sum_{k=1}^m (p_{k-1} + p_k) / 2 * (r_k - r_{k-1})
```

The same construction applies to the ROC curve (swap precision/recall for
TPR/FPR) — it's just "area of a trapezoid," repeated for every pair of
adjacent threshold points, which is exactly what
`sklearn.metrics.roc_auc_score` / `auc()` compute under the hood.

#### Average Precision: exact formula

**Average Precision (AP)** approximates the same integral for the PR curve,
but is built directly from the step function traced out as the
classification threshold is lowered one prediction at a time — recall only
increases as the threshold drops (`TP` grows), while precision moves
non-monotonically:

```
AP = integral_0^1 p(r) dr  ~=  sum_{k=1}^m p_k * (r_k - r_{k-1})
```

which is equivalent to averaging precision at the rank position of each true
positive:

```
AP = (1/P) * sum_{i=1}^P Precision@k_i
```

where `P` is the total number of positive examples and `Precision@k_i` is
precision computed at the rank of the `i`-th positive example, once
predictions are sorted by descending score. This rank-based form is how
`sklearn.metrics.average_precision_score` actually computes AP, and it's why
"AP" and "PR-AUC" are often used interchangeably even though AP is a
slightly different (unweighted-by-width, left-Riemann-sum-like)
approximation than the trapezoidal rule above.

### Log-loss (cross-entropy) and probability quality

*How good are the predicted **probabilities**, not just the final class
label?* Log-loss (a.k.a. binary cross-entropy) scores probabilistic
predictions directly, and is what logistic regression (and most neural-net
classifiers) actually optimizes during training:

```
LogLoss = -(1/N) * sum_i [ y_i * log(y_hat_i) + (1 - y_i) * log(1 - y_hat_i) ]
```

where `y_hat_i = sigma(z_i)`, `z_i = w^T x_i + b`, and the sigmoid function
and its derivative are:

```
sigma(z)  = 1 / (1 + e^(-z))
sigma'(z) = sigma(z) * (1 - sigma(z))
```

**Why log-loss, specifically, and not some other penalty for wrong
probabilities?** Model each label `y_i` as a Bernoulli random variable with
success probability `y_hat_i`. The likelihood of the observed labels is
`prod_i y_hat_i^(y_i) * (1 - y_hat_i)^(1 - y_i)`. Taking the negative log
(to turn the product into a sum, and maximization into minimization) gives
exactly the log-loss formula above — so **minimizing log-loss is maximum
likelihood estimation (MLE)** for a Bernoulli/sigmoid model. See
[MLE and Loss Functions](../probabilistic-ml/mle-and-loss-functions.md) for
the general MLE-to-loss-function derivation (squared error falls out of a
Gaussian likelihood the same way log-loss falls out of a Bernoulli one).

The clean derivative `sigma'(z) = sigma(z)(1 - sigma(z))` is why logistic
regression's gradient has such a simple closed form
(`gradient = X^T (y_hat - y)`), and it's the same identity reused in
backprop through any sigmoid output layer.

**Unlike accuracy/precision/recall/F1 or even AUC-ROC**, log-loss is
sensitive to *how confident* a wrong prediction was — predicting 0.99 for
the wrong class is punished far more than predicting 0.51. This makes
log-loss a **calibration-sensitive** metric: a model can have perfect
ranking (AUC-ROC = 1) and still post a poor log-loss if its predicted
probabilities are miscalibrated. See [Probability Calibration](calibration.md)
for how to measure and fix that directly.

## When to use / when not

Use precision when false positives are the expensive error; recall when
false negatives are the expensive error; F1/PR-AUC as a balanced view,
especially under class imbalance; ROC-AUC when classes are roughly balanced
and you care about ranking quality independent of the operating threshold.

## Common interview questions

- Precision vs. recall — define both, give a real-world example where you'd
  optimize for each.
- Why is accuracy misleading on imbalanced data? Give a concrete example.
- What does F1 do that a simple average of precision and recall doesn't?
  (Harmonic mean punishes a large imbalance between the two more heavily.)
- Macro vs. weighted vs. micro average — when would each give very different
  numbers?
- Why prefer PR-AUC over ROC-AUC for imbalanced classification?
- What's on each axis of an ROC curve? Of a PR curve?
- What does an AUC of 0.5 mean? Of 1.0?
- How do you choose a classification threshold if the default 0.5 isn't
  appropriate? (Pick the point on the PR/ROC curve matching your
  cost-of-error tradeoff, e.g. maximize F1, or fix recall at a business
  requirement and read off precision.)
- Give the pairwise/probabilistic definition of AUC-ROC. Why is it invariant
  to monotonic transformations of the model's scores?
- Why is a random classifier's PR-AUC baseline equal to the positive-class
  fraction, while its ROC-AUC baseline is always 0.5?
- Derive log-loss from the Bernoulli likelihood. Why is log-loss described
  as "MLE for a sigmoid model"?
- Can a model have AUC-ROC = 1 but bad log-loss? What does that tell you
  about the model?

## Common mistakes

- Reporting only accuracy on an imbalanced dataset.
- Using ROC-AUC as the headline metric for a rare-positive-class problem
  instead of PR-AUC.
- Averaging precision/recall/F1 with the wrong scheme (macro when weighted
  was intended, or vice versa) and drawing the wrong conclusion about
  minority-class performance.
- Computing metrics on the training set instead of a held-out
  set — see [Cross-Validation](cross-validation.md) and
  [Data Leakage](data-leakage.md).
- Picking a decision threshold by eyeballing accuracy instead of the metric
  that actually matches the business cost of each error type.

## Example

```python
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)[:, 1]

print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))  # precision/recall/F1/support + macro/weighted avg

print("ROC-AUC:", roc_auc_score(y_val, y_proba))
print("PR-AUC (average precision):", average_precision_score(y_val, y_proba))

from sklearn.metrics import log_loss
print("Log-loss:", log_loss(y_val, y_proba))
```

See also: [Imbalanced Data](../imbalanced-data.md),
[Cross-Validation](cross-validation.md),
[Bias-Variance Tradeoff](bias-variance-tradeoff.md),
[Regression Metrics](regression-metrics.md) — the continuous-target
analogue of everything above,
[Probability Calibration](calibration.md) — for when the actual predicted
probability value matters, not just ranking,
[MLE and Loss Functions](../probabilistic-ml/mle-and-loss-functions.md) —
why log-loss is the "correct" loss for a probabilistic classifier.
