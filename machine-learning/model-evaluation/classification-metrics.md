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
```

See also: [Imbalanced Data](../imbalanced-data.md),
[Cross-Validation](cross-validation.md),
[Bias-Variance Tradeoff](bias-variance-tradeoff.md).
