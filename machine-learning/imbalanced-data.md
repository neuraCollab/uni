# Imbalanced Data

## What is it?

A classification problem where one class vastly outnumbers another (e.g.
fraud detection: 0.1% positive, 99.9% negative). Standard training and
evaluation both quietly assume roughly balanced classes, and both break down
in specific, predictable ways when that assumption fails.

## Why accuracy misleads

A model that always predicts the majority class on a 99.9%/0.1% dataset
scores **99.9% accuracy** while catching zero positives — the metric looks
excellent while the model is useless for the actual task (catching the rare,
usually-more-important class). Accuracy weighs every point equally, so on
imbalanced data it's dominated by how well you classify the class you
already didn't need help with.

## Choosing the right metric

Since accuracy is unreliable here, pick a metric that actually reflects
performance on the minority class — see
[Classification Metrics](model-evaluation/classification-metrics.md) for the
full precision/recall/F1/ROC toolkit; the imbalance-specific guidance:

- **PR-AUC over ROC-AUC when the positive class is rare.** ROC-AUC's false
  positive rate is computed against the (huge) negative class, so it stays
  visually flattering even when the model produces many false positives in
  absolute terms — a small FPR is still a lot of actual false positives when
  negatives vastly outnumber positives. Precision-Recall AUC instead tracks
  precision directly, which degrades visibly as false positives pile up
  relative to true positives, making it far more sensitive to
  minority-class performance.
- **F1 score** — harmonic mean of precision and recall, a reasonable single
  number when you care about both roughly equally and want to avoid the
  accuracy trap.
- **Recall at a fixed precision (or vice versa)** — often the actual business
  question is "how many fraud cases do we catch, if we require ≥90%
  precision so we're not flooding a review team with false alarms" — report
  the metric that matches the real operating constraint, not just AUC in the
  abstract.

## Resampling

**SMOTE (Synthetic Minority Oversampling TEchnique)** — generates new
synthetic minority-class points by interpolating between existing minority
points and their nearest minority neighbors (conceptually related to kNN —
see [kNN](knn.md)), rather than just duplicating existing rows. **Risk**:
interpolating near the class boundary can create synthetic points that land
inside majority-class territory (noisy/unrealistic points that don't
represent a real minority pattern), especially when minority points are
sparse or the boundary is complex — this can make the decision boundary
worse, not better, if applied carelessly. Always evaluate on real
(non-resampled) held-out data, and apply SMOTE only inside the training fold
— resampling before a train/test split leaks synthetic near-duplicates of
test points into training.

**Random oversampling** — duplicate existing minority rows. Simple, no risk
of inventing unrealistic points, but can cause overfitting to the exact
duplicated points (the model can effectively memorize them).

**Random undersampling** — drop majority-class rows down to match the
minority count. Cheap and fast, but throws away potentially useful majority
data — risky when the majority class itself has diverse
sub-patterns worth keeping.

## Class weighting

Instead of changing the data, change the **loss function** to penalize
minority-class errors more — `class_weight="balanced"` in scikit-learn
(automatically weights each class inversely proportional to its frequency),
or manually specified weights. Achieves a similar effect to resampling
without duplicating/synthesizing data or discarding majority samples, and is
usually the simplest first thing to try — most sklearn classifiers, plus
LightGBM/CatBoost/XGBoost, support this directly.

## Threshold tuning (post-hoc, no retraining needed)

A classifier's default 0.5 probability threshold is arbitrary — it's not
calibrated to your actual cost tradeoff between false positives and false
negatives. After training (on any classifier, imbalanced or not), you can
move the decision threshold to trade precision for recall without touching
the model at all: lower the threshold to catch more positives (higher
recall, lower precision) or raise it to reduce false alarms (higher
precision, lower recall). Pick the threshold using the PR curve against your
actual operating requirement (e.g. "the highest recall achievable at ≥90%
precision"). This is often the cheapest, lowest-risk lever to pull —
resampling and reweighting both change what the model learns; threshold
tuning just changes how you read out a prediction it already made.

## When to use which

- Start with **class weighting** — free, no data manipulation, works with
  most model APIs.
- Add **threshold tuning** always — it's nearly free and should be done
  regardless of what else you try.
- Reach for **SMOTE/resampling** when class weighting alone isn't enough, or
  for algorithms without a native class-weight option — but validate
  carefully, since synthetic points can hurt if the minority class is very
  sparse or the boundary is complex.
- Reach for **undersampling** mainly when the majority class is so large
  that training time/cost is itself a constraint, and you can afford to
  discard majority data.

## Common interview questions

- Why is accuracy a poor metric for imbalanced classification? Give a
  concrete example.
- Why is PR-AUC often preferred over ROC-AUC for rare-positive-class
  problems?
- How does SMOTE generate synthetic samples, and what can go wrong?
- What's the difference between class weighting and resampling, and when
  would you pick one over the other?
- How would you pick a decision threshold in a fraud-detection setting?
- Why must resampling (SMOTE included) be applied only inside the training
  fold, never before a train/test split?

## Common mistakes

- Reporting accuracy as the headline metric on an imbalanced problem.
- Applying SMOTE (or any resampling) before splitting into train/test —
  leaks synthetic near-duplicates of test-set points into training,
  inflating validation metrics.
- Using ROC-AUC as the only metric and being surprised the model floods
  production with false positives despite a "good" AUC.
- Never touching the decision threshold and leaving it at the default 0.5
  even when the real-world cost of false positives vs. false negatives is
  highly asymmetric.
- Oversampling/undersampling on the full dataset once, then cross-validating
  — the same leakage-across-folds mistake as fitting a scaler globally, see
  [Scaling & Categorical Encoding](preprocessing/scaling-categorical.md).

## Example

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score

# Cheapest first lever: class weighting, no data changes.
clf = LogisticRegression(class_weight="balanced")
clf.fit(X_train, y_train)

probs = clf.predict_proba(X_test)[:, 1]
print("PR-AUC:", average_precision_score(y_test, probs))

# Threshold tuning: pick the threshold hitting a target precision.
precision, recall, thresholds = precision_recall_curve(y_test, probs)
target_precision = 0.9
idx = next(i for i, p in enumerate(precision) if p >= target_precision)
chosen_threshold = thresholds[idx]
```

## See also

- [Classification Metrics](model-evaluation/classification-metrics.md)
- [kNN](knn.md)
- [Random Forest](trees-ensembles/random-forest.md) /
  [Gradient Boosting](trees-ensembles/gradient-boosting-catboost-lgbm.md) —
  both support `class_weight`/similar parameters directly.
