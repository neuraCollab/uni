# Probability Calibration

## What is it?

A classifier's predicted probabilities are **calibrated** when they mean
what they say: among all the objects the model assigns a score of 0.67,
roughly 67% of them should actually be positive.

$$P(y_i = 1 \mid q(x_i) = \hat{p}) = \hat{p}$$

where $q(x)$ is the model's output score. The left side is the *true*,
empirical fraction of positives among objects the model assigned score
$\hat{p}$; calibration means that empirical fraction should equal $\hat{p}$
itself.

**Concrete example:** if the model assigns $\hat{p} = 0.67$ to a thousand
different objects, roughly 67% of that thousand should truly be class 1. If,
on the real data, only 30% of those objects turn out to be class 1, the
model's probabilities are badly miscalibrated — even if its *ranking* of
those same objects is fine.

## Why does it matter?

A model can have excellent ranking quality (high AUC-ROC — see
[Classification Metrics](classification-metrics.md)) while producing
useless probabilities, because AUC-ROC only cares about relative order, not
the actual numeric values — it's a rank statistic, invariant to any
monotonic transform of the scores (see the pairwise definition of AUC-ROC
in [Classification Metrics](classification-metrics.md)). Calibration
matters whenever you need the *actual number*, not just the ranking:

- **Cost-sensitive decision-making** — e.g. multiplying $P(\text{fraud})$ by a
  dollar cost matrix to decide whether to block a transaction; the decision
  threshold comes from real costs, not an arbitrary rank cutoff.
- **Combining or ensembling probabilities from multiple models** —
  averaging or otherwise blending scores only makes sense if each model's
  number means the same thing on the same scale.
- **Showing probabilities to end users** — a "70% chance of rain," or a
  risk score shown to a doctor or loan officer, is only trustworthy if it's
  calibrated.
- Gradient-boosted trees and SVMs in particular tend to produce poorly
  calibrated scores by default (systematically over- or under-confident)
  even when they rank very well — their training objectives aren't a proper
  probabilistic (MLE) fit the way logistic regression's log-loss is (see
  [Classification Metrics: Log-loss](classification-metrics.md)).

**When calibration does *not* matter much:** pure ranking/retrieval tasks
(top-k recommendation, search relevance, or any setting where only the
relative order of scores is ever consumed) — and any evaluation that is
itself rank-based, like AUC-ROC.

## How do you fix miscalibration?

### Histogram (binning) calibration

1. Split $[0, 1]$ into $k$ bins — either **equal-width** or
   **equal-frequency** (equal-mass) bins. On each bin $B_j$, predict a
   single constant probability $\theta_j$ for every object whose raw score
   $q(x_i)$ falls in $B_j$.
2. Choose the $\theta_j$ to best match the true average label within each
   bin — solve:

$$\sum_{j=1}^{k} \left| \frac{\sum_{i=1}^{N} \mathbb{1}[q(x_i) \in B_j] \, y_i}{|B_j|} - \theta_j \right| \to \min_{\theta_1, \ldots, \theta_k}$$

In words: $\theta_j$ is set to (as close as possible to) the empirical
fraction of positives among the calibration-set objects that landed in bin
$B_j$.

### Isotonic regression calibration

Similar idea, but the bin **boundaries are also learned**, and the sequence
of bin levels is constrained to be non-decreasing (isotonic = monotonic).
Boundaries $0 = b_0 \leq b_1 \leq \ldots \leq b_k = 1$ define bins
$B_j = \{ t : b_{j-1} \leq t < b_j \}$, with $\theta_1 \leq \theta_2 \leq \ldots \leq \theta_k$.
Both the boundaries $b_j$ and the levels $\theta_j$ are fit by approximating
$y_i$ with a piecewise-constant, monotonic function $g$ of $q(x_i)$:

$$\sum_i (y_i - g(q(x_i)))^2 \to \min_{g} \quad (g \text{ piecewise-constant, monotonic})$$

This is strictly more flexible than fixed-width/fixed-frequency histogram
binning — it adapts bin widths to wherever the data actually needs
correcting — at the cost of more free parameters, and therefore more risk
of overfitting on a small calibration set.

### Platt scaling (and beta calibration)

A simpler, parametric alternative: fit a single **sigmoid** to map raw
scores to calibrated probabilities
($p_{\text{calibrated}} = \text{sigmoid}(A \cdot \text{score} + B)$, with $A, B$ fit by logistic
regression on held-out data). More restrictive than isotonic regression (it
assumes the miscalibration curve itself has a sigmoid shape), but far less
prone to overfitting when calibration data is scarce. A more general,
nonparametric alternative sitting between the two is **beta calibration**.
In practice: `sklearn.calibration.CalibratedClassifierCV(method='sigmoid' | 'isotonic')`
— always fit the calibrator on a held-out split, never on the same data the
base model was trained on, or you'll calibrate to the model's already
overfit training scores.

## How do you measure calibration quality?

### Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)

Bin $[0, 1]$ by **predicted** probability (same idea as histogram
calibration above), then compare, per bin, the empirical fraction of
positives $\bar{y}(B_j)$ against the average predicted probability
$\bar{q}(B_j)$:

$$ECE = \sum_{j=1}^{k} \frac{|B_j|}{N} \left| \bar{y}(B_j) - \bar{q}(B_j) \right|$$

$$MCE = \max_{j=1,\ldots,k} \left| \bar{y}(B_j) - \bar{q}(B_j) \right|$$

ECE is the size-weighted average gap, across bins, between "what the model
said" and "what actually happened"; MCE is the single worst-bin gap. Both
flag **systematic over- or under-confidence** — a large gap in a specific
probability range means the model consistently lies about its confidence
there (e.g. everything it calls "90% confident" is only right 70% of the
time).

### Brier Score

$$\text{BrierScore} = \frac{1}{N} \sum_{i=1}^{N} (y_i - q(x_i))^2$$

This is literally **[MSE](regression-metrics.md) applied to probabilities
instead of a continuous target** — treat $y_i \in \{0, 1\}$ as the "true
value" and $q(x_i)$ as the "prediction." Lower is better; $0$ is perfect.
Brier Score is generally **preferred over ECE/MCE** as a single headline
number, because it's a smooth, strictly proper scoring rule (it's minimized
exactly when the predicted probabilities equal the true probabilities)
rather than a binned approximation sensitive to bin choice — but ECE/MCE
remain more *diagnostic*, since they show *where* (in which probability
range) the miscalibration actually lives.

## When to use / when not

Check and fix calibration whenever probabilities feed a cost matrix, get
combined across models, or get shown to a human. Skip it when only the
ranking is ever consumed — pure retrieval/top-k recommendation, or whenever
the downstream evaluation is itself rank-based (AUC-ROC).

## Common interview questions

- What does it mean for a classifier to be "calibrated"? Give a concrete
  example of a model that's miscalibrated despite high AUC.
- Why can a model have AUC-ROC = 1 and still be poorly calibrated?
- Histogram calibration vs. isotonic regression — what's the practical
  tradeoff?
- Why must the calibration set be held out from the model's training data?
- Brier Score vs. ECE/MCE — when would you reach for each?
- Which model families tend to need calibration the most, and why?

## Common mistakes

- Calibrating on the same data the model was trained on (overfits the
  calibrator to already-overfit training scores).
- Treating a high-AUC model as automatically trustworthy for
  probability-dependent decisions (cost matrices, thresholds derived from
  real-world costs).
- Using too few bins for ECE/MCE (hides miscalibration) or too many
  relative to the sample size (each bin becomes noisy and the metric
  becomes unstable).
- Defaulting to isotonic regression on a small calibration set — it has
  more free parameters than Platt scaling and overfits more easily there.

## Example

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

calibrated_model = CalibratedClassifierCV(estimator=model, method="isotonic", cv=5)
calibrated_model.fit(X_train, y_train)

p = calibrated_model.predict_proba(X_val)[:, 1]
print("Brier score:", brier_score_loss(y_val, p))

true_frac, pred_frac = calibration_curve(y_val, p, n_bins=10)  # reliability diagram
```

## Related notes

- [Classification Metrics](classification-metrics.md) — log-loss,
  ROC-AUC/PR-AUC, and why a rank-based metric can hide miscalibration.
- [Regression Metrics](regression-metrics.md) — Brier Score is MSE applied
  to probability predictions.
