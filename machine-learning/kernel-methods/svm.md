# Support Vector Machines (SVM)

## What is it?

A supervised classifier (and, via SVR, a regressor) that finds the
**maximum-margin hyperplane** separating classes: the decision boundary that
sits as far as possible from the nearest points of each class. Those nearest
points — the ones that actually determine where the boundary sits — are the
**support vectors**. Every other training point could be deleted without
changing the fitted boundary at all.

## Why?

Many hyperplanes can perfectly separate two linearly-separable classes.
Maximizing the margin (rather than picking any separating line) is a form of
built-in regularization: a wider margin generalizes better to unseen data,
because the boundary isn't hugging any particular training point. This idea —
optimize for the worst-case gap, not just zero training error — is the core
SVM contribution.

## How does it work?

**Hard margin (separable case).** Find `w`, `b` maximizing the margin
`2/‖w‖` subject to every point being correctly classified with at least unit
margin: `y_i (w·x_i + b) ≥ 1` for all `i`. Equivalent to minimizing `‖w‖²`
under those constraints — a convex quadratic program.

**Soft margin (the realistic case) + hinge loss.** Real data usually isn't
perfectly separable, so SVM allows violations via slack variables `ξ_i ≥ 0`
and minimizes:

```
L = (1/2)‖w‖² + C · Σ ξ_i
  = (1/2)‖w‖² + C · Σ max(0, 1 - y_i(w·x_i + b))
```

The second form is the **hinge loss**: zero once a point is correctly
classified *and* outside the margin, and growing linearly for points that are
misclassified or sit inside the margin. Points with zero hinge loss and
margin > 1 don't affect the solution at all — only margin-violating and
boundary points (the support vectors) do.

**The `C` parameter — margin width vs. misclassification tradeoff:**
- **Small `C`** → the penalty for margin violations is cheap → the optimizer
  prioritizes a **wide margin**, tolerating more misclassified/inside-margin
  points. More regularization, more bias, less variance. Underfits if too small.
- **Large `C`** → violations are expensive → the optimizer prioritizes
  **classifying every point correctly**, accepting a narrower margin. Less
  regularization, more variance, can overfit noisy/overlapping data.
- `C → ∞` approaches the hard-margin SVM (if separable).

This is the same knob as `alpha` in Ridge/Lasso, just inverted: see
[Regularization](../linear-models/regularization.md) for the
`alpha = 1/C`-style correspondence.

**The kernel trick.** SVM's optimization only ever needs **dot products**
between data points (`x_i · x_j`), never the raw feature vectors themselves.
That means you can replace the dot product with a **kernel function**
`K(x_i, x_j) = φ(x_i)·φ(x_j)` that computes the inner product *as if* the data
had been mapped into some higher-dimensional space `φ(x)`, without ever
computing `φ(x)` explicitly. This lets a linear-boundary algorithm produce
nonlinear decision boundaries in the original feature space, cheaply.

Common kernels:
- **Linear**: `K(x_i, x_j) = x_i · x_j`. No transformation; use when classes
  are (roughly) linearly separable or `n_features` is already large relative
  to `n_samples` (e.g. text/TF-IDF — see
  [TF-IDF](../text-features-tfidf.md)).
- **Polynomial**: `K(x_i, x_j) = (γ x_i·x_j + r)^d`. Captures feature
  interactions up to degree `d`.
- **RBF / Gaussian**: `K(x_i, x_j) = exp(-γ‖x_i - x_j‖²)`. Maps into an
  infinite-dimensional space; the default go-to nonlinear kernel. `γ`
  controls how far a single training point's influence reaches — large `γ`
  → tight, wiggly boundaries (overfitting risk), small `γ` → smoother,
  near-linear boundaries.

`C` and the kernel's own hyperparameters (`γ`, `degree`) should always be
tuned together via cross-validation (see
[Cross-Validation](../model-evaluation/cross-validation.md)) — they trade off
against each other.

**Multiclass strategies.** SVM is inherently binary; scikit-learn extends it via:
- **One-vs-Rest (OvR)**: train `K` binary classifiers, each "class `k` vs.
  everyone else"; predict the class whose classifier scores highest.
  `K` classifiers total, each trained on the full dataset.
- **One-vs-One (OvO)**: train a binary classifier for every pair of classes
  (`K(K-1)/2` classifiers); predict by majority vote. `SVC` uses OvO by
  default because it scales better than OvR for the underlying QP solver
  (each pairwise problem only sees the two relevant classes' data), even
  though it fits more models.

## When to use / when not

**Use when:**
- High-dimensional feature space, especially `n_features` comparable to or
  larger than `n_samples` (text classification, bioinformatics).
- Classes have a clear margin of separation.
- Small-to-medium dataset size (thousands, not millions of rows).
- You need a strong nonlinear model without hand-engineering features (via
  RBF/poly kernels).

**Avoid when:**
- Very large datasets — training is `O(n²)` to `O(n³)` in the number of
  samples for kernelized SVM, which becomes impractical past ~100k rows
  (`LinearSVC`, which uses a different liblinear solver, scales much better
  for the linear-kernel case specifically).
- Classes heavily overlap / very noisy labels — the margin concept breaks
  down and you'll spend all your tuning budget fighting `C`.
- You need calibrated probability estimates out of the box (`SVC`'s
  `predict_proba` requires an extra internal 5-fold CV + Platt scaling,
  `probability=True`, which is slow and only approximate).
- You need built-in interpretability — unlike a single decision tree, kernel
  SVM decision boundaries aren't easily human-readable (see
  [Decision Trees](../trees-ensembles/decision-trees.md) for the alternative).

## Common interview questions

- What are support vectors, and why can non-support-vector points be removed
  without changing the model?
- Derive/explain the hinge loss and how it differs from logistic loss.
- What does `C` control, and what happens as `C → 0` / `C → ∞`?
- Explain the kernel trick — why does SVM only need dot products?
- Why is RBF's feature space infinite-dimensional?
- Compare `C` and `γ` for RBF-SVM: what does each control, and how do they
  interact? (`γ` too large + `C` too large is a classic overfitting combo.)
- One-vs-Rest vs. One-vs-One — tradeoffs in number of models trained vs.
  training-set size per model?
- Why does SVM need feature scaling? (Margin/distance-based — like KNN,
  unscaled features with larger numeric ranges dominate the distance
  computation.)
- SVM vs. logistic regression — when would you pick one over the other?
  (Logistic regression gives calibrated probabilities and scales better;
  SVM with RBF gives a stronger nonlinear boundary out of the box but no
  native probabilities and worse scaling.)

## Common mistakes

- Forgetting to scale features (`StandardScaler`) before fitting — SVM is a
  distance/margin-based method, so unscaled features distort the margin.
- Using `SVC` (kernelized) on a very large dataset and being surprised
  training doesn't finish — use `LinearSVC` or `SGDClassifier(loss="hinge")`
  for the linear case at scale.
- Treating `predict_proba` from `SVC(probability=True)` as a first-class
  probability estimate — it's a post-hoc calibration, can be inconsistent
  with `predict`'s hard decision, and is expensive to compute.
- Tuning `C` alone and leaving `γ` at its default for RBF kernels — the two
  need to be searched jointly (e.g. a grid/log-scale search over both).
- Assuming a bigger margin always means a better model regardless of `C` —
  too wide a margin (too small `C`) underfits just as surely as too narrow a
  margin overfits.

## Example

```python
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))

param_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": [0.001, 0.01, 0.1, 1],
}
search = GridSearchCV(pipe, param_grid, cv=5, scoring="f1_macro")
search.fit(X_train, y_train)
print(search.best_params_)
```

See also: [Kernel Ridge Regression](kernel-ridge-regression.md) (the kernel
trick applied to Ridge regression instead of max-margin classification) and
[Classification Metrics](../model-evaluation/classification-metrics.md) for
evaluating the resulting classifier.
