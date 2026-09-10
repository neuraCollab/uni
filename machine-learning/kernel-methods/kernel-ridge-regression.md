# Kernel Ridge Regression (KRR)

## What is it?

Ordinary [Ridge regression](../linear-models/regularization.md) with the
**kernel trick** applied, so it can learn nonlinear relationships while
keeping Ridge's closed-form solution and squared-loss objective.

## The kernel trick, generally

Many linear methods (Ridge, SVM, PCA) can be rewritten so that the data only
ever appears inside **dot products** $x_i \cdot x_j$. If that's true, you can
substitute a **kernel function** $K(x_i, x_j) = \phi(x_i) \cdot \phi(x_j)$ that
computes the dot product *as if* both points had first been mapped into some
(possibly much higher-dimensional, even infinite-dimensional) feature space
$\phi(\cdot)$ — without ever materializing $\phi(x)$ itself. This is often far cheaper
than the explicit transform-then-dot-product would be, and it's what lets
linear algorithms fit fundamentally nonlinear functions of the original
features. Common kernels: linear, polynomial, RBF/Gaussian (see
[SVM](svm.md) for the formulas — the same kernel menu applies here).

## Why Ridge + kernel trick?

Plain Ridge minimizes $\|y - Xw\|^2 + \alpha\|w\|^2$, which only fits functions
linear in the original features. Rewriting the Ridge solution in its **dual
form** expresses the prediction for a new point $x$ entirely in terms of dot
products between $x$ and the training points:

$$f(x) = \sum_i c_i (x_i \cdot x)$$

(using $c$ for the dual coefficients here, to keep $\alpha$ free for the
regularization strength below — some sources use $\alpha$ for both, which
is where the confusion in "KRR vs. SVM regularization" comparisons often
comes from.)

Swap that dot product for a kernel $K(x_i, x)$ and you get Kernel Ridge
Regression — Ridge's L2-penalized squared-loss objective, but now fitting a
nonlinear function through an implicit feature space. The dual/kernelized
closed-form solution is:

$$c = (K + \alpha I)^{-1} y$$

where $K$ is the $n \times n$ matrix of pairwise kernel values between all
training points (the **Gram matrix**). Note this is $O(n^3)$ to solve and
$O(n^2)$ just to store $K$ — KRR doesn't scale to large $n$ the way linear
Ridge does.

## KRR vs. SVR — the corrected comparison

KRR is frequently confused with **SVR** (Support Vector Regression, the
regression analog of SVM) because both use the kernel trick and both are
"kernelized regression." They differ in loss function and, critically, in
**what kind of solution that loss produces**:

| | Kernel Ridge Regression | SVR |
|---|---|---|
| Loss | **Squared loss**: $(y - f(x))^2$ | **$\varepsilon$-insensitive loss**: $0$ if $\|y - f(x)\| < \varepsilon$, else $\|y - f(x)\| - \varepsilon$ |
| Penalty | L2 ($\alpha\|w\|^2$) | L2 (via the margin/support-vector formulation), plus $C$ for the loss-vs-margin tradeoff |
| Solution form | **Closed-form** (solve one linear system) | Convex QP, solved iteratively (SMO-style) |
| Solution density | **Dense** — every training point gets a nonzero $c_i$ and contributes to every prediction | **Sparse** — points within the $\varepsilon$-tube have exactly zero contribution; only support vectors (points on/outside the tube) matter |
| Prediction cost | $O(n)$ kernel evaluations against *all* training points | $O(n_{\text{support vectors}})$ — often much less than $n$ |
| Training cost | One linear solve, $O(n^3)$ | Iterative QP solve; typically comparable or worse asymptotically, but benefits from sparsity in practice |

> A common but **incorrect** claim is that KRR uses an "L1-ish" loss and SVM
> uses a quadratic one — it's the other way around in the relevant sense:
> KRR's *loss* is squared (L2) error with no dead zone, while SVR's
> $\varepsilon$-insensitive loss has a flat (zero-loss) region and grows *linearly*
> outside it — closer in spirit to an L1/hinge-style loss than KRR's is. The
> "L1 vs quadratic" framing is best avoided; the interview-safe way to state
> the distinction is **squared loss + dense dual solution (KRR)** vs.
> **$\varepsilon$-insensitive loss + sparse dual solution (SVR)**.

Because KRR has no $\varepsilon$-insensitive dead zone, *every* training point pulls on
the fit, which is why it can't produce a sparse solution the way SVR can.

## When to use / when not

**Use KRR when:** you want the kernel trick's nonlinearity but prefer a
closed-form fit (no iterative solver, no convergence tuning) and the dataset
is small enough that $O(n^2)$/$O(n^3)$ is affordable (roughly up to a few
thousand rows).

**Use SVR instead when:** you want a sparse model (faster prediction, less
memory at inference time) or the $\varepsilon$-insensitive loss's built-in tolerance to
small errors is desirable (errors under $\varepsilon$ are free, unlike KRR where every
deviation is penalized).

**Avoid either when:** the dataset is large (tens of thousands+ rows) — both
scale poorly with $n$; prefer a linear model on engineered features, a tree
ensemble, or a Nystroem/random-features kernel approximation instead.

## Common interview questions

- What does the kernel trick actually save you from computing?
- Derive (or describe) the dual form of Ridge regression and where the
  kernel substitution happens.
- Why does KRR's solution use *every* training point while SVR's uses only
  support vectors?
- What loss does KRR minimize vs. what loss does SVR minimize?
- Why is KRR expensive for large $n$, and what would you do instead? (Nyström
  approximation, random Fourier features, switch to a linear model, or
  subsample.)
- If you need calibrated regression on a huge dataset, would you choose KRR?
  Why not?

## Common mistakes

- Repeating the "KRR = L1 loss, SVM = quadratic loss" claim — backwards, and
  loss-type isn't even the cleanest way to state the real distinction
  (density of the solution is). See the corrected comparison above.
- Forgetting that KRR predictions require kernel evaluations against *all*
  training points at inference time — unlike SVR, there's no free lunch from
  sparsity, so prediction latency scales with training-set size.
- Not scaling features before choosing `gamma` for an RBF kernel — same pitfall
  as with SVM.
- Tuning `alpha` alone and forgetting the kernel's own hyperparameters
  (`gamma`, `degree`) — they need a joint search, same as SVM's `C` and `gamma`.

## Example

```python
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

krr = GridSearchCV(
    KernelRidge(kernel="rbf"),
    param_grid={"alpha": [0.1, 1.0, 10.0], "gamma": [0.01, 0.1, 1.0]},
    cv=5,
    scoring="neg_root_mean_squared_error",
)
krr.fit(X_train, y_train)

svr = GridSearchCV(
    SVR(kernel="rbf"),
    param_grid={"C": [1, 10, 100], "epsilon": [0.01, 0.1, 0.5], "gamma": [0.01, 0.1, 1.0]},
    cv=5,
    scoring="neg_root_mean_squared_error",
)
svr.fit(X_train, y_train)

# Compare: SVR's fitted estimator typically has far fewer "active" points
# (support vectors) than KRR, which uses all n_train points at predict time.
print("SVR support vectors:", len(svr.best_estimator_.support_))
print("KRR training points used per prediction:", X_train.shape[0])
```

See also: [SVM](svm.md) for the max-margin/hinge-loss side of the kernel
family, and [Regularization](../linear-models/regularization.md) for
un-kernelized Ridge.
