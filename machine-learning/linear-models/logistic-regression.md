# Logistic Regression

## What is it?

A linear model for **classification** that predicts the probability of a class via the **sigmoid** function applied to a linear combination of features, despite the name containing "regression."

## How does it work?

### Binary case

Linear score (logit): `z = w·x + b`.

**Sigmoid function** maps the score to a probability in (0, 1):

```
p = sigma(z) = 1 / (1 + exp(-z))
```

**Decision boundary:** predict class 1 if `p >= 0.5`, i.e. `z >= 0` — a **linear** decision boundary (a hyperplane) in feature space, since `z = w·x + b = 0` is linear in `x`.

**Loss: log-loss (binary cross-entropy).** For a single example with true label `y in {0, 1}` and predicted probability `p`:

```
L(y, p) = -[ y*ln(p) + (1-y)*ln(1-p) ]
```

**Why log-loss, not squared error?** Log-loss comes from maximizing the likelihood of a Bernoulli-distributed target — `p` is literally the Bernoulli parameter, and `-log p(y | x)` is exactly this expression. It's also convex in `w` for the logistic model (squared error on top of a sigmoid is not, which would make optimization harder), and it penalizes confident-wrong predictions much more heavily than squared error would (as `p -> 1` while `y = 0`, loss `-> infinity`).

This unit deviance is also the **Bernoulli deviance** from the GLM framework — logistic regression is the Bernoulli-family GLM with a logit link (see [Generalized Linear Models](generalized-linear-models.md)).

### Multiclass

Two standard strategies:

- **One-vs-Rest (OvR):** train one binary logistic classifier per class (class `k` vs. everyone else), predict the class whose classifier gives the highest score/probability. Simple, but the per-class probabilities aren't guaranteed to sum to 1 without renormalizing.
- **Softmax (multinomial) regression:** a direct generalization — one linear score `z_k` per class, then

```
p_k = exp(z_k) / sum_j exp(z_j)
```

which *does* produce a valid probability distribution over all classes simultaneously, and is trained by minimizing multiclass cross-entropy. This is what `LogisticRegression(multi_class='multinomial')` uses (and is the default in modern sklearn for most solvers).

### Regularization

`LogisticRegression` supports the same penalty families as linear regression, applied to the log-loss instead of squared error:
- `penalty='l2'` (default) — Ridge-style, smooth shrinkage.
- `penalty='l1'` — Lasso-style, sparse coefficients, needs `solver='liblinear'` or `'saga'`.
- `penalty='elasticnet'` — mix of both, needs `solver='saga'`, with `l1_ratio` controlling the mix.
- Note sklearn parametrizes strength via `C = 1/alpha` (inverse regularization strength) — **larger `C` means weaker regularization**, the opposite convention from `Ridge`/`Lasso`'s `alpha`.

### Interpreting coefficients: log-odds

Because `z = w·x + b = ln(p / (1-p))` (the **logit**, or log-odds), each coefficient `w_j` is the change in **log-odds** of the positive class per one-unit increase in `x_j`, holding other features fixed. Equivalently, `exp(w_j)` is the multiplicative change in the **odds** (`p / (1-p)`) per unit increase in `x_j`. This is different from a linear-regression coefficient, which is a direct change in the *predicted value* — logistic coefficients act on the *log-odds scale*, not the probability scale directly (the effect on probability itself is nonlinear, largest near `p=0.5`).

### class_weight vs. sample_weight

Both let you tell the model "some examples/classes matter more," but at different granularity:

```python
class_weight = {0: 1.0, 1: 5.0}       # per-class multiplier, applied to every example of that class
sample_weight = np.array([1, 1, 5, 1, 10, ...])  # per-example multiplier, len == n_samples
```

- `class_weight` — useful for **class imbalance** (e.g. `class_weight='balanced'` auto-weights inversely to class frequency).
- `sample_weight` — useful when *individual examples* have different importance/reliability (e.g. more trustworthy labels, or examples that should count more due to survey weighting), independent of class membership.
- Neither is learned by the model — both are set externally, based on domain knowledge or a fixed imbalance-correction heuristic.

## When to use / when not to use

**Use when:** you need a fast, interpretable, well-calibrated probabilistic classifier as a baseline; the decision boundary is plausibly linear (or close to it after feature engineering); you need coefficient-level interpretability (log-odds).

**Avoid when:** the true decision boundary is strongly nonlinear and you don't want to hand-engineer interaction/polynomial features — a kernelized ([SVM](../kernel-methods/svm.md)) or tree-based model may fit better; or when there's severe multicollinearity and you need feature selection (favor L1/ElasticNet penalty in that case).

## Common interview questions

- Derive log-loss from the Bernoulli likelihood.
- Why is log-loss convex for logistic regression but squared error is not?
- Difference between One-vs-Rest and softmax/multinomial for multiclass?
- How do you interpret a logistic regression coefficient?
- What's the difference between `class_weight` and `sample_weight`?
- Why does `LogisticRegression` use `C` instead of `alpha`, and how does it relate to `Ridge`'s `alpha`?
- Is logistic regression a linear or nonlinear model? *(Linear decision boundary in the original feature space; the sigmoid is a nonlinear squashing function on top, but it doesn't change the linearity of the boundary itself.)*

## Common mistakes

- Believing accuracy is always the right metric — on imbalanced classes, always check precision/recall/F1 and consider `class_weight` (see [Classification Metrics](../model-evaluation/classification-metrics.md)).
- Interpreting coefficients on the probability scale directly instead of the log-odds scale.
- Forgetting to scale features when using L1/L2 regularization (same issue as [Ridge/Lasso](regularization.md)).
- Assuming `LogisticRegression` outputs are perfectly calibrated probabilities out of the box for every solver/penalty combination — worth checking with a calibration curve if downstream probability values matter, not just the ranking.

## Example

See [`code/glm.py`](code/glm.py) for the general GLM framing, and the elastic-net MNIST digit-recognition example the archive used to demonstrate sparsity under L1-leaning regularization:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=5000, test_size=10000, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = LogisticRegression(
    penalty="elasticnet", solver="saga", l1_ratio=0.5,
    C=50 / 5000, max_iter=5000, tol=1e-3, random_state=0,
)
clf.fit(X_train, y_train)

sparsity = np.mean(clf.coef_ == 0) * 100
print(f"Sparsity with elasticnet penalty: {sparsity:.2f}%")
print(f"Test accuracy: {clf.score(X_test, y_test):.4f}")
```

## Related notes

- [Generalized Linear Models](generalized-linear-models.md) — logistic regression as the Bernoulli GLM.
- [Regularization](regularization.md) — same L1/L2/ElasticNet penalties, applied here to log-loss.
- [Classification Metrics](../model-evaluation/classification-metrics.md) — evaluating the resulting classifier properly.
- [Kernel Methods: SVM](../kernel-methods/svm.md) — a margin-based (rather than probabilistic) alternative for linear/kernelized classification.
