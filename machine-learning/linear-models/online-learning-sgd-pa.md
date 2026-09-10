# Online Learning: SGD and Passive-Aggressive

## What is it?

**Online learning** algorithms update the model **incrementally, one sample (or mini-batch) at a time**, instead of solving a closed-form or batch-optimization problem over the whole dataset at once. This matters when data arrives as a stream, or when the full dataset doesn't fit in memory. `SGDClassifier`/`SGDRegressor` and `PassiveAggressiveClassifier`/`Regressor` are sklearn's two main linear online learners.

## Stochastic Gradient Descent (SGD)

**How it works:** instead of computing the gradient of the loss over the *entire* dataset each step (as batch gradient descent does), SGD estimates the gradient from **one sample at a time** and takes a step:

```
w <- w - eta * grad(loss(w; x_i, y_i))
```

- **`eta` (learning rate) schedules:** `'constant'` (fixed `eta0`), `'optimal'` (sklearn's default heuristic decay based on the regularization strength), `'invscaling'` (`eta0 / t^power_t`), `'adaptive'` (halve `eta` when training loss stops improving). The schedule matters a lot — too high and training diverges/oscillates, too low and convergence is painfully slow.
- **Important implementation detail:** sklearn's `SGDClassifier`/`SGDRegressor` process the dataset **one sample at a time per update** (true stochastic GD), not mini-batches — unlike typical deep learning "SGD" which usually means mini-batch. For huge datasets it still needs multiple passes (`max_iter` epochs) over the data.
- **Why feature scaling matters so much here:** the same learning rate `eta` is applied across all features simultaneously. If one feature has a much larger scale than another, the gradient step in that direction is disproportionately large/small relative to what's "right" for that feature, causing slow/unstable convergence. Always `StandardScaler` (or similar) before SGD — this matters far more for SGD than for closed-form solvers like OLS/Ridge, which aren't iterative and don't have this scale-dependent step-size problem.
- `loss='log_loss'` → logistic regression trained via SGD instead of a batch solver; `loss='hinge'` → linear SVM trained via SGD; `penalty='l2'|'l1'|'elasticnet'` → same regularization options as the batch linear models.
- **When to use:** very large datasets/streaming data where batch solvers are too slow or don't fit in memory; when you want a linear model trained progressively as new data arrives.

## Passive-Aggressive (PA)

**How it works:** an online, **margin-based** algorithm (conceptually close to a streaming SVM). For each incoming sample:
- If the sample is already correctly classified with **sufficient margin**, the algorithm stays **passive** — no update at all.
- If the sample is misclassified or falls inside the margin, the algorithm is **aggressive** — it makes the *smallest possible update* to `w` that fixes the margin violation for this sample (a closed-form step, not a fixed-size gradient step like SGD).

This "smallest sufficient correction" property is what distinguishes PA from SGD: SGD always takes a step of a size set by `eta`; PA takes exactly as large a step as needed to satisfy the margin constraint on the current example (and no more).

**Aggressiveness parameter `C`:**
- Controls the tradeoff between fitting the *current* example perfectly and preserving the *previously learned* weights.
- Larger `C` → more aggressive updates, more willing to move `w` a lot to fully satisfy the current example.
- **PA-I** — a hard cap on step size (soft-margin, uses hinge loss with a linear penalty on the slack), controlled linearly by `C`.
- **PA-II** — a variant with a quadratic penalty term added to the update-size objective, more tolerant of noisy/mislabeled samples (a single bad example influences the weights less abruptly than under PA-I). `C -> infinity` recovers the original hard-margin Passive-Aggressive rule (no tolerance for margin violations at all) for both variants.
- sklearn's `PassiveAggressiveClassifier(loss='hinge')` implements PA-I, `loss='squared_hinge'` implements PA-II.

## SGD vs. Passive-Aggressive

| | SGD | Passive-Aggressive |
|---|---|---|
| Update size | fixed by learning-rate schedule `eta` | data-dependent: exactly enough to satisfy the margin |
| Update trigger | every sample (some tiny gradient step even if already correct) | only on misclassification/margin violation ("passive" otherwise) |
| Key hyperparameter | `eta0` + schedule | `C` (aggressiveness) |
| Loss | configurable (log, hinge, squared, etc.) | hinge-family (margin-based) |
| Sensitivity to feature scale | high | lower (though scaling is still good practice) |

## Common interview questions

- Why is sklearn's "SGD" per-sample rather than mini-batch, and how is that different from deep-learning SGD?
- How does a learning-rate schedule affect convergence?
- What does "passive" vs "aggressive" refer to in PA, concretely?
- How does PA-I differ from PA-II?
- Why does `C -> infinity` recover the hard-margin update?
- Why is feature scaling more critical for SGD than for a closed-form solver like Ridge?

## Common mistakes

- Forgetting to scale features before SGD, leading to slow/unstable convergence that looks like "the model doesn't learn."
- Assuming PA is loss-free/model-free — it's still a linear model with a margin-based (SVM-like) decision rule; it just updates differently online.
- Treating sklearn's `SGDClassifier` as automatically doing mini-batches — it processes samples one at a time (`partial_fit` lets you feed genuine mini-batches/chunks yourself for streaming use cases).

## Example

See [`code/sgd.py`](code/sgd.py) for an `SGDClassifier` demo (log-loss, i.e. online logistic regression, on MNIST) and [`code/pa.py`](code/pa.py) for a `PassiveAggressiveClassifier` (PA-II via `C`) demo on a binarized MNIST task.

## Related notes

- [Logistic Regression](logistic-regression.md) — the batch-solver counterpart to `SGDClassifier(loss='log_loss')`.
- [Kernel Methods: SVM](../kernel-methods/svm.md) — the margin/hinge-loss idea PA is built on.
