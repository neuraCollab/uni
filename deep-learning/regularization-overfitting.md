# Regularization & Overfitting in Deep Learning

> Source note: the concrete example this file is built from — [`overUnderFitting.py`](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit)-style Tiny/Small/Medium/Large model-capacity comparison — is adapted from a **TensorFlow/Keras** tutorial (the archived source: `tensorFlow/tfText/overUnderFitting.py`), since it was the only complete treatment of this topic across both source repos. This knowledge base is **PyTorch-primary**; the concepts below (L1/L2, dropout, early stopping, augmentation, batchnorm-as-regularizer) are framework-agnostic and apply identically in PyTorch — the illustrative code snippets here are PyTorch.

## What is it?

**Overfitting** is when a model fits the training data (including its noise) so closely that it fails to generalize to unseen data — training loss keeps dropping while validation loss stops improving or gets worse. **Regularization** is any technique that constrains a model's effective capacity or otherwise biases it toward simpler, more generalizable solutions.

## Why?

Larger/more flexible networks can represent more functions, including ones that memorize the training set rather than learning the underlying pattern. The archived experiment makes this concrete: it trains four architectures of increasing capacity (Tiny: one 16-unit layer; Small: two 16-unit layers; Medium: three 64-unit layers; Large: four 512-unit layers) on the same data and shows validation loss diverging from training loss earlier and more severely as capacity grows — the textbook overfitting curve. Regularization exists to let you use a model with enough capacity to learn the real pattern without also memorizing noise.

## How does it work?

### L1 vs. L2 weight regularization

Both add a penalty on weight magnitude to the loss, discouraging large weights (which tend to correspond to the model relying heavily on a few features / fitting sharp, idiosyncratic patterns).

- **L2 (weight decay)**: penalty `lambda * sum(w^2)`. Shrinks weights smoothly toward zero but rarely exactly to zero. The most common choice for deep nets.
- **L1**: penalty `lambda * sum(|w|)`. Pushes many weights to exactly zero — produces sparse weight vectors, useful for implicit feature selection.

In "add a penalty term to the loss" form, weight decay is written directly as:

```
L_new = L + λ||W||_2
```

Note on notation: some sources (and the terse form above) write the penalty as `||W||_2`, the norm itself; in practice essentially every framework implements the *squared* L2 norm, `||W||_2^2 = sum(w^2)` (i.e. the `lambda * sum(w^2)` above) — its gradient is the simple, everywhere-differentiable `2*w`, whereas the un-squared norm has an undefined gradient at `W = 0`. When someone says "L2 regularization" or "weight decay" in an interview, they mean the squared form unless they say otherwise.

### Activation-based (sparsity) regularization

A less commonly taught alternative/supplement to dropout: instead of penalizing weight magnitude, penalize a hidden unit's *average activation* for deviating from some target sparsity level `p` (a small number, e.g. `0.05`) — pushing each unit to fire rarely rather than on most inputs:

```
L_new = L - λ * Σ_k [ p * log(p̂_k) ]
```

where `p̂_k = f(x_i, θ)` is unit `k`'s average activation over the batch (or a running estimate over training). This is a simplified/one-sided version of the fuller KL-divergence sparsity penalty used in sparse autoencoders:

```
Σ_k [ p*log(p/p̂_k) + (1-p)*log((1-p)/(1-p̂_k)) ]     # KL(Bernoulli(p) || Bernoulli(p̂_k)), summed over hidden units
```

which penalizes both "unit fires too often" (`p̂_k` too high) and "unit fires too rarely" (`p̂_k` too low) relative to the target `p`. Less common than dropout/weight-decay in modern practice, but worth recognizing — it's the standard regularizer in sparse autoencoders and shows up in interview questions about representation learning.

Source example (Keras):
```python
layers.Dense(512, activation='elu', kernel_regularizer=regularizers.l2(0.001))
```

PyTorch equivalent — L2 is usually applied via the optimizer's `weight_decay` argument rather than a per-layer regularizer term:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # L2
```
(For true L1, or L2 restricted to specific parameters, add the penalty term to the loss manually.)

### Dropout

Randomly zeroes a fraction of activations (a different random subset each forward pass, only during training) with probability `p`:
```python
layers.Dropout(0.5)   # source example
```
```python
self.drop = nn.Dropout(p=0.5)   # PyTorch equivalent
```

**Why it works**: it prevents **co-adaptation** — neurons can't rely on any specific other neuron always being present, since any of them might be zeroed out on a given step, so each neuron is forced to learn features that are useful more independently/robustly. There's also an **ensemble interpretation**: training with dropout is approximately equivalent to training an exponential number of thinned sub-networks (one per dropout mask) with shared weights, and averaging their predictions at test time — a cheap approximation to ensembling.

### The critical PyTorch trap: `model.train()` vs. `model.eval()`

Dropout and BatchNorm behave **differently** depending on mode:
- `model.train()`: dropout actively zeroes activations; BatchNorm uses the current batch's statistics and updates its running mean/variance.
- `model.eval()`: dropout is a no-op (passes everything through, no zeroing); BatchNorm uses its stored running statistics instead of the current batch's.

```python
model.train()
for x, y in train_loader:
    ...                       # dropout active, batchnorm uses batch stats

model.eval()
with torch.no_grad():
    preds = model(x_val)      # dropout off, batchnorm uses running stats
```

**Forgetting `model.eval()` at inference time is a classic, extremely common bug** — the model silently keeps dropping activations and/or normalizing against batch statistics of a (possibly tiny or single-example) inference batch, producing noisy, inconsistent, or just wrong predictions, with no error raised. Always pair training loops with an explicit mode switch before evaluation, and switch back to `.train()` before resuming training.

### Early stopping

Monitor validation loss (or another metric) during training and stop once it stops improving for `patience` epochs, keeping the best checkpoint rather than the final one.
```python
tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200)
```
This is exactly how the archived Tiny/Small/Medium/Large comparison avoids each model training to a wildly different number of epochs — every model stops once its own validation metric plateaus. In PyTorch this is implemented manually (track best val loss, save `state_dict()` on improvement, break the loop after `patience` epochs without improvement) or via a library callback (e.g. in PyTorch Lightning).

### Data augmentation

Applying label-preserving random transformations to training inputs (crops, flips, color jitter for images; synonym replacement/back-translation for text) so the model sees more effective variation without collecting more labeled data. Directly attacks overfitting by making memorization harder — the model rarely sees the exact same input twice.

### Batch normalization — the exact mechanics

BatchNorm is two steps applied per-channel/per-feature, not one. Given a mini-batch with mean `μ` and variance `σ²` (computed over the batch, for a given channel):

**Step 1 — normalize** to zero mean / unit variance:

```
x^(k+1) = (x^k - μ) / sqrt(σ² + ε)
```

(`ε` is a small constant for numerical stability, avoiding division by zero.)

**Step 2 — running statistics for inference.** Training sees only mini-batch statistics, but at inference time there may be no batch (or a batch of 1) to compute `μ`/`σ²` from — so during training, BatchNorm also maintains an exponential moving average of the batch statistics, updated at *every* training forward pass:

```
μ*  = λ * μ*  + (1 - λ) * μ
σ*² = λ * σ*² + (1 - λ) * σ²
```

where `λ` is a momentum hyperparameter of the layer. At inference time, the layer normalizes using the stored `μ*`/`σ*²` instead of recomputing batch statistics. **This is exactly why `model.eval()` matters for BatchNorm, not just Dropout** (see the train/eval section above) — in eval mode, BatchNorm switches from "normalize using this batch's own mean/variance" to "normalize using the running `μ*`/`σ*²` accumulated during training." Forgetting `model.eval()` means BatchNorm keeps normalizing against whatever tiny/skewed batch happens to be in front of it at inference.

**Step 3 — learned affine scale-and-shift.** This is the detail most often left out when people explain BatchNorm in interviews: normalization doesn't end at step 1. A learned, per-channel affine transform is applied *after* normalizing:

```
y = γ * x_norm + β
```

`γ` (scale) and `β` (shift) are learned parameters, one pair per channel/feature, trained by backprop exactly like any other weight (in PyTorch, `BatchNorm1d/2d/3d.weight` is `γ` and `.bias` is `β`). This gives the network an escape hatch: if forcing a channel to exactly zero-mean/unit-variance is *not* actually optimal, the network can learn a `γ`/`β` that undoes the normalization (e.g. `γ = sqrt(σ²+ε)`, `β = μ` recovers the original scale/shift exactly). So BatchNorm is "normalize, then let the network learn whatever mean/scale it actually wants" — not a hard constraint to zero-mean/unit-variance.

(Notation note: some handwritten/lecture sources write this step as `x^(k+2) = β·x^(k+1) + γ`, i.e. with `β` as the multiplicative/scale term and `γ` as the additive/shift term — the reverse of the naming convention above. The names are arbitrary; what's invariant is that there's one learned multiplicative and one learned additive parameter per channel, applied after normalization. This note uses the more common `γ`-scale/`β`-shift convention, matching the original BatchNorm paper and PyTorch's attribute names.)

### Batch normalization as an implicit regularizer

Not in the source file, but a very common interview follow-up: **BatchNorm** normalizes each layer's activations using the current mini-batch's mean/variance during training. Beyond its main purpose (stabilizing/accelerating training by keeping activation distributions consistent across layers, reducing sensitivity to initialization and learning rate), it has a mild regularizing side effect — because the batch statistics are noisy estimates that vary batch-to-batch, each activation gets a slightly different normalization each step, acting like a mild form of noise injection similar in spirit to dropout. This is why networks with heavy BatchNorm use sometimes need less dropout to reach the same generalization.

## Example

```python
class RegularizedMLP(nn.Module):
    def __init__(self, in_dim, hidden=512, p_drop=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)

model = RegularizedMLP(28)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # L2

model.train()
# ...training loop...

model.eval()
with torch.no_grad():
    preds = model(x_val)  # dropout off, batchnorm uses running stats
```

## When to use

Apply regularization whenever validation loss diverges from training loss (the textbook overfitting signature). Dropout and weight decay are near-default for large MLPs/CNNs; BatchNorm is close to default in CNNs regardless of overfitting concerns (for training stability); data augmentation is close to default whenever the domain supports label-preserving transforms; early stopping costs nothing and should almost always be on. See also [`hyperparameter-tuning.md`](hyperparameter-tuning.md) for how dropout rate / weight-decay strength are themselves tuned.

## Common interview questions

- **What's the difference between L1 and L2 regularization?** L1 encourages sparsity (many weights exactly zero); L2 shrinks weights smoothly without necessarily zeroing them.
- **Why does dropout work?** Prevents co-adaptation of neurons by randomly removing them during training; approximates averaging over an exponential ensemble of sub-networks.
- **What's the difference between `model.train()` and `model.eval()`, and why does it matter?** Controls whether Dropout/BatchNorm use their training-time stochastic/batch-statistics behavior or their fixed inference-time behavior; forgetting `eval()` at inference silently degrades predictions.
- **How does early stopping act as regularization?** It limits effective training time/capacity usage, stopping before the model has enough gradient steps to overfit the training set.
- **Is BatchNorm a regularizer?** Primarily a training-stabilization technique, but yes, it has a secondary mild regularizing effect from the noise introduced by per-batch statistics.

## Common mistakes

- Forgetting `model.eval()` before validation/inference (and forgetting to switch back to `model.train()` afterward) — the single most common PyTorch regularization bug.
- Applying dropout at inference time (equivalent to the above) or computing BatchNorm statistics from a single inference example.
- Using L2 weight decay together with Adam without realizing that naive `weight_decay` in the optimizer interacts with Adam's adaptive scaling differently than true decoupled weight decay (this is what `AdamW` was specifically designed to fix).
- Tuning capacity (bigger model) as the first response to underfitting without first checking learning rate/training length — and conversely, reaching for a bigger model when the real problem is data quantity/quality.
- Adding regularization strength blindly without a validation curve to actually check it's helping — over-regularizing causes underfitting just as surely as under-regularizing causes overfitting.
