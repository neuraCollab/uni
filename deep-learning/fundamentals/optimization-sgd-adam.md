# Optimization: SGD, Momentum, RMSprop, Adam

## What is it?

The family of algorithms that update a network's parameters to minimize the loss, given the gradients computed by [backpropagation](neural-networks-backprop.md). They differ in how much history/adaptivity they use beyond the raw gradient at the current step.

## Why?

Plain gradient descent uses a single global step size and treats every parameter identically. In practice, loss surfaces are ill-conditioned (steep in some directions, flat in others), noisy (mini-batch gradients are estimates, not the true gradient), and non-convex (saddle points, local structure that a naive step size handles badly). Momentum and adaptive learning rates were developed to make optimization faster and more robust to these issues.

## How does it work?

**Gradient Descent (batch).** Compute the gradient of the loss over the *entire* dataset, take one step: $\theta \leftarrow \theta - \text{lr} \cdot \nabla_\theta L$. Accurate but far too slow/expensive per step for deep learning.

**SGD (stochastic / mini-batch).** Estimate the gradient from a small batch instead of the full dataset. Noisier, but each step is cheap, and the noise itself can help escape shallow local minima/saddle points.

$$\theta \leftarrow \theta - \text{lr} \cdot \nabla_\theta L(\theta;\, \text{batch})$$

**SGD + Momentum.** Accumulate a running (exponentially-weighted) average of past gradients — a "velocity" — and step in that direction instead of the raw gradient. Damps oscillation across steep/narrow ravines and accelerates along consistent directions.

$$
\begin{aligned}
v &\leftarrow \beta v + (1 - \beta) \nabla_\theta L \qquad (\beta \approx 0.9) \\
\theta &\leftarrow \theta - \text{lr} \cdot v
\end{aligned}
$$

**RMSprop.** Adapts the learning rate *per parameter* by dividing by the root of a running average of squared gradients. Parameters with consistently large gradients get their effective step size shrunk; parameters with small/sparse gradients get relatively larger steps.

$$
\begin{aligned}
s &\leftarrow \beta s + (1 - \beta) (\nabla_\theta L)^2 \qquad (\beta \approx 0.99) \\
\theta &\leftarrow \theta - \text{lr} \cdot \frac{\nabla_\theta L}{\sqrt{s} + \varepsilon}
\end{aligned}
$$

**Adam (Adaptive Moment Estimation).** Combines both ideas: a momentum term (1st moment, mean of gradients) and an RMSprop-style term (2nd moment, mean of squared gradients), plus a bias-correction step because both running averages are initialized at zero and are biased toward zero early in training.

$$
\begin{aligned}
m &\leftarrow \beta_1 m + (1 - \beta_1) \nabla_\theta L &&\text{1st moment (momentum)} \\
s &\leftarrow \beta_2 s + (1 - \beta_2) (\nabla_\theta L)^2 &&\text{2nd moment (RMSprop-style)} \\[4pt]
\hat{m} &= \frac{m}{1 - \beta_1^t} &&\text{bias correction} \\
\hat{s} &= \frac{s}{1 - \beta_2^t} \\[4pt]
\theta &\leftarrow \theta - \text{lr} \cdot \frac{\hat{m}}{\sqrt{\hat{s}} + \varepsilon}
\end{aligned}
$$

Defaults that work well almost everywhere: $\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$. Adam is the default choice for most deep learning problems because it's robust to the initial learning rate choice and converges fast; plain SGD+momentum sometimes generalizes *better* on large vision models when tuned carefully (a known empirical gap), which is why you still see it in some training recipes (e.g. ResNet/ImageNet).

### Learning rate: the single most important hyperparameter

Everything above still needs a base learning rate `lr`. Too high: loss diverges or oscillates. Too low: painfully slow convergence, or gets stuck in a poor region. Before tuning anything else (architecture width, batch size, regularization strength), get the learning rate roughly right — usually via a quick range test (try `1e-1 ... 1e-5` on a log scale, watch which ones actually descend without diverging).

**LR schedules** — the LR usually shouldn't stay constant through training:

- **Step decay** — multiply LR by a factor (e.g. 0.1) every N epochs. Simple, widely used.
- **Cosine annealing** — smoothly decay LR following a cosine curve from the initial value down to ~0 over training. Popular in modern recipes because it avoids the abrupt drops of step decay.
- **Warmup** — start with a small LR and ramp up linearly for the first few hundred/thousand steps before switching to the main schedule. Stabilizes early training when weights are far from any good region and gradients/adaptive-moment estimates are still unreliable (especially important for Adam and for transformers).

### Batch size effects

Larger batches give a lower-variance (smoother, more accurate) estimate of the true gradient, which lets you push the learning rate higher and parallelize better on GPU. But there's a well-documented **generalization gap**: models trained with very large batch sizes often converge to sharper minima and generalize *worse* on held-out data than models trained with smaller batches, even at equal training loss. The common mitigations are to scale the learning rate up with batch size (linear scaling rule) and add a warmup period. This tradeoff — large batch = faster/smoother training but potentially worse generalization — is a frequent interview question.

## Example (PyTorch)

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

for epoch in range(num_epochs):
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    scheduler.step()
```

Note that [`unet_segment.py`](../pytorch/code/unet_segment.py) uses plain `optim.RMSprop` — a perfectly reasonable choice, just not the most common default; don't assume every PyTorch training loop uses Adam.

## When to use

- **Adam (or AdamW)**: default starting point for most problems — CNNs, RNNs, transformers.
- **SGD + momentum**: when you have the compute budget to tune it carefully; still common for large-scale vision training where it can generalize better than Adam.
- **RMSprop**: reasonable default for RNNs historically; largely superseded by Adam but still seen.

## Common interview questions

- **Derive the Adam update rule, including why bias correction is needed.** Without it, `m` and `s` are biased toward 0 at early timesteps since they're initialized at 0 and only partially updated each step.
- **Why does momentum help?** Averages out gradient noise and accelerates movement in directions where the gradient is consistent, damping oscillation in directions where it flips sign.
- **What happens if the learning rate is too high? Too low?** Divergence/oscillation vs. slow convergence or getting stuck.
- **Large batch vs. small batch — tradeoffs?** Smoother/faster-to-compute gradient estimate and better hardware utilization vs. worse generalization (sharp minima) unless LR/warmup are adjusted.
- **Why use a learning rate schedule instead of a constant LR?** Large steps early help fast progress when far from the optimum; small steps late help fine-grained convergence without overshooting.

## Common mistakes

- Not tuning the learning rate at all and blaming the architecture when training doesn't converge.
- Using a very large batch size without also scaling up the learning rate or adding warmup, and being surprised by worse validation performance.
- Forgetting `optimizer.zero_grad()` before `loss.backward()` (see [`neural-networks-backprop.md`](neural-networks-backprop.md)) — an optimizer bug, not an optimizer-*choice* bug, but shows up in the same place.
- Assuming Adam is strictly "better" than SGD — it converges faster but SGD+momentum can reach flatter, better-generalizing minima with enough tuning.
