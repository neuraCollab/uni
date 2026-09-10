# Weight Initialization

## What is it?

The choice of starting values for a network's weights, before any training step has run. Training a neural network is non-convex optimization — gradient descent finds *a* local solution reachable from wherever it starts, not *the* global optimum — so where you start matters.

## Why?

Bad initialization can prevent training from working at all (symmetric weights never break symmetry — see below), or slow it dramatically (activations that shrink or blow up as they pass through layers before training even gets going). Good initialization is a cheap, one-time fix for problems that would otherwise look like vanishing/exploding gradients from step one.

## How does it work?

### All-zeros initialization — and why it fails

Setting every weight to `0` seems harmless (or even principled — "start neutral"), but it's fatal:

```
W = 0   =>   ∂L/∂W = 0 for every weight, identically, for every neuron in a layer
```

The reason is a **symmetry** argument, and it's one of the most commonly asked interview questions on this topic: if every neuron in a layer starts with identical weights, every neuron computes the identical function of the input, so every neuron receives the identical gradient during backprop. They then get updated by the identical amount, so they *stay* identical — forever. The layer never differentiates into diverse feature detectors; effectively, a layer of `n` identical neurons behaves like a layer of 1 neuron no matter how wide you make it. (Biases can safely start at zero — it's specifically the *weights feeding into the same neurons* that must be broken out of symmetry. Zero bias with random weights is fine and common.)

### Naive random initialization

The obvious fix is to break symmetry with randomness: draw every weight i.i.d. from a fixed distribution, e.g.

```
W ~ N(μ=0, σ²)
```

This breaks symmetry, but a *fixed* variance `σ²` doesn't account for how many inputs feed into a neuron (the layer's fan-in). Each unit's pre-activation is a sum of `n_in` weighted terms, so its variance scales with `n_in * σ²`:

- If `σ²` is too small relative to the layer width, the variance of the pre-activations shrinks layer after layer — activations (and later, gradients flowing back through them) collapse toward zero deeper into the network: **vanishing**.
- If `σ²` is too large, the opposite happens — activations and gradients grow layer after layer: **exploding**.

Either failure mode makes deep networks slow or impossible to train, which is exactly the vanishing/exploding gradient problem described in [`neural-networks-backprop.md`](neural-networks-backprop.md#common-interview-questions).

### Xavier / Glorot initialization

Xavier (Glorot) initialization fixes the fan-in blind spot by calibrating the distribution's spread using *both* the number of input units (`n_in`) and output units (`n_out`) of the layer, with the explicit design goal of keeping the variance of activations — and of gradients flowing backward — roughly constant as you move through the network, instead of systematically shrinking or growing layer by layer. The uniform-distribution form:

```
W_i ~ U[ -sqrt(6 / (n_in + n_out)),  sqrt(6 / (n_in + n_out)) ]
```

(A Gaussian version exists too, `W ~ N(0, 2/(n_in+n_out))`, with the same calibration idea — the uniform form above is the one transcribed here.) Xavier's derivation assumes roughly linear/symmetric activations around 0 (it was designed with tanh/sigmoid in mind), which is where it's most theoretically justified.

### He initialization (not from the source material, included for completeness)

ReLU breaks Xavier's assumption: it zeros out roughly half its inputs (everything negative), which halves the effective variance passed forward. **He initialization** is the ReLU-specific correction, scaling the variance up to compensate:

```
W ~ N(0, 2 / n_in)
```

Rule of thumb: Xavier/Glorot for tanh/sigmoid-style networks, He for ReLU/Leaky-ReLU-style networks (`nn.init.kaiming_normal_` / `nn.init.xavier_uniform_` in PyTorch, respectively).

## Example

```python
import torch.nn as nn

layer = nn.Linear(256, 128)

# Xavier/Glorot — pairs well with tanh/sigmoid activations
nn.init.xavier_uniform_(layer.weight)
nn.init.zeros_(layer.bias)

# He/Kaiming — pairs well with ReLU-family activations
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
nn.init.zeros_(layer.bias)
```

In practice you rarely call these manually for standard layers — `nn.Linear`, `nn.Conv2d`, etc. already ship with a sensible default (PyTorch's `nn.Linear` defaults to a Kaiming-uniform variant) — but it's worth knowing how to override it, and essential to know *why* it matters when asked in an interview.

## When to use

- **Always** use a variance-calibrated scheme (Xavier or He) over naive fixed-variance random init for anything beyond a couple of layers — the deeper the network, the more the fan-in/fan-out mismatch compounds.
- Match the scheme to the activation family: Xavier for tanh/sigmoid, He for ReLU/Leaky ReLU/PReLU.
- Never initialize weights to all-zeros (or any other constant) for a layer with more than one neuron — see the symmetry argument above. (An all-zero *bias* alongside random weights is fine.)
- Modern architectures with residual connections, BatchNorm/LayerNorm, and careful init are jointly what makes very deep networks trainable at all — initialization alone isn't a complete fix for vanishing/exploding gradients in very deep nets, but it's the cheapest first line of defense. See [`regularization-overfitting.md`](../regularization-overfitting.md) for BatchNorm's role in this.

## Common interview questions

- **Why does initializing all weights to zero break training?** The symmetry argument: identical weights → identical neuron outputs → identical gradients → identical updates, forever. The layer never diversifies. (Be ready to state this precisely — it's a very common question.)
- **What goes wrong with naive fixed-variance random initialization in a deep network?** Doesn't account for layer width (fan-in/fan-out); too-small variance vanishes activations deeper into the network, too-large variance explodes them.
- **What does Xavier/Glorot initialization do differently, and why does it use both `n_in` and `n_out`?** Calibrates the initialization variance using fan-in and fan-out so that activation (and gradient) variance stays roughly constant across layers, rather than systematically shrinking/growing with depth.
- **Why does He initialization exist separately from Xavier?** ReLU zeros out about half its inputs, which effectively halves the variance passed to the next layer relative to what Xavier assumes (roughly linear/symmetric activations); He compensates by scaling the variance up.
- **Is zero-initializing biases a problem?** No — the failure mode is specifically about *weights feeding into the same neuron pool* being identical; zero bias with randomly-initialized weights breaks symmetry fine.

## Common mistakes

- Initializing every weight (not just bias) to zero or any other single constant value, and being surprised the network doesn't train.
- Using a fixed-variance random init (e.g. `N(0, 0.01)`) regardless of layer width — works fine for small/shallow nets, silently degrades as networks get deeper or layers get wider/narrower.
- Mismatching the initialization scheme and activation function (e.g. He init tuned for ReLU's half-zeroed variance, used with tanh) — usually not catastrophic, but leaves performance on the table.
- Treating initialization as a complete fix for vanishing/exploding gradients in very deep networks — it helps at the start of training, but doesn't substitute for architectural fixes (residual connections, normalization layers) once depth gets large.

## Related notes

- [Neural Networks & Backpropagation](neural-networks-backprop.md) — the vanishing/exploding gradient problem this initialization is designed to mitigate.
- [Regularization & Overfitting](../regularization-overfitting.md) — BatchNorm, which reduces a network's sensitivity to initialization choice.
