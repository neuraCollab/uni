# Neural Networks & Backpropagation

## What is it?

A neural network is a stack of parameterized linear transformations ($Wx + b$) interleaved with non-linear activation functions. Backpropagation is the algorithm that computes the gradient of a loss function with respect to every parameter in the network, using the chain rule, so an optimizer can update those parameters.

Code reference: [`pytorch/code/simple_mlp.py`](../pytorch/code/simple_mlp.py) — a minimal 2-layer MLP (`Linear -> tanh -> Linear -> tanh`) trained end-to-end.

## Why?

A single linear layer can only represent linear functions of the input. Real-world data (images, text, tabular relationships) is rarely linearly separable, so we need a function approximator that can represent arbitrary non-linear mappings. Stacking linear layers with non-linearities between them gives a *universal function approximator* (in the limit of enough width/depth).

## How does it work?

**Forward pass.** Input $x$ flows through each layer:

$$
\begin{aligned}
z_1 &= W_1 x + b_1 \\
a_1 &= \text{activation}(z_1) \\
z_2 &= W_2 a_1 + b_2 \\
\hat{y} &= \text{activation}(z_2) \quad \text{(or identity for regression / logits)}
\end{aligned}
$$

**Loss.** Compare $\hat{y}$ to the target $y$ with a loss function (MSE for regression, cross-entropy for classification).

**Backward pass (backpropagation).** Using the chain rule, propagate the gradient of the loss backward through the graph, layer by layer:

- $\dfrac{\partial L}{\partial \hat{y}}$ is computed directly from the loss function.
- $\dfrac{\partial L}{\partial z_2} = \dfrac{\partial L}{\partial \hat{y}} \cdot \text{activation}'(z_2)$
- $\dfrac{\partial L}{\partial W_2} = \dfrac{\partial L}{\partial z_2}\, a_1^\top$, $\dfrac{\partial L}{\partial b_2} = \dfrac{\partial L}{\partial z_2}$
- $\dfrac{\partial L}{\partial a_1} = W_2^\top \dfrac{\partial L}{\partial z_2}$
- $\dfrac{\partial L}{\partial z_1} = \dfrac{\partial L}{\partial a_1} \cdot \text{activation}'(z_1)$
- ... and so on back to the input.

Each layer only needs to know the gradient flowing in from the layer *after* it and the local derivative of its own operation — that's the chain rule doing the heavy lifting. In PyTorch this whole process is automatic (`loss.backward()`); see [`tensors-autograd.md`](../pytorch/tensors-autograd.md) for how the underlying autograd engine builds and walks this graph.

**Weight update.** An optimizer (SGD, Adam, ...) uses the accumulated gradients to update each parameter: $W \leftarrow W - \text{lr} \cdot \dfrac{\partial L}{\partial W}$. See [`optimization-sgd-adam.md`](optimization-sgd-adam.md).

**Why activations must be non-linear.** If every activation were the identity (or any linear function), the whole network would collapse algebraically into a single linear transform: $W_2 (W_1 x) = (W_2 W_1) x = W_{\text{combined}} x$. No matter how many layers you stack, the network could only ever represent what a single linear layer represents. Non-linearities (ReLU, tanh, sigmoid, GELU, ...) are what let depth actually buy you extra representational power.

### The precise statement: backprop is the chain rule via transposed Jacobians

The hand-wavy "each layer only needs its local derivative and the upstream gradient" claim above has an exact statement behind it. Consider a composition $f = g \circ h$: $h$ maps a point $x_0$ to $h(x_0)$, and $g$ is applied after it. Write $D_{x_0}h$ for the Jacobian of $h$ at $x_0$ (a linear map). The first-order approximation of the composition near $x_0$ is:

$$
\begin{aligned}
[D_{h(x_0)}g]\big([D_{x_0}h](x - x_0)\big)
&= \left\langle \nabla_{h(x_0)}g,\ [D_{x_0}h](x - x_0) \right\rangle \\
&= \left\langle [D_{x_0}h]^\top \nabla_{h(x_0)}g,\ x - x_0 \right\rangle
\end{aligned}
$$

- First line: a directional derivative equals the inner product with the gradient.
- Second line: move the linear map $[D_{x_0}h]$ to the other operand of the inner product — that move is exactly what "transpose" means.

Matching this against the definition of the gradient of $f$ at $x_0$ gives:

$$\nabla_{x_0} f = [D_{x_0}h]^\top \nabla_{h(x_0)}g$$

In words: **the gradient flowing backward through a layer is that layer's local Jacobian, transposed, applied to the gradient flowing in from the layer after it.** ("Transpose" here is really the adjoint of the linear map $D_{x_0}h$; in finite dimensions with the standard inner product, the adjoint of a linear map is just its matrix transpose — which is why this is stated as "transpose the local Jacobian.") Every line in the informal derivation above is an instance of this — $\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot \text{activation}'(z)$ is $J^\top \nabla g$ for a diagonal (elementwise) Jacobian, and $\frac{\partial L}{\partial a_1} = W_2^\top \frac{\partial L}{\partial z_2}$ is $J^\top \nabla g$ for the Jacobian of a linear layer, which is just the weight matrix itself — hence the $W_2^\top$.

**The general backprop algorithm, stated precisely:**

1. **Forward pass**: compute and store every intermediate representation $x = x_0, x_1, \ldots, x_m = y$. These stored activations are needed again during step 2 — this is exactly why activation memory (not just parameter count) drives training memory usage.
2. **Backward pass**: compute every gradient by walking the graph in reverse (from the loss back to the inputs), applying $\nabla f = J^\top \nabla g$ at each node using that node's local Jacobian.
3. **Optimization step**: hand the computed gradients to an optimizer (SGD, Adam, ...) to update the parameters — see [`optimization-sgd-adam.md`](optimization-sgd-adam.md).

### Worked example: a small computational graph

Take input $X$ (shape $n \times D$ — $n$ examples, $D$ features), a first weight matrix $U$ ($D \times k$), an elementwise nonlinearity $g$, a second weight vector $W$ ($k \times 1$), a second elementwise nonlinearity $h$, and a loss $L$:

**Forward pass** (shapes annotated at each node):

$$
X_{(n\times D)} \xrightarrow{\ \cdot\, U_{(D\times k)}\ } XU_{(n\times k)} \xrightarrow{\ g\ } g(XU)_{(n\times k)} \xrightarrow{\ \cdot\, W_{(k\times 1)}\ } g(XU)W_{(n\times 1)} \xrightarrow{\ h\ } h(g(XU)W)_{(n\times 1)} \xrightarrow{\ L(\cdot,\,y)\ } \text{loss}_{(\text{scalar})}
$$

**Backward pass** — walk the same graph right to left. At each node, apply that node's local Jacobian, transposed, to the gradient arriving from its right — a matrix-multiply Jacobian transposes to the matrix transpose ($W^\top$, $U^\top$); an elementwise-nonlinearity Jacobian is diagonal, so "transpose and multiply" reduces to an elementwise product ($\odot$):

$$
\begin{aligned}
\nabla_{\hat{y}} L \\
\delta &:= \nabla_{\hat{y}} L \odot h'(g(XU)W) \\
\frac{\partial L}{\partial W} &= g(XU)^\top \delta \\
\frac{\partial L}{\partial [g(XU)]} &= \delta\, W^\top \\
\frac{\partial L}{\partial [XU]} &= \left[\delta\, W^\top\right] \odot g'(XU) \\
\frac{\partial L}{\partial U} &= X^\top \frac{\partial L}{\partial [XU]}
\end{aligned}
$$

- $\nabla_{\hat{y}} L$ — gradient of the loss w.r.t. $\hat{y} = h(g(XU)W)$
- $\delta$ — back through $h$: elementwise product with $h$'s local derivative ($n \times 1$)
- $\frac{\partial L}{\partial W}$ — back through "$\cdot W$": local Jacobian is $W$ itself → transpose to $g(XU)^\top$ ($k \times 1$, matches $W$)
- $\frac{\partial L}{\partial [g(XU)]}$ — back through "$\cdot W$" the other way: gradient w.r.t. the left operand ($n \times k$)
- $\frac{\partial L}{\partial [XU]}$ — back through $g$: elementwise product with $g$'s local derivative ($n \times k$)
- $\frac{\partial L}{\partial U}$ — back through "$\cdot U$": local Jacobian is $U$ itself → transpose to $X^\top$ ($D \times k$, matches $U$)

Every arrow above is literally $\nabla f = J^\top \nabla g$ from the previous section: $W^\top$ and $U^\top$ are exact transposes of the forward-pass weight matrices, and the $\odot\, g'(\cdot)$ / $\odot\, h'(\cdot)$ terms are the (diagonal) Jacobians of the elementwise nonlinearities — a diagonal matrix transposed is itself, which is why those steps collapse to plain elementwise multiplication.

## Activation functions reference

| Function | Formula | Notes |
|---|---|---|
| Sigmoid | $\dfrac{1}{1 + e^{-x}}$ | Saturates for large $\lvert x \rvert$ → vanishing gradient far from 0. Still the standard choice for binary-classification output layers (squashes to a $(0,1)$ probability). |
| ReLU | $\max(0, x)$ | No saturation for $x > 0$, cheap to compute and differentiate. Suffers from "dying ReLU": if a neuron's input is always negative, its gradient is always exactly 0 and it never updates again. |
| Leaky ReLU | $\max(\alpha x, x)$, with $0 < \alpha < 1$ a **fixed hyperparameter** | Fixes dying ReLU by giving a small nonzero gradient on the negative side; $\alpha$ is chosen ahead of time, not learned. |
| PReLU (Parametric ReLU) | $\max(\alpha x, x)$, with $0 < \alpha < 1$ | Same functional form as Leaky ReLU, but $\alpha$ is a **learned parameter**, trained via backprop like any weight (typically one $\alpha$ per channel) — the network picks its own negative-side slope instead of it being fixed. |
| Tanh | $\dfrac{e^x - e^{-x}}{e^x + e^{-x}}$ | Zero-centered (unlike sigmoid), but still saturates at both ends → vanishing gradient for large $\lvert x \rvert$. |

## Example

```python
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_hidden, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, num_hidden)
        self.layer2 = nn.Linear(num_hidden, output_dim)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        return x
```

Full runnable training loop: [`pytorch/code/simple_mlp.py`](../pytorch/code/simple_mlp.py).

## When to use

MLPs (fully-connected stacks) are the right default for tabular data and as the "head" on top of a feature extractor (CNN backbone, transformer encoder, etc.). For structured input (images, sequences) you generally want an architecture with an appropriate inductive bias instead — convolutions for spatial data ([`cnn-segmentation-unet.md`](../pytorch/cnn-segmentation-unet.md)), recurrence or attention for sequential data ([`rnn.md`](../pytorch/rnn.md), [`attention-transformers.md`](../attention-transformers.md)).

## Common interview questions

- **Derive backprop for a 2-layer network.** Be able to write out the chain-rule steps above without hand-waving.
- **Why do we need non-linear activations?** See above — linear layers collapse into one.
- **What causes vanishing/exploding gradients?** Repeated multiplication of derivatives through many layers; if those derivatives are consistently `< 1` (e.g. sigmoid saturating) the gradient shrinks toward zero, if consistently `> 1` it blows up. Mitigations: better activations (ReLU family), normalization (BatchNorm/LayerNorm), residual connections, careful initialization (Xavier/He), gradient clipping.
- **What does `.backward()` actually do?** Walks the dynamically-built computation graph in reverse topological order, applying the chain rule at each node, accumulating gradients into `.grad` on leaf tensors that have `requires_grad=True`.
- **Why do we zero gradients each step?** PyTorch *accumulates* gradients into `.grad` by default (useful for gradient accumulation across mini-batches); without `optimizer.zero_grad()`, gradients from the previous step would add into the current one.
- **What's the difference between a loss function and a cost function?** Loss is typically per-example; cost is the aggregate (e.g. mean) over a batch/dataset. Used interchangeably in casual conversation, but interviewers sometimes probe this distinction.

## Common mistakes

- Forgetting `optimizer.zero_grad()` before `loss.backward()`, silently accumulating gradients across steps.
- Using a linear (or no) activation between every layer — network can't learn non-linear decision boundaries.
- Applying the final activation *and* using a loss function that already expects logits (e.g. `Sigmoid()` + `BCEWithLogitsLoss` — `BCEWithLogitsLoss` wants raw logits, not probabilities; combine `Sigmoid()` only with plain `BCELoss`, or drop the `Sigmoid()` and use `BCEWithLogitsLoss` for numerical stability).
- Not calling `model.train()` / `model.eval()` around training vs. inference when the model has layers with different train/eval behavior (Dropout, BatchNorm) — see [`regularization-overfitting.md`](../regularization-overfitting.md).
