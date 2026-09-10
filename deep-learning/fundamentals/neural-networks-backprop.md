# Neural Networks & Backpropagation

## What is it?

A neural network is a stack of parameterized linear transformations (`Wx + b`) interleaved with non-linear activation functions. Backpropagation is the algorithm that computes the gradient of a loss function with respect to every parameter in the network, using the chain rule, so an optimizer can update those parameters.

Code reference: [`pytorch/code/simple_mlp.py`](../pytorch/code/simple_mlp.py) — a minimal 2-layer MLP (`Linear -> tanh -> Linear -> tanh`) trained end-to-end.

## Why?

A single linear layer can only represent linear functions of the input. Real-world data (images, text, tabular relationships) is rarely linearly separable, so we need a function approximator that can represent arbitrary non-linear mappings. Stacking linear layers with non-linearities between them gives a *universal function approximator* (in the limit of enough width/depth).

## How does it work?

**Forward pass.** Input `x` flows through each layer:

```
z1 = W1 @ x + b1
a1 = activation(z1)
z2 = W2 @ a1 + b2
y_hat = activation(z2)   # or identity for regression / logits
```

**Loss.** Compare `y_hat` to the target `y` with a loss function (MSE for regression, cross-entropy for classification).

**Backward pass (backpropagation).** Using the chain rule, propagate the gradient of the loss backward through the graph, layer by layer:

- `dL/dy_hat` is computed directly from the loss function.
- `dL/dz2 = dL/dy_hat * activation'(z2)`
- `dL/dW2 = dL/dz2 @ a1^T`, `dL/db2 = dL/dz2`
- `dL/da1 = W2^T @ dL/dz2`
- `dL/dz1 = dL/da1 * activation'(z1)`
- ... and so on back to the input.

Each layer only needs to know the gradient flowing in from the layer *after* it and the local derivative of its own operation — that's the chain rule doing the heavy lifting. In PyTorch this whole process is automatic (`loss.backward()`); see [`tensors-autograd.md`](../pytorch/tensors-autograd.md) for how the underlying autograd engine builds and walks this graph.

**Weight update.** An optimizer (SGD, Adam, ...) uses the accumulated gradients to update each parameter: `W -= lr * dL/dW`. See [`optimization-sgd-adam.md`](optimization-sgd-adam.md).

**Why activations must be non-linear.** If every activation were the identity (or any linear function), the whole network would collapse algebraically into a single linear transform: `W2 @ (W1 @ x) = (W2 @ W1) @ x = W_combined @ x`. No matter how many layers you stack, the network could only ever represent what a single linear layer represents. Non-linearities (ReLU, tanh, sigmoid, GELU, ...) are what let depth actually buy you extra representational power.

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
