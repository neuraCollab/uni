# PyTorch Tensors & Autograd

## What is it?

`torch.Tensor` is PyTorch's core n-dimensional array type — like a numpy array, but it can live on a GPU and can track the operations performed on it so gradients can be computed automatically. **Autograd** is the engine that records those operations into a dynamic computation graph and walks it backward to compute gradients via the chain rule.

Code reference: [`pytorch/code/tensors_basics.py`](code/tensors_basics.py), [`pytorch/code/autograd_basics.py`](code/autograd_basics.py).

## Why?

Every training loop needs two things: a data structure to hold activations/parameters efficiently (on CPU or GPU), and a way to compute `dL/dparam` for every parameter without hand-deriving gradients for every architecture change. Autograd gives you that second part for free — write the forward pass, call `.backward()`, done.

## How does it work?

### Tensor creation, dtype, device

```python
t = torch.tensor([3, 5, 2])                     # dtype inferred (int64)
f64 = torch.tensor([2.0, 3.0], dtype=torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
on_device = torch.randn(3, 3, device=device)     # allocate directly on a device
moved = t.to(device)                             # or move an existing tensor
```

Tensors on different devices can't be combined in one op — move one first or you get a `RuntimeError`.

`torch.from_numpy(arr)` **shares memory** with the numpy array — mutating one mutates the other. `torch.tensor(data)` always **copies**. This is a classic interview gotcha:

```python
arr = np.array([[1, 2], [3, 4]])
t = torch.from_numpy(arr)
t[0, 0] = 5
assert arr[0, 0] == 5   # arr changed too - same underlying buffer
```

### Broadcasting

Two tensors are broadcastable if, comparing their shapes from the *trailing* dimension, every pair of dimensions is either equal or one of them is 1 (missing leading dims are treated as 1). Example:

```
A: (4, 1, 3)
B:    (2, 3)
```
Align from the right: `(3)` vs `(3)` — equal, OK. `(1)` vs `(2)` — one is 1, broadcasts to 2. `(4)` vs (missing, treated as 1) — broadcasts to 4. Result shape: `(4, 2, 3)`.

```python
p = torch.rand(4)
scalar_broadcast = p - 3   # scalar broadcasts across every element
```

### In-place vs out-of-place ops

Any op with a trailing underscore (`add_`, `mul_`, `fill_`, ...) mutates the tensor **in place**, no new memory allocated. The equivalent without the underscore (`add`, `mul`, `+`) returns a **new** tensor, leaving the original untouched.

```python
p.add(q)     # out-of-place: returns a new tensor
p.add_(q)    # in-place: mutates p directly
```

**Autograd gotcha:** an in-place op on a *leaf* tensor that `requires_grad=True` raises `RuntimeError: a leaf Variable that requires grad is being used in an in-place operation`. PyTorch refuses this because it would silently invalidate values the graph needs to compute gradients during `backward()`. Even on non-leaf tensors, in-place ops can corrupt values needed later in the backward pass and raise similar errors.

```python
leaf = torch.tensor([1.0, 2.0], requires_grad=True)
# leaf.add_(1.0)          # RuntimeError
with torch.no_grad():
    leaf.add_(1.0)         # fine - mutation explicitly excluded from tracking
```

### requires_grad, backward(), .grad

```python
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([-4.0], requires_grad=True)

f = (x + y) ** 2 + 2 * x * y
f.backward()          # computes df/dx and df/dy, accumulates into .grad

print(x.grad)          # df/dx = 2(x+y) + 2y
print(y.grad)          # df/dy = 2(x+y) + 2x
```

- `requires_grad=True` marks a tensor as a graph "leaf" whose gradient should be tracked.
- `.backward()` only works directly on a **scalar** output (otherwise you must pass a gradient tensor matching the output's shape — a vector-Jacobian product).
- `.grad` **accumulates** across multiple `.backward()` calls. You must call `optimizer.zero_grad()` (or set `tensor.grad = None`) between training steps, or gradients from the previous step add into the current one.

### `torch.no_grad()` and `.detach()`

```python
with torch.no_grad():
    z = x * y + 1        # not recorded - no graph built, no memory for backward

detached = f.detach()     # same values, severed from the graph - new leaf, requires_grad=False
```

Both exist to avoid unnecessary graph-building at inference time: building the graph costs memory (every intermediate activation needed for backward is kept alive) and compute. During evaluation/inference you never call `.backward()`, so tracking is pure waste — always wrap inference in `torch.no_grad()`, and use `.detach()` when you need a tensor's *values* without dragging along its history (e.g. logging a loss value, or feeding a tensor into a second model you don't want gradients flowing back through).

## Example

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([-4.0], requires_grad=True)
f = (x + y) ** 2 + 2 * x * y
f.backward()
print(x.grad, y.grad)
```

Full runnable versions: [`pytorch/code/tensors_basics.py`](code/tensors_basics.py) (creation/indexing/broadcasting/matmul/linalg), [`pytorch/code/autograd_basics.py`](code/autograd_basics.py) (scalar autograd + the in-place gotcha).

## When to use

Every PyTorch model, always. The one real choice is remembering to disable graph-tracking (`no_grad`/`detach`/`model.eval()`) whenever you're not calling `.backward()` — training vs. inference.

## Common interview questions

- **Why does `torch.from_numpy` share memory while `torch.tensor` doesn't?** `from_numpy` wraps the existing numpy buffer directly (zero-copy); `torch.tensor(...)` always allocates a fresh buffer and copies the data in.
- **What's the difference between `.view()` and `.reshape()`?** `.view()` requires the tensor to be contiguous-compatible and never copies; `.reshape()` behaves like `.view()` when possible but silently falls back to a copy otherwise.
- **Why can't you call `.backward()` on a non-scalar tensor without arguments?** Because `.backward()` computes a vector-Jacobian product; for a non-scalar output you must supply the "upstream gradient" tensor (same shape as the output) to contract the Jacobian with.
- **Why do in-place ops break autograd on leaf tensors?** They'd overwrite values in the same memory that the graph needs for computing gradients during the backward pass; PyTorch raises rather than silently giving wrong gradients.
- **What's the practical difference between `torch.no_grad()` and `.detach()`?** `no_grad()` is a context manager disabling graph tracking for a block of code; `.detach()` returns a new tensor (sharing storage) severed from the graph. Use `no_grad()` around whole inference blocks, `.detach()` for pulling one tensor out of a graph mid-computation.

## Common mistakes

- Forgetting `optimizer.zero_grad()` and having gradients silently accumulate across steps.
- Running inference without `torch.no_grad()` — wastes memory building a graph that's never used for `.backward()`, and can OOM on large models.
- Calling an in-place op on a leaf `requires_grad=True` tensor and being confused by the `RuntimeError`.
- Mixing tensors on different devices (CPU vs GPU) in the same op.
- Assuming `.view()` always works — it fails on non-contiguous tensors (e.g. after `.permute()`) where `.reshape()` (or `.contiguous().view()`) is needed instead.
