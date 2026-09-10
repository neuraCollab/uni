# Autograd basics: requires_grad, .backward(), .grad, and a tiny from-scratch
# forward-pass example wired up with a hand-picked weight matrix.
# Consolidated from the archive's backward.py (autograd) and tensors_autocomp.py
# (a manual 2-layer perceptron forward pass - kept here as a "what would autograd
# be differentiating" illustration, with grad tracking added).

# %% Scalar autograd: requires_grad, backward(), .grad
import torch

x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([-4.0], requires_grad=True)

f = (x + y) ** 2 + 2 * x * y
f.backward()  # computes d(f)/d(x) and d(f)/d(y), accumulates them into .grad

print(x.data, x.grad)   # df/dx = 2(x+y) + 2y
print(y.data, y.grad)   # df/dy = 2(x+y) + 2x

# Key points for interviews:
# - requires_grad=True marks a tensor as a "leaf" that needs its gradient tracked.
# - .backward() only works on a scalar output (or you must pass a gradient tensor
#   matching the output's shape, for vector-Jacobian products).
# - .grad ACCUMULATES across multiple .backward() calls - you must call
#   optimizer.zero_grad() (or tensor.grad = None) between training steps, or
#   gradients from previous steps will add into the current ones.

# %% no_grad() / detach() for inference - stop tracking the graph
with torch.no_grad():
    # operations here are not recorded, so no graph is built and no memory
    # is spent on intermediate activations needed for backward(). Standard
    # practice for evaluation/inference.
    z = x * y + 1

detached = f.detach()  # same values, but severed from the graph - a new leaf
                        # with requires_grad=False. Useful when you need a
                        # tensor's values without dragging its whole history along.

# %% A tiny manual "forward pass" you could imagine autograd differentiating
# (original archive version used plain floats/threshold activations with no
# grad tracking at all - rewritten here so it's differentiable end-to-end).
def act(x):
    # hard threshold - NOT differentiable (zero gradient almost everywhere).
    # Kept only to mirror the original toy example; a real network would use
    # a smooth activation like sigmoid/ReLU/tanh instead.
    return (x >= 0.5).float()

def perceptron(house, rock, attr):
    X = torch.tensor([house, rock, attr], dtype=torch.float32)
    Wh = torch.tensor([[0.3, 0.3, 0.0], [0.4, -0.5, 1.0]])
    Wout = torch.tensor([-1.0, 1.0])

    Zh = torch.mv(Wh, X)
    Uh = act(Zh)
    Zout = torch.dot(Wout, Uh)
    return act(Zout)

out = perceptron(house=1, rock=0, attr=1)
print("yes" if out == 1 else "no")

# %% In-place ops + autograd: the classic gotcha
leaf = torch.tensor([1.0, 2.0], requires_grad=True)
# leaf.add_(1.0)  # -> RuntimeError: a leaf Variable that requires grad is
                  # being used in an in-place operation. PyTorch refuses this
                  # because it would silently invalidate values the graph
                  # needs for backward(). Use out-of-place ops on leaves that
                  # require grad (leaf = leaf + 1.0), or wrap the mutation in
                  # torch.no_grad() if you really mean to update it in place
                  # (e.g. manual SGD without an optimizer).
with torch.no_grad():
    leaf.add_(1.0)  # fine: the mutation is explicitly excluded from tracking
