# Tensor basics: creation, dtypes, indexing, broadcasting, in-place ops,
# matmul/bmm/dot/outer, and a few linalg utilities.
# Consolidated + lightly cleaned up from the archive's tensors_*.py cell scripts
# (tensors_intro, tensors_funcs, tensors_indexes, tensors_operations,
# tensors_vectorMatrix, backward's creation cells). Run cell-by-cell in
# VS Code / Jupyter (each `# %%` is a separate cell) or top-to-bottom as a script.

# %% Creation & dtypes
import torch
import numpy as np

t = torch.tensor([3, 5, 2])                       # inferred dtype (int64)
e = torch.empty(3)                                 # uninitialized memory - garbage values
f64 = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)

data = [[1, 2, 3], [2, 3, 4], [312, 12, 321]]
t_int32 = torch.tensor(data, dtype=torch.int32)

# torch.from_numpy SHARES memory with the numpy array - mutating one mutates
# the other. This is a classic interview gotcha (unlike torch.tensor(), which copies).
arr = np.array(data)
t_from_np = torch.from_numpy(arr)
t_from_np[0, 0] = 5
assert arr[0, 0] == 5  # arr changed too - same underlying buffer

t_int32_float = t_int32.float()  # .float()/.double()/.long() etc always COPY

# %% Other constructors
zeros = torch.zeros(2, 3, dtype=torch.int32)
identity = torch.eye(3, 2)                # non-square "identity-like" matrix
filled = torch.full((2, 4), 5)
ranged = torch.arange(7, 10, 2)           # like Python range: start, stop, step
lin = torch.linspace(1, 5, 4)             # 4 points evenly spaced in [1, 5], inclusive
uniform = torch.rand(3, 3)                # U(0, 1)
normal = torch.randn(2, 3)                # N(0, 1)

# in-place constructors mutate the tensor that already exists rather than
# allocating a new one - useful for reusing pre-allocated buffers
buf = torch.IntTensor(4, 3).zero_()
buf.fill_(3)

# %% Views, reshape, permute, squeeze
x = torch.arange(27)
view = x.view(3, 9)          # view SHARES storage with x - no copy, must be contiguous-compatible
reshaped = x.reshape(3, 3, 3)  # like view, but falls back to a copy if a view isn't possible
permuted = reshaped.permute(2, 1, 0)  # reorders axes, does NOT move data (returns a view)
transposed_last2 = view.mT     # shorthand for transposing the last two dims

squeezed = reshaped.squeeze(0)   # drops a size-1 dim at index 0 (no-op here since dim 0 has size 3)

# %% Indexing & slicing (same semantics as numpy)
a = torch.arange(12)
a[2]              # single element -> 0-dim tensor
a[-2] = 100        # negative indexing + in-place assignment
b = a[2:4:1]       # slice: start:stop:step
mask_select = torch.randn(3, 3)
positive_only = mask_select[mask_select > 0.7]   # boolean mask indexing -> 1D tensor of matches

# %% Broadcasting + in-place vs out-of-place ops
p = torch.rand(4)
q = torch.zeros(4).fill_(1)

sum_new = p + q          # out-of-place: allocates a new tensor
sum_new2 = p.add(q)       # same as above, functional form
p.add_(q)                 # IN-PLACE (trailing underscore convention): mutates p directly, no new allocation
                           # Caution: in-place ops on a leaf tensor with requires_grad=True
                           # can raise "a leaf Variable that requires grad is being used in
                           # an in-place operation" or silently corrupt values needed for
                           # backward() - see autograd_basics.py for details.

scalar_broadcast = p - 3  # scalar is broadcast across every element

# %% Matrix / vector / batch ops
m1 = torch.arange(1, 10).view(3, 3)
m2 = torch.arange(10, 19).view(3, 3)

elementwise = m1 * m2          # Hadamard product, NOT matrix multiplication
matmul = torch.matmul(m1, m2)  # true matrix multiplication
matmul2 = m1.mm(m2)            # .mm is matmul restricted to 2D tensors

batch1 = torch.randn(7, 3, 5)
batch2 = torch.randn(7, 5, 4)
batched = torch.bmm(batch1, batch2)   # batched matmul: (B,n,m) @ (B,m,p) -> (B,n,p)

v1 = torch.arange(3).random_(4, 5)
v2 = torch.arange(3).random_(1, 3)
dot = torch.dot(v1, v2)        # 1D . 1D -> scalar
dot_op = v1 @ v2               # @ operator dispatches to matmul/dot/mm depending on rank
outer = torch.outer(v1, v2)    # outer product -> matrix

mat_vec = m1.mv(v1)            # matrix-vector product

# %% A little linalg
rank = torch.linalg.matrix_rank(m1.float())
solved = torch.linalg.solve(m1.float(), v1.float())  # solves m1 @ x = v1 (needs m1 non-singular)
inv = torch.linalg.inv(m1.float())

# %% Aggregate stats
sample = torch.FloatTensor((2, 3))
sample.mean(); sample.max(); sample.min(); sample.sum()
sample.max(dim=0).values      # max along a dim returns a (values, indices) namedtuple
torch.median(sample); torch.var(sample); torch.std(sample)

stacked = torch.vstack([sample, torch.FloatTensor((2, 3))])
stacked.corrcoef()  # correlation matrix between rows
stacked.cov()        # covariance matrix between rows

# %% CPU <-> GPU device placement
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
on_device = torch.randn(3, 3, device=device)   # allocate directly on a device
moved = m1.to(device)                          # or move an existing tensor
back_to_cpu = moved.cpu()
# Tensors on different devices cannot be combined in an op directly - you must
# move one of them first, or you'll get a RuntimeError.
