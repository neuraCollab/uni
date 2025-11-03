# %%
import torch
# %%
a = torch.rand(22)
a
# %%
a - 3
# %%
a
a.shape
# %%
b = torch.zeros(22).fill_(1)
b
# %%
a + b
# %%
a.add(b)
# %%
a.add_(b)
# %%
a
# %%
