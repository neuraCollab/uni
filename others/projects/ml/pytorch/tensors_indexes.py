# %%
import torch
# %%
a = torch.arange(12)
a
# %%
a[2]
# %%
a[-2]
# %%
a[-3].item()
# %%
a[-2] = 100
# %%
b = a[2:4:1]
b
# %%
c = b[0:-1:1]
c
# %%
d = a[:-1]
d
# %%
k = a[-1:]
k
# %%
a[:4] = torch.empty(4)
a

# %%
p = torch.randn(3,3)
p[:, 1:2]
# %%
p[p>0.7]
# %%
