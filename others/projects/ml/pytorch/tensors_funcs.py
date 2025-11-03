# %%
import torch
# %%
a = torch.FloatTensor((2, 3))
a
# %%
a.mean()
# %%
a.max()
# %%
a.min()
# %%
a.sum()
# %%
a.max(dim=0)
# %%
a.max(dim=0).values
# %%
torch.log(a)
# %%
a.sin_()
# %%
torch.median(a)
# %%
torch.var(a)
# %%
torch.std(a)
# %%
torch.corrcoef(a)
# %%
torch.cov(a)
# %%
b = torch.FloatTensor((2, 3))
# %%
ab = torch.vstack([a,b])
ab
# %%
ab.corrcoef()
# %%
ab.cov()
# %%
