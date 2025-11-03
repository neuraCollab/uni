# %% 
import torch
# %%
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([-4.0], requires_grad=True)

# %%
f = (x + y) ** 2 + 2 *x * y
f.backward()
# %%
f
# %%
print(x.data, x.grad)
print(y.data, y.grad)
# %%
