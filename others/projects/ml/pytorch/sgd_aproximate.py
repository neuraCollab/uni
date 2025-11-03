# %%
import torch
from random import randint
# %%
def model(X, w):
    return X @ w
# %%
N = 2
w = torch.FloatTensor(N).uniform_(-1e-5, 1e-5)
w.requires_grad_(True)
x = torch.arange(0,3,0.1)
# %%
# not complited