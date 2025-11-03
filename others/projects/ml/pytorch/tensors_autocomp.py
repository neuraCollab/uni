# %%
import torch
# %%
tz = torch.zeros(2,3)
# %%
tz = torch.zeros(2,3, dtype=torch.int32)
# %%
tz
# %%
a = torch.eye(3,2)
a
# %%
t = torch.full((2,4), 5)
t
# %%
b = torch.arange(7, 10, 2)
b
# %%
f = torch.linspace(1, 5, 3)
f

# %%
f = torch.linspace(1, 5, 4)
f
# %%
j = torch.rand(3,3)
j
# %%
p = torch.randn(2,3)
p
# %%
o = torch.IntTensor(4,3).zero_() # меняет текущий при изменении, а 
                                # а не создает новый
o
# %%
o.fill_(3)
o
# %%
j.float()
j.uniform_(0,1)
# %%
x = torch.rand(5,2)
x
# %%
x = torch.arange(27)
# %%
d = x.view(3,9)
d # d and x use general tensor storage, but different views
# %%
r = x.reshape(3,3,3)
r
# x.resize(9,3)
# x
# %%
r.permute(2,1, 0)
# %%
d.mT
# %%
r.size()
# %%
r.squeeze(0)
# %%
r.size()
# %%
r.squeeze(1)
# %%
r.size()
# %%
