# %%
import torch
# %%
a = torch.arange(1, 10).view(3,3)
b = torch.arange(10, 19).view(3,3)
# %%
r1 = a * b
r1
# %%
c = torch.matmul(a,b)
c # multiple like matrix not elements
# %%
c = a.mm(b)
c
# %%
bx = torch.randn(7,3,5)
by = torch.randn(7,5,4)
# %%
bc = torch.bmm(bx,by)
bc.size()
# %%
a = torch.arange(3).random_(4,5)
b = torch.arange(3).random_(1,3)
# %%
c = torch.dot(a, b)
c
# %%
a @ b
# %%
c = torch.outer(a,b)
c
# %%
v = torch.arange(1, 10).view(3,3)
i = torch.arange(10, 19).view(3,3)
v
# %%
r = v.mv(a)
r
# %%
# torch.linalg.matrix_rank(v.float(), a)
torch.linalg.matrix_rank(v.float())
# %%
torch.linalg.solve(v.float(), a.float()) # Ну кароч не сингулярная
# %%
invV = torch.linalg.inv(v.float())
# %%
