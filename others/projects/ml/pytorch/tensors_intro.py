# %%
import torch

# %%
t = torch.tensor([3,5,2])
e = torch.empty(3)


# %%
e.dtype
# %%
torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
# %%
d = [[1,2,3], [2,3,4], [312, 12,321]]

t_d = torch.tensor(d, dtype=torch.int32)
# %%
import numpy as np

arr = np.array(d)

t_arr = torch.from_numpy(arr)
# %%
t_arr
# %%
t_arr[0,0] = 5
# %%
t_arr
# %%
t_d[0,0] = 433
# %%
t_d
# %%
tf = t_d.float()
# %%
tf
# %%
