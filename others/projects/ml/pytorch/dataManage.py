# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data 
# %%
class MyDataset(data.Dataset):
    def __init__(self, data):
        pass
    def __getitem__(self, index):
        return super().__getitem__(index)
    def __len__(self):
        pass
    
# %%
d_train = MyDataset([])

# %%
x_i, y_i = d_train[0]
data_sz = len(d_train)
# %%
# Dataloader сначала перемешиывает датасет, а потом разбивает на batch"и
train_data = data.DataLoader(d_train, 1, shuffle=True, drop_last=False)

# for x_train, y_train in train_data: