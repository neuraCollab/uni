# %% 
import torch 
import torch.nn as nn
import torchvision.transforms.v2 as tfs
import torchvision.models as models
# %%
# просто пример как использовать resNet на своих данных (для 10 классов вместо 1000)
resnet_weights = models.ResNet50_Weights.DEFAULT
transforms = resnet_weights.transforms()

model = models.resnet50(resnet_weights)
model.requires_grad_(False)

model.fc = nn.Linear(512*4, 10)
model.fc.requires_grad_(True)
