# Transfer learning pattern: load an ImageNet-pretrained ResNet50, freeze the
# backbone, replace the classification head for a new number of classes.
# Ported from the archive's transfer_learning.py.

# %%
import torch.nn as nn
import torchvision.models as models

# %% Load pretrained weights + the transforms they were trained with
resnet_weights = models.ResNet50_Weights.DEFAULT
transforms = resnet_weights.transforms()  # use this to preprocess your own
                                           # images so they match ImageNet
                                           # normalization/resizing exactly

model = models.resnet50(weights=resnet_weights)

# %% Freeze the whole backbone - no gradients computed for these params
model.requires_grad_(False)

# %% Replace the head for 10 classes instead of ImageNet's 1000
# ResNet50's final feature dim is 512 * 4 (bottleneck block expansion factor)
model.fc = nn.Linear(512 * 4, 10)
model.fc.requires_grad_(True)  # newly created layers require grad by default,
                                # but this line makes the intent explicit

# From here: train normally with an optimizer that only receives
# model.fc.parameters() (or filter(lambda p: p.requires_grad, model.parameters())),
# since the frozen backbone params don't need an optimizer state at all.

# %% Partial unfreezing (fine-tuning) - common follow-up step
# Once the new head has converged, unfreeze the last block(s) for a low-LR
# fine-tuning pass, e.g.:
# for param in model.layer4.parameters():
#     param.requires_grad_(True)
# Then use a much smaller learning rate for these than you'd use from scratch,
# since the pretrained weights are already close to a good optimum and large
# updates would destroy that.
