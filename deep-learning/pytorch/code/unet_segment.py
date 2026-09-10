# Full U-Net for binary image segmentation: encoder/decoder with skip
# connections, transposed-conv upsampling, a combined BCE + Soft Dice loss,
# training loop, checkpointing, and inference + visualization.
#
# Ported from the archive's unet_segment.py (the strongest single file in the
# source material) with light cleanup. This is DATASET-DEPENDENT by nature of
# being a segmentation example - it expects a folder `dataset_seg/images` and
# `dataset_seg/masks` with matching filenames, and an inference image
# `car_1.jpg`. Swap in your own paths to run it; the point is the architecture
# and training pattern, not this specific dataset.

# %%
import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs_v2
import torch.nn as nn
import torch.optim as optim


# %% Dataset: loads (image, mask) pairs, binarizes the mask
class SegmentDataset(data.Dataset):
    def __init__(self, path, transform_img=None, transform_mask=None):
        self.path = path
        self.transform_img = transform_img
        self.transform_mask = transform_mask

        images_dir = os.path.join(self.path, 'images')
        list_files = os.listdir(images_dir)
        self.length = len(list_files)
        self.images = [os.path.join(images_dir, f) for f in list_files]

        masks_dir = os.path.join(self.path, 'masks')
        list_files = os.listdir(masks_dir)
        self.masks = [os.path.join(masks_dir, f) for f in list_files]

    def __getitem__(self, item):
        path_img, path_mask = self.images[item], self.masks[item]
        img = Image.open(path_img).convert('RGB')
        mask = Image.open(path_mask).convert('L')  # grayscale

        if self.transform_img:
            img = self.transform_img(img)

        if self.transform_mask:
            mask = self.transform_mask(mask)
            # binarize: anything below 250 becomes foreground (1), else background (0)
            mask[mask < 250] = 1
            mask[mask >= 250] = 0

        return img, mask

    def __len__(self):
        return self.length


# %% U-Net architecture
class UNetModel(nn.Module):
    class _TwoConvLayers(nn.Module):
        """Conv -> ReLU -> BN, twice. The basic building block used at every
        resolution level of both the encoder and the decoder."""
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
            )

        def forward(self, x):
            return self.model(x)

    class _EncoderBlock(nn.Module):
        """Convs at this resolution, then downsample. Returns BOTH the
        downsampled tensor (fed to the next encoder block) and the
        pre-downsample tensor (kept for the matching decoder's skip connection)."""
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.block = UNetModel._TwoConvLayers(in_channels, out_channels)
            self.max_pool = nn.MaxPool2d(2)

        def forward(self, x):
            x = self.block(x)
            y = self.max_pool(x)
            return y, x

    class _DecoderBlock(nn.Module):
        """Upsample via transposed conv, concatenate with the encoder's skip
        tensor at the same resolution, then run convs to fuse them."""
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.transpose = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
            self.block = UNetModel._TwoConvLayers(in_channels, out_channels)

        def forward(self, x, skip):
            x = self.transpose(x)
            u = torch.cat([x, skip], dim=1)  # channel-wise concat: this IS the skip connection
            u = self.block(u)
            return u

    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.enc_block1 = self._EncoderBlock(in_channels, 64)
        self.enc_block2 = self._EncoderBlock(64, 128)
        self.enc_block3 = self._EncoderBlock(128, 256)
        self.enc_block4 = self._EncoderBlock(256, 512)

        self.bottleneck = self._TwoConvLayers(512, 1024)

        self.dec_block1 = self._DecoderBlock(1024, 512)
        self.dec_block2 = self._DecoderBlock(512, 256)
        self.dec_block3 = self._DecoderBlock(256, 128)
        self.dec_block4 = self._DecoderBlock(128, 64)

        self.out = nn.Conv2d(64, num_classes, 1)  # 1x1 conv -> per-pixel logits

    def forward(self, x):
        x, skip1 = self.enc_block1(x)
        x, skip2 = self.enc_block2(x)
        x, skip3 = self.enc_block3(x)
        x, skip4 = self.enc_block4(x)

        x = self.bottleneck(x)

        x = self.dec_block1(x, skip4)
        x = self.dec_block2(x, skip3)
        x = self.dec_block3(x, skip2)
        x = self.dec_block4(x, skip1)

        return self.out(x)  # raw logits - pair with BCEWithLogitsLoss, not BCELoss


# %% Loss: Soft Dice (handles pixel-class imbalance) combined with BCE
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth  # avoids division by zero when a mask is empty

    def forward(self, logits, targets):
        num = targets.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = m1 * m2

        score = 2 * (intersection.sum(1) + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        score = 1 - score.sum() / num
        return score


# %% Data pipeline
tr_img = tfs_v2.Compose([tfs_v2.ToImage(), tfs_v2.ToDtype(torch.float32, scale=True)])
tr_mask = tfs_v2.Compose([tfs_v2.ToImage(), tfs_v2.ToDtype(torch.float32)])

d_train = SegmentDataset(r"dataset_seg", transform_img=tr_img, transform_mask=tr_mask)
train_data = data.DataLoader(d_train, batch_size=2, shuffle=True)

model = UNetModel()

optimizer = optim.RMSprop(params=model.parameters(), lr=0.001)
loss_bce = nn.BCEWithLogitsLoss()
loss_dice = SoftDiceLoss()

# %% Training loop
epochs = 10
model.train()

for _e in range(epochs):
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=True)
    for x_train, y_train in train_tqdm:
        predict = model(x_train)
        loss = loss_bce(predict, y_train) + loss_dice(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{_e + 1}/{epochs}], loss_mean={loss_mean:.3f}")

st = model.state_dict()
torch.save(st, 'model_unet_seg.tar')

# %% Load checkpoint back
st = torch.load('model_unet_seg.tar', weights_only=False)
model.load_state_dict(st)

# %% Inference + visualization
model.eval()
img = Image.open(r"car_1.jpg").convert('RGB')
img = tr_img(img).unsqueeze(0)

with torch.no_grad():
    p = model(img).squeeze(0)
    x = torch.sigmoid(p.permute(1, 2, 0))

x = x.numpy() * 255
x = np.clip(x, 0, 255).astype('uint8')
plt.imshow(x, cmap='gray')
plt.show()
