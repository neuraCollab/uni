# CNNs & U-Net Segmentation

## What is it?

A Convolutional Neural Network (CNN) applies learned filters that slide over an image, exploiting spatial locality and weight-sharing. **U-Net** is an encoder-decoder CNN architecture originally designed for biomedical image segmentation, notable for its **skip connections** between matching encoder/decoder resolutions — it's the standard architecture for pixel-wise (dense) prediction tasks.

Code reference: [`pytorch/code/unet_segment.py`](code/unet_segment.py) — full encoder/decoder with skip connections, transposed-conv upsampling, combined BCE + Soft Dice loss, training loop, checkpointing, inference.

## Why?

Image classification only needs a single label per image, so a CNN can progressively downsample and throw away spatial resolution. **Segmentation** needs a label *per pixel* — you can't afford to lose spatial precision. U-Net solves this by keeping a symmetric decoder that upsamples back to full resolution, while skip connections from the encoder recover the fine spatial detail that downsampling destroyed.

## How does it work?

### Conv/pooling/stride/padding basics

- **Convolution**: a small learned kernel (e.g. 3x3) slides across the input, computing a weighted sum at each position — the same weights are reused everywhere (weight sharing), which is what gives CNNs translation-invariance and far fewer parameters than a fully-connected layer over the same input.
- **Padding**: adding zeros around the input border so the output spatial size doesn't shrink after convolution (`padding=1` with a 3x3 kernel preserves size — `unet_segment.py`'s `_TwoConvLayers` uses exactly this).
- **Stride**: how many pixels the kernel moves per step; stride > 1 downsamples.
- **Pooling** (e.g. `MaxPool2d(2)`): a non-learned downsampling operation, taking the max (or average) over small windows — halves spatial resolution while keeping the strongest activations.

### Receptive field

The **receptive field** of a unit is the region of the original input that can influence its value. A single 3x3 conv sees a 3x3 patch. Stack a second 3x3 conv on top and each unit now indirectly "sees" a 5x5 patch of the original input — receptive field grows with depth. This is why stacking small convolutions (rather than using one huge kernel) is standard: it's cheaper (fewer parameters), adds more non-linearities (better representational power), and still grows the effective receptive field to cover large context by the time you reach deeper layers — essential for tasks like segmentation where a pixel's correct label can depend on distant context.

### Why U-Net's skip connections matter

The encoder downsamples repeatedly (`_EncoderBlock` runs convs then `MaxPool2d(2)`), building up a small, highly-abstract, semantically-rich feature map — good for "what is in this image" but with badly degraded spatial precision (a 512x512 image might be down to 32x32 by the bottleneck). If the decoder only had that bottleneck to work from, upsampling back to 512x512 would produce blurry, spatially-imprecise masks — the fine boundary details were destroyed by pooling and can't be recovered from a low-resolution representation alone.

**Skip connections** fix this: each encoder block returns *both* the downsampled tensor (fed deeper) and the pre-downsample tensor (kept for later). The matching decoder block concatenates that saved high-resolution tensor with its own upsampled output before further convs:

```python
def forward(self, x, skip):
    x = self.transpose(x)                  # upsample
    u = torch.cat([x, skip], dim=1)        # skip connection: channel-wise concat
    u = self.block(u)
    return u
```

This gives the decoder access to precise spatial information (from the skip) *and* rich semantic information (from the upsampled path) at every resolution level — exactly what's needed for pixel-accurate boundaries.

### Encoder-decoder architectures generally

U-Net is one instance of a broader pattern: compress the input into an information-dense bottleneck (encoder), then reconstruct a full-resolution output from it (decoder). The same skeleton shows up in autoencoders/[VAEs](vae.md), image-to-image translation, and semantic segmentation — what differs is the loss, whether there's a skip connection, and whether the bottleneck is deterministic or probabilistic.

### Dice loss vs BCE for segmentation

`unet_segment.py` combines both:

```python
loss = loss_bce(predict, y_train) + loss_dice(predict, y_train)
```

- **BCE (binary cross-entropy)**, applied per-pixel, treats every pixel as an independent binary classification problem. Simple and well-behaved for optimization, but under **class imbalance** — e.g. a small foreground object on a large background, very common in segmentation — the loss is dominated by the (easy, numerous) background pixels, and the model can get a deceptively low loss while doing a poor job on the (rare, hard) foreground.
- **Dice loss** directly optimizes the **overlap** between prediction and ground truth (it's a differentiable relaxation of the Dice coefficient, $\dfrac{2\lvert A \cap B \rvert}{\lvert A \rvert + \lvert B \rvert}$), which is invariant to the *size* imbalance between foreground and background — a missed 5-pixel object hurts the Dice score just as much proportionally as a missed 500-pixel object. This makes it much better suited to imbalanced segmentation masks than BCE alone.

```python
class SoftDiceLoss(nn.Module):
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        m1, m2 = probs.view(num, -1), targets.view(num, -1)
        intersection = m1 * m2
        score = 2 * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        return 1 - score.sum() / num
```

Combining BCE + Dice is a common practical recipe: BCE gives smooth, well-conditioned per-pixel gradients early in training; Dice pushes directly toward better region overlap and compensates for class imbalance.

Note the model outputs raw logits (`self.out = nn.Conv2d(64, num_classes, 1)`, a 1x1 conv producing per-pixel logits), paired with `BCEWithLogitsLoss` rather than `Sigmoid()` + `BCELoss` — for numerical stability (see the common-mistakes note in [`neural-networks-backprop.md`](../fundamentals/neural-networks-backprop.md)).

## Example

U-Net's overall shape (4 encoder blocks, a bottleneck, 4 decoder blocks with skip connections, a final 1x1 conv):

```python
class UNetModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.enc_block1 = self._EncoderBlock(in_channels, 64)
        ...
        self.bottleneck = self._TwoConvLayers(512, 1024)
        self.dec_block1 = self._DecoderBlock(1024, 512)
        ...
        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x, skip1 = self.enc_block1(x)
        ...
        x = self.bottleneck(x)
        x = self.dec_block1(x, skip4)   # skip connections wire encoder -> decoder
        ...
        return self.out(x)
```

Full runnable training + inference pipeline: [`pytorch/code/unet_segment.py`](code/unet_segment.py).

## When to use

Semantic/instance segmentation, medical imaging (U-Net's original domain), any dense pixel-prediction task (depth estimation, image restoration). Not needed for plain classification, where you only need the encoder half plus a pooled head — see [`transfer-learning.md`](transfer-learning.md) for that pattern with a ResNet backbone.

## Common interview questions

- **Why does U-Net need skip connections instead of just a deeper decoder?** Downsampling in the encoder destroys spatial precision that no amount of decoder depth alone can recover — skip connections re-inject it directly.
- **Why combine BCE and Dice loss instead of using just one?** BCE gives stable per-pixel gradients but is dominated by class imbalance; Dice directly targets overlap and is imbalance-robust but can have less stable gradients early on. Combining gets the benefits of both.
- **What does receptive field mean and why does it grow with depth?** The region of input pixels that can influence a given unit's activation; each additional conv layer extends how far information can propagate, so stacking layers lets deep units "see" large image regions cheaply.
- **Why use transposed convolutions (or upsampling + conv) in the decoder instead of just resizing?** Transposed convs are *learned* upsampling — the network can learn how to best fill in detail, rather than relying on a fixed interpolation rule.
- **What's the difference between semantic and instance segmentation?** Semantic segmentation labels every pixel by class (all cars get the same label); instance segmentation additionally distinguishes separate object instances of the same class.

## Common mistakes

- Using plain BCE alone on a heavily imbalanced mask and being surprised the model "cheats" by predicting mostly background.
- Applying `Sigmoid()` and then `BCEWithLogitsLoss` (which expects raw logits) — double-applies the sigmoid and destabilizes training.
- Forgetting `model.eval()` before inference — `unet_segment.py`'s encoder/decoder blocks use `BatchNorm2d`, which behaves differently in train vs. eval mode (see [`regularization-overfitting.md`](../regularization-overfitting.md)).
- Mismatched spatial sizes between an encoder skip tensor and the corresponding decoder tensor when input dimensions aren't a clean power of two relative to the number of downsampling steps — a common source of `torch.cat` shape errors.
