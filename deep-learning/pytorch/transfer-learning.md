# Transfer Learning

## What is it?

Reusing a model already trained on a large dataset (typically ImageNet for vision) as the starting point for a new, usually smaller, task — instead of training a new architecture from randomly-initialized weights.

Code reference: [`pytorch/code/transfer_learning.py`](code/transfer_learning.py) — ImageNet-pretrained ResNet50, frozen backbone, replaced classification head.

## Why?

Training a large CNN from scratch requires a huge labeled dataset and a lot of compute. Most task-specific datasets are far smaller than ImageNet. Transfer learning works because the **low-level features learned by early/middle layers of a CNN trained on a large, diverse image dataset — edges, textures, color blobs, simple shapes — are generic** and transfer across visual domains; only the later, more task-specific layers need to be relearned (or replaced) for a new task.

## How does it work?

### Load pretrained weights

```python
resnet_weights = models.ResNet50_Weights.DEFAULT
transforms = resnet_weights.transforms()   # use to preprocess inputs identically
                                            # to how the pretrained model was trained
model = models.resnet50(weights=resnet_weights)
```

Using the weights' own `.transforms()` matters — the pretrained backbone expects inputs normalized/resized exactly the way it was trained (ImageNet mean/std, specific resize/crop), and mismatching this quietly degrades performance.

### Freeze vs. fine-tune vs. partial unfreeze

- **Freeze the backbone**: `model.requires_grad_(False)` — no gradients computed for any backbone parameter; only the new head is trained. Cheapest, fastest, least prone to overfitting on a small dataset.
- **Fine-tune**: unfreeze some or all backbone layers and continue training them (usually at a much lower learning rate) alongside the new head.
- **Partial unfreezing**: a middle ground — freeze most of the backbone but unfreeze the last block(s) closest to the output, since those layers encode the most task/domain-specific features and benefit most from adapting to the new data.

```python
model.requires_grad_(False)                 # freeze everything
model.fc = nn.Linear(512 * 4, 10)            # new head for 10 classes
model.fc.requires_grad_(True)                # new layers require grad by default; explicit here

# later, as a follow-up fine-tuning pass:
for param in model.layer4.parameters():
    param.requires_grad_(True)               # unfreeze just the last block
```

Only parameters with `requires_grad=True` need to be passed to the optimizer (or filter for them): frozen params don't need optimizer state (e.g. Adam's per-parameter moment buffers) at all, saving memory.

### The standard decision matrix

| Dataset size vs. pretrained domain | Recommendation |
|---|---|
| Small dataset, **similar** to pretraining domain | Freeze backbone, train only the new head — high risk of overfitting if you fine-tune with so little data |
| Small dataset, **different** from pretraining domain | Freeze most of the backbone, fine-tune only the last block(s) with a low LR |
| Large dataset, similar domain | Fine-tune more/all of the backbone — enough data to safely update more parameters |
| Large dataset, very different domain | Fine-tune extensively, or consider training from scratch if the domain gap is large enough that pretrained features aren't a good prior |

### Learning rate when fine-tuning

Pretrained weights already sit close to a good optimum. A learning rate appropriate for training from scratch (which starts from random weights and needs large early updates) would apply overly large updates to already-good weights and can destroy the pretrained representations ("catastrophic forgetting" of the useful features). Fine-tuning therefore typically uses a **much smaller LR** (often 10-100x smaller) than training from scratch, sometimes with different LRs per layer group (discriminative learning rates — smaller for earlier/more-generic layers, larger for later/more task-specific layers and the new head).

## Example

```python
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.requires_grad_(False)
model.fc = nn.Linear(512 * 4, 10)   # new head, unfrozen by default

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
# train the head only...

# fine-tuning pass, later:
for p in model.layer4.parameters():
    p.requires_grad_(True)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
```

Full example: [`pytorch/code/transfer_learning.py`](code/transfer_learning.py). See also the TensorFlow/Keras equivalent pattern: [`keras-tf-reference/code/transfer_learning_tf.py`](../keras-tf-reference/code/transfer_learning_tf.py).

## When to use

Almost always the right default for vision tasks with limited labeled data — training from scratch is rarely worth it unless your domain is very unlike natural images (e.g. some medical/satellite/scientific imaging) or you have a genuinely large labeled dataset. Also applies beyond vision: fine-tuning pretrained language models ([`attention-transformers.md`](../attention-transformers.md)) follows the exact same freeze/fine-tune logic.

## Common interview questions

- **Why does transfer learning work at all?** Early CNN layers learn generic, reusable low-level features (edges, textures) that transfer across visual domains; only later layers are task-specific.
- **When would you freeze vs. fine-tune the backbone?** Small/similar dataset -> freeze; large and/or different domain -> fine-tune more layers. See decision matrix above.
- **Why use a lower learning rate when fine-tuning vs. training from scratch?** To avoid large updates destroying already-good pretrained weights.
- **What happens if you fine-tune the whole network with a small dataset and a normal (not reduced) learning rate?** High risk of overfitting and of catastrophically forgetting the useful pretrained features.
- **What's the difference between feature extraction and fine-tuning in this context?** Feature extraction = frozen backbone, only the head trains (backbone used purely as a fixed feature extractor); fine-tuning = backbone weights also get updated.

## Common mistakes

- Forgetting to freeze the backbone (`requires_grad_(False)`) and accidentally fine-tuning the whole network with a learning rate meant for a from-scratch head only.
- Not using the pretrained model's own preprocessing/normalization (`resnet_weights.transforms()`), causing a silent distribution mismatch between training and pretraining.
- Passing all `model.parameters()` to the optimizer even though most are frozen — wastes memory on unused optimizer state (harmless in effect since frozen params get zero gradient, but wasteful and confusing).
- Using the same learning rate for fine-tuning as for training the new head from scratch.
