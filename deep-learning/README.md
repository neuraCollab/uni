# Deep Learning

Interview-cram notes on neural networks, covering fundamentals, architectures, and training practice. **PyTorch is the primary framework for this knowledge base** — most code references live under [`pytorch/code/`](pytorch/code/) and are meant to be read alongside their matching note. A handful of notes below (regularization, hyperparameter tuning, embeddings) are sourced from TensorFlow/Keras material instead, because that's where the only complete worked examples for those topics existed in the source archives — the *concepts* are framework-agnostic, so they're included here rather than left as gaps. Those notes say so explicitly and give PyTorch-equivalent snippets where useful.

## Fundamentals

- [Neural Networks & Backpropagation](fundamentals/neural-networks-backprop.md) — forward/backward pass, the chain rule, why non-linear activations matter. Code: [`simple_mlp.py`](pytorch/code/simple_mlp.py).
- [Optimization: SGD, Momentum, RMSprop, Adam](fundamentals/optimization-sgd-adam.md) — gradient descent variants, learning rate schedules, batch size / generalization-gap tradeoff.

## PyTorch

- [Tensors & Autograd](pytorch/tensors-autograd.md) — tensor basics, broadcasting, in-place vs. out-of-place ops, `requires_grad`/`.backward()`/`.grad`, `no_grad()`/`.detach()`. Code: [`tensors_basics.py`](pytorch/code/tensors_basics.py), [`autograd_basics.py`](pytorch/code/autograd_basics.py).
- [CNNs & U-Net Segmentation](pytorch/cnn-segmentation-unet.md) — conv/pooling/receptive field, encoder-decoder architectures, skip connections, Dice vs. BCE loss. Code: [`unet_segment.py`](pytorch/code/unet_segment.py).
- [Transfer Learning](pytorch/transfer-learning.md) — freeze vs. fine-tune vs. partial unfreeze, the small/large + similar/different decision matrix, fine-tuning learning rates. Code: [`transfer_learning.py`](pytorch/code/transfer_learning.py).
- [RNNs, LSTM/GRU, Bidirectional Models](pytorch/rnn.md) — vanishing/exploding gradients, gating mechanisms, bidirectional RNNs, many-to-one vs. many-to-many. Code: [`rnn.py`](pytorch/code/rnn.py).
- [Variational Autoencoders (VAE)](pytorch/vae.md) — autoencoder vs. VAE, the reparameterization trick, KL divergence, beta-VAE. Code: [`vae.py`](pytorch/code/vae.py).

## Cross-cutting topics

- [Regularization & Overfitting](regularization-overfitting.md) — L1/L2, dropout, `model.train()`/`model.eval()`, early stopping, augmentation, BatchNorm as implicit regularizer. *(TensorFlow-sourced example, PyTorch-equivalent snippets included.)*
- [Hyperparameter Tuning for Deep Learning](hyperparameter-tuning.md) — grid/random/Bayesian search applied to DL hyperparameters, why DL search is expensive, Hyperband/early-stopping-based pruning. See also [`../machine-learning/hyperparameter-optimization.md`](../machine-learning/hyperparameter-optimization.md) for the general (framework-agnostic) search-strategy background. *(TensorFlow-sourced example.)*
- [Embeddings](embeddings.md) — learned lookup tables vs. one-hot, dimensionality tradeoffs, pretrained (word2vec/GloVe) vs. learned-from-scratch, generalization to tabular/recommender embeddings. *(TensorFlow-sourced example.)*
- [Attention & Transformers](attention-transformers.md) — self-attention, scaled dot-product formula, multi-head attention, positional encoding, encoder/decoder block structure, BERT vs. GPT. *(Written from general knowledge — absent from both source repos.)*

## Secondary reference: Keras/TensorFlow

[`keras-tf-reference/code/`](keras-tf-reference/code/) holds Keras/TF equivalents of a few core patterns — useful if an interview leans TensorFlow, or just to see the same idea in a second framework:

- [`regression.py`](keras-tf-reference/code/regression.py) — Normalization layer -> Sequential -> compile -> fit -> evaluate, on the Auto MPG dataset.
- [`cnn_architecture.py`](keras-tf-reference/code/cnn_architecture.py) — basic Conv2D/MaxPooling stack for Fashion-MNIST.
- [`save_load_model.py`](keras-tf-reference/code/save_load_model.py) — checkpoint callbacks, `save_weights`/`load_weights`, SavedModel/HDF5 formats.
- [`transfer_learning_tf.py`](keras-tf-reference/code/transfer_learning_tf.py) — Keras-side transfer learning pattern (pairs with [`pytorch/transfer-learning.md`](pytorch/transfer-learning.md)).

No dedicated notes were written for these Keras files — they exist purely as secondary code reference alongside the PyTorch-primary notes above.
