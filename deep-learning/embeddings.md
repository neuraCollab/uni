# Embeddings

> Source example: the archived `tensorFlow/tfText/wordEmbeddings.py` — a complete IMDB sentiment pipeline using `TextVectorization`, a custom text-standardization function, an `Embedding` layer, and `GlobalAveragePooling1D`. TensorFlow/Keras-sourced (this KB is PyTorch-primary), but the embedding concept itself is identical across frameworks — PyTorch's `nn.Embedding` is a direct equivalent.

## What is it?

An embedding layer is a **learned lookup table** that maps discrete tokens (word IDs, category IDs, user/item IDs) to dense, continuous, fixed-size vectors. It replaces sparse one-hot representations with compact, trainable representations.

## Why?

A one-hot encoding of a vocabulary of size `V` needs a `V`-dimensional vector per token — mostly zeros, no notion of similarity between tokens (any two distinct one-hot vectors are equally "far apart"), and directly feeding one-hot vectors into a dense layer is equivalent to (but far more wasteful than) a lookup table. An embedding layer:

1. **Dramatically reduces dimensionality** — a 10,000-word vocabulary can be represented in, say, 16 or 300 dimensions instead of 10,000.
2. **Captures semantic similarity** — because the vectors are learned by gradient descent to be useful for the downstream task, tokens that behave similarly in that task end up with similar vectors (nearby in embedding space), which one-hot encoding cannot represent at all.

## How does it work?

An embedding layer is literally a weight matrix of shape `(vocab_size, embedding_dim)`; looking up token `i` just indexes row `i`. The gradient of the loss with respect to that row updates it like any other parameter — so embeddings are learned jointly with the rest of the network, purely from the task's own training signal (no separate self-supervised step required, though one is often used anyway — see pretrained embeddings below).

```python
# Keras source example
embedding_layer = tf.keras.layers.Embedding(1000, 5)   # 1000-word vocab -> 5-dim vectors
result = embedding_layer(tf.constant([1, 2, 3]))       # looks up rows 1, 2, 3
```

```python
# PyTorch equivalent
embedding = nn.Embedding(num_embeddings=1000, embedding_dim=5)
vecs = embedding(torch.tensor([1, 2, 3]))   # (3, 5)
```

### Full pipeline pattern (from the source example)

1. **Text standardization + vectorization**: lowercase, strip HTML, strip punctuation, then map each token to an integer ID from a fixed-size vocabulary (`TextVectorization`/`nn.Embedding`'s input side in PyTorch is a tokenizer + vocab you build yourself, e.g. via `torchtext` or a custom tokenizer):

```python
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')

vectorize_layer = TextVectorization(standardize=custom_standardization,
                                     max_tokens=vocab_size, output_sequence_length=sequence_length)
```

2. **Embedding layer**: maps each integer ID in the (padded/truncated) sequence to its dense vector, producing a `(batch, seq_len, embedding_dim)` tensor.

3. **Pooling** (`GlobalAveragePooling1D`, i.e. mean over the sequence dimension): collapses the variable-length sequence of per-token embeddings into one fixed-size vector per example — a simple, cheap way to get a sequence-level representation without an RNN/attention mechanism.

```python
model = Sequential([
    vectorize_layer,
    Embedding(vocab_size, embedding_dim, name="embedding"),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1),
])
```

### Embedding dimensionality tradeoffs

Larger embedding dimension: can capture more nuance/more distinctions between tokens, but more parameters (risk of overfitting on small datasets, more memory), and diminishing returns past a point set by how much genuine semantic variation the vocabulary actually has. Smaller: cheaper, more regularized, but may underfit / collapse genuinely distinct tokens into overly-similar vectors. Common practical range for word embeddings: tens (small vocabularies/tasks) to a few hundred dimensions (large vocabularies, e.g. 300 for word2vec/GloVe).

### Pretrained embeddings (word2vec/GloVe) vs. learned from scratch

- **Learn from scratch** (what the source example does): the embedding matrix is randomly initialized and trained jointly with the rest of the network on your task's data. Works well when you have **enough task-specific labeled data** for the embeddings to learn good structure purely from that signal.
- **Pretrained embeddings** (word2vec, GloVe, or embeddings extracted from a pretrained language model): initialize (and optionally freeze, or fine-tune at a low learning rate — same logic as [`transfer-learning.md`](pytorch/transfer-learning.md)) the embedding matrix from vectors trained on a much larger, generic corpus. Preferred when your task's own labeled dataset is **small** — the pretrained vectors already encode broad semantic structure that a small dataset couldn't learn on its own.

This is the exact same freeze-vs-fine-tune transfer-learning tradeoff as with pretrained CNN backbones, just applied to a lookup table instead of a stack of convolutions.

### Beyond NLP: embeddings as a general pattern

The same idea generalizes far beyond words:

- **Tabular deep learning**: categorical features (e.g. `city`, `product_category`) with many possible values are embedded into dense vectors instead of one-hot encoded, letting the network learn which categories behave similarly — much more parameter-efficient than one-hot for high-cardinality categorical columns, and standard practice in modern tabular deep learning.
- **Recommender systems**: user embeddings and item embeddings, trained (often jointly, e.g. via matrix-factorization-style or two-tower models) so that the dot product / distance between a user's and an item's embedding predicts affinity (rating, click probability, purchase probability). This is one of the most common industrial applications of the embedding idea outside of NLP.

In every case the underlying mechanism is identical: replace a large, sparse, similarity-free discrete representation with a small, dense, learned one that gradient descent shapes to be useful for the task.

## Example

```python
class TabularEmbeddingModel(nn.Module):
    def __init__(self, num_categories, embed_dim, num_numeric_features, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(num_categories, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim + num_numeric_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, cat_ids, numeric_features):
        cat_vec = self.embed(cat_ids)
        x = torch.cat([cat_vec, numeric_features], dim=1)
        return self.head(x)
```

Full NLP pipeline this note is based on: the archived `tensorFlow/tfText/wordEmbeddings.py` (IMDB sentiment classification with a learned embedding + average pooling).

## When to use

Any time you have a discrete, potentially high-cardinality input that a one-hot encoding would represent wastefully and without any notion of similarity: vocabulary tokens, categorical tabular features, user/item IDs, graph node IDs. Use pretrained embeddings when labeled data is scarce; learn from scratch when you have ample task-specific data.

## Common interview questions

- **Why use embeddings instead of one-hot encoding?** Lower dimensionality, and learned proximity between similar tokens/categories — one-hot vectors are all equidistant from each other, embeddings aren't.
- **How is an embedding layer trained?** Like any other weight matrix — indexed lookup is differentiable with respect to the looked-up row, so gradients from the downstream loss update exactly the rows that were used in a given batch.
- **When would you use pretrained embeddings vs. train from scratch?** Pretrained when your own labeled dataset is small relative to the vocabulary/task complexity; from scratch when you have enough task-specific data (same logic as CNN transfer learning).
- **How do embeddings generalize outside NLP?** Categorical feature embeddings in tabular deep learning; user/item embeddings in recommender systems; node embeddings in graph learning — any high-cardinality discrete input.
- **What does `GlobalAveragePooling1D` do in a text embedding pipeline, and what's a limitation of it?** Averages token embeddings across the sequence into one fixed-size vector — simple and fast, but ignores word order entirely (a bag-of-embeddings model), unlike an RNN ([`pytorch/rnn.md`](pytorch/rnn.md)) or attention-based model ([`attention-transformers.md`](attention-transformers.md)) which can use order/position.

## Common mistakes

- Using one-hot encoding for a high-cardinality categorical feature (thousands of categories) instead of an embedding — wastes memory/parameters and gives the model no way to learn cross-category similarity.
- Choosing an embedding dimension far larger than the data supports, overfitting on rare tokens/categories that appear only a handful of times.
- Fine-tuning pretrained embeddings with too high a learning rate and destroying the useful pretrained structure (same fine-tuning-LR caution as [`transfer-learning.md`](pytorch/transfer-learning.md)).
- Not accounting for out-of-vocabulary tokens/unseen categories at inference time (no reserved "unknown" embedding row) — silently errors or crashes on production data the training vocabulary never saw.
