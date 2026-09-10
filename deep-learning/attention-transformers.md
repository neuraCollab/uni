# Attention & Transformers

> Not present in either source repo (PyTorch or TensorFlow archives) — written from general knowledge, kept intentionally concise as a cram note rather than a full course.

## What is it?

**Self-attention** lets every position in a sequence directly look at and weigh every other position when computing its representation, rather than only accessing information sequentially through a recurrent hidden state ([`pytorch/rnn.md`](pytorch/rnn.md)). The **Transformer** is the architecture built almost entirely out of attention layers (plus feedforward blocks) that made this the dominant approach for sequence modeling, especially in NLP.

## Why?

RNNs process a sequence step by step, so information from position 1 has to pass through `t-1` intermediate hidden-state updates before reaching position `t` — which is both slow (inherently sequential, can't parallelize across time) and prone to vanishing gradients over long distances ([`pytorch/rnn.md`](pytorch/rnn.md)). Attention lets any position attend directly to any other position in a single step: no chain of intermediate updates to degrade through, and because every position's computation is independent given the input, it's **fully parallelizable** across the sequence during training (a major reason transformers scale so well on GPUs/TPUs).

## How does it work?

### Scaled dot-product self-attention

Each input token is projected (via three learned weight matrices) into a **query (Q)**, **key (K)**, and **value (V)** vector. The attention score between two positions is the dot product of one's query and the other's key, scaled and normalized:

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

- `Q K^T` — for every pair of positions, how relevant is this key to that query (dot product = similarity).
- `/ sqrt(d_k)` — scales down the dot products (which grow with dimension `d_k`) so the softmax doesn't saturate into near-one-hot outputs with vanishing gradients.
- `softmax(...)` — turns each row into a probability distribution over "how much to attend to every other position."
- `... V` — the output at each position is a weighted sum of every position's value vector, weighted by that attention distribution.

Every position's output can be computed independently of every other output (given Q, K, V for the whole sequence), which is exactly what makes this parallelizable, unlike an RNN's inherently sequential recurrence.

### Multi-head attention

Instead of a single Q/K/V projection, use several ("heads") in parallel, each with its own learned projections, each therefore free to learn a different notion of "relevance" (e.g. one head might track syntactic dependency, another might track coreference). The heads' outputs are concatenated and linearly projected back to the model dimension. This gives the model several independent "views" of the relationships between positions instead of averaging everything into one.

### Positional encoding

Attention as defined above is **permutation-invariant** — swap two tokens' positions and the set of attention outputs is unaffected (unlike an RNN, which is inherently order-sensitive because it processes the sequence step by step). Since word/token order usually carries meaning, position information has to be injected explicitly, typically by adding a positional encoding vector to each token's embedding before the first attention layer:

- **Sinusoidal** (original Transformer paper): fixed, non-learned functions of position using sine/cosine at different frequencies — has a nice property that relative offsets correspond to a fixed linear transformation, and it extrapolates to sequence lengths not seen in training.
- **Learned**: a trainable embedding per position, like a lookup table indexed by position instead of token identity (see [`embeddings.md`](embeddings.md)) — simpler but doesn't extrapolate beyond the trained max length as gracefully.

### Transformer block structure

Each encoder/decoder layer repeats the same skeleton:

```
x = x + SelfAttention(LayerNorm(x))     # residual connection + self-attention
x = x + FeedForward(LayerNorm(x))       # residual connection + position-wise feedforward (2-layer MLP)
```

(Exact placement of LayerNorm relative to the residual — "pre-norm" as shown vs. "post-norm" — varies between implementations; both are common.) The **residual connections** are what make stacking many such blocks trainable at all — same rationale as in very deep CNNs/ResNets: they give gradients a direct path backward through the network, mitigating vanishing gradients across depth. **LayerNorm** stabilizes activation scale similarly to how BatchNorm does in CNNs ([`regularization-overfitting.md`](regularization-overfitting.md)), but normalizes across the feature dimension per example instead of across the batch — important because it makes LayerNorm's behavior independent of batch size/composition, unlike BatchNorm.

The **decoder** additionally uses **masked** self-attention (each position can only attend to earlier positions, preserving autoregressive generation) plus a **cross-attention** layer that attends over the encoder's output — this is how encoder-decoder transformers (e.g. for translation) connect the two stacks.

### BERT vs. GPT

- **BERT** (encoder-only): trained with a masked-language-modeling objective (predict randomly-masked tokens using *bidirectional* context — every position can attend to every other position, both before and after). Good for understanding/classification tasks (sentiment, NER, question answering) where you have the whole input available and just need a rich representation of it.
- **GPT** (decoder-only): trained autoregressively (predict the next token given only preceding tokens — masked/causal self-attention). Good for open-ended generation, since it's built to produce one token at a time conditioned only on what came before, matching how it's actually used at inference.

These are the two dominant transformer "families"; many modern large language models are decoder-only (GPT-style) since autoregressive generation is the more general capability (can be adapted to classification/understanding tasks via prompting, whereas an encoder-only bidirectional model can't naturally generate open-ended text).

## Example

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / d_k ** 0.5      # (batch, seq_len, seq_len)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    return weights @ v                                  # (batch, seq_len, d_v)
```

PyTorch provides this as a building block directly: `torch.nn.MultiheadAttention` and a full encoder layer via `torch.nn.TransformerEncoderLayer` — in practice you rarely hand-roll attention, but being able to derive/explain the formula above is the interview-relevant skill.

## When to use

Sequence tasks where long-range dependencies and training throughput both matter — which today means most large-scale NLP, and increasingly vision (Vision Transformers) and multimodal models. For small datasets/short sequences/low-resource settings, RNNs or simpler models can still be a reasonable, cheaper choice ([`pytorch/rnn.md`](pytorch/rnn.md)) — transformers have no inherent sequential inductive bias, so they typically need more data (or pretraining) to learn ordering/locality patterns that an RNN or CNN gets "for free" from its architecture.

## Common interview questions

- **Write out the scaled dot-product attention formula and explain each term.** `softmax(QK^T / sqrt(d_k)) V` — see the "Scaled dot-product self-attention" section above.
- **Why divide by `sqrt(d_k)`?** Without scaling, dot products grow with dimensionality, pushing softmax into a saturated near-one-hot regime with very small gradients; scaling keeps the softmax input in a well-conditioned range.
- **Why do transformers need positional encoding but RNNs don't?** Self-attention is permutation-invariant by construction (no inherent notion of order); RNNs are inherently order-sensitive because they process tokens sequentially.
- **What's the advantage of multi-head attention over single-head?** Lets the model attend to information from different representation subspaces/relationship types in parallel, instead of averaging everything into one attention pattern.
- **Why are transformers more parallelizable than RNNs during training?** Every position's attention output can be computed independently given Q/K/V for the whole sequence, with no sequential dependency between timesteps — unlike an RNN's step-by-step recurrence.
- **BERT vs. GPT — what's the core architectural and training difference?** BERT is encoder-only with bidirectional masked-language-model pretraining (good for understanding tasks); GPT is decoder-only with causal/autoregressive pretraining (good for generation).

## Common mistakes

- Confusing self-attention (Q, K, V all derived from the same sequence) with cross-attention (Q from one sequence/the decoder, K/V from another/the encoder) — they use the same formula but different sources for Q vs. K/V.
- Forgetting the causal mask in decoder self-attention, which would let the model "see the future" during autoregressive training/generation — a subtle bug that inflates training performance unrealistically.
- Assuming transformers have some inherent notion of word order without positional encoding — they don't; omitting it (or misconfiguring the max sequence length it was designed for) silently degrades order-sensitive tasks.
- Conflating LayerNorm and BatchNorm — LayerNorm normalizes per-example across features (batch-size independent), BatchNorm normalizes per-feature across the batch (needs a large enough batch, and behaves differently in train/eval mode, see [`regularization-overfitting.md`](regularization-overfitting.md)).
