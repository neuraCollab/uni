# RNNs, LSTM/GRU, and Bidirectional Sequence Models

## What is it?

A Recurrent Neural Network (RNN) processes a sequence one timestep at a time, maintaining a hidden state that's updated at each step and carries information forward from earlier timesteps. LSTM and GRU are gated variants designed to fix vanilla RNNs' inability to learn long-range dependencies.

Code reference: [`pytorch/code/rnn.py`](code/rnn.py) — a unidirectional `nn.RNN` sequence-classification head (`TextRNN`) and a bidirectional variant (`WordsRnn`) that concatenates forward/backward final hidden states.

## Why?

Feedforward networks and CNNs assume fixed-size, order-independent (or locally-ordered) inputs. Sequences (text, time series, audio) have variable length and order-dependent meaning — RNNs handle this by sharing the same weights across every timestep and passing a hidden state along, so the network's output at step `t` can depend on everything seen at steps `1..t`.

## How does it work?

### Vanilla RNN and the vanishing/exploding gradient problem

At each timestep: $h_t = \text{activation}(W_h h_{t-1} + W_x x_t + b)$. Training uses backpropagation through time (BPTT) — the same chain rule as normal backprop, but unrolled across every timestep. The gradient flowing back to an early timestep is a product of many Jacobians (one per timestep in between). Over long sequences:

- If those per-step derivatives are consistently `< 1` (common with `tanh`/sigmoid activations, which saturate), the gradient **vanishes** — shrinks toward zero, so the network effectively can't learn dependencies spanning more than a few dozen steps.
- If consistently `> 1`, the gradient **explodes** — blows up numerically, destabilizing training (usually mitigated with gradient clipping).

This is the same underlying phenomenon as in very deep feedforward networks (see [`neural-networks-backprop.md`](../fundamentals/neural-networks-backprop.md)), just unrolled across time instead of across layers — and it's the specific reason vanilla RNNs struggle with long sequences.

### LSTM and GRU: why gating fixes this

LSTM and GRU introduce a **cell state** (LSTM) or simplified hidden state (GRU) that information can flow through with much more direct, largely-linear paths, controlled by learned **gates** — small sigmoid-output subnetworks that decide what to keep, discard, or write at each step, rather than forcing everything through a repeated non-linear squashing at every timestep.

**LSTM gates**, conceptually:
- **Forget gate** — decides what fraction of the existing cell state to discard.
- **Input gate** — decides what new information to write into the cell state.
- **Output gate** — decides what part of the (updated) cell state to expose as the hidden state output at this timestep.

Because the cell-state update is close to additive/linear (gated addition rather than a full non-linear transform at every step), gradients can flow backward through many timesteps largely unimpeded — directly addressing vanishing gradients. (Note: `rnn.py` only uses plain `nn.RNN` for its demos — LSTM/GRU aren't in the ported code, but understanding why they exist is a near-certain interview topic whenever RNNs come up.)

**GRU** simplifies LSTM's three gates down to two (update gate and reset gate) and merges the cell state and hidden state into one, giving comparable performance to LSTM on many tasks with fewer parameters and a slightly simpler structure.

### Bidirectional RNNs

A unidirectional RNN's hidden state at step `t` only depends on steps `1..t` — it has no access to future context. A **bidirectional RNN** runs two RNNs over the sequence, one forward and one backward, and combines their hidden states (commonly by concatenation) — so the representation at every position has access to context from both directions.

```python
rnn = nn.RNN(300, 16, batch_first=True, bidirectional=True)
y, h = rnn(torch.randn(8, 3, 300))
# y: (batch=8, seq_len=3, hidden*2=32) - outputs at every timestep, fwd+bwd concatenated
# h: (num_layers*2=2, batch=8, hidden=16) - final hidden states, one per direction
```

```python
class WordsRnn(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.hidden_size = 16
        self.rnn = nn.RNN(in_features, self.hidden_size, batch_first=True, bidirectional=True)
        self.out = nn.Linear(self.hidden_size * 2, out_features)   # *2 for concatenated directions

    def forward(self, x):
        _, h = self.rnn(x)
        # h[-2] = last layer's forward-direction hidden state
        # h[-1] = last layer's backward-direction hidden state
        hh = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        return self.out(hh)
```

Bidirectional RNNs need the **entire sequence available upfront** — they can't be used for online/streaming/autoregressive generation where future tokens don't exist yet, only for tasks where the full input is available at inference time (e.g. classification, tagging of a complete sentence).

### Sequence classification vs. sequence-to-sequence

- **Many-to-one (sequence classification)**: consume the whole sequence, produce a single output from the final hidden state. This is what both `TextRNN` and `WordsRnn` do:

```python
class TextRNN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.hidden_size = 64
        self.rnn = nn.RNN(in_features, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, out_features)

    def forward(self, x):
        _, h = self.rnn(x)              # only need the final hidden state
        return self.out(h.squeeze(0))
```

- **Many-to-many (sequence-to-sequence)**: produce an output at every timestep (or a different-length output sequence entirely, via an encoder-decoder pair) — e.g. tagging every word in a sentence, or translation. Uses the per-timestep output tensor (`y` above) rather than only the final hidden state, or a separate decoder RNN entirely.

## When to use

RNN/LSTM/GRU: any variable-length sequential data where you want a relatively lightweight, sequential-by-construction model — time series, smaller-scale text tasks, on-device or low-resource settings. For large-scale NLP with long-range dependencies and where full-sequence context and parallel training matter most, attention-based transformers have largely superseded RNNs — see [`attention-transformers.md`](../attention-transformers.md) for why (parallelizable, no vanishing gradient over distance).

## Common interview questions

- **Why do vanilla RNNs struggle with long sequences?** Repeated multiplication of per-timestep Jacobians during BPTT causes vanishing or exploding gradients, preventing the network from learning dependencies across many steps.
- **How does LSTM address vanishing gradients?** A largely-additive/linear cell-state update path controlled by gates, instead of forcing information through a repeated non-linear squash at every step.
- **What's the difference between LSTM and GRU?** GRU merges the cell and hidden states and uses two gates (update, reset) instead of LSTM's three (forget, input, output) — fewer parameters, often comparable performance.
- **Why can't you use a bidirectional RNN for autoregressive generation?** It requires the whole sequence upfront (the backward pass runs from the end), so it's unusable when future tokens don't exist yet at inference time.
- **Many-to-one vs many-to-many — give an example of each.** Many-to-one: sentiment classification (whole sentence -> one label). Many-to-many: part-of-speech tagging (label per token) or machine translation (sequence -> sequence, generally different lengths, via an encoder-decoder).

## Common mistakes

- Using a bidirectional RNN in a setting that requires streaming/causal predictions (future context isn't actually available).
- Forgetting to concatenate (not just index) both directions' final hidden states in a bidirectional classifier — using only `h[-1]` silently drops the forward direction's information.
- Not clipping gradients on vanilla RNNs/LSTMs over long sequences, risking exploding-gradient instability.
- Assuming LSTM/GRU are immune to vanishing gradients — they're much more resistant, not immune; extremely long sequences can still be difficult.
