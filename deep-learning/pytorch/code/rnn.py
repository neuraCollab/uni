# RNN basics: a plain nn.RNN sequence-classification head, and a bidirectional
# variant that concatenates forward/backward final hidden states.
#
# Merged from the archive's recurrentNN.py and dualRecurent.py. Both files
# defined an incomplete `Dataset` for text preprocessing, explicitly marked
# by the original author as unfinished ("дальше идет предобработка текста,
# мне очень не хочется ее писать" - "text preprocessing goes here, I really
# don't feel like writing it") - that stub is intentionally dropped here,
# keeping only the model definitions, which are complete and correct.

# %%
import torch
import torch.nn as nn


# %% Unidirectional RNN -> single-vector sequence classification
class TextRNN(nn.Module):
    """Takes a batch of sequences (batch, seq_len, in_features) and produces
    one output vector per sequence, using the RNN's final hidden state."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.hidden_size = 64
        self.in_features = in_features
        self.out_features = out_features

        self.rnn = nn.RNN(in_features, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, out_features)

    def forward(self, x):
        _, h = self.rnn(x)     # h: (num_layers * num_directions, batch, hidden_size)
        y = self.out(h.squeeze(0))  # drop the num_layers*directions=1 dim before the Linear
        return y


# %% Bidirectional RNN demo - shapes
rnn = nn.RNN(300, 16, batch_first=True, bidirectional=True)
y, h = rnn(torch.randn(8, 3, 300))
print("y:", y.size())  # (batch=8, seq_len=3, hidden*2=32) - outputs at every timestep
print("h:", h.size())  # (num_layers*2=2, batch=8, hidden=16) - final hidden states, fwd+bwd


# %% Bidirectional RNN -> sequence classification
class WordsRnn(nn.Module):
    """Bidirectional RNN classifier: concatenates the last forward hidden
    state with the last backward hidden state before the output layer, so
    the classifier sees context from both directions of the sequence."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.hidden_size = 16
        self.in_features = in_features
        self.out_features = out_features

        self.rnn = nn.RNN(
            in_features, self.hidden_size, batch_first=True, bidirectional=True
        )
        self.out = nn.Linear(self.hidden_size * 2, out_features)

    def forward(self, x):
        _, h = self.rnn(x)
        # h[-2] = last layer's forward-direction hidden state,
        # h[-1] = last layer's backward-direction hidden state
        hh = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        y = self.out(hh)
        return y
