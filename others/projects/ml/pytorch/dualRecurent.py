# %%
import torch
import torch.nn as nn

# %%
rnn = nn.RNN(300, 16, batch_first=True, bidirectional=True)
y, h = rnn(torch.randn(8, 3, 300))
# %%
print("y:", y.size())
print("h:", h.size())
# %%


class WordsRnn(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.hidden_size = 16
        self.in_features = in_features
        self.out_features = out_features

        self.rnn = nn.RNN(
            in_features, self.hidden_size, batch_first=True, bidirectional=True
        )
        self.out = nn.Linear(self.hidden_size * 2, out_features=out_features)

    def forward(self, x):
        x, h = self.rnn(x)
        hh = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        y = self.out(hh)
        return y


# %%
