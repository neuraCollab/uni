# %%
import torch
import torch.nn as nn
from torch.nn import RNN
from torch.utils import data
# %%

class TextRNN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.hidden_size = 64
        self.in_features = in_features
        self.out_features = out_features
        
        self.rnn = RNN(in_features, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, out_features=out_features)

    def forward(self, x):
        x, h = self.rnn(x)
        y = self.out(h)

        return y
# %%
class CharsDataset(data.Dataset):
    def __init__(self, path, prev_chars=3):
        self.prev_chars = prev_chars

        with open(path, 'r', encoding='utf8') as f:
            self.text = f.read()
            self.text 
            
# дальше идет предобработка текста, мне очень не хочется ее писать, смотрите 
# полный пример в руководстве