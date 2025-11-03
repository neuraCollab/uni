# %%
import torch
import torch.nn as nn


# %%
class autorEncoderMNIST(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128, bias=False),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64, bias=False),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, self.hidden_dim),
        )

        self.h_mean = nn.Linear(64, self.hidden_dim)
        self.h_log_var = nn.Linear(64, self.hidden_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 64, bias=False),
            nn.ELU(inplase=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128, bias=False),
            nn.ELU(inplace=True),
            nn.Linear(128, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        enc = self.encoder(x)

        h_mean = self.h_mean(enc)
        h_log_var = self.h_log_var(enc)

        noice = torch.randn_like(h_mean)
        h = noice * torch.exp(h_log_var / 2) + h_mean
        x = self.decoder(h)

        return x, h, h_mean, h_log_var


# %%


class VAELoss(nn.Module):
    def forward(self, x, y, h_mean, h_log_var):
        img_loss = torch.sum(torch.square(x - y), dim=1)
        kl_loss = -0.5 * torch.sum(
            1 + h_log_var - torch.square(h_mean) - torch.exp(h_log_var), dim=1
        )
        return torch.mean(img_loss + kl_loss)
