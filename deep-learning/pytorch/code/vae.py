# MLP Variational Autoencoder: encoder -> (mean, log-variance) -> reparameterization
# trick -> decoder, plus the VAE loss (reconstruction + KL divergence).
# Ported from the archive's varAutoEnc.py.
#
# BUG FIX: the original decoder had `nn.ELU(inplase=True)` - a typo of the
# `inplace` kwarg. Since `ELU` has no `inplase` parameter, this raises
# `TypeError: __init__() got an unexpected keyword argument 'inplase'` at
# construction time, before the model ever runs. Fixed to `inplace=True` below.

# %%
import torch
import torch.nn as nn


# %%
class VAE(nn.Module):
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

        # Two separate heads off the encoder's last hidden layer: predict the
        # mean and log-variance of the approximate posterior q(z|x).
        self.h_mean = nn.Linear(64, self.hidden_dim)
        self.h_log_var = nn.Linear(64, self.hidden_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 64, bias=False),
            nn.ELU(inplace=True),   # FIXED: was `inplase=True` (typo) - TypeError at construction
            nn.BatchNorm1d(64),
            nn.Linear(64, 128, bias=False),
            nn.ELU(inplace=True),
            nn.Linear(128, output_dim),
            nn.Sigmoid(),           # output in [0, 1] - matches normalized pixel targets
        )

    def forward(self, x):
        enc = self.encoder(x)

        h_mean = self.h_mean(enc)
        h_log_var = self.h_log_var(enc)

        # Reparameterization trick: sample z = mean + std * epsilon, where
        # epsilon ~ N(0, 1) is drawn OUTSIDE the computation graph. This makes
        # sampling differentiable w.r.t. h_mean/h_log_var, because the random
        # part (epsilon) carries no gradient requirement - you can't backprop
        # through `torch.randn` sampling directly if the sampling itself
        # depends on the parameters, but you CAN backprop through a
        # deterministic function of the parameters and an independent noise source.
        noise = torch.randn_like(h_mean)
        z = noise * torch.exp(h_log_var / 2) + h_mean  # std = exp(log_var / 2)

        x_recon = self.decoder(z)

        return x_recon, z, h_mean, h_log_var


# %% VAE loss = reconstruction term + KL divergence term
class VAELoss(nn.Module):
    def forward(self, x, y, h_mean, h_log_var):
        # Reconstruction: sum of squared error per sample (MSE-style; a BCE
        # term is also common when x is binary/normalized pixel data).
        recon_loss = torch.sum(torch.square(x - y), dim=1)

        # KL(q(z|x) || N(0, I)), closed form for two diagonal Gaussians.
        # Pulls the learned latent distribution toward a standard normal,
        # which is what makes the latent space smooth/sample-able.
        kl_loss = -0.5 * torch.sum(
            1 + h_log_var - torch.square(h_mean) - torch.exp(h_log_var), dim=1
        )
        return torch.mean(recon_loss + kl_loss)

# Note on beta-VAE: multiplying kl_loss by a factor beta > 1 trades off
# reconstruction fidelity for a more disentangled/regularized latent space -
# see vae.md for when that tradeoff is worth making.
