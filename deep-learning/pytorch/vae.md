# Variational Autoencoders (VAE)

## What is it?

An autoencoder that learns a **probability distribution** over the latent space instead of a single deterministic latent vector. It's trained to both reconstruct its input and keep its learned latent distribution close to a standard normal, which makes the latent space smooth and sample-able (you can draw a random `z` and decode it into a plausible new output — generative, unlike a plain autoencoder).

Code reference: [`pytorch/code/vae.py`](code/vae.py) — MLP encoder -> (mean, log-variance) heads -> reparameterization -> MLP decoder, plus the combined reconstruction + KL loss.

## Why?

A plain (deterministic) autoencoder learns a bottleneck that compresses input to a code and reconstructs it, but the latent space it learns has no particular structure — nearby points in latent space don't necessarily decode to similar, meaningful outputs, and there's no principled way to sample new latent points to *generate* new data. A VAE instead treats the encoder as producing a distribution `q(z|x)` (parameterized by a mean and variance) and regularizes that distribution toward `N(0, I)`, giving a continuous, structured latent space you can sample from — turning the autoencoder into a proper generative model.

## How does it work?

### Autoencoder vs. VAE

| | Autoencoder | VAE |
|---|---|---|
| Bottleneck | A single deterministic vector $z = \text{encoder}(x)$ | Parameters of a distribution: `mean`, `log_var` |
| Latent space structure | No guarantee of smoothness/continuity | Regularized toward $\mathcal{N}(0, I)$ -> smooth, sample-able |
| Can generate new samples? | Not reliably (no idea what an arbitrary latent point decodes to) | Yes — sample $z \sim \mathcal{N}(0, I)$ and decode |

### The reparameterization trick

The encoder produces `h_mean` and `h_log_var`. Naively you'd sample $z \sim \mathcal{N}(\text{h\_mean}, \exp(\text{h\_log\_var}))$ directly — but **you can't backpropagate through a random sampling operation**: gradients don't flow through `torch.randn(...)` calls whose parameters depend on the network, because sampling isn't a differentiable function of its parameters.

The fix: sample the randomness from an independent, parameter-free source and combine it deterministically with the encoder's outputs:

```python
noise = torch.randn_like(h_mean)                 # epsilon ~ N(0, 1), independent of the network
z = noise * torch.exp(h_log_var / 2) + h_mean     # z = mean + std * epsilon
```

Now `z` is a *deterministic, differentiable* function of `h_mean`, `h_log_var`, and the (non-parametric) noise `epsilon` — gradients can flow back through `mean` and `log_var` into the encoder normally, while the actual randomness is isolated in `epsilon`, which needs no gradient. This is the reparameterization trick, and it's what makes VAEs trainable end-to-end with standard backprop ([`tensors-autograd.md`](tensors-autograd.md)).

### The KL divergence term

```python
kl_loss = -0.5 * torch.sum(
    1 + h_log_var - torch.square(h_mean) - torch.exp(h_log_var), dim=1
)
```

This is the closed-form KL divergence between the learned posterior $q(z|x) = \mathcal{N}(\text{mean}, \exp(\text{log\_var}))$ and the prior $\mathcal{N}(0, I)$, for diagonal Gaussians. It penalizes the encoder for drifting the latent distribution away from a standard normal. Without this term, the model could minimize reconstruction loss by mapping each input to an arbitrarily narrow, far-apart region of latent space (effectively memorizing inputs with a near-zero-variance "distribution" per example) — that would reconstruct perfectly but produce a latent space with huge gaps that don't decode to anything meaningful, defeating the purpose of having a probabilistic structure at all. The KL term forces distributions to overlap and cluster near the origin, keeping the space continuous and generative.

### Reconstruction loss vs. KL loss tradeoff (beta-VAE)

```python
recon_loss = torch.sum(torch.square(x - y), dim=1)   # MSE-style; BCE is also common for [0,1] targets
total_loss = torch.mean(recon_loss + kl_loss)
```

These two terms pull in different directions: reconstruction loss wants the latent code to preserve as much input-specific information as possible (favoring wide/precise, less-regularized `q(z|x)`); KL loss wants every `q(z|x)` to collapse toward the same `N(0, I)` (favoring less information retained). **Beta-VAE** generalizes this by weighting the KL term with a factor $\beta$: $\text{recon\_loss} + \beta \cdot \text{kl\_loss}$. `beta > 1` pushes harder toward a more regularized, more disentangled latent space at the cost of reconstruction fidelity (useful when you want interpretable latent dimensions); `beta < 1` favors sharper reconstructions at the cost of a less-structured latent space.

## Example

```python
class VAE(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(...)              # -> shared hidden representation
        self.h_mean = nn.Linear(64, hidden_dim)
        self.h_log_var = nn.Linear(64, hidden_dim)
        self.decoder = nn.Sequential(..., nn.Sigmoid())  # output in [0, 1]

    def forward(self, x):
        enc = self.encoder(x)
        h_mean, h_log_var = self.h_mean(enc), self.h_log_var(enc)

        noise = torch.randn_like(h_mean)
        z = noise * torch.exp(h_log_var / 2) + h_mean    # reparameterization trick

        x_recon = self.decoder(z)
        return x_recon, z, h_mean, h_log_var
```

Full model + loss: [`pytorch/code/vae.py`](code/vae.py).

**Typo fix note:** the source archive's decoder had `nn.ELU(inplase=True)` — a typo of the `inplace` kwarg, which would raise `TypeError: __init__() got an unexpected keyword argument 'inplase'` at model construction (before any forward pass runs, since it fails inside `nn.ELU.__init__`). This was already fixed to `inplace=True` in the ported [`vae.py`](code/vae.py) by the prior porting pass — verified while writing this note, no further action needed.

## When to use

Generative modeling where you want a smooth, sample-able latent space (image generation, anomaly detection via reconstruction error, semi-supervised learning, data augmentation by sampling/interpolating in latent space). If you only need compression/dimensionality reduction with no generative requirement, a plain (deterministic) autoencoder is simpler and often sufficient. For higher-fidelity image generation, diffusion models and GANs generally outperform vanilla VAEs on sample sharpness, though VAEs remain popular as a building block (e.g. the latent space in latent diffusion models).

## Common interview questions

- **Why is the reparameterization trick necessary?** Backprop can't flow through a stochastic sampling node whose distribution parameters depend on the network; reparameterizing moves the randomness into an independent, parameter-free noise term so the rest of the computation is deterministic and differentiable.
- **What does the KL term in the VAE loss actually do, and what happens without it?** Regularizes the learned per-example latent distribution toward $\mathcal{N}(0, I)$. Without it, the model can "cheat" by collapsing each input to a near-deterministic point far from others, defeating the point of a continuous, sample-able latent space (this failure mode is sometimes called posterior collapse in the opposite direction — here it's closer to encoder "memorization").
- **How would you generate a new sample from a trained VAE?** Sample $z \sim \mathcal{N}(0, I)$ directly (no encoder needed) and run it through the decoder.
- **What's the effect of increasing beta in a beta-VAE?** More disentangled/regularized latent space, at the cost of reconstruction fidelity.
- **VAE vs. GAN — high level differences?** VAE optimizes an explicit likelihood-based objective (reconstruction + KL) and gives a well-defined encoder for inference over latents; typically produces blurrier samples. GAN trains a generator against a discriminator adversarially, no explicit likelihood, typically sharper samples but less stable training and no built-in encoder.

## Common mistakes

- Sampling `z` directly from a distribution parameterized by the network output without reparameterizing — breaks gradient flow to the encoder.
- Forgetting the KL term entirely and ending up with a VAE that behaves like (and has the same latent-space problems as) a plain autoencoder.
- Mismatching the reconstruction loss to the output activation/target range — e.g. using squared-error reconstruction loss meaningfully differs from BCE when targets are normalized pixel probabilities; the source code pairs `Sigmoid()` output in `[0, 1]` with a squared-error reconstruction term, which is a valid but not the only common choice (BCE reconstruction loss is equally common here).
- Treating `h_log_var` as if it were the variance directly instead of the log-variance — note the code uses `torch.exp(h_log_var / 2)` for standard deviation, not `h_log_var` itself.
