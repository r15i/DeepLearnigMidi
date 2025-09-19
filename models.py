import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import torchvision.utils as vutils

# --- Assumed Local Imports (User's files) ---
import DBAdapters as dba


# # PyTorch VAE (Conv) - encoder -> mu, logvar; reparameterization; decoder -> logits
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.latent_dim = latent_dim
        # Encoder convs (input B,1,28,28)
        self.enc_conv = nn.Sequential(
            # 1 immagine a 32 (32 filtri qu)
            nn.Conv2d(1, 32, 4, 2, 1),  # -> 14x14
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # -> 7x7
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),  # -> ~3x3
            nn.ReLU(True),
        )
        self._enc_out_channels = 128
        self._enc_feat_size = 3  # finale spaziale (per 28x28 input con i layer sopra)
        self.fc_mu = nn.Linear(
            self._enc_out_channels * self._enc_feat_size * self._enc_feat_size,
            latent_dim,
        )
        self.fc_logvar = nn.Linear(
            self._enc_out_channels * self._enc_feat_size * self._enc_feat_size,
            latent_dim,
        )

        # Decoder
        self.fc_dec = nn.Linear(
            latent_dim,
            self._enc_out_channels * self._enc_feat_size * self._enc_feat_size,
        )
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            # poi crop/interpolate a 28x28
        )

    def encode(self, x):
        h = self.enc_conv(x)
        B = h.shape[0]
        h_flat = h.view(B, -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        B = h.shape[0]
        h = h.view(B, self._enc_out_channels, self._enc_feat_size, self._enc_feat_size)
        x_logits = self.dec_conv(h)
        x_logits = F.interpolate(
            x_logits, size=(28, 28), mode="bilinear", align_corners=False
        )
        return x_logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_logits = self.decode(z)
        return x_logits, mu, logvar


def vae_loss(x, x_logits, mu, logvar):
    # recon è negativo
    recon_loss = F.binary_cross_entropy_with_logits(x_logits, x, reduction="sum")
    var = logvar.exp()
    # slide 67 autoencoder
    kl = 0.5 * torch.sum(mu.pow(2) + var - logvar - 1.0)
    return recon_loss + kl, recon_loss, kl
