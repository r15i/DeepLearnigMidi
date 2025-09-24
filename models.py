import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: (B, 1, 128, 16) -> (B, latent_dim)
        self.encoder_conv = nn.Sequential(
            # Input: (B, 1, 128, 16)
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # -> (B, 32, 64, 8)
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> (B, 64, 32, 4)
            nn.ReLU(True),
            nn.Conv2d(
                64, 128, kernel_size=4, stride=2, padding=1
            ),  # -> (B, 128, 16, 2)
            nn.ReLU(True),
        )

        # Calculate the flattened size of the encoder's output feature map
        self._enc_out_shape = (128, 16, 2)
        enc_out_dim = (
            self._enc_out_shape[0] * self._enc_out_shape[1] * self._enc_out_shape[2]
        )

        # Fully connected layers to get mu and logvar
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)

        # Decoder: (B, latent_dim) -> (B, 1, 128, 16)
        self.decoder_fc = nn.Linear(latent_dim, enc_out_dim)
        self.decoder_conv = nn.Sequential(
            # Input: (B, 128, 16, 2)
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),  # -> (B, 64, 32, 4)
            nn.ReLU(True),
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),  # -> (B, 32, 64, 8)
            nn.ReLU(True),
            nn.ConvTranspose2d(
                32, 1, kernel_size=4, stride=2, padding=1
            ),  # -> (B, 1, 128, 16)
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        h_flat = h.view(h.size(0), -1)  # Flatten the feature map
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # Project and reshape
        h = self.decoder_fc(z)
        h = h.view(h.size(0), *self._enc_out_shape)
        # Apply transposed convolutions
        x_logits = self.decoder_conv(h)
        return x_logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_logits = self.decode(z)
        return x_logits, mu, logvar


def vae_loss(x, x_logits, mu, logvar, beta=1.0):
    """
    Calculates the VAE loss, which is a sum of reconstruction loss and KL divergence.
    Loss is averaged over the batch.
    """
    batch_size = x.size(0)

    # Reconstruction Loss (Binary Cross-Entropy)
    # Measures how well the VAE reconstructs the input piano roll.
    recon_loss = F.binary_cross_entropy_with_logits(x_logits, x, reduction="sum")

    # KL Divergence
    # A regularizer that forces the latent space to be a smooth, continuous distribution (a Gaussian).
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Combine losses and average over the batch
    total_loss = (recon_loss + beta * kl_div) / batch_size

    return total_loss, recon_loss / batch_size, kl_div / batch_size
