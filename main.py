import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import torchvision.utils as vutils

# dataset MNIST (esempio), training loop minimale
from torchvision import datasets, transforms


# --- Assumed Local Imports (User's files) ---
import DBAdapters as dba
import models as md

# ==============================================================================
# 1. Configuration and Data Loading
# ==============================================================================

# --- Paths and Directories ---
csv_file_path = "./dataset/MAESTRO_Dataset/maestro-v3.0.0.csv"
midi_base_path = "./dataset/MAESTRO_Dataset/maestro-v3.0.0"
output_dir = "training_output"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)

# --- Hyperparameters ---
h = 128  # MIDI notes (height)
w = 16  # Time steps (width)
# noise_dim = 100  TBD for noise


# NOTE added for variational autoencoders


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = md.ConvVAE(latent_dim=16).to(device)
optimiser = optim.Adam(model.parameters(), lr=1e-3)

transform = transforms.Compose([transforms.ToTensor()])  # MNIST in [0,1]
train_ds = datasets.MNIST(root=".", train=True, download=True, transform=transform)
loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)

for epoch in range(1, 11):
    model.train()
    total_loss = 0.0
    for xb, _ in loader:
        xb = xb.to(device)
        x_logits, mu, logvar = model(xb)
        loss, recon, kl = md.vae_loss(xb, x_logits, mu, logvar)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}  avg loss per batch: {total_loss / len(loader):.2f}")


model.eval()
with torch.no_grad():
    z = torch.randn(64, model.latent_dim).to(device)  # sample from N(0,I)
    logits = model.decode(z)
    samples = torch.sigmoid(logits)  # in [0,1]
    # ora samples è (64,1,28,28) pronta per visualizzare/salvare

exit()

# --- Dataset and DataLoader ---
my_dataset = dba.MaestroMIDIDataset(
    csv_file=csv_file_path, midi_base_path=midi_base_path
)
data_loader = DataLoader(
    my_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)

# ==============================================================================
# 2. Model, Optimizers, and Loss Functions
# ==============================================================================

# --- IMPORTANT NOTE ON MODEL ARCHITECTURE ---
# The training logic below requires the Discriminator to return an intermediate
# feature map for the feature matching loss. You must modify your 'md.Discriminator'
# class to support this.

# --- Instantiate the models ---
# netG = md.Generator(pitch_range=w)
# netD = md.Discriminator(pitch_range=w)
# netC = md.Conditioner(pitch_range=w)


# --- Loss functions and Optimizers ---
# Directly uses the optimized, built-in PyTorch loss functions:
#         nn.BCEWithLogitsLoss()
#         nn.MSELoss()
# Second Script (Custom Wrappers): Wraps the standard PyTorch functions
# inside its own custom functions (sigmoid_cross_entropy_with_logits, l2_loss, lrelu).
criterion_adv = nn.BCEWithLogitsLoss()  # For adversarial loss (more stable)
criterion_l2 = nn.MSELoss()  # For L2 feature matching losses

# optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
# optimizerG = optim.Adam(
#     list(netG.parameters()) + list(netC.parameters()), lr=lr, betas=(0.5, 0.999)
# )

# --- Move to GPU if available ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# netG.to(device)
# netD.to(device)
# netC.to(device)

# ==============================================================================
# 3. Training Loop
# ==============================================================================

print("\nStarting MidiNet Training Loop...")
fixed_noise = torch.randn(batch_size, noise_dim, device=device)

for epoch in range(num_epochs):
    # Initialize the condition for the first bar of each sequence in the batch
    previous_bar_melody_condition = torch.zeros((batch_size, 1, h, w), device=device)

    for i, batch in enumerate(data_loader):
        if batch is None:
            print(f"Skipping a bad batch at index {i}.")
            continue

        real_melody_bar = batch["real_melody_bar"].to(device)
        chord_condition = batch["chord_condition"].to(device)

        # --- Log to console ---
        if i % 100 == 0:
            print(
                f"[{epoch + 1}/{num_epochs}][{i}/{len(data_loader)}] | "
                # f"Loss_D: {errD.item():.4f} | Loss_G: {errG.item():.4f} | "
                # f"D(x): {D_x:.4f} | D(G(z)): {D_G_z1:.4f} -> {D_G_z2:.4f}"
            )

    # --- End of Epoch ---
    print(f"\n===> Epoch {epoch + 1} Complete\n")


print("Training finished.")
