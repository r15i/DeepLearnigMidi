import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import torchvision.utils as vutils

# # dataset MNIST (esempio), training loop minimale
# from torchvision import datasets, transforms


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
batch_size = 32 #depends on the GPU
num_epochs = 10
# noise_dim = 100  TBD for noise



# --- Dataset and DataLoader ---
# num_workers can be changed (depends on what you run the code)
my_dataset = dba.MaestroMIDIDataset(
    csv_file=csv_file_path, midi_base_path=midi_base_path
)
data_loader = DataLoader(
    my_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=dba.collate_fn_skip_error,
)


# ==============================================================================
# 2. Model, Optimizers, and Loss Functions
# ==============================================================================

# --- IMPORTANT NOTE ON MODEL ARCHITECTURE ---
# The training logic below requires to modify the latent dimension depending on how much general should be
# should be the output. If the dataset contains different important features (different patterns/ poliphonic music)
# an higher dimension of teh latent variable is needed [>=32]

# --- Instantiate the model ---
model = md.ConvVAE(latent_dim=32)


# --- Loss functions and Optimizers ---
# Directly uses the optimized, built-in PyTorch loss functions:
#         nn.BCEWithLogitsLoss(): in our case is enough
#         nn.MSELoss(): better if we want to add the velocity condition
# EMILIO check the two line below
# Second Script (Custom Wrappers): Wraps the standard PyTorch functions
# inside its own custom functions (binary_cross_entropy_with_logits, l2_loss, lrelu).
# criterion_adv = nn.BCEWithLogitsLoss()  # For adversarial loss (more stable)
# criterion_l2 = nn.MSELoss()  # For L2 feature matching losses

optimiser = optim.Adam(model.parameters(), lr=1e-3)
# optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
# optimizerG = optim.Adam(
#     list(netG.parameters()) + list(netC.parameters()), lr=lr, betas=(0.5, 0.999)
# )

# --- Move to GPU if available ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)
# netG.to(device)
# netD.to(device)
# netC.to(device)

# ==============================================================================
# 3. Training Loop
# ==============================================================================

print("\nStarting MidiNet Training Loop...")
#fixed_noise = torch.randn(batch_size, noise_dim, device=device) I don't remember EMILIO HELP
for epoch in range(num_epochs):
    model.train()
    total_loss, total_recon, total_kl = 0, 0, 0
    for i, batch in enumerate(data_loader):
        if batch is None:
            print(f"Skipping a bad batch at index {i}.")
            continue

        x = batch["real_melody_bar"].to(device)   # (B,1,128,16)

        # Forward
        x_logits, mu, logvar = model(x)
        loss, recon, kl = md.vae_loss(x, x_logits, mu, logvar)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        # --- Log to console ---
        if i % 100 == 0:
            print(
                f"[{epoch + 1}/{num_epochs}][{i}/{len(data_loader)}] | "
                # f"Loss_D: {errD.item():.4f} | Loss_G: {errG.item():.4f} | "
                # f"D(x): {D_x:.4f} | D(G(z)): {D_G_z1:.4f} -> {D_G_z2:.4f}"
            )

    # --- End of Epoch ---
    print(f"\n===> Epoch {epoch + 1} Complete\n",
          f"\n recon={total_recon/len(data_loader):.2f}\n ",
          f"\n kl={total_kl/len(data_loader):.2f}\n")
    
# ==============================================================================
# 4. Evaluation
# ==============================================================================
model.eval()
with torch.no_grad():
    z = torch.randn(8, model.latent_dim).to(device)
    logits = model.decode(z)
    samples = torch.sigmoid(logits)  # (8,1,128,16)

    print("Generated samples shape:", samples.shape)
    # Visualizza il primo sample
    dba.visualize_midi(samples[0].cpu())

exit()

# ==============================================================================
# 3. Training Loop with chord
# ==============================================================================

print("\nStarting MidiNet Training Loop with conditioned bars...")
#fixed_noise = torch.randn(batch_size, noise_dim, device=device) I don't remember EMILIO HELP
num_epochs = 10
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
