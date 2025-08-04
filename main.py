import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import torchvision.utils as vutils

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
noise_dim = 100
n_filters = 256
batch_size = 32
lr = 0.0002
num_epochs = 20
fm_l2_weight = 0.1  # Feature matching L2 loss weight
mean_img_l2_weight = 0.01  # Mean image L2 loss weight

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
netG = md.Generator(pitch_range=w)
netD = md.Discriminator(pitch_range=w)
netC = md.Conditioner(pitch_range=w)


# --- Loss functions and Optimizers ---
# Directly uses the optimized, built-in PyTorch loss functions:
#         nn.BCEWithLogitsLoss()
#         nn.MSELoss()
# Second Script (Custom Wrappers): Wraps the standard PyTorch functions
# inside its own custom functions (sigmoid_cross_entropy_with_logits, l2_loss, lrelu).
criterion_adv = nn.BCEWithLogitsLoss()  # For adversarial loss (more stable)
criterion_l2 = nn.MSELoss()  # For L2 feature matching losses

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(
    list(netG.parameters()) + list(netC.parameters()), lr=lr, betas=(0.5, 0.999)
)

# --- Move to GPU if available ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
netG.to(device)
netD.to(device)
netC.to(device)

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

        # Ensure batch size consistency, especially for the last batch
        current_batch_size = real_melody_bar.size(0)
        if current_batch_size != batch_size:
            previous_bar_melody_condition = previous_bar_melody_condition[
                :current_batch_size
            ]
            current_fixed_noise = fixed_noise[:current_batch_size]
        else:
            current_fixed_noise = fixed_noise

        real_melody_bar = real_melody_bar.unsqueeze(1)  # Add channel dimension

        ############################
        # (1) Update Discriminator
        ############################
        netD.zero_grad()

        # --- Train with real batch ---
        label_real = torch.full(
            (current_batch_size,), 0.9, dtype=torch.float, device=device
        )  # One-sided label smoothing
        D_logits_real, fm_real = netD(real_melody_bar)
        errD_real = criterion_adv(D_logits_real, label_real)
        errD_real.backward(retain_graph=True)  # Retain graph to use fm_real later
        D_x = torch.sigmoid(D_logits_real).mean().item()

        # --- Train with fake batch ---
        noise = torch.randn(current_batch_size, noise_dim, device=device)
        conditioner_output = netC(previous_bar_melody_condition)
        fake_melody_bar = netG(noise, conditioner_output, chord_condition)

        label_fake = torch.full(
            (current_batch_size,), 0.0, dtype=torch.float, device=device
        )
        D_logits_fake, _ = netD(fake_melody_bar.detach())
        errD_fake = criterion_adv(D_logits_fake, label_fake)
        errD_fake.backward()

        errD = errD_real + errD_fake
        optimizerD.step()
        D_G_z1 = torch.sigmoid(D_logits_fake).mean().item()

        ####################################
        # (2) Update Generator & Conditioner
        ####################################
        netG.zero_grad()
        netC.zero_grad()

        # Generator wants D to output 1 for its fakes
        label_g = torch.full(
            (current_batch_size,), 1.0, dtype=torch.float, device=device
        )
        D_logits_g, fm_fake = netD(fake_melody_bar)

        # --- Calculate Generator Losses ---
        errG_adv = criterion_adv(D_logits_g, label_g)
        errG_fm_l2 = (
            criterion_l2(torch.mean(fm_fake, 0), torch.mean(fm_real, 0)) * fm_l2_weight
        )
        errG_mean_img_l2 = (
            criterion_l2(torch.mean(fake_melody_bar, 0), torch.mean(real_melody_bar, 0))
            * mean_img_l2_weight
        )

        # --- Total Generator Loss ---
        errG = errG_adv + errG_fm_l2 + errG_mean_img_l2
        errG.backward()
        optimizerG.step()
        D_G_z2 = torch.sigmoid(D_logits_g).mean().item()

        # --- Update previous bar condition for the next iteration ---
        previous_bar_melody_condition = real_melody_bar.detach().clone()

        # --- Log to console ---
        if i % 100 == 0:
            print(
                f"[{epoch + 1}/{num_epochs}][{i}/{len(data_loader)}] | "
                f"Loss_D: {errD.item():.4f} | Loss_G: {errG.item():.4f} | "
                f"D(x): {D_x:.4f} | D(G(z)): {D_G_z1:.4f} -> {D_G_z2:.4f}"
            )

    # --- End of Epoch ---
    print(f"\n===> Epoch {epoch + 1} Complete\n")

    # Save generated image samples for inspection
    with torch.no_grad():
        conditioner_output_fixed = netC(previous_bar_melody_condition)
        fake_samples = netG(pitch_range=w).detach().cpu()
    vutils.save_image(
        fake_samples,
        f"{output_dir}/fake_samples_epoch_{epoch + 1:03d}.png",
        normalize=True,
    )

    # Save model checkpoints
    torch.save(netG.state_dict(), f"{output_dir}/models/netG_epoch_{epoch + 1}.pth")
    torch.save(netD.state_dict(), f"{output_dir}/models/netD_epoch_{epoch + 1}.pth")
    torch.save(netC.state_dict(), f"{output_dir}/models/netC_epoch_{epoch + 1}.pth")

print("Training finished.")
