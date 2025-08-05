import torch
import torch.nn as nn
import os
import torchvision.utils as vutils
import numpy as np

# --- Assumed Local Imports (Your project's files) ---
# Make sure these files are in the same directory or your Python path
import DBAdapters as dba
import models as md

# ==============================================================================
# 1. Configuration
# ==============================================================================

# --- Paths ---
# IMPORTANT: Point these to your existing dataset and output directories
csv_file_path = "./dataset/MAESTRO_Dataset/maestro-v3.0.0.csv"
midi_base_path = "./dataset/MAESTRO_Dataset/maestro-v3.0.0"
model_dir = "training_output/models"
output_dir_test = "testing_output"
os.makedirs(output_dir_test, exist_ok=True)

# --- Parameters to Load ---
# *** SET THE EPOCH YOU WANT TO TEST ***
epoch_to_load = 20  # Change this to the epoch number of the saved models

# --- Model & Data Hyperparameters (Must match training script) ---
h = 128  # MIDI notes (height)
w = 16  # Time steps (width)
noise_dim = 100

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==============================================================================
# 2. Load Models with Trained Weights
# ==============================================================================

print(f"\nLoading models from epoch {epoch_to_load}...")

# --- Instantiate the models with the same architecture as in training ---
# NOTE: The pitch_range=w argument is used here to exactly match the
# instantiation in your training script.
netG = md.Generator(pitch_range=w)
netD = md.Discriminator(pitch_range=w)
netC = md.Conditioner(pitch_range=w)

# --- Construct file paths for the weights ---
path_G = os.path.join(model_dir, f"netG_epoch_{epoch_to_load}.pth")
path_D = os.path.join(model_dir, f"netD_epoch_{epoch_to_load}.pth")
path_C = os.path.join(model_dir, f"netC_epoch_{epoch_to_load}.pth")

# --- Load the state dictionaries ---
try:
    netG.load_state_dict(torch.load(path_G, map_location=device))
    netD.load_state_dict(torch.load(path_D, map_location=device))
    netC.load_state_dict(torch.load(path_C, map_location=device))
except FileNotFoundError as e:
    print(f"Error: Model file not found. {e}")
    print("Please ensure 'epoch_to_load' is set correctly and the files exist.")
    exit()

# --- Move models to device and set to evaluation mode ---
netG.to(device)
netD.to(device)
netC.to(device)

netG.eval()
netD.eval()
netC.eval()

print("Models loaded successfully.")


# ==============================================================================
# 3. Prepare Conditioning Data
# ==============================================================================

# We need a sample from the dataset to provide the conditions for the generator.
print("\nLoading a sample from the dataset for conditioning...")

# --- Load the dataset ---
my_dataset = dba.MaestroMIDIDataset(
    csv_file=csv_file_path, midi_base_path=midi_base_path
)

# --- Get a single item for conditioning ---
# You can change the index to get different conditioning data
sample_idx = 50
if len(my_dataset) <= sample_idx:
    print(
        f"Error: sample_idx {sample_idx} is out of bounds for the dataset size {len(my_dataset)}."
    )
    exit()

conditioning_data = my_dataset[sample_idx]

# This bar will act as the "previous bar" condition
# It's also our "real" sample for the discriminator to evaluate
previous_bar_condition = conditioning_data["real_melody_bar"].to(device)
chord_condition = conditioning_data["chord_condition"].to(device)

# --- Reshape for batch size of 1 and add channel dimension ---
previous_bar_condition = previous_bar_condition.unsqueeze(0).unsqueeze(
    0
)  # Shape: [1, 1, h, w]
chord_condition = chord_condition.unsqueeze(0)  # Shape: [1, h, w]

print(f"Using data from sample index {sample_idx} for conditioning.")


# ==============================================================================
# 4. Generate and Evaluate a Sample
# ==============================================================================

print("\nGenerating a new melody bar...")

with torch.no_grad():  # No need to calculate gradients
    # --- Create random noise vector ---
    noise = torch.randn(1, noise_dim, device=device)

    # --- (1) Get conditioner output ---
    conditioner_output = netC(previous_bar_condition)

    # --- (2) Generate a fake melody bar ---
    generated_bar = netG(noise, conditioner_output, chord_condition)

    # --- (3) Use the Discriminator to evaluate both real and fake bars ---
    # The discriminator returns (logits, intermediate_features)
    # We only need the logits for evaluation.

    # Evaluate the REAL bar (the one we used for conditioning)
    real_logits, _ = netD(previous_bar_condition)
    prob_real = torch.sigmoid(real_logits).item()

    # Evaluate the FAKE bar (the one we just generated)
    fake_logits, _ = netD(generated_bar)
    prob_fake = torch.sigmoid(fake_logits).item()

print("\n--- Discriminator Evaluation ---")
print(f"Probability D thinks the REAL bar is real: {prob_real:.4f}")
print(f"Probability D thinks the GENERATED bar is real: {prob_fake:.4f}")
print("--------------------------------")


# ==============================================================================
# 5. Save the Output
# ==============================================================================

# Save the generated piano-roll image
output_filename = os.path.join(
    output_dir_test, f"generated_sample_epoch_{epoch_to_load}.png"
)
vutils.save_image(generated_bar, output_filename, normalize=True)

# Also save the conditioning bar for comparison
condition_filename = os.path.join(output_dir_test, "conditioning_bar_real.png")
vutils.save_image(previous_bar_condition, condition_filename, normalize=True)

print(f"\nSuccessfully saved generated sample to: {output_filename}")
print(f"Saved the real conditioning bar for comparison to: {condition_filename}")
print("\nTesting finished.")
