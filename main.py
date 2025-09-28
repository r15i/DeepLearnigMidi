# main.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import torchvision.utils as vutils
from tqdm import tqdm
import argparse

# Local imports
import DBAdapters as dba
import models as md

# ==============================================================================
# 1. Configuration and Data Loading
# ==============================================================================

# --- Paths and Directories ---
# NOTE: We now point directly to the directory with the preprocessed .pt files.
PROCESSED_DIR = "./dataset/processed"
OUTPUT_DIR = "training_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)


# --- Argument Parser Setup ---
parser = argparse.ArgumentParser(
    description="Preprocess, train a VAE, and visualize the MAESTRO dataset.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Shows default values in help message
)

# --- VAE Training Hyperparameters ---
parser.add_argument(
    "-e", "--epochs", type=int, default=10, help="Number of training epochs."
)
parser.add_argument(
    "-b", "--batch-size", type=int, default=32, help="Batch size for training."
)
parser.add_argument(
    "-lr",
    "--learning-rate",
    type=float,
    default=1e-3,
    help="Learning rate for the Adam optimizer.",
)
parser.add_argument(
    "-ld",
    "--latent-dim",
    type=int,
    default=32,
    help="Dimension of the VAE's latent space.",
)
parser.add_argument(
    "--beta",
    type=float,
    default=1.0,
    help="Weight of the KL divergence term in the VAE loss.",
)
parser.add_argument(
    "--seq-len",
    type=int,
    default=16,
    help="Number of time steps (width) per MIDI segment.",
)

# Parse the arguments
args = parser.parse_args()


# --- Hyperparameters ---
# H is a constant based on the MIDI standard, not an argument.
H = 128
# All other hyperparameters are now loaded from the command-line arguments.
W = args.seq_len
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
LATENT_DIM = args.latent_dim
BETA = args.beta

# --- Dataset and DataLoader ---
print("Loading dataset from preprocessed .pt files...")

# Use the new, faster dataset class that loads from .pt files
# Ensure your DBAdapters.py has the MaestroPreprocessedDataset class
my_dataset = dba.MaestroPreprocessedDataset(processed_dir=PROCESSED_DIR)

data_loader = DataLoader(
    my_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,  # Adjust based on your system's capabilities
    pin_memory=True,
    # collate_fn is not strictly needed now but doesn't hurt
    # collate_fn=dba.collate_fn_skip_error,
)
print(f"Dataset loaded successfully with {len(my_dataset)} samples.")

# ==============================================================================
# 2. Model, Optimizer, and Loss
# ==============================================================================

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Instantiate the model ---
model = md.ConvVAE(latent_dim=LATENT_DIM).to(device)

# --- Optimizer ---
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==============================================================================
# 3. Training Loop
# ==============================================================================

print("\n🚀 Starting VAE Training Loop...")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss, total_recon, total_kl = 0, 0, 0

    # Using tqdm for a nice progress bar
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

    for batch in progress_bar:
        if batch is None:
            continue

        x = batch["real_melody_bar"].to(device)  # Shape: (B, 1, H, W)

        # --- Forward pass ---
        x_logits, mu, logvar = model(x)
        loss, recon, kl = md.vae_loss(x, x_logits, mu, logvar, beta=BETA)

        # --- Backward pass and optimization ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- Accumulate losses for logging ---
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()

        # Update progress bar description
        progress_bar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Recon": f"{recon.item():.2f}",
                "KL": f"{kl.item():.2f}",
            }
        )

    # --- End of Epoch ---
    avg_loss = total_loss / len(data_loader)
    avg_recon = total_recon / len(data_loader)
    avg_kl = total_kl / len(data_loader)
    print(f"\n===> Epoch {epoch + 1} Complete:")
    print(
        f"    Average Loss: {avg_loss:.4f} | Avg Recon Loss: {avg_recon:.2f} | Avg KL Div: {avg_kl:.2f}\n"
    )

print("✅ Training finished.")

# ==============================================================================
# 4. Save Model and Generate Samples
# ==============================================================================
print("💾 Saving final model...")
model_save_path = os.path.join(OUTPUT_DIR, "models", "vae_final.pth")
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

print("\n🎶 Generating new MIDI samples from random noise...")
model.eval()
with torch.no_grad():
    # Create random latent vectors
    z = torch.randn(64, model.latent_dim).to(device)

    # Decode them into piano roll logits
    logits = model.decode(z)

    # Apply sigmoid to get probabilities and create binary samples by thresholding
    samples = (torch.sigmoid(logits) > 0.5).float()  # (64, 1, 128, 16)

    # Save a grid of the generated samples
    sample_grid_path = os.path.join(OUTPUT_DIR, "generated_samples.png")
    vutils.save_image(samples, sample_grid_path, normalize=True)
    print(f"Generated samples saved to {sample_grid_path}")

    # Visualize the first generated sample
    dba.visualize_midi(samples[0], title="First Generated Sample")
