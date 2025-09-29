# main.py

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import torchvision.utils as vutils
from tqdm import tqdm
import argparse

# Local imports
import DBAdapters as dba
import models as md
from evaluator import MusicEvaluator  


def main():
    # ==============================================================================
    # 1. Configuration and Data Loading
    # ==============================================================================

    # --- Paths and Directories ---
    # NOTE: We now point directly to the directory with the preprocessed .pt files.
    PROCESSED_DIR = "./dataset/MAESTRO_Dataset/processed"  # point directly to the processed directory
    OUTPUT_DIR = "training_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)

    # --- Argument Parser Setup ---
    parser = argparse.ArgumentParser(
        description="Preprocess, train a VAE, and visualize the MAESTRO dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Shows default values in help message
    )

    # --- VAE model ---
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="conv",
        choices=["conv", "res"],
        help="Tipo di VAE da usare: 'conv' o 'res'.",
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

    # Split into training (80%), validation (10%), and testing (10%)
    train_size = int(0.8 * len(my_dataset))
    val_size = int(0.1 * len(my_dataset))
    test_size = len(my_dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(
        my_dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # Adjust based on your system's capabilities
        pin_memory=True,
        # collate_fn is not strictly needed now but doesn't hurt
        # collate_fn=dba.collate_fn_skip_error,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # Adjust based on your system's capabilities
        pin_memory=True,
        # collate_fn is not strictly needed now but doesn't hurt
        # collate_fn=dba.collate_fn_skip_error,
    )

    test_loader = DataLoader(
        test_set,
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

    if args.model == "conv":
        model = md.ConvVAE(latent_dim=LATENT_DIM).to(device)
        print("\n Loading ConvVAE model...")
    elif args.model == "res":
        model = md.ResVAE(latent_dim=LATENT_DIM).to(device)
        print("\n Loading ResVAE model...")
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # --- Optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ======================================================================
    # Early Stopping setup
    # ======================================================================
    best_val_loss = float("inf")
    patience = 10  # numero massimo di epoche senza miglioramento
    counter = 0
    delta = 0.5  # minimo miglioramento richiesto sulla val_loss

    # ==============================================================================
    # 3. Training  & Validation Loop
    # ==============================================================================
    # Loss Vectors
    train_losses, train_recons, train_kls = [], [], []
    val_losses, val_recons, val_kls = [], [], []

    print("\n Starting VAE Training Loop...")
    for epoch in range(NUM_EPOCHS):
        # --- Training ---
        model.train()
        total_loss, total_recon, total_kl = 0, 0, 0

        # Using tqdm for a nice progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

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
        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon / len(train_loader)
        avg_kl = total_kl / len(train_loader)

        # Question: è meglio plottare la loss o avg_loss?
        train_losses.append(avg_loss)
        train_recons.append(avg_recon)
        train_kls.append(avg_kl)

        print(f"\n===> Epoch {epoch + 1} Complete:")
        print(
            f"    Average Loss: {avg_loss:.4f} | Avg Recon: {avg_recon:.2f} | Avg KL Div: {avg_kl:.2f}\n"
        )

        # --- Validation ---
        model.eval()
        val_loss, val_recon, val_kl = 0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                x = batch["real_melody_bar"].to(device)

                # Forward pass
                x_logits, mu, logvar = model(x)
                loss, recon, kl = md.vae_loss(x, x_logits, mu, logvar, beta=BETA)

                val_loss += loss.item()
                val_recon += recon.item()
                val_kl += kl.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_recon = val_recon / len(val_loader)
        avg_val_kl = val_kl / len(val_loader)
        val_losses.append(avg_val_loss)
        val_recons.append(avg_val_recon)
        val_kls.append(avg_val_kl)

        print(
            f"    Average Val Loss:   {avg_val_loss:.4f} | Avg Recon: {avg_val_recon:.2f} | Avg KL Div: {avg_val_kl:.2f}\n"
        )

        # --- Early Stopping ---
        if avg_val_loss < best_val_loss - delta:
            best_val_loss = avg_val_loss
            counter = 0
            model_save_path = os.path.join(
                OUTPUT_DIR, "models", f"{args.model}_vae_final.pth"
            )
            torch.save(model.state_dict(), model_save_path)
            print(" New best model saved!")
        else:
            counter += 1
            print(f" EarlyStopping counter: {counter}/{patience}")
            if counter >= patience:
                print(" Early stopping triggered!")
                break

    print("✅ Training  and Validation finished.")

    # ==============================================================================
    # 4. Train VS Val loss & Recon and KL term SUBPLOTS
    # ==============================================================================

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss totale
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (-ELBO)")
    plt.title("Training vs Validation Loss")
    plt.legend()

    # Recon & KL
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_recons, label="Train Recon")
    plt.plot(epochs, val_recons, label="Val Recon")
    plt.plot(epochs, train_kls, label="Train KL")
    plt.plot(epochs, val_kls, label="Val KL")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Recon vs KL")
    plt.legend()

    # Saving the figure
    print(" Saving Losses plot...")

    loss_plot_dir = os.path.join(OUTPUT_DIR, "loss_plot")
    os.makedirs(loss_plot_dir, exist_ok=True)
    plot_save_path = os.path.join(
        loss_plot_dir, f"{args.model}_loss_plot.png"
    )  # nome dell'immagine da modificare in base al tipo di training
    plt.savefig(plot_save_path, dpi=300, bbox_inches="tight")

    # ==============================================================================
    # 5. Generate Samples
    # ==============================================================================

    print("\n Generating new MIDI samples from random noise...")
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

    # ==============================================================================
    # 6. Model Evaluation
    # ==============================================================================
    # ----Instatiate the evaluator------
    evaluator = MusicEvaluator(model, device)

    # compute the metrics
    metrics = evaluator.evaluate(test_loader, num_batches=10, latent_dim=LATENT_DIM)

    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
