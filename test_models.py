import torch
from torch.utils.data import DataLoader
import os
import argparse

# Import locali
import models as md
import DBAdapters as dba
from evaluator import MusicEvaluator



# --- Argument Parser Setup ---
parser = argparse.ArgumentParser(
    description="Evaluating the test dataset with 3 different metrics",
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
parser.add_argument("--latent-dim", type=int, default=32)
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--lr-kind", type=str, default="2", choices=["1","2","3"], help= "tipo di LR; 1 sta per 10^-2, 2 sta per 10^-3, 3 stra per 10^-4")

parser.add_argument("--batch-size", type=int, default=32)

args = parser.parse_args()

# --- Config ---
MODEL_PATH = f"training_output/models/{args.model}_vae_lr{args.lr_kind}.pth"
PROCESSED_DIR = "./dataset/MAESTRO_Dataset/processed"
BATCH_SIZE = args.batch_size
LATENT_DIM = args.latent_dim
 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset ---
print("Loading dataset...")
my_dataset = dba.MaestroPreprocessedDataset(processed_dir=PROCESSED_DIR)

# Usa solo test set per evaluation
test_size = int(0.1 * len(my_dataset))
train_size = int(0.8 * len(my_dataset))
val_size = len(my_dataset) - train_size - test_size
_, _, test_set = torch.utils.data.random_split(my_dataset, [train_size, val_size, test_size])

test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# --- Loading model ---
if args.model == "conv":
    model = md.ConvVAE(latent_dim=LATENT_DIM).to(DEVICE)
elif args.model == "res":
    model = md.ResVAE(latent_dim=LATENT_DIM).to(DEVICE)
else:
    raise ValueError(f"Unknown model type: {args.model}")
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"Model loaded from {MODEL_PATH}")

# --- Evaluation ---
evaluator = MusicEvaluator(model, DEVICE)
metrics = evaluator.evaluate(test_loader, num_batches=10, latent_dim=LATENT_DIM)

print("Evaluation Results:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")