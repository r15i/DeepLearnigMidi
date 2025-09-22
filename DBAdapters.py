# In DBAdapters.py, replace MaestroMIDIDataset with this new class:

import torch
from torch.utils.data import Dataset
import os
from pathlib import Path
import matplotlib.pyplot as plt


class MaestroPreprocessedDataset(Dataset):
    """
    Loads data directly from preprocessed .pt files in a directory.
    Each .pt file is expected to contain a tensor of piano roll segments.
    """

    def __init__(self, processed_dir="./dataset/MAESTRO_Dataset/processed"):
        self.processed_dir = Path(processed_dir)

        print(f"Looking for preprocessed files in: {self.processed_dir}")
        self.file_paths = list(self.processed_dir.glob("*.pt"))

        if not self.file_paths:
            raise FileNotFoundError(
                f"No .pt files found in {self.processed_dir}. Please run the preprocessing script first."
            )

        print(
            "Loading all preprocessed segments into memory... This might take a moment."
        )
        # This loop loads all segments from all files into one giant list.
        self.all_segments = []
        for path in self.file_paths:
            segments_tensor = torch.load(path)
            # We add each segment individually to the list.
            self.all_segments.extend(list(segments_tensor))

        print(
            f"Successfully loaded {len(self.all_segments)} segments from {len(self.file_paths)} files."
        )

    def __len__(self):
        return len(self.all_segments)

    def __getitem__(self, idx):
        # Retrieve the pre-loaded segment and add the channel dimension
        segment = self.all_segments[idx]  # Shape: (128, 16)

        # Unsqueeze adds the channel dimension: (1, 128, 16)
        return {"real_melody_bar": segment.unsqueeze(0).float()}


def visualize_midi(piano_roll_tensor, title="Piano Roll Visualization"):
    """
    Visualizes a piano roll tensor as a heatmap plot.
    """
    # Ensure tensor is on the CPU and detached from the graph for plotting
    if piano_roll_tensor.is_cuda:
        piano_roll_tensor = piano_roll_tensor.cpu()
    piano_roll = piano_roll_tensor.squeeze().detach().numpy()

    plt.figure(figsize=(10, 8))
    plt.imshow(
        piano_roll,
        aspect="auto",
        origin="lower",
        cmap="binary",  # Use 'binary' for black and white
        interpolation="nearest",
    )
    plt.title(title, fontsize=16)
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("MIDI Pitch", fontsize=12)
    plt.colorbar()
    plt.show()
