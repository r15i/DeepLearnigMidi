import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import pretty_midi
import matplotlib.pyplot as plt
from collections import defaultdict


class MaestroMIDIDataset(Dataset):
    def __init__(
        self,
        csv_file="./dataset/MAESTRO_Dataset/maestro-v3.0.0.csv",
        midi_base_path="./dataset/MAESTRO_Dataset/maestro-v3.0.0",
        transform=None,
        h=128,  # MIDI notes (0-127)
        w=16,  # Time steps per bar (16th notes)
    ):
        self.midi_base_path = midi_base_path
        self.transform = transform
        self.h = h
        self.w = w

        # Load and filter the CSV file
        data_frame = pd.read_csv(csv_file)
        self.data_frame = data_frame.dropna(subset=["midi_filename"])

        print(
            "Initializing dataset... This may take a moment as we pre-process MIDI files."
        )
        self.all_samples = self._create_samples()
        print(
            f"Initialization complete. Found {len(self.all_samples)} valid (previous_bar, current_bar) pairs."
        )

    def _create_samples(self):
        """
        Pre-processes all MIDI files to create a list of all valid training samples.
        Each sample is a tuple: (previous_bar_matrix, current_bar_matrix, chord_vector).
        """
        all_samples = []
        for idx in range(len(self.data_frame)):
            midi_relative_path = self.data_frame.iloc[idx]["midi_filename"]
            midi_full_path = os.path.join(self.midi_base_path, midi_relative_path)

            if not os.path.exists(midi_full_path):
                continue

            try:
                midi_data = pretty_midi.PrettyMIDI(midi_full_path)

                # --- 1. Get Bar Timings ---
                bar_times = midi_data.get_downbeats()
                if len(bar_times) < 2:
                    continue

                # --- 2. Create a full piano roll for the entire song ---
                piano_roll = midi_data.get_piano_roll(fs=100)
                piano_roll_binary = (piano_roll > 0).astype(np.float32)

                # --- 3. Slice the piano roll into bars and create samples ---
                for i in range(len(bar_times) - 1):
                    start_time = bar_times[i]
                    end_time = bar_times[i + 1]

                    # Convert bar start/end times to piano roll indices
                    start_idx = int(start_time * 100)
                    end_idx = int(end_time * 100)

                    # Slice the bar from the full piano roll
                    bar_slice = piano_roll_binary[:, start_idx:end_idx]
                    if bar_slice.shape[1] == 0:
                        continue

                    # Resample the bar to the desired dimensions (h x w)
                    time_indices = np.linspace(
                        0, bar_slice.shape[1] - 1, self.w, dtype=int
                    )
                    current_bar_matrix = bar_slice[:, time_indices]

                    # --- Get the previous bar ---
                    if i == 0:
                        # For the first bar, the previous bar is all zeros
                        previous_bar_matrix = np.zeros(
                            (self.h, self.w), dtype=np.float32
                        )
                    else:
                        # Get the previous bar's slice and resample it
                        prev_start_time = bar_times[i - 1]
                        prev_end_time = bar_times[i]
                        prev_start_idx = int(prev_start_time * 100)
                        prev_end_idx = int(prev_end_time * 100)
                        prev_bar_slice = piano_roll_binary[
                            :, prev_start_idx:prev_end_idx
                        ]
                        if prev_bar_slice.shape[1] == 0:
                            continue
                        prev_time_indices = np.linspace(
                            0, prev_bar_slice.shape[1] - 1, self.w, dtype=int
                        )
                        previous_bar_matrix = prev_bar_slice[:, prev_time_indices]

                    # --- Extract Chord for the current bar ---
                    chord_vector = self._extract_chord_from_bar(current_bar_matrix)

                    # Add the complete sample to our list
                    all_samples.append(
                        (previous_bar_matrix, current_bar_matrix, chord_vector)
                    )

            except Exception as e:
                # print(f"Could not process file {midi_full_path}: {e}")
                continue

        return all_samples

    def _extract_chord_from_bar(self, bar_matrix):
        """
        Analyzes a bar's piano roll to determine the most likely chord.
        Returns a 13-dimensional vector (12 for pitch class, 1 for major/minor).
        This is a simplified chord detection method.
        """
        notes_present = np.where(np.sum(bar_matrix, axis=1) > 0)[0]
        if len(notes_present) == 0:
            return np.zeros(13, dtype=np.float32)

        # Get the pitch classes (0-11) of the notes present
        pitch_classes = sorted(list(set(note % 12 for note in notes_present)))

        if not pitch_classes:
            return np.zeros(13, dtype=np.float32)

        # Use the lowest note as the potential root
        root = pitch_classes[0]

        # Determine quality (major/minor) based on the third
        is_major = False
        major_third = (root + 4) % 12
        minor_third = (root + 3) % 12

        if major_third in pitch_classes:
            is_major = True
        elif minor_third not in pitch_classes:
            # If neither third is present, default to major
            is_major = True

        chord_vector = np.zeros(13, dtype=np.float32)
        chord_vector[root] = 1
        chord_vector[12] = 1.0 if is_major else 0.0

        return chord_vector

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Retrieve the pre-processed sample
        previous_bar, current_bar, chord = self.all_samples[idx]

        # The .unsqueeze(0) adds the required "channel" dimension.
        # Shape changes from [128, 16] to [1, 128, 16].
        # The DataLoader will then batch this to [batch_size, 1, 128, 16].
        sample = {
            "real_melody_bar": torch.from_numpy(current_bar).unsqueeze(0),
            "previous_bar_melody_condition": torch.from_numpy(previous_bar).unsqueeze(
                0
            ),
            "chord_condition": torch.from_numpy(chord),  # Chord vector remains 1D
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


# Helper functions for visualization and collating
# These were missing in the original code but are needed for the debug script.
def collate_fn_skip_error(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else None


def visualize_midi(piano_roll_tensor):
    """
    Visualize a piano roll tensor as a heatmap.
    """
    plt.figure(figsize=(10, 5))
    piano_roll = piano_roll_tensor.squeeze().cpu().numpy().T
    plt.imshow(
        piano_roll,
        aspect="auto",
        origin="lower",
        cmap="binary",
        interpolation="nearest",
    )
    plt.title("Piano Roll Visualization")
    plt.xlabel("Pitch (MIDI Note)")
    plt.ylabel("Time Step")
    plt.show()


# --- Comprehensive Debugging Test Script ---
if __name__ == "__main__":
    # --- Setup ---
    output_dir = "debug_output"
    os.makedirs(output_dir, exist_ok=True)

    # NOTE: You will need to change these paths to point to your local dataset
    csv_path = "./dataset/MAESTRO_Dataset/maestro-v3.0.0.csv"
    midi_path = "./dataset/MAESTRO_Dataset/maestro-v3.0.0"

    print("--- Initializing Eager Loading Dataset for Debugging ---")

    # The MaestroMIDIDataset now loads all data at initialization
    eager_dataset = MaestroMIDIDataset(csv_file=csv_path, midi_base_path=midi_path)

    eager_loader = DataLoader(
        eager_dataset,
        batch_size=4,
        shuffle=True,  # You can shuffle the eager dataset
        num_workers=0,
        collate_fn=collate_fn_skip_error,
    )

    print("\nIterating through DataLoader...")
    for i, batch in enumerate(eager_loader):
        if batch is None:
            print(f"Batch {i + 1}: SKIPPED.")
            continue

        print(f"\n--- Processing Batch {i + 1} ---")
        first_item_melody = batch["real_melody_bar"][0]
        first_item_prev_melody = batch["previous_bar_melody_condition"][0]

        print("\n  --- Debugging first item in batch ---")
        print(f"real_melody_bar shape: {first_item_melody.shape}")
        print(f"previous_bar_melody_condition shape: {first_item_prev_melody.shape}")

        visualize_midi(first_item_melody)

        if i >= 2:
            print("\n--- Comprehensive debug test finished successfully! ---")
            break
