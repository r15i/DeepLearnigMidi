import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import pretty_midi
import matplotlib.pyplot as plt
from collections import defaultdict


# eager loading implementation
# TODO: probably a can turn the float32 into a binary at runtime to spare space in memory
# TODO: understand better midi to be able to split it better
class MaestroMIDIDataset(Dataset):
    def __init__(
        self,
        csv_file="./dataset/MAESTRO_Dataset/maestro-v3.0.0.csv",
        midi_base_path="./dataset/MAESTRO_Dataset/maestro-v3.0.0",
        transform=None,
        h=128,  # MIDI notes
        w=16,  # Time steps per bar (16th notes)
    ):
        """
        Args:
            csv_file (string): Path to the MAESTRO csv file.
            midi_base_path (string): Base directory where MIDI files are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
            h (int): Height of the piano roll (number of MIDI pitches).
            w (int): Width of the piano roll (number of time steps per bar).
        """
        self.data_frame = pd.read_csv(csv_file)
        self.midi_base_path = midi_base_path
        self.transform = transform
        self.h = h
        self.w = w

        # Filter out rows where midi_filename might be missing
        self.data_frame = self.data_frame.dropna(subset=["midi_filename"])

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
                # Get the end time of each bar based on the MIDI's time signature changes.
                bar_times = midi_data.get_downbeats()
                if len(bar_times) < 2:
                    continue  # Skip files with less than one full bar

                # --- 2. Create a full piano roll for the entire song ---
                # NOTE: need to account for the tempo based on the midi
                # Let's assume a bar duration (e.g., 4 beats at 120bpm = 2 seconds).
                # fs = w / bar_duration (e.g., 16 time steps / 2 seconds = 8 fs)
                # fs (frames per second) = w / bar_duration_in_seconds = 16 / 2 = 8
                # fs_for_bar = (8  # This 'fs' is conceptual for a single bar's representation.)
                # fs (frames per second): This argument controls the temporal resolution of the piano roll.

                # NOTE:: need to tune the timings
                # We use a high temporal resolution (fs) to accurately capture note timings.
                piano_roll = midi_data.get_piano_roll(fs=100)
                # Binarize the piano roll (we only care if a note is on or off)jjj
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
                        continue  # Skip empty slices

                    # --- Resample the bar to the desired dimensions (h x w) ---
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
                            continue  # Skip if previous slice is empty
                        prev_time_indices = np.linspace(
                            0, prev_bar_slice.shape[1] - 1, self.w, dtype=int
                        )
                        previous_bar_matrix = prev_bar_slice[:, prev_time_indices]

                    # --- Extract Chord for the current bar ---
                    # uses an euristic to get the most possible chord
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

        # ### CORRECTION HERE ###
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


# --- Test Script ---
if __name__ == "__main__":
    csv_file_path = "./dataset/MAESTRO_Dataset/maestro-v3.0.0.csv"
    midi_base_path = "./dataset/MAESTRO_Dataset/maestro-v3.0.0"
    my_dataset = MaestroMIDIDataset(
        csv_file=csv_file_path, midi_base_path=midi_base_path
    )

    # This custom collate function filters out None values that might be returned
    # by a dataset's __getitem__ method if a file is corrupted.
    def collate_fn_skip_error(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    data_loader = DataLoader(
        my_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Set to > 0 for faster loading if your system supports it
        collate_fn=collate_fn_skip_error,
    )

    plot_output_dir = "debug_plots"
    os.makedirs(plot_output_dir, exist_ok=True)

    print("\nIterating through DataLoader to test the new Dataset class:")
    for i, batch in enumerate(data_loader):
        if batch is None:
            print(f"Batch {i + 1}: Skipped due to loading errors in all batch items.")
            continue

        print(f"--- Batch {i + 1} ---")
        real_melody_bar = batch["real_melody_bar"]
        previous_bar_melody = batch["previous_bar_melody_condition"]
        chord_condition = batch["chord_condition"]

        # ### CORRECTION HERE ###
        # The shapes printed below will now be 4D for the melody bars, which is correct.
        print(
            f"  Real Melody Bar shape: {real_melody_bar.shape}"
        )  # Should be [batch_size, 1, 128, 16]
        print(
            f"  Previous Bar Melody shape: {previous_bar_melody.shape}"
        )  # Should be [batch_size, 1, 128, 16]
        print(
            f"  Chord Condition shape: {chord_condition.shape}"
        )  # Should be [batch_size, 13]

        # --- Plotting the first sample in the batch ---
        # .squeeze() removes the channel dimension for plotting
        melody_to_plot = real_melody_bar[0].squeeze().cpu().numpy()
        fig1 = plt.figure(figsize=(8, 4))
        plt.imshow(melody_to_plot, aspect="auto", origin="lower", cmap="binary")
        plt.title(f"Batch {i + 1}: Real Melody Bar")
        plt.xlabel("Time Steps (w=16)")
        plt.ylabel("MIDI Notes (h=128)")
        plt.savefig(os.path.join(plot_output_dir, f"batch_{i + 1}_real_melody.png"))
        plt.close(fig1)

        prev_to_plot = previous_bar_melody[0].squeeze().cpu().numpy()
        if np.any(prev_to_plot):
            fig2 = plt.figure(figsize=(8, 4))
            plt.imshow(prev_to_plot, aspect="auto", origin="lower", cmap="binary")
            plt.title(f"Batch {i + 1}: Previous Bar Melody Condition")
            plt.xlabel("Time Steps (w=16)")
            plt.ylabel("MIDI Notes (h=128)")
            plt.savefig(
                os.path.join(plot_output_dir, f"batch_{i + 1}_previous_bar.png")
            )
            plt.close(fig2)

        chord_to_plot = chord_condition[0].squeeze().cpu().numpy()
        fig3 = plt.figure(figsize=(6, 2))
        plt.bar(range(13), chord_to_plot)
        plt.title(f"Batch {i + 1}: Chord Condition")
        plt.xlabel("Chord Dimension")
        plt.xticks(range(13), [f"D{j}" for j in range(12)] + ["Maj/Min"])
        plt.savefig(os.path.join(plot_output_dir, f"batch_{i + 1}_chord.png"))
        plt.close(fig3)

        if i >= 2:
            print("\nTest finished successfully.")
            break
