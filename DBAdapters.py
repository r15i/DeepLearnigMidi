import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os  # To join paths
import pretty_midi  # For MIDI file processing
import matplotlib.pyplot as plt

# NOTE: this class rappresent how we parse the datesets


# NOTE:
# Maestro headers:
# canonical_composer,canonical_title,split,year,midi_filename,audio_filename,duration
# Alban Berg,Sonata Op. 1,train,2018,2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.midi,2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.wav,698.661160312


class MaestroMIDIDataset(Dataset):
    def __init__(
        self,
        csv_file="./dataset/MAESTRO_Dataset/maestro-v3.0.0.csv",
        midi_base_path="./dataset/MAESTRO_Dataset/maestro-v3.0.0",
        transform=None,
    ):
        """
        Args:
            csv_file (string): Path to the csv file (e.g., maestro-v1.0.0.csv).
            midi_base_path (string): The base directory where MIDI files are stored.
                                     e.g., if midi_filename in CSV is '2018/xyz.midi',
                                     and your files are in '/path/to/maestro/2018/xyz.midi',
                                     then midi_base_path would be '/path/to/maestro'.
            transform (callable, optional): Optional transform to be applied
                on a sample. This could be for data augmentation or further processing.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.midi_base_path = midi_base_path
        self.transform = transform

        # Filter out rows where midi_filename might be missing or invalid if necessary
        self.data_frame = self.data_frame.dropna(subset=["midi_filename"])

        # Store the full paths to MIDI files
        self.midi_file_paths = [
            os.path.join(self.midi_base_path, filename)
            for filename in self.data_frame["midi_filename"]
        ]

        print(
            f"Initialized MaestroMIDIDataset with {len(self.midi_file_paths)} MIDI files."
        )
        # Optional: You might want to filter out non-existent files here
        # self.midi_file_paths = [p for p in self.midi_file_paths if os.path.exists(p)]
        # print(f"After filtering, {len(self.midi_file_paths)} MIDI files remain.")

    def __len__(self):
        return len(self.midi_file_paths)

    # TODO: implement for the current nn architecture
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        midi_path = self.midi_file_paths[idx]

        try:
            # this will be called to actually retrive the data to serve to the network
            # so it will be used to retrive the de file midi , format it and give it to the network

            # For MidiNet, the input is a 2-D matrix representing notes over time in a bar.
            # The dimensions are h (number of MIDI notes, 128 in their implementation)
            # by w (number of time steps in a bar, 16 for sixteenth notes).

            # MidiNet generates melodies one bar after another.
            # When conditioning on a previous bar's melody, that melody is also an h-by-w matrix.
            # When conditioning on a chord sequence, a 13-dimensional vector is used per bar.

            # TODO:
            # Real Melody Bar (X): A binary matrix of shape h×w (128 MIDI notes × 16 time steps).
            # This represents a single bar of real melody from the training dataset.
            # Previous Bar Melody (2-D Condition): A binary matrix of shape h×w (128 MIDI notes × 16 time steps).

            # TODO:
            # This is either a real melody bar from the dataset (e.g., the bar immediately preceding
            # X) or an all-zeros matrix if no prior context is provided (e.g., for the very first bar in a sequence).

            # TODO:
            # Chord Sequence (1-D Condition): A 13-dimensional vector. This represents the chord for the current bar,
            # with 12 dimensions for the key and 1 for major/minor type.

            # NOTE:
            # this can be probably be generated at training time
            # Random Noise (z): A vector of random Gaussian noise of length 100.

            # Handle for the file
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            h = 128  # Number of MIDI notes
            w = 16  # Number of time steps in a bar (sixteenth notes)

            # NOTE: need to account for the tempo based on the midi
            # Let's assume a bar duration (e.g., 4 beats at 120bpm = 2 seconds).
            # fs = w / bar_duration (e.g., 16 time steps / 2 seconds = 8 fs)
            # fs (frames per second) = w / bar_duration_in_seconds = 16 / 2 = 8
            fs_for_bar = 8  # This 'fs' is conceptual for a single bar's representation.
            # fs (frames per second): This argument controls the temporal resolution of the piano roll.
            # fs=8 --> 8 slices per sec
            full_piano_roll = midi_data.get_piano_roll(fs=fs_for_bar)

            # Ensure the piano roll has enough time steps for one bar.
            # If not, pad or skip (depending on your dataset strategy).
            if full_piano_roll.shape[1] >= w:
                # Take the first 'w' time steps for simplicity as one bar
                bar_piano_roll = full_piano_roll[:, :w]
            else:
                # Pad with zeros if the MIDI is shorter than a bar (unlikely for full songs)
                # Or, handle this by returning None and filtering in DataLoader
                bar_piano_roll = np.zeros((h, w))
                bar_piano_roll[:, : full_piano_roll.shape[1]] = full_piano_roll

            # Convert to binary representation as MidiNet often uses binary matrices.
            # (Note: velocity information is neglected for simplicity in their implementation).
            melody_matrix = (bar_piano_roll > 0).astype(np.float32)  # Binary (0 or 1)
            # Convert to PyTorch tensor. MidiNet uses a 2-D matrix for notes and time steps.
            # It will be h-by-w.
            # So, the shape is (128, 16) for a single bar.
            processed_melody_tensor = torch.from_numpy(melody_matrix).float()

            # NOTE: Placeholder, assume no prior bar for this sample all zero
            # this is the previous bar, probably this can be easily implemented by
            # using a global variable that tracks the previusly used bar
            # if not it is needed to re retrive the current midi and previous each time
            previous_bar_melody = torch.zeros((h, w)).float()

            # NOTE: Placeholder, assuming random chord
            # Dummy chord condition (1-D condition).
            # random 13-dimensional vector.
            chord_condition = torch.randint(
                0, 2, (13,)
            ).float()  # Random binary chord (major/minor, key)

            # The 'sample' returned by __getitem__ should contain everything needed for a single training step.
            # For MidiNet, this typically involves:
            # 1. A real melody bar (X) to train the discriminator.
            # 2. Conditional information (previous_bar_melody, chord_condition) for the generator.

            sample = {
                "real_melody_bar": processed_melody_tensor,  # X from p_data(X)
                "previous_bar_melody_condition": previous_bar_melody,  # 2-D condition for Generator
                "chord_condition": chord_condition,  # 1-D condition for Generator
            }

            if self.transform:
                sample = self.transform(sample)

            return sample

        except Exception as e:
            print(f"Error processing MIDI file {midi_path}: {e}")
            # Handle corrupted files: you might return None, or raise an error,
            # or skip this sample. Returning None would require a custom collate_fn
            # in your DataLoader to filter out None values.
            return None


# TODO: here goes the tests for each db
# NOTE:
# EACH OBJECT IS AN ADAPTER FOR THE SINGLE IMPLEMENTATION OF THE NN
# so we need an adapter for each couple (DB, INPUT_OF_THE_MODEL)
# Test to verify dbs and library:

if __name__ == "__main__":
    import DBAdapters as dba

    csv_file_path = "./dataset/MAESTRO_Dataset/maestro-v3.0.0.csv"
    midi_base_path = "./dataset/MAESTRO_Dataset/maestro-v3.0.0"
    my_dataset = dba.MaestroMIDIDataset(
        csv_file=csv_file_path, midi_base_path=midi_base_path
    )

    batch_size = 4
    shuffle = True
    num_workers = 0

    data_loader = DataLoader(
        my_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    plot_output_dir = "debug_plots"
    os.makedirs(plot_output_dir, exist_ok=True)

    print("\nIterating through DataLoader:")
    for epoch in range(1):
        print(f"--- Epoch {epoch + 1} ---")
        for i, batch in enumerate(data_loader):
            if batch is None:
                print(f"Batch {i + 1}: Skipped due to error in loading a MIDI file.")
                continue

            real_melody_bar = batch["real_melody_bar"]
            previous_bar_melody_condition = batch["previous_bar_melody_condition"]
            chord_condition = batch["chord_condition"]

            print(f"Batch {i + 1}:")
            print(f"  Real Melody Bar shape: {real_melody_bar.shape}")
            print(
                f"  Previous Bar Melody Condition shape: {previous_bar_melody_condition.shape}"
            )
            print(f"  Chord Condition shape: {chord_condition.shape}")

            # --- ADDING RAW MATRIX PRINTING FOR DEBUGGING ---
            print(
                f"\n--- Raw Real Melody Bar Matrix (first sample in batch {i + 1}) ---"
            )
            # Convert to NumPy and print the array
            print(real_melody_bar[0].squeeze().cpu().numpy())
            print(f"----------------------------------------------------\n")

            if np.any(previous_bar_melody_condition[0].squeeze().cpu().numpy()):
                print(
                    f"\n--- Raw Previous Bar Melody Condition Matrix (first sample in batch {i + 1}) ---"
                )
                print(previous_bar_melody_condition[0].squeeze().cpu().numpy())
                print(f"----------------------------------------------------\n")

            print(
                f"\n--- Raw Chord Condition Vector (first sample in batch {i + 1}) ---"
            )
            print(chord_condition[0].squeeze().cpu().numpy())
            print(f"----------------------------------------------------\n")

            # --- Plotting and Saving Images (as before, these ARE matrix plots) ---

            melody_to_plot = real_melody_bar[0].squeeze().cpu().numpy()
            fig1 = plt.figure(figsize=(8, 4))
            plt.imshow(melody_to_plot, aspect="auto", origin="lower", cmap="binary")
            plt.title(
                f"Epoch {epoch + 1}, Batch {i + 1}: Real Melody Bar (First Sample)"
            )
            plt.xlabel("Time Steps (w=16)")
            plt.ylabel("MIDI Notes (h=128)")
            plt.colorbar(label="Note On (1) / Note Off (0)")
            fig1.savefig(
                os.path.join(
                    plot_output_dir,
                    f"epoch{epoch + 1}_batch{i + 1}_real_melody_plot.png",
                )
            )
            plt.close(fig1)

            condition_to_plot = previous_bar_melody_condition[0].squeeze().cpu().numpy()
            if np.any(condition_to_plot):
                fig2 = plt.figure(figsize=(8, 4))
                plt.imshow(
                    condition_to_plot, aspect="auto", origin="lower", cmap="binary"
                )
                plt.title(
                    f"Epoch {epoch + 1}, Batch {i + 1}: Previous Bar Melody Condition (First Sample)"
                )
                plt.xlabel("Time Steps (w=16)")
                plt.ylabel("MIDI Notes (h=128)")
                plt.colorbar(label="Note On (1) / Note Off (0)")
                fig2.savefig(
                    os.path.join(
                        plot_output_dir,
                        f"epoch{epoch + 1}_batch{i + 1}_prev_bar_condition_plot.png",
                    )
                )
                plt.close(fig2)

            chord_to_plot = chord_condition[0].squeeze().cpu().numpy()
            fig3 = plt.figure(figsize=(6, 2))
            plt.bar(range(len(chord_to_plot)), chord_to_plot)
            plt.title(
                f"Epoch {epoch + 1}, Batch {i + 1}: Chord Condition (First Sample)"
            )
            plt.xlabel("Chord Dimension")
            plt.ylabel("Value")
            plt.xticks(range(13), [f"Dim {j}" for j in range(12)] + ["Type"])
            fig3.savefig(
                os.path.join(
                    plot_output_dir,
                    f"epoch{epoch + 1}_batch{i + 1}_chord_condition_plot.png",
                )
            )
            plt.close(fig3)

            if i >= 1:
                break
