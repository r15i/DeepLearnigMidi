import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os  # To join paths
import pretty_midi  # For MIDI file processing
import matplotlib.pyplot as plt
import utils as ut
from collections import defaultdict
import sys
import subprocess

from utils import visualize_midi_custom


# NOTE: this class rappresent how we parse the datesets


# NOTE:
# Maestro headers:
# canonical_composer,canonical_title,split,year,midi_filename,audio_filename,duration
# Alban Berg,Sonata Op. 1,train,2018,2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.midi,2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.wav,698.661160312


class MaestroMIDIDatasetLazy(Dataset):
    def __init__(
        self,
        csv_file="./dataset/MAESTRO_Dataset/maestro-v3.0.0.csv",
        midi_base_path="./dataset/MAESTRO_Dataset/maestro-v3.0.0",
        transform=None,
        h=128,  # MIDI notes
        w=16,  # Time steps per bar (16th notes)
    ):
        """
        Implements "lazy loading". MIDI files are processed on-the-fly in __getitem__.
        This has a very fast startup time and low memory usage, but can be slower
        during training if data processing becomes a bottleneck.
        """
        self.midi_base_path = midi_base_path
        self.transform = transform
        self.h = h
        self.w = w

        # NOTE: reduced dataset for debug
        data_frame = pd.read_csv(csv_file).dropna(subset=["midi_filename"])[:10]

        # data_frame = pd.read_csv(csv_file).dropna(subset=["midi_filename"])[:10]

        print("Initializing lazy dataset... Scanning MIDI files to create an index.")
        # self.sample_map will store tuples of (file_path, bar_index)
        self.sample_map = []
        for idx in range(len(data_frame)):
            midi_relative_path = data_frame.iloc[idx]["midi_filename"]
            midi_full_path = os.path.join(self.midi_base_path, midi_relative_path)

            if not os.path.exists(midi_full_path):
                continue

            try:
                # We still need to open the file once to see how many bars it has.
                midi_data = pretty_midi.PrettyMIDI(midi_full_path)
                num_bars = len(midi_data.get_downbeats()) - 1
                if num_bars > 0:
                    for bar_idx in range(num_bars):
                        self.sample_map.append((midi_full_path, bar_idx))
            except Exception:
                continue

        print(
            f"Initialization complete. Found {len(self.sample_map)} total bars across all MIDI files."
        )

    def __len__(self):
        return len(self.sample_map)

    def __getitem__(self, idx):
        # check if at that position there is a tensor
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Look up the file path and bar index for the requested sample
        midi_path, bar_idx = self.sample_map[idx]

        try:
            # Pitches (128)
            # ^
            # | [0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,...]  (Note G5)
            # | [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,...]  (Note E5)
            # | [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,...]
            # | [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,...]  (Note C4)
            # +---------------------------------------------------------------------> Time (many thousands of columns)

            # 2. Load and process the MIDI file ON-THE-FLY
            midi_data = pretty_midi.PrettyMIDI(midi_path)

            # Calculates the precise start times of all musical bars within
            # that song based on its tempo and time signature.

            # Pitches (128)
            # ^
            # | [0,0,0,1,1,1,1,0 | 0,0,0,0,0,0,0,0,0 | 0,0,0,0,1,1,1,0,0,...]
            # | [0,0,0,0,0,0,0,0 | 0,1,1,1,1,1,0,0,0 | 0,0,0,0,0,0,0,0,0,...]
            # | [0,0,0,0,0,0,0,0 | 0,0,0,0,0,0,0,0,0 | 0,0,0,0,0,0,0,0,0,...]
            # | [1,1,1,1,1,1,1,1 | 1,1,1,1,1,1,1,1,1 | 1,1,1,1,1,1,1,1,1,...]
            # +------------------|-------------------|-------------------|-----> Time
            #   Bar 1            Bar 2               Bar 3
            # (0.0s -> 2.1s)   (2.1s -> 4.2s)      (4.2s -> 6.3s)

            bar_times = midi_data.get_downbeats()

            # NOTE: need to account for the tempo based on the midi
            # Let's assume a bar duration (e.g., 4 beats at 120bpm = 2 seconds).
            # fs = w / bar_duration (e.g., 16 time steps / 2 seconds = 8 fs)
            # fs (frames per second) = w / bar_duration_in_seconds = 16 / 2 = 8
            # fs_for_bar = 8  # This 'fs' is conceptual for a single bar's representation.
            # fs (frames per second): This argument controls the temporal resolution of the piano roll.
            # fs=8 --> 8 slices per sec
            piano_roll = midi_data.get_piano_roll(fs=100)

            # Simplifies the piano roll into a binary format:
            # 1 if a note was played (velocity > 0), 0 otherwise.
            piano_roll_binary = (piano_roll > 0).astype(np.float32)

            # 3. Slice and resample the CURRENT bar (relative to tempo )
            # Gets the start and end timestamps for the specific bar we need,
            # identified by bar_idx.
            # NOTE: this make an assumption on the first bar , probably would be better to take an average
            # over all
            start_time = bar_times[bar_idx]
            end_time = bar_times[bar_idx + 1]

            # Uses the timestamps (multiplied by the sampling rate fs=100) to slice the
            # exact columns corresponding to this bar from the full piano roll.
            bar_slice = piano_roll_binary[
                :, int(start_time * 100) : int(end_time * 100)
            ]

            if bar_slice.shape[1] == 0:
                # lA safety check. If the slice is empty, it returns None, indicating this sample should be skipped.
                return None  # Handle potential empty slice

            # This resamples the bar. It creates self.w (e.g., 16) evenly spaced indices and selects those columns from bar_slice
            # , resizing it to the fixed width required by the network.
            time_indices = np.linspace(0, bar_slice.shape[1] - 1, self.w, dtype=int)
            current_bar_matrix = bar_slice[:, time_indices]

            # 4. Slice and resample the PREVIOUS bar
            if bar_idx == 0:
                previous_bar_matrix = np.zeros((self.h, self.w), dtype=np.float32)
            else:
                prev_start_time = bar_times[bar_idx - 1]
                prev_end_time = bar_times[bar_idx]
                prev_bar_slice = piano_roll_binary[
                    :, int(prev_start_time * 100) : int(prev_end_time * 100)
                ]
                if prev_bar_slice.shape[1] == 0:
                    return None  # Handle potential empty slice
                prev_time_indices = np.linspace(
                    0, prev_bar_slice.shape[1] - 1, self.w, dtype=int
                )
                previous_bar_matrix = prev_bar_slice[:, prev_time_indices]

            # 5. Extract the chord and create the sample dictionary
            chord_vector = ut.extract_chord_from_bar(current_bar_matrix)

            sample = {
                "real_melody_bar": torch.from_numpy(current_bar_matrix).unsqueeze(0),
                "previous_bar_melody_condition": torch.from_numpy(
                    previous_bar_matrix
                ).unsqueeze(0),
                "chord_condition": torch.from_numpy(chord_vector),
            }

            if self.transform:
                sample = self.transform(sample)

            return sample

        except Exception as e:
            # print(f"Error processing {midi_path} at bar {bar_idx}: {e}")
            return (
                None  # Return None if there's an error, will be filtered by collate_fn
            )


# ==============================================================================
# --- Comprehensive Debugging Test Script ---
# ==============================================================================
if __name__ == "__main__":
    # --- Setup ---
    output_dir = "debug_output"
    os.makedirs(output_dir, exist_ok=True)

    print("--- Initializing Lazy Loading Dataset for Debugging ---")
    lazy_dataset = MaestroMIDIDatasetLazy()

    lazy_loader = DataLoader(
        lazy_dataset,
        batch_size=4,
        # keep same order
        shuffle=False,
        num_workers=0,
        collate_fn=ut.collate_fn_skip_error,
    )

    print("\nIterating through DataLoader...")
    for i, batch in enumerate(lazy_loader):
        if batch is None:
            print(f"Batch {i + 1}: SKIPPED.")
            continue

        print(f"\n--- Processing Batch {i + 1} ---")
        # gets only the first
        first_item_melody = batch["real_melody_bar"][0]
        first_item_prev_melody = batch["previous_bar_melody_condition"][0]

        print("\n  --- Debugging first item in batch ---")

        # path_midi = os.path.join(output_dir, f"debug_batch_{i + 1}_melody.mid")
        # ut.play_piano_roll(
        #     first_item_melody,
        #     path_midi,
        # )
        # if torch.any(first_item_prev_melody):
        #     ut.play_piano_roll(
        #         first_item_prev_melody,
        #         os.path.join(output_dir, f"debug_batch_{i + 1}_prev_melody.mid"),
        #     )

        ut.visualize_midi(first_item_melody)

        # alternative visualization
        # ut.visualize_midi_custom(path_midi)

        if i >= 2:
            print("\n--- Comprehensive debug test finished successfully! ---")
            break


class RecursiveMIDIDataset(Dataset):
    def __init__(
        self,
        midi_base_path="../DeepLearningMidi/dataset/clean_midi",
        transform=None,
        csv_out_path=None,
        load_from_csv=None,
    ):
        """
        Args:
            midi_base_path (str): Root directory to search for MIDI files.
            transform (callable, optional): Optional transform to apply on samples.
            csv_out_path (str, optional): If provided, saves valid MIDI paths to this CSV.
            load_from_csv (str, optional): If provided, skips scan and loads paths from CSV.
        """
        self.transform = transform

        if load_from_csv is not None and os.path.exists(load_from_csv):
            self.midi_file_paths = pd.read_csv(load_from_csv)["midi_path"].tolist()
            print(f"Loaded {len(self.midi_file_paths)} MIDI paths from {load_from_csv}")
        else:
            self.midi_file_paths = []
            for dirpath, _, filenames in os.walk(midi_base_path):
                for filename in filenames:
                    if filename.lower().endswith((".mid", ".midi")):
                        filepath = os.path.join(dirpath, filename)
                        try:
                            _ = pretty_midi.PrettyMIDI(filepath)  # Check validity
                            self.midi_file_paths.append(filepath)
                        except Exception as e:
                            print(f"Skipping invalid file: {filepath} ({e})")

            print(f"Found {len(self.midi_file_paths)} valid MIDI files.")

            if csv_out_path:
                df = pd.DataFrame({"midi_path": self.midi_file_paths})
                df.to_csv(csv_out_path, index=False)
                print(f"Saved valid MIDI paths to CSV: {csv_out_path}")

    def __len__(self):
        return len(self.midi_file_paths)

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

            # TODO: MOVED to main.py
            # This is either a real melody bar from the dataset (e.g., the bar immediately preceding
            # X) or an all-zeros matrix if no prior context is provided (e.g., for the very first bar in a sequence).

            # TODO:
            # Chord Sequence (1-D Condition): A 13-dimensional vector. This represents the chord for the current bar,
            # with 12 dimensions for the key and 1 for major/minor type.

            # Handle for the file
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            h = 128  # Number of MIDI notes
            w = 16  # Number of time steps in a bar (sixteenth notes)

            # NOTE: in this case the midi files are songs (they are only piano rolls).
            # a possible solution could be FILTERS
            # we need to create filters that take out from the midi files all the unnecessary notes (bass, guitar, drums ...)
            # a second option is to not use Prettymidi (check if it works also for general songs)

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
                # NOTE: placeholder to handle better
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

            # NOTE: to move to main.py
            # Placeholder, assume no prior bar for this sample all zero
            # this is the previous bar, probably this can be easily implemented by
            # using a global variable that tracks the previusly used bar
            # if not it is needed to re retrive the current midi and previous each time
            # previous_bar_melody = torch.zeros((h, w)).float()

            # NOTE: Placeholder, assuming random chord
            # Dummy chord condition (1-D condition).
            # random 13-dimensional vector.

            ##chord_condition = torch.randint(0, 2, (13,)).float()  # Random binary chord (major/minor, key)

            # The 'sample' returned by __getitem__ should contain everything needed for a single training step.
            # For MidiNet, this typically involves:
            # 1. A real melody bar (X) to train the discriminator.
            # 2. Conditional information (previous_bar_melody, chord_condition) for the generator.

            sample = {
                "melody_tensor": processed_melody_tensor,
                "midi_data": midi_data,  # X from p_data(X)
                "midi_path": midi_path,  # 1-D condition for Generator
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
# Test to verify dbs and library: WRITTEN BY EMILIO

# if __name__ == "__main__":
#     import DBAdapters as dba

#     csv_file_path = "./dataset/MAESTRO_Dataset/maestro-v3.0.0.csv"
#     midi_base_path = "./dataset/MAESTRO_Dataset/maestro-v3.0.0"
#     my_dataset = dba.MaestroMIDIDataset(
#         csv_file=csv_file_path, midi_base_path=midi_base_path
#     )

#     batch_size = 4
#     shuffle = True
#     num_workers = 0

#     data_loader = DataLoader(
#         my_dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers,
#     )

#     plot_output_dir = "debug_plots"
#     os.makedirs(plot_output_dir, exist_ok=True)

#     print("\nIterating through DataLoader:")
#     for epoch in range(1):
#         print(f"--- Epoch {epoch + 1} ---")
#         for i, batch in enumerate(data_loader):
#             if batch is None:
#                 print(f"Batch {i + 1}: Skipped due to error in loading a MIDI file.")
#                 continue

#             real_melody_bar = batch["real_melody_bar"]
#             previous_bar_melody_condition = batch["previous_bar_melody_condition"]
#             chord_condition = batch["chord_condition"]

#             print(f"Batch {i + 1}:")
#             print(f"  Real Melody Bar shape: {real_melody_bar.shape}")
#             print(
#                 f"  Previous Bar Melody Condition shape: {previous_bar_melody_condition.shape}"
#             )
#             print(f"  Chord Condition shape: {chord_condition.shape}")

#             # --- ADDING RAW MATRIX PRINTING FOR DEBUGGING ---
#             print(
#                 f"\n--- Raw Real Melody Bar Matrix (first sample in batch {i + 1}) ---"
#             )
#             # Convert to NumPy and print the array
#             print(real_melody_bar[0].squeeze().cpu().numpy())
#             print(f"----------------------------------------------------\n")

#             if np.any(previous_bar_melody_condition[0].squeeze().cpu().numpy()):
#                 print(
#                     f"\n--- Raw Previous Bar Melody Condition Matrix (first sample in batch {i + 1}) ---"
#                 )
#                 print(previous_bar_melody_condition[0].squeeze().cpu().numpy())
#                 print(f"----------------------------------------------------\n")

#             print(
#                 f"\n--- Raw Chord Condition Vector (first sample in batch {i + 1}) ---"
#             )
#             print(chord_condition[0].squeeze().cpu().numpy())
#             print(f"----------------------------------------------------\n")

#             # --- Plotting and Saving Images (as before, these ARE matrix plots) ---

#             melody_to_plot = real_melody_bar[0].squeeze().cpu().numpy()
#             fig1 = plt.figure(figsize=(8, 4))
#             plt.imshow(melody_to_plot, aspect="auto", origin="lower", cmap="binary")
#             plt.title(
#                 f"Epoch {epoch + 1}, Batch {i + 1}: Real Melody Bar (First Sample)"
#             )
#             plt.xlabel("Time Steps (w=16)")
#             plt.ylabel("MIDI Notes (h=128)")
#             plt.colorbar(label="Note On (1) / Note Off (0)")
#             fig1.savefig(
#                 os.path.join(
#                     plot_output_dir,
#                     f"epoch{epoch + 1}_batch{i + 1}_real_melody_plot.png",
#                 )
#             )
#             plt.close(fig1)

#             condition_to_plot = previous_bar_melody_condition[0].squeeze().cpu().numpy()
#             if np.any(condition_to_plot):
#                 fig2 = plt.figure(figsize=(8, 4))
#                 plt.imshow(
#                     condition_to_plot, aspect="auto", origin="lower", cmap="binary"
#                 )
#                 plt.title(
#                     f"Epoch {epoch + 1}, Batch {i + 1}: Previous Bar Melody Condition (First Sample)"
#                 )
#                 plt.xlabel("Time Steps (w=16)")
#                 plt.ylabel("MIDI Notes (h=128)")
#                 plt.colorbar(label="Note On (1) / Note Off (0)")
#                 fig2.savefig(
#                     os.path.join(
#                         plot_output_dir,
#                         f"epoch{epoch + 1}_batch{i + 1}_prev_bar_condition_plot.png",
#                     )
#                 )
#                 plt.close(fig2)

#             chord_to_plot = chord_condition[0].squeeze().cpu().numpy()
#             fig3 = plt.figure(figsize=(6, 2))
#             plt.bar(range(len(chord_to_plot)), chord_to_plot)
#             plt.title(
#                 f"Epoch {epoch + 1}, Batch {i + 1}: Chord Condition (First Sample)"
#             )
#             plt.xlabel("Chord Dimension")
#             plt.ylabel("Value")
#             plt.xticks(range(13), [f"Dim {j}" for j in range(12)] + ["Type"])
#             fig3.savefig(
#                 os.path.join(
#                     plot_output_dir,
#                     f"epoch{epoch + 1}_batch{i + 1}_chord_condition_plot.png",
#                 )
#             )
#             plt.close(fig3)

#             if i >= 1:
#                 break

# if __name__ == "__main__":
#     # use this command if you don't have the csv_file
#     # dataset = RecursiveMIDIDataset(
#     #     midi_base_path="../DeepLearningMidi/dataset/clean_midi",
#     #     csv_out_path="valid_paths.csv"
#     # )
#     # Use the following command in case the csv file is already done
#     dataset = RecursiveMIDIDataset(load_from_csv="valid_paths.csv")
#
#     def custom_collate_fn(batch):
#         return {
#             "melody_tensor": torch.stack([item["melody_tensor"] for item in batch]),
#             "midi_data": [item["midi_data"] for item in batch],  # lascia come lista
#             "midi_path": [item["midi_path"] for item in batch],  # lascia come lista
#         }
#
#     dataloader = DataLoader(
#         dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=custom_collate_fn
#     )
#
#     print("\n--- Inizio debug DataLoader ---\n")
#
#     plot_output_dir = "debug_plots"
#     os.makedirs(plot_output_dir, exist_ok=True)
#
#     print("\nIterating through DataLoader:")
#     for epoch in range(1):
#         print(f"--- Epoch {epoch + 1} ---")
#         for i, batch in enumerate(dataloader):
#             if batch is None:
#                 print(f"Batch {i + 1}: Skipped due to error in loading a MIDI file.")
#                 continue
#
#             real_melody_bar = batch["melody_tensor"]
#             # previous_bar_melody_condition = batch["previous_bar_melody_condition"]
#             # chord_condition = batch["chord_condition"]
#
#             print(f"Batch {i + 1}:")
#             print(f"  Real Melody Bar shape: {real_melody_bar.shape}")
#             # print(
#             #     f"  Previous Bar Melody Condition shape: {previous_bar_melody_condition.shape}"
#             # )
#             # print(f"  Chord Condition shape: {chord_condition.shape}")
#
#             # --- ADDING RAW MATRIX PRINTING FOR DEBUGGING ---
#             print(
#                 f"\n--- Raw Real Melody Bar Matrix (first sample in batch {i + 1}) ---"
#             )
#             # Convert to NumPy and print the array
#             print(real_melody_bar[0].squeeze().cpu().numpy())
#             print(f"----------------------------------------------------\n")
#
#             # if np.any(previous_bar_melody_condition[0].squeeze().cpu().numpy()):
#             #     print(
#             #         f"\n--- Raw Previous Bar Melody Condition Matrix (first sample in batch {i + 1}) ---"
#             #     )
#             #     print(previous_bar_melody_condition[0].squeeze().cpu().numpy())
#             #     print(f"----------------------------------------------------\n")
#
#             # print(
#             #     f"\n--- Raw Chord Condition Vector (first sample in batch {i + 1}) ---"
#             # )
#             # print(chord_condition[0].squeeze().cpu().numpy())
#             # print(f"----------------------------------------------------\n")
#
#             # --- Plotting and Saving Images (as before, these ARE matrix plots) ---
#
#             melody_to_plot = real_melody_bar[0].squeeze().cpu().numpy()
#             fig1 = plt.figure(figsize=(8, 4))
#             plt.imshow(melody_to_plot, aspect="auto", origin="lower", cmap="binary")
#             plt.title(
#                 f"Epoch {epoch + 1}, Batch {i + 1}: Real Melody Bar (First Sample)"
#             )
#             plt.xlabel("Time Steps (w=16)")
#             plt.ylabel("MIDI Notes (h=128)")
#             plt.colorbar(label="Note On (1) / Note Off (0)")
#             fig1.savefig(
#                 os.path.join(
#                     plot_output_dir,
#                     f"epoch{epoch + 1}_batch{i + 1}_real_melody_plot.png",
#                 )
#             )
#             plt.close(fig1)
#
#             if i >= 1:
#                 break  # Limita a 2 batch
