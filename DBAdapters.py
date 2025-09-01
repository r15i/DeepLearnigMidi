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
<<<<<<< HEAD
        Args:
            csv_file (string): Path to the MAESTRO csv file.
            midi_base_path (string): Base directory where MIDI files are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
            h (int): Height of the piano roll (number of MIDI pitches).
            w (int): Width of the piano roll (number of time steps per bar).
=======
        Implements "lazy loading". MIDI files are processed on-the-fly in __getitem__.
        This has a very fast startup time and low memory usage, but can be slower
        during training if data processing becomes a bottleneck.
>>>>>>> 762ba2e656ef25155743896bb7e2828ef09e4849
        """
        self.midi_base_path = midi_base_path
        self.transform = transform
        self.h = h
        self.w = w

<<<<<<< HEAD
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
=======
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
>>>>>>> 762ba2e656ef25155743896bb7e2828ef09e4849

    def __getitem__(self, idx):
        # check if at that position there is a tensor
        if torch.is_tensor(idx):
            idx = idx.tolist()

<<<<<<< HEAD
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


<<<<<<< HEAD
            # Handle for the file
=======
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
>>>>>>> 762ba2e656ef25155743896bb7e2828ef09e4849
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


