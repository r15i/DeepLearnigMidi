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


<<<<<<< HEAD
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
            #Placeholder, assume no prior bar for this sample all zero
            # this is the previous bar, probably this can be easily implemented by
            # using a global variable that tracks the previusly used bar
            # if not it is needed to re retrive the current midi and previous each time
            #previous_bar_melody = torch.zeros((h, w)).float()

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

class RecursiveMIDIDataset(Dataset):
    def __init__(self, 
                 midi_base_path="../DeepLearningMidi/dataset/clean_midi", 
                 transform=None, 
                 csv_out_path=None, 
                 load_from_csv=None):
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
            #Placeholder, assume no prior bar for this sample all zero
            # this is the previous bar, probably this can be easily implemented by
            # using a global variable that tracks the previusly used bar
            # if not it is needed to re retrive the current midi and previous each time
            #previous_bar_melody = torch.zeros((h, w)).float()

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

if __name__ == "__main__":
    # use this command if you don't have the csv_file
    # dataset = RecursiveMIDIDataset(
    #     midi_base_path="../DeepLearningMidi/dataset/clean_midi",
    #     csv_out_path="valid_paths.csv"
    # )
    # Use the following command in case the csv file is already done
    dataset = RecursiveMIDIDataset(load_from_csv="valid_paths.csv")

    def custom_collate_fn(batch):
        return {
        "melody_tensor": torch.stack([item["melody_tensor"] for item in batch]),
        "midi_data": [item["midi_data"] for item in batch],       # lascia come lista
        "midi_path": [item["midi_path"] for item in batch],       # lascia come lista
    }

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0,collate_fn=custom_collate_fn)

    print("\n--- Inizio debug DataLoader ---\n")
=======
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
>>>>>>> 3c1fd1d049fbafd764a48d51708d040b8db8dc68

    plot_output_dir = "debug_plots"
    os.makedirs(plot_output_dir, exist_ok=True)

<<<<<<< HEAD
    print("\nIterating through DataLoader:")
    for epoch in range(1):
        print(f"--- Epoch {epoch + 1} ---")
        for i, batch in enumerate(dataloader):
            if batch is None:
                print(f"Batch {i + 1}: Skipped due to error in loading a MIDI file.")
                continue

            real_melody_bar = batch["melody_tensor"]
            # previous_bar_melody_condition = batch["previous_bar_melody_condition"]
            # chord_condition = batch["chord_condition"]

            print(f"Batch {i + 1}:")
            print(f"  Real Melody Bar shape: {real_melody_bar.shape}")
            # print(
            #     f"  Previous Bar Melody Condition shape: {previous_bar_melody_condition.shape}"
            # )
            # print(f"  Chord Condition shape: {chord_condition.shape}")
=======
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
>>>>>>> 3c1fd1d049fbafd764a48d51708d040b8db8dc68

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

<<<<<<< HEAD
            # if np.any(previous_bar_melody_condition[0].squeeze().cpu().numpy()):
            #     print(
            #         f"\n--- Raw Previous Bar Melody Condition Matrix (first sample in batch {i + 1}) ---"
            #     )
            #     print(previous_bar_melody_condition[0].squeeze().cpu().numpy())
            #     print(f"----------------------------------------------------\n")

            # print(
            #     f"\n--- Raw Chord Condition Vector (first sample in batch {i + 1}) ---"
            # )
            # print(chord_condition[0].squeeze().cpu().numpy())
            # print(f"----------------------------------------------------\n")

            # --- Plotting and Saving Images (as before, these ARE matrix plots) ---

            melody_to_plot = real_melody_bar[0].squeeze().cpu().numpy()
            fig1 = plt.figure(figsize=(8, 4))
            plt.imshow(melody_to_plot, aspect="auto", origin="lower", cmap="binary")
            plt.title(
                f"Epoch {epoch + 1}, Batch {i + 1}: Real Melody Bar (First Sample)"
            )
=======
        prev_to_plot = previous_bar_melody[0].squeeze().cpu().numpy()
        if np.any(prev_to_plot):
            fig2 = plt.figure(figsize=(8, 4))
            plt.imshow(prev_to_plot, aspect="auto", origin="lower", cmap="binary")
            plt.title(f"Batch {i + 1}: Previous Bar Melody Condition")
>>>>>>> 3c1fd1d049fbafd764a48d51708d040b8db8dc68
            plt.xlabel("Time Steps (w=16)")
            plt.ylabel("MIDI Notes (h=128)")
            plt.savefig(
                os.path.join(plot_output_dir, f"batch_{i + 1}_previous_bar.png")
            )
            plt.close(fig2)

<<<<<<< HEAD
            if i >= 1:
                break  # Limita a 2 batch
=======
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
>>>>>>> 3c1fd1d049fbafd764a48d51708d040b8db8dc68
