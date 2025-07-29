import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os  # To join paths


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
            import pretty_midi
            # NOTE:--- MIDI Loading and Processing Placeholder ---
            # THIS IS A PLACEHOLDER
            # this will be called to actually retrive the data to serve to the network
            # so it will be used to retrive the de file midi , format it and give it to the network

            # NOTE: START PLACEHOLDER
            # This is where you'll load the MIDI file and convert it into a
            # numerical representation (e.g., piano roll, sequence of events).
            # The exact representation depends on your generative model architecture.
            # Example: Using pretty_midi to get a piano roll
            # This is a simple example; you'll likely need more sophisticated
            # processing (e.g., fixed length sequences, quantization).
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            # Get a piano roll, e.g., for C major scale at 120 bpm, 1 second long
            # 'fs' is the frames per second. Higher fs means more detailed time resolution.
            # 'piano_roll' will be a (128, num_frames) numpy array (pitch x time)
            # You might want to sum across instruments or select a specific one.
            piano_roll = midi_data.get_piano_roll(fs=100)  # 100 frames per second
            # Normalize or scale the piano roll values if needed
            # For generative models, values are often between 0 and 1 or -1 and 1
            piano_roll = piano_roll / 127.0  # Assuming velocities up to 127
            # Convert to PyTorch tensor. Generative models often expect (batch, channels, height, width)
            # or (batch, sequence_length, features)
            # For piano roll (pitch x time), it could be (1, pitch, time) if treating pitch as a channel
            # or (time, pitch) if treating time as sequence length.
            # Let's assume (pitch, time) and add a batch dimension later.
            processed_midi_tensor = torch.from_numpy(piano_roll).float()
            # For generative models, often the input is also the "output" (reconstruction, generation target)
            # So, you might just return the processed_midi_tensor itself.
            sample = {"midi_data": processed_midi_tensor}

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
