import os
import pandas as pd
import pretty_midi
import numpy as np
import torch
from tqdm import tqdm
import argparse

# Your script's configuration and other functions (like visualization)
# should also be in this file. This is just the corrected core function.

# --- Constants ---
CSV_FILE = "./dataset/MAESTRO_Dataset/maestro-v3.0.0.csv"
MIDI_ROOT = "./dataset/MAESTRO_Dataset/maestro-v3.0.0"
PROCESSED_DIR = "./dataset/MAESTRO_Dataset/processed"
PITCH_RANGE = 128
SEQUENCE_LENGTH = 16  # Adjust if needed


def preprocess_and_save_dataset(limit=None, sampling_frequency=20):
    """
    Processes MIDI files into space-efficient, uint8 piano roll tensors.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print(f"Output directory for processed files: {PROCESSED_DIR}")

    try:
        metadata = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {CSV_FILE}. Please check the path.")
        return

    if limit:
        print(f"--- Running in limited mode. Processing the first {limit} files. ---")
        metadata = metadata.head(limit)

    print("Starting MIDI file processing...")
    for index, row in tqdm(
        metadata.iterrows(), total=metadata.shape[0], desc="Processing MIDI files"
    ):
        midi_filename = row["midi_filename"]
        midi_path = os.path.join(MIDI_ROOT, midi_filename)

        if not os.path.exists(midi_path):
            continue

        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            end_time = midi_data.get_end_time()
            total_steps = int(end_time * sampling_frequency)

            # 1. Create piano roll with the CORRECT data type
            piano_roll = np.zeros((PITCH_RANGE, total_steps), dtype=np.uint8)

            for instrument in midi_data.instruments:
                if not instrument.is_drum:
                    for note in instrument.notes:
                        start_step = int(note.start * sampling_frequency)
                        end_step = int(note.end * sampling_frequency)
                        piano_roll[note.pitch, start_step:end_step] = 1

            piano_roll = piano_roll.T
            num_sequences = len(piano_roll) // SEQUENCE_LENGTH

            for i in range(num_sequences):
                start_idx = i * SEQUENCE_LENGTH
                end_idx = start_idx + SEQUENCE_LENGTH

                # 2. Make an independent COPY of the slice
                sequence = piano_roll[start_idx:end_idx].T.copy()

                if sequence.shape == (PITCH_RANGE, SEQUENCE_LENGTH):
                    # NOTE: could be go with lower dimension using numpy but pt prevents it
                    tensor = torch.from_numpy(sequence)
                    # print(f"DEBUG: Tensor dtype is {tensor.dtype} right before saving.")
                    base_midi_name = os.path.splitext(os.path.basename(midi_filename))[
                        0
                    ]
                    output_filename = f"{base_midi_name}_segment_{i}.pt"
                    output_path = os.path.join(PROCESSED_DIR, output_filename)
                    torch.save(tensor, output_path)

        except Exception as e:
            print(f"\nCould not process file {midi_path}. Error: {e}")

    print("\n🎉 Dataset processing complete!")


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MAESTRO dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Limit to N files.")
    parser.add_argument("--sf", type=int, default=20, help="Sampling frequency in Hz.")
    args = parser.parse_args()

    print(
        f"Processing: {'ALL' if not args.limit else args.limit} files at {args.sf} Hz"
    )
    preprocess_and_save_dataset(limit=args.limit, sampling_frequency=args.sf)
    # You can add your visualization call back here if you like
