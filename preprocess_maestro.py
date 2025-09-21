import os
import pandas as pd
import pretty_midi
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import random
import argparse  # Import the argparse library


parser = argparse.ArgumentParser(
    description="Preprocess the MAESTRO MIDI dataset into piano roll tensors.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Limit processing to the first N files. Processes all files by default.",
)
parser.add_argument(
    "--sf",
    type=int,
    default=20,
    help="Sampling frequency for the piano roll in Hz.",
)
args = parser.parse_args()


# ==============================================================================
# 1. Configuration (Unchanged)
# ==============================================================================
CSV_FILE = "./dataset/MAESTRO_Dataset/maestro-v3.0.0.csv"
MIDI_ROOT = "./dataset/MAESTRO_Dataset/maestro-v3.0.0"
PROCESSED_DIR = "./dataset/MAESTRO_Dataset/processed"
SAMPLING_FREQUENCY = args.sf
TEST_LIMIT = args.limit
SEQUENCE_LENGTH = 16
PITCH_RANGE = 128


# ==============================================================================
# 2. Preprocessing Function (CORRECTED)
# ==============================================================================
def preprocess_and_save_dataset(limit=None):
    """
    Processes MIDI files and saves them as tensors using a robust, manual
    piano roll creation method to avoid "stuck note" errors.
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

            # --- THIS IS THE NEW ROBUST LOGIC ---
            # Manually create the piano roll to be robust against faulty MIDI files.

            # 1. Get the total length of the song in time steps
            end_time = midi_data.get_end_time()
            total_steps = int(end_time * SAMPLING_FREQUENCY)

            # 2. Create an empty piano roll (all zeros)
            piano_roll = np.zeros((PITCH_RANGE, total_steps), dtype=np.float32)

            # 3. Iterate through each note and "draw" it onto the piano roll
            for instrument in midi_data.instruments:
                if not instrument.is_drum:
                    for note in instrument.notes:
                        start_step = int(note.start * SAMPLING_FREQUENCY)
                        end_step = int(note.end * SAMPLING_FREQUENCY)
                        # Fill in the 1s for the duration of the note
                        piano_roll[note.pitch, start_step:end_step] = 1

            # ----------------------------------------

            piano_roll = piano_roll.T  # Transpose for slicing
            num_sequences = len(piano_roll) // SEQUENCE_LENGTH

            for i in range(num_sequences):
                start_idx = i * SEQUENCE_LENGTH
                end_idx = start_idx + SEQUENCE_LENGTH
                sequence = piano_roll[start_idx:end_idx].T

                if sequence.shape == (PITCH_RANGE, SEQUENCE_LENGTH):
                    tensor = torch.from_numpy(sequence)
                    base_midi_name = os.path.splitext(os.path.basename(midi_filename))[
                        0
                    ]
                    output_filename = f"{base_midi_name}_segment_{i}.pt"
                    output_path = os.path.join(PROCESSED_DIR, output_filename)
                    torch.save(tensor, output_path)
        except Exception as e:
            print(f"\nCould not process file {midi_path}. Error: {e}")

    print("\n🎉 Dataset processing complete!")


# ==============================================================================
# 3. Visualization "Engine" Function (UPDATED)
# ==============================================================================
def visualize_reconstruction(
    original_midi_path,
    processed_pt_path,
    output_image_path="reconstruction_verification.png",
):
    """
    Compares a segment of a raw MIDI file with its processed .pt tensor.
    The bottom plot now shows discrete time steps.
    """
    try:
        segment_tensor = torch.load(processed_pt_path)
        match = re.search(r"_segment_(\d+)\.pt$", processed_pt_path)
        if not match:
            return
        segment_index = int(match.group(1))

        start_step = segment_index * SEQUENCE_LENGTH
        start_time = start_step / SAMPLING_FREQUENCY
        end_time = (start_step + SEQUENCE_LENGTH) / SAMPLING_FREQUENCY

        midi_data = pretty_midi.PrettyMIDI(original_midi_path)

        # --- UPDATE: Set sharex=False since axes are now different ---
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
        fig.suptitle(
            f"Verification: Original MIDI vs. Processed Segment #{segment_index}",
            fontsize=16,
        )

        # Top plot: Raw MIDI (in seconds)
        ax_raw = axes[0]
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    if note.start < end_time and note.end > start_time:
                        visible_start, visible_end = (
                            max(note.start, start_time),
                            min(note.end, end_time),
                        )
                        rect = plt.Rectangle(
                            (visible_start, note.pitch - 0.4),
                            visible_end - visible_start,
                            0.8,
                            edgecolor="none",
                            facecolor="deepskyblue",
                        )
                        ax_raw.add_patch(rect)

        ax_raw.set_xlim(start_time, end_time)
        ax_raw.set_ylim(20, 100)
        ax_raw.set_title(
            f"Raw MIDI: '{os.path.basename(original_midi_path)}' ({start_time:.2f}s - {end_time:.2f}s)"
        )
        ax_raw.set_ylabel("MIDI Pitch")
        ax_raw.set_xlabel("Time (seconds)")
        ax_raw.grid(True, linestyle=":", linewidth=0.5)

        # Bottom plot: Processed Tensor (in time steps)
        ax_processed = axes[1]

        # --- UPDATE: Remove 'extent' to use matrix indices (time steps) ---
        ax_processed.imshow(
            segment_tensor.numpy(), aspect="auto", origin="lower", cmap="viridis"
        )
        ax_processed.set_ylim(20, 100)
        ax_processed.set_title(
            f"Processed Tensor: '{os.path.basename(processed_pt_path)}'"
        )
        ax_processed.set_ylabel("MIDI Pitch")
        ax_processed.set_xlabel("Time Steps (Matrix Columns)")
        # Set ticks to be integers from 0 to 15
        ax_processed.set_xticks(np.arange(-0.5, 16, 2))
        ax_processed.set_xticklabels(np.arange(0, 17, 2))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_image_path)
        print(f"\n✅ Verification image saved to '{output_image_path}'")
    except Exception as e:
        print(f"\nCould not generate reconstruction visualization. Error: {e}")


# ==============================================================================
# 4. & 5. (Unchanged)
# ==============================================================================
# TODO: add the possibility to select a specific segment
def run_random_visualization(limit=None):
    # This function is the same as the last version
    print("\n--- Running Random Visualization ---")
    try:
        metadata = pd.read_csv(CSV_FILE)
        if limit:
            print(f"Sampling from the first {limit} files processed.")
            metadata = metadata.head(limit)
        if not os.path.exists(PROCESSED_DIR) or not os.listdir(PROCESSED_DIR):
            print("Processed data directory is empty. Please run preprocessing first.")
            return
        max_tries = 10
        for i in range(max_tries):
            print(
                f"Attempt {i + 1}/{max_tries}: Searching for a random segment with notes..."
            )
            random_song = metadata.sample(1).iloc[0]
            midi_filename = random_song["midi_filename"]
            midi_path = os.path.join(MIDI_ROOT, midi_filename)
            if not os.path.exists(midi_path):
                continue
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            duration = midi_data.get_end_time()
            num_segments = int(duration * SAMPLING_FREQUENCY) // SEQUENCE_LENGTH
            if num_segments == 0:
                continue
            random_segment_id = random.randint(0, num_segments - 1)
            base_name = os.path.splitext(os.path.basename(midi_filename))[0]
            pt_filename = f"{base_name}_segment_{random_segment_id}.pt"
            pt_path = os.path.join(PROCESSED_DIR, pt_filename)
            if not os.path.exists(pt_path):
                continue
            segment_tensor = torch.load(pt_path)
            if torch.sum(segment_tensor) > 0:
                print(f"Found a non-silent segment!")
                print(f"Selected song: '{midi_filename}'")
                print(f"Selected segment: #{random_segment_id}")
                visualize_reconstruction(midi_path, pt_path)
                return
        print(
            f"\nCould not find a non-silent segment after {max_tries} tries. Please process more files or try again."
        )
    except Exception as e:
        print(f"An error occurred during random visualization: {e}")


if __name__ == "__main__":
    if TEST_LIMIT == None:
        print(f"Prosessing : ALL files at {SAMPLING_FREQUENCY} per second ")
    else:
        print(f"Proscessing : {TEST_LIMIT} files at {SAMPLING_FREQUENCY} per second ")
    preprocess_and_save_dataset(limit=TEST_LIMIT)
    run_random_visualization(limit=TEST_LIMIT)
