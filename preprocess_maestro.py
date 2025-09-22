import os
import re
import random
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import numpy as np
import pandas as pd
import pretty_midi
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configuration (can be moved to a separate config file) ---
PITCH_RANGE = 128
SEQUENCE_LENGTH = 16  # The number of time steps in each segment


class MaestroProcessor:
    """
    Handles the preprocessing of the MAESTRO MIDI dataset.

    This class reads MIDI files, converts them into piano roll representations,
    segments them, and saves them efficiently as PyTorch tensors.
    """

    def __init__(
        self,
        csv_path: str,
        midi_root: str,
        processed_dir: str,
        sampling_frequency: int = 20,
    ):
        self.csv_path = Path(csv_path)
        self.midi_root = Path(midi_root)
        self.processed_dir = Path(processed_dir)
        self.sampling_frequency = sampling_frequency

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Metadata CSV not found at: {self.csv_path}")
        if not self.midi_root.is_dir():
            raise FileNotFoundError(
                f"MIDI root directory not found at: {self.midi_root}"
            )

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory for processed files: {self.processed_dir}")

    def _midi_to_segments(self, midi_path: Path) -> Optional[torch.Tensor]:
        """Converts a single MIDI file into a tensor of piano roll segments."""
        try:
            midi_data = pretty_midi.PrettyMIDI(str(midi_path))
            end_time = midi_data.get_end_time()
            total_steps = int(end_time * self.sampling_frequency)

            if total_steps == 0:
                return None

            # Create piano roll with an efficient data type
            piano_roll = np.zeros((PITCH_RANGE, total_steps), dtype=np.uint8)

            for instrument in midi_data.instruments:
                if not instrument.is_drum:
                    for note in instrument.notes:
                        start_step = int(note.start * self.sampling_frequency)
                        end_step = int(note.end * self.sampling_frequency)
                        piano_roll[note.pitch, start_step:end_step] = 1

            # Transpose to (time, pitch) for easy slicing
            piano_roll = piano_roll.T
            num_sequences = len(piano_roll) // SEQUENCE_LENGTH

            if num_sequences == 0:
                return None

            # Collect all valid segments
            segments = []
            for i in range(num_sequences):
                start_idx = i * SEQUENCE_LENGTH
                end_idx = start_idx + SEQUENCE_LENGTH
                segment = piano_roll[
                    start_idx:end_idx
                ]  # Shape: (SEQUENCE_LENGTH, PITCH_RANGE)
                # Transpose to (PITCH_RANGE, SEQUENCE_LENGTH) for convention
                segments.append(segment.T)

            # Stack into a single tensor for efficient storage
            return torch.from_numpy(np.stack(segments))

        except Exception as e:
            print(f"\nCould not process file {midi_path.name}. Error: {e}")
            return None

    def process_dataset(self, limit: Optional[int] = None):
        """
        Processes the entire dataset based on the metadata CSV.
        """
        metadata = pd.read_csv(self.csv_path)
        if limit:
            print(
                f"--- Running in limited mode. Processing the first {limit} files. ---"
            )
            metadata = metadata.head(limit)

        print("Starting MIDI file processing...")
        for _, row in tqdm(
            metadata.iterrows(), total=metadata.shape[0], desc="Processing MIDI files"
        ):
            midi_filename = row["midi_filename"]
            midi_path = self.midi_root / midi_filename

            if not midi_path.exists():
                continue

            segments_tensor = self._midi_to_segments(midi_path)

            if segments_tensor is not None:
                output_filename = f"{midi_path.stem}.pt"
                output_path = self.processed_dir / output_filename
                torch.save(segments_tensor, output_path)

        print("\n🎉 Dataset processing complete!")


class MaestroVisualizer:
    """
    Handles visualization and verification of the processed MAESTRO data.
    """

    def __init__(
        self,
        csv_path: str,
        midi_root: str,
        processed_dir: str,
        sampling_frequency: int = 20,
    ):
        self.csv_path = Path(csv_path)
        self.midi_root = Path(midi_root)
        self.processed_dir = Path(processed_dir)
        self.sampling_frequency = sampling_frequency

    def visualize_reconstruction(
        self,
        original_midi_path: Path,
        segment_tensor: torch.Tensor,
        segment_index: int,
        output_image_path: str = "reconstruction_verification.png",
    ):
        """
        Compares a segment of a raw MIDI file with its processed .pt tensor.
        """
        start_step = segment_index * SEQUENCE_LENGTH
        start_time = start_step / self.sampling_frequency
        end_time = (start_step + SEQUENCE_LENGTH) / self.sampling_frequency

        midi_data = pretty_midi.PrettyMIDI(str(original_midi_path))

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
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
                        rect = plt.Rectangle(
                            (max(note.start, start_time), note.pitch - 0.4),
                            min(note.end, end_time) - max(note.start, start_time),
                            0.8,
                            facecolor="deepskyblue",
                        )
                        ax_raw.add_patch(rect)

        ax_raw.set_xlim(start_time, end_time)
        ax_raw.set_ylim(20, 100)
        ax_raw.set_title(
            f"Raw MIDI: '{original_midi_path.name}' ({start_time:.2f}s - {end_time:.2f}s)"
        )
        ax_raw.set_ylabel("MIDI Pitch")
        ax_raw.set_xlabel("Time (seconds)")
        ax_raw.grid(True, linestyle=":", linewidth=0.5)

        # Bottom plot: Processed Tensor (in time steps)
        ax_processed = axes[1]
        ax_processed.imshow(
            segment_tensor.numpy(), aspect="auto", origin="lower", cmap="viridis"
        )
        ax_processed.set_ylim(20, 100)
        ax_processed.set_title(f"Processed Tensor Segment #{segment_index}")
        ax_processed.set_ylabel("MIDI Pitch")
        ax_processed.set_xlabel("Time Steps (Matrix Columns)")
        ax_processed.set_xticks(np.arange(-0.5, SEQUENCE_LENGTH, 2))
        ax_processed.set_xticklabels(np.arange(0, SEQUENCE_LENGTH + 1, 2))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_image_path)
        print(f"\n✅ Verification image saved to '{output_image_path}'")
        plt.close()

    def run_random_visualization(
        self, limit: Optional[int] = None, chosen_segment_id: Optional[int] = None
    ):
        """Selects a random, non-silent processed segment and visualizes it."""
        print("\n--- Running Random Visualization ---")

        # Find the original MIDI filenames from the metadata
        metadata = pd.read_csv(self.csv_path)
        if limit:
            metadata = metadata.head(limit)

        # Map processed filenames back to original MIDI file paths
        processed_files = list(self.processed_dir.glob("*.pt"))
        if not processed_files:
            print("Processed data directory is empty. Please run preprocessing first.")
            return

        max_tries = 10
        for i in range(max_tries):
            print(
                f"Attempt {i + 1}/{max_tries}: Searching for a random segment with notes..."
            )

            # Select a random processed file
            random_pt_path = random.choice(processed_files)

            # Find the corresponding original midi file
            original_midi_name_stem = random_pt_path.stem
            midi_filename_matches = metadata[
                metadata["midi_filename"].str.contains(
                    original_midi_name_stem, regex=False
                )
            ]
            if midi_filename_matches.empty:
                continue

            original_midi_filename = midi_filename_matches.iloc[0]["midi_filename"]
            midi_path = self.midi_root / original_midi_filename

            if not midi_path.exists():
                continue

            all_segments = torch.load(random_pt_path)
            num_segments = all_segments.shape[0]

            if chosen_segment_id is not None and chosen_segment_id < num_segments:
                segment_id = chosen_segment_id
            else:
                segment_id = random.randint(0, num_segments - 1)

            segment_tensor = all_segments[segment_id]

            if torch.sum(segment_tensor) > 0:
                print("Found a non-silent segment!")
                print(f"Selected song: '{midi_path.name}'")
                print(f"Selected segment: #{segment_id}")
                self.visualize_reconstruction(midi_path, segment_tensor, segment_id)
                return

        print(f"\nCould not find a non-silent segment after {max_tries} tries.")


# --- Main execution block ---
if __name__ == "__main__":
    # Define paths relative to the script location or as absolute paths
    BASE_DIR = Path(__file__).parent
    DATASET_DIR = BASE_DIR / "dataset" / "MAESTRO_Dataset"

    CSV_FILE = DATASET_DIR / "maestro-v3.0.0.csv"
    MIDI_ROOT = DATASET_DIR / "maestro-v3.0.0"
    PROCESSED_DIR = DATASET_DIR / "processed"

    parser = argparse.ArgumentParser(
        description="Preprocess and visualize the MAESTRO dataset."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit processing to the first N files."
    )
    parser.add_argument("--sf", type=int, default=20, help="Sampling frequency in Hz.")
    parser.add_argument(
        "--skip-process", action="store_true", help="Skip the preprocessing step."
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Run visualization after processing."
    )
    args = parser.parse_args()

    if not args.skip_process:
        processor = MaestroProcessor(
            csv_path=str(CSV_FILE),
            midi_root=str(MIDI_ROOT),
            processed_dir=str(PROCESSED_DIR),
            sampling_frequency=args.sf,
        )
        processor.process_dataset(limit=args.limit)

    if args.visualize:
        visualizer = MaestroVisualizer(
            csv_path=str(CSV_FILE),
            midi_root=str(MIDI_ROOT),
            processed_dir=str(PROCESSED_DIR),
            sampling_frequency=args.sf,
        )
        visualizer.run_random_visualization(limit=args.limit)
