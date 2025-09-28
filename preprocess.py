#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import argparse
from pathlib import Path
from typing import Optional, List

import torch
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configuration ---
PITCH_RANGE = 128
SEQUENCE_LENGTH = 16  # The number of time steps in each segment


class MidiProcessor:
    """
    Handles the preprocessing of any directory of MIDI files.

    This class discovers MIDI files, converts them into piano roll representations,
    segments them, and saves them efficiently as PyTorch tensors.
    """

    def __init__(
        self,
        midi_dir: str,
        processed_dir: str,
        sampling_frequency: int = 20,
    ):
        self.midi_dir = Path(midi_dir)
        self.processed_dir = Path(processed_dir)
        self.sampling_frequency = sampling_frequency

        if not self.midi_dir.is_dir():
            raise FileNotFoundError(
                f"MIDI source directory not found at: {self.midi_dir}"
            )

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory for processed files: {self.processed_dir}")

    def _midi_to_segments(self, midi_path: Path) -> Optional[torch.Tensor]:
        """Converts a single MIDI file into a tensor of piano roll segments."""
        try:
            midi_data = pretty_midi.PrettyMIDI(str(midi_path))
            end_time = midi_data.get_end_time()
            total_steps = int(end_time * self.sampling_frequency)

            if total_steps < SEQUENCE_LENGTH:
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
                segment = piano_roll[start_idx:end_idx]
                # Transpose to (PITCH_RANGE, SEQUENCE_LENGTH) for convention
                segments.append(segment.T)

            # Stack into a single tensor for efficient storage
            return torch.from_numpy(np.stack(segments).astype(np.float32))

        except Exception as e:
            print(f"\nCould not process file {midi_path.name}. Error: {e}")
            return None

    def process_dataset(self, limit: Optional[int] = None):
        """
        Processes the entire dataset by scanning the midi_dir.
        """
        # Discover all MIDI files in the source directory
        midi_files = sorted(
            list(self.midi_dir.glob("*.mid")) + list(self.midi_dir.glob("*.midi"))
        )

        if limit:
            print(f"--- Running in limited mode. Processing {limit} files. ---")
            midi_files = midi_files[:limit]

        if not midi_files:
            print(
                f"Warning: No MIDI files found in {self.midi_dir}. Nothing to process."
            )
            return

        print(f"Found {len(midi_files)} MIDI files to process...")
        for midi_path in tqdm(midi_files, desc="Processing MIDI files"):
            segments_tensor = self._midi_to_segments(midi_path)

            if segments_tensor is not None and segments_tensor.shape[0] > 0:
                output_filename = f"{midi_path.stem}.pt"
                output_path = self.processed_dir / output_filename
                torch.save(segments_tensor, output_path)

        print("\n🎉 Dataset processing complete!")


class MidiVisualizer:
    """
    Handles visualization and verification of the processed MIDI data.
    """

    def __init__(
        self,
        midi_dir: str,
        processed_dir: str,
        sampling_frequency: int = 20,
    ):
        self.midi_dir = Path(midi_dir)
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

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
        fig.suptitle(
            f"Verification: '{original_midi_path.name}' | Segment #{segment_index}",
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
        ax_raw.set_title(f"Original MIDI ({start_time:.2f}s - {end_time:.2f}s)")
        ax_raw.set_ylabel("MIDI Pitch")
        ax_raw.set_xlabel("Time (seconds)")
        ax_raw.grid(True, linestyle=":", linewidth=0.5)

        # Bottom plot: Processed Tensor (in time steps)
        ax_processed = axes[1]
        ax_processed.imshow(
            segment_tensor.numpy(), aspect="auto", origin="lower", cmap="viridis"
        )
        ax_processed.set_ylim(20, 100)
        ax_processed.set_title(f"Processed Tensor")
        ax_processed.set_ylabel("MIDI Pitch")
        ax_processed.set_xlabel("Time Steps (Matrix Columns)")
        ax_processed.set_xticks(np.arange(-0.5, SEQUENCE_LENGTH, 2))
        ax_processed.set_xticklabels(np.arange(0, SEQUENCE_LENGTH + 1, 2))

        plt.savefig(output_image_path)
        print(f"\n✅ Verification image saved to '{output_image_path}'")
        plt.close()

    def run_random_visualization(self):
        """Selects a random, non-silent processed segment and visualizes it."""
        print("\n--- Running Random Visualization ---")
        processed_files = list(self.processed_dir.glob("*.pt"))
        if not processed_files:
            print("❌ Processed data directory is empty. Run processing first.")
            return

        max_tries = 20
        for i in range(max_tries):
            print(f"Attempt {i + 1}/{max_tries}: Searching for a segment with notes...")

            # Select a random processed file
            random_pt_path = random.choice(processed_files)

            # Find the corresponding original midi file by checking extensions
            original_midi_path = self.midi_dir / f"{random_pt_path.stem}.mid"
            if not original_midi_path.exists():
                original_midi_path = self.midi_dir / f"{random_pt_path.stem}.midi"
                if not original_midi_path.exists():
                    continue

            all_segments = torch.load(random_pt_path)
            num_segments = all_segments.shape[0]

            segment_id = random.randint(0, num_segments - 1)
            segment_tensor = all_segments[segment_id]

            if torch.sum(segment_tensor) > 0:
                print("Found a non-silent segment!")
                print(f"  Song: '{original_midi_path.name}'")
                print(f"  Segment: #{segment_id}")
                self.visualize_reconstruction(
                    original_midi_path,
                    segment_tensor,
                    segment_id,
                    output_image_path="random_reconstruction.png",
                )
                return

        print(f"❌ Could not find a non-silent segment after {max_tries} tries.")

    def visualize_specific_song(self, song_filename: str, segment_index: int):
        """Visualizes a specific segment from a specific song."""
        print(f"\n--- Visualizing '{song_filename}', Segment #{segment_index} ---")

        # Construct paths
        midi_path = self.midi_dir / song_filename
        processed_path = self.processed_dir / f"{Path(song_filename).stem}.pt"

        # Validate paths
        if not midi_path.exists():
            print(f"❌ Error: MIDI file not found at '{midi_path}'")
            return
        if not processed_path.exists():
            print(f"❌ Error: Processed tensor not found at '{processed_path}'")
            print("Please ensure you have run the processing step first.")
            return

        all_segments = torch.load(processed_path)
        num_segments = all_segments.shape[0]

        if not (0 <= segment_index < num_segments):
            print(
                f"❌ Error: Invalid segment index. "
                f"Please choose an index between 0 and {num_segments - 1}."
            )
            return

        segment_tensor = all_segments[segment_index]
        output_filename = (
            f"specific_reconstruction_{Path(song_filename).stem}_seg{segment_index}.png"
        )

        self.visualize_reconstruction(
            midi_path, segment_tensor, segment_index, output_filename
        )


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess and visualize a dataset of MIDI files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--midi-dir",
        type=str,
        required=True,
        help="Directory containing the input MIDI files.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="./processed_data",
        help="Directory to save the processed tensors.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit processing to the first N files found.",
    )
    parser.add_argument(
        "--sf",
        type=int,
        default=20,
        help="Sampling frequency in Hz for the piano roll.",
    )
    parser.add_argument(
        "--skip-process",
        action="store_true",
        help="Skip the preprocessing step and only run visualization.",
    )

    # Visualization arguments
    vis_group = parser.add_argument_group("Visualization Options")
    vis_group.add_argument(
        "--visualize-random",
        action="store_true",
        help="Visualize a random, non-empty segment from the processed data.",
    )
    vis_group.add_argument(
        "--visualize-song",
        type=str,
        help="Filename of a specific song in --midi-dir to visualize.",
    )
    vis_group.add_argument(
        "--segment-id",
        type=int,
        default=0,
        help="The segment ID to visualize when using --visualize-song.",
    )

    args = parser.parse_args()

    if not args.skip_process:
        processor = MidiProcessor(
            midi_dir=args.midi_dir,
            processed_dir=args.processed_dir,
            sampling_frequency=args.sf,
        )
        processor.process_dataset(limit=args.limit)

    if args.visualize_random or args.visualize_song:
        visualizer = MidiVisualizer(
            midi_dir=args.midi_dir,
            processed_dir=args.processed_dir,
            sampling_frequency=args.sf,
        )
        if args.visualize_random:
            visualizer.run_random_visualization()

        if args.visualize_song:
            visualizer.visualize_specific_song(args.visualize_song, args.segment_id)
