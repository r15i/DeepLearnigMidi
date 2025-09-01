import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os  # To join paths
import pretty_midi  # For MIDI file processing
import matplotlib.pyplot as plt
from roll import MidiFile


def collate_fn_skip_error(batch):
    """Filters out None values from a batch before collating."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def visualize_midi_custom(path):
    mid = MidiFile(path)
    # mid = MidiFile("test_file/1.mid")
    # get the list of all events
    # events = mid.get_events()
    # get the np array of piano roll image
    roll = mid.get_roll()
    # draw piano roll by pyplot
    mid.draw_roll()


def visualize_midi(first_item_melody):
    # --- MODIFIED: Display plot instead of saving ---
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(
        first_item_melody.squeeze().cpu().numpy(),
        aspect="auto",
        origin="lower",
        cmap="binary",
    )
    plt.title(f"Batch {i + 1} - Piano Roll Visualization (Close to Continue)")
    plt.xlabel("Time Steps (w=16)")
    plt.ylabel("MIDI Notes (h=128)")
    print("  -> Displaying plot...")
    plt.show()  # This will pause the script until you close the plot window.
    plt.close(fig)


def extract_chord_from_bar(self, bar_matrix):
    # 1. Find all notes that were played at least once in the bar.
    # np.sum(bar_matrix, axis=1) counts how many times each of the 128 notes was active.
    # np.where(... > 0)[0] gives the indices (the MIDI note numbers) of the notes that were active.
    notes_present = np.where(np.sum(bar_matrix, axis=1) > 0)[0]
    if len(notes_present) == 0:
        return np.zeros(13, dtype=np.float32)
    # 2. Convert note numbers to "pitch classes" (0-11).
    # This ignores the octave and just focuses on the note name (C, C#, D, etc.).
    # 60 % 12 = 0 (C), 64 % 12 = 4 (E), 67 % 12 = 7 (G).
    pitch_classes = sorted(list(set(note % 12 for note in notes_present)))
    if not pitch_classes:
        return np.zeros(13, dtype=np.float32)

    # 3. Guess the root of the chord.
    # This is a very simple heuristic: it assumes the LOWEST note is the root.
    # In our example, 0 (C) is the lowest.
    root = pitch_classes[0]

    # 4. Guess if the chord is Major or Minor.
    # This is the cleverest part. It checks for the presence of the "third".
    # A major third is 4 semitones above the root. (0 + 4) % 12 = 4 (E)
    # A minor third is 3 semitones above the root. (0 + 3) % 12 = 3 (D#)
    is_major = (root + 4) % 12 in pitch_classes or (root + 3) % 12 not in pitch_classes
    chord_vector = np.zeros(13, dtype=np.float32)
    chord_vector[root] = 1
    chord_vector[12] = 1.0 if is_major else 0.0
    return chord_vector


def play_piano_roll(piano_roll_tensor, output_filename, tempo=120):
    """Converts a piano roll tensor back to an audible MIDI file and opens it."""
    if piano_roll_tensor.ndim == 3:
        pr_squeezed = piano_roll_tensor.squeeze(0).cpu().numpy()
    else:
        pr_squeezed = piano_roll_tensor.cpu().numpy()

    pr_binary = (pr_squeezed > 0) * 100

    midi_output = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0)

    seconds_per_step = 60.0 / (tempo * 4)

    for pitch in range(pr_binary.shape[0]):
        for time_step in range(pr_binary.shape[1]):
            if pr_binary[pitch, time_step] > 0:
                note = pretty_midi.Note(
                    velocity=int(pr_binary[pitch, time_step]),
                    pitch=pitch,
                    start=time_step * seconds_per_step,
                    end=(time_step + 1) * seconds_per_step,
                )
                instrument.notes.append(note)

    midi_output.instruments.append(instrument)
    midi_output.write(output_filename)
    print(f"  -> Saved audible MIDI to '{output_filename}'")

    # --- ADDED: Attempt to play the file immediately ---
    # try:
    #     if sys.platform == "win32":
    #         os.startfile(output_filename)
    #     elif sys.platform == "darwin":  # macOS
    #         subprocess.run(["open", output_filename])
    #     else:  # Linux
    #         subprocess.run(["xdg-open", output_filename])
    # except Exception as e:
    #     print(f"  -> Could not automatically play file: {e}")
