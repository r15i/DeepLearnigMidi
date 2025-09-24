import torch
import pretty_midi
import os
import argparse
import time

# --- Local Imports ---
# Make sure your models.py file is accessible
import models as md

# --- Optional Imports ---
# Pygame is used for optional playback
try:
    import pygame

    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


def tensor_to_midi(piano_roll_tensor, output_path="generation_output.mid", tempo=120):
    """
    Converts a piano roll tensor into a MIDI file.

    Args:
        piano_roll_tensor (torch.Tensor): A tensor of shape (num_segments, 1, 128, 16)
                                          or a single segment of shape (1, 128, 16).
        output_path (str): Path to save the output MIDI file.
        tempo (int): Tempo of the resulting MIDI file in beats per minute.
    """
    # Ensure tensor is on CPU and is a boolean tensor (True for note-on, False for note-off)
    piano_roll = piano_roll_tensor.squeeze().cpu().numpy() > 0.5

    # Create a PrettyMIDI object
    midi_data = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0)  # Program 0 is Acoustic Grand Piano

    # Calculate the duration of a single time step (16th note)
    sixteenth_note_duration = 60.0 / tempo / 4.0

    # If the input is a single segment, add a dimension to loop through it
    if piano_roll.ndim == 2:
        piano_roll = piano_roll[None, :, :]

    current_time = 0.0
    for segment in piano_roll:
        for pitch in range(128):
            # Find the start and end times of notes for this pitch in this segment
            note_ons = segment[pitch, :] == 1
            if note_ons.any():
                start_step = None
                for step, is_on in enumerate(note_ons):
                    if is_on and start_step is None:
                        start_step = step
                    elif not is_on and start_step is not None:
                        # Note ended, create the MIDI note object
                        start_time = current_time + start_step * sixteenth_note_duration
                        end_time = current_time + step * sixteenth_note_duration
                        note = pretty_midi.Note(
                            velocity=100, pitch=pitch, start=start_time, end=end_time
                        )
                        instrument.notes.append(note)
                        start_step = None
                # If a note is on at the end of the segment, close it
                if start_step is not None:
                    start_time = current_time + start_step * sixteenth_note_duration
                    end_time = current_time + len(note_ons) * sixteenth_note_duration
                    note = pretty_midi.Note(
                        velocity=100, pitch=pitch, start=start_time, end=end_time
                    )
                    instrument.notes.append(note)

        # Advance time to the next segment
        current_time += 16 * sixteenth_note_duration

    # Add the instrument to the MIDI data
    midi_data.instruments.append(instrument)

    # Write out the MIDI data
    midi_data.write(output_path)
    print(f"✅ MIDI file successfully saved to: {output_path}")


def play_midi(file_path):
    """
    Plays a MIDI file using pygame.
    """
    if not PYGAME_AVAILABLE:
        print("\n⚠️ Pygame not found. Cannot play MIDI file.")
        print("   Please install it by running: pip install pygame")
        return

    try:
        print(f"\n▶️ Playing {file_path}...")
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            # Wait for the music to finish
            time.sleep(1)
    except Exception as e:
        print(
            f"\n❌ Could not play MIDI file. Please ensure your system's audio is configured."
        )
        print(f"   Error details: {e}")
    finally:
        # Clean up pygame resources
        if PYGAME_AVAILABLE:
            pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description="Generate MIDI music from a trained VAE model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="training_output/models/vae_final.pth",
        help="Path to the trained VAE model (.pth) file.",
    )
    parser.add_argument(
        "-n",
        "--num-bars",
        type=int,
        default=8,
        help="Number of musical bars (segments) to generate.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="generated_music.mid",
        help="Name of the output MIDI file.",
    )
    parser.add_argument(
        "-ld",
        "--latent-dim",
        type=int,
        default=32,
        help="Latent dimension of the VAE model.",
    )
    parser.add_argument(
        "--play",
        action="store_true",  # This makes it a flag, e.g., --play
        help="Play the generated MIDI file after saving.",
    )
    args = parser.parse_args()

    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found at '{args.model_path}'")
        return

    # --- 2. Load Model ---
    print(f"Loading model from {args.model_path}...")
    model = md.ConvVAE(latent_dim=args.latent_dim)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()  # Set model to evaluation mode
    print("Model loaded successfully.")

    # --- 3. Generate Music ---
    print(f"\n🎶 Generating {args.num_bars} bars of music...")
    with torch.no_grad():
        # Create random latent vectors (one for each bar)
        z = torch.randn(args.num_bars, model.latent_dim).to(device)

        # Decode them into piano roll logits
        logits = model.decode(z)

        # Apply sigmoid to get probabilities
        samples = torch.sigmoid(logits)

    # --- 4. Convert to MIDI and Save ---
    tensor_to_midi(samples, output_path=args.output_file)

    # --- 5. Optionally Play the MIDI File ---
    if args.play:
        play_midi(args.output_file)


if __name__ == "__main__":
    main()
