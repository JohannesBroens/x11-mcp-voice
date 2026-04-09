#!/usr/bin/env python3
"""Record wake word samples for training a custom openwakeword model.

Guides you through recording "Hey Nox" multiple times with a simple
terminal UI. Saves each sample as a 16kHz mono WAV file.

Usage: python scripts/record-wake-word.py [--output-dir ./samples] [--count 50]
"""
import argparse
import os
import sys
import time
import wave

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS = 1
DURATION_S = 2.0  # each recording is 2 seconds


def record_one(index: int, total: int) -> np.ndarray:
    """Record a single sample, returns audio as numpy array."""
    print(f"\n  [{index}/{total}]  Press ENTER, then say 'Hey Nox'...", end="", flush=True)
    input()
    print(f"           🔴 Recording... ", end="", flush=True)

    audio = sd.rec(
        int(DURATION_S * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
    )
    sd.wait()

    print("done!")
    return audio


def save_wav(audio: np.ndarray, path: str) -> None:
    """Save audio as 16kHz mono WAV."""
    with wave.open(path, "wb") as f:
        f.setnchannels(CHANNELS)
        f.setsampwidth(2)  # int16
        f.setframerate(SAMPLE_RATE)
        f.writeframes(audio.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Record 'Hey Nox' wake word samples")
    parser.add_argument("--output-dir", default="./wake_word_samples/hey_nox", help="Where to save WAV files")
    parser.add_argument("--count", type=int, default=50, help="Number of samples to record")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Check how many already exist (for resuming)
    existing = len([f for f in os.listdir(args.output_dir) if f.endswith(".wav")])

    print()
    print("  ========================================")
    print("    Hey Nox — Wake Word Recording Tool")
    print("  ========================================")
    print()
    print(f"  Target: {args.count} samples")
    print(f"  Output: {args.output_dir}/")
    if existing:
        print(f"  Found {existing} existing samples — will continue from there")
    print()
    print("  Tips for good samples:")
    print("    - Vary your distance (close, arm's length, across room)")
    print("    - Vary your tone (normal, quiet, loud, tired, excited)")
    print("    - Vary your speed (fast, slow, deliberate)")
    print("    - Say it naturally, like you would in daily use")
    print("    - Background noise is OK — it helps the model generalize")
    print()
    print("  Press Ctrl+C at any time to stop and keep what you have.")
    print()

    recorded = 0
    start_index = existing + 1

    try:
        for i in range(start_index, start_index + args.count - existing):
            audio = record_one(i, args.count)

            # Quick check: is there actual audio or just silence?
            peak = np.max(np.abs(audio))
            if peak < 500:
                print("           ⚠  Very quiet — might be silence. Keeping it anyway.")

            path = os.path.join(args.output_dir, f"hey_nox_{i:03d}.wav")
            save_wav(audio, path)
            recorded += 1

    except KeyboardInterrupt:
        print("\n\n  Stopped early.")

    total = existing + recorded
    print(f"\n  Recorded {recorded} new samples ({total} total in {args.output_dir}/)")

    if total >= 30:
        print("  You have enough samples to train a model.")
        print(f"  Next step: train with openwakeword using these samples as positive examples.")
    else:
        print(f"  Need at least 30 samples for decent accuracy ({30 - total} more to go).")
        print(f"  Run this script again to resume — it picks up where you left off.")


if __name__ == "__main__":
    main()
