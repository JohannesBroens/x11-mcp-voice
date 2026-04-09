#!/usr/bin/env python3
"""Train a custom 'Hey Nox' wake word model from real + synthetic recordings.

Extracts openwakeword embeddings from WAV files, then trains a small DNN
classifier. Outputs an ONNX model compatible with openwakeword.

Usage: python scripts/train-wake-word.py

Requires:
    - Generated synthetic clips in training/hey_nox_output/
    - Real recordings in wake_word_samples/hey_nox/
    - Pre-computed ACAV100M features in training/
    - Validation features in training/
"""
import os
import sys
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

PROJECT_DIR = Path(__file__).parent.parent
TRAINING_DIR = PROJECT_DIR / "training"
SAMPLES_DIR = PROJECT_DIR / "wake_word_samples" / "hey_nox"
OUTPUT_DIR = TRAINING_DIR / "hey_nox_output" / "hey_nox"

# openwakeword embedding model (converts audio -> 96-dim features)
EMBEDDING_MODEL = TRAINING_DIR / "openwakeword" / "openwakeword" / "resources" / "models" / "embedding_model.onnx"
MELSPEC_MODEL = TRAINING_DIR / "openwakeword" / "openwakeword" / "resources" / "models" / "melspectrogram.onnx"

# Negative feature data
ACAV_FEATURES = TRAINING_DIR / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
VALIDATION_FEATURES = TRAINING_DIR / "validation_set_features.npy"

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # 80ms at 16kHz — openwakeword's native frame size


def load_wav(path: str) -> np.ndarray:
    """Load a WAV file as float32 numpy array."""
    with wave.open(path, "rb") as f:
        raw = f.readframes(f.getnframes())
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def extract_embeddings(wav_dir: str, melspec_sess, embed_sess, max_files: int = 0) -> np.ndarray:
    """Extract openwakeword embeddings from all WAV files in a directory.

    Pipeline per file:
      1. Feed full audio to melspec model → [1, 1, n_frames, 32]
      2. Slide a 76-frame window across mel frames, extract embedding for each
      3. Embedding model: [1, 76, 32, 1] → [1, 1, 1, 96] → 96-dim feature vector
    """
    files = sorted([f for f in os.listdir(wav_dir) if f.endswith(".wav")])
    if max_files > 0:
        files = files[:max_files]

    all_embeddings = []

    for i, fname in enumerate(files):
        if i % 200 == 0:
            print(f"  Extracting embeddings: {i}/{len(files)}...", flush=True)

        audio = load_wav(os.path.join(wav_dir, fname))

        # Pad to at least 1.5 seconds
        min_samples = int(1.5 * SAMPLE_RATE)
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)))

        # Feed full audio to melspec: [1, n_samples] → [1, 1, n_frames, 32]
        mel_input = audio.reshape(1, -1).astype(np.float32)
        mel_out = melspec_sess.run(None, {"input": mel_input})[0]
        # mel_out: [1, 1, n_frames, 32] → extract frames: [n_frames, 32]
        mel_frames = mel_out[0, 0]  # [n_frames, 32]

        if mel_frames.shape[0] < 76:
            continue

        # Slide window of 76 frames, stride of 8 (~160ms steps)
        for start in range(0, mel_frames.shape[0] - 76 + 1, 8):
            window = mel_frames[start : start + 76]  # [76, 32]
            # Embedding expects [batch, 76, 32, 1]
            emb_input = window.reshape(1, 76, 32, 1).astype(np.float32)
            embedding = embed_sess.run(None, {"input_1": emb_input})[0]
            all_embeddings.append(embedding.flatten())  # 96-dim

    if not all_embeddings:
        return np.zeros((0, 96), dtype=np.float32)

    return np.array(all_embeddings, dtype=np.float32)


class WakeWordDNN(nn.Module):
    """DNN classifier matching openwakeword's expected input: [batch, 16, 96].

    Takes 16 frames of 96-dim embeddings, flattens to 1536-dim,
    then classifies through a small feedforward network.
    This matches how openwakeword feeds data to wake word models.
    """

    def __init__(self, n_frames=16, embed_dim=96, hidden_size=64, n_layers=3):
        super().__init__()
        input_size = n_frames * embed_dim  # 16 * 96 = 1536
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch, 16, 96] → flatten to [batch, 1536]
        return self.net(x.flatten(start_dim=1))


def main():
    print("=" * 50)
    print("  Hey Nox — Wake Word Model Training")
    print("=" * 50)
    print()

    # Check files exist
    for path, desc in [
        (EMBEDDING_MODEL, "Embedding model"),
        (MELSPEC_MODEL, "Melspec model"),
        (ACAV_FEATURES, "ACAV100M negative features"),
        (VALIDATION_FEATURES, "Validation features"),
    ]:
        if not path.exists():
            print(f"  ERROR: {desc} not found: {path}")
            sys.exit(1)

    # Load ONNX models for feature extraction
    print("[1/5] Loading embedding models...")
    melspec_sess = ort.InferenceSession(str(MELSPEC_MODEL))
    embed_sess = ort.InferenceSession(str(EMBEDDING_MODEL))

    # Extract positive embeddings from synthetic clips
    positive_train_dir = OUTPUT_DIR / "positive_train"
    positive_test_dir = OUTPUT_DIR / "positive_test"

    print(f"[2/5] Extracting positive features from {positive_train_dir}...")
    pos_embeds_train = extract_embeddings(str(positive_train_dir), melspec_sess, embed_sess, max_files=2000)
    print(f"  Got {len(pos_embeds_train)} raw positive embeddings")

    print(f"  Extracting positive test features from {positive_test_dir}...")
    pos_embeds_test = extract_embeddings(str(positive_test_dir), melspec_sess, embed_sess, max_files=500)
    print(f"  Got {len(pos_embeds_test)} raw positive test embeddings")

    # Group positive embeddings into windows of 16 frames → [N, 16, 96]
    # This matches openwakeword's input format
    N_FRAMES = 16
    def group_frames(embeds, n_frames=N_FRAMES):
        """Group sequential embeddings into windows of n_frames."""
        groups = []
        for i in range(0, len(embeds) - n_frames + 1, n_frames // 2):  # stride = 8 (overlap)
            groups.append(embeds[i : i + n_frames])
        if groups:
            return np.array(groups, dtype=np.float32)
        return np.zeros((0, n_frames, 96), dtype=np.float32)

    pos_train = group_frames(pos_embeds_train)
    pos_test = group_frames(pos_embeds_test)
    print(f"  Grouped into windows: train={pos_train.shape}, test={pos_test.shape}")

    # Load pre-computed negative features (ACAV100M)
    # Shape: [N, 16, 96] — already in the right format!
    print("[3/5] Loading negative features...")
    neg_all = np.load(str(ACAV_FEATURES), mmap_mode="r")
    print(f"  ACAV100M shape: {neg_all.shape}")
    n_neg_train = min(len(neg_all), len(pos_train) * 20)  # 20:1 neg:pos ratio
    neg_indices = np.random.choice(len(neg_all), size=n_neg_train, replace=False)
    neg_train = neg_all[neg_indices].astype(np.float32)  # [n, 16, 96] — same shape as positive
    print(f"  Sampled {n_neg_train} negative training embeddings (shape: {neg_train.shape})")

    val_raw = np.load(str(VALIDATION_FEATURES))
    # Validation features are [N, 96] — group into [N', 16, 96] windows
    val_features = group_frames(val_raw)
    val_hours = len(val_raw) * 0.08 / 3600  # each embedding = ~80ms
    print(f"  Loaded {len(val_features)} validation windows (~{val_hours:.1f} hours)")

    # Prepare training data
    print("[4/5] Training model...")
    X_train = np.vstack([pos_train, neg_train])
    y_train = np.concatenate([
        np.ones(len(pos_train), dtype=np.float32),
        np.zeros(len(neg_train), dtype=np.float32),
    ])

    # Shuffle
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    # Convert to tensors
    X_tensor = torch.from_numpy(X_train)
    y_tensor = torch.from_numpy(y_train).unsqueeze(1)

    # Train — input is [batch, 16, 96]
    model = WakeWordDNN(n_frames=N_FRAMES, embed_dim=96, hidden_size=64, n_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    batch_size = 512
    n_epochs = 20
    best_loss = float("inf")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for i in range(0, len(X_tensor), batch_size):
            batch_X = X_tensor[i : i + batch_size]
            batch_y = y_tensor[i : i + batch_size]

            pred = model(batch_X)
            loss = criterion(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        # Evaluate on positive test set
        model.eval()
        with torch.no_grad():
            pos_test_tensor = torch.from_numpy(pos_test)
            pos_preds = model(pos_test_tensor).numpy().flatten()
            pos_acc = (pos_preds > 0.5).mean()

            # False positive rate on validation set
            val_tensor = torch.from_numpy(val_features.astype(np.float32))
            val_preds = model(val_tensor).numpy().flatten()
            fp_rate = (val_preds > 0.5).mean()
            # Convert to false positives per hour (~11 hours in validation set)
            fp_per_hour = fp_rate * len(val_features) / 11.0

        print(f"  Epoch {epoch+1:2d}/{n_epochs}: loss={avg_loss:.4f}  "
              f"pos_acc={pos_acc:.3f}  fp/hr={fp_per_hour:.1f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Load best model
    model.load_state_dict(best_state)

    # Export to ONNX
    print("[5/5] Exporting ONNX model...")
    output_path = PROJECT_DIR / "models" / "hey_nox.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy = torch.randn(1, N_FRAMES, 96)  # [1, 16, 96] — matches openwakeword's pipeline
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11,
    )

    print()
    print(f"  Model saved to: {output_path}")
    print(f"  Model size: {output_path.stat().st_size / 1024:.1f} KB")
    print()
    print("  To use it, update config.yaml:")
    print('    wake_word:')
    print('      model: "hey_nox"')
    print('      threshold: 0.5')
    print()
    print("  Then: nox restart")


if __name__ == "__main__":
    main()
