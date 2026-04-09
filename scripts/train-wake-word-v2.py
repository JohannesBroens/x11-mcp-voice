#!/usr/bin/env python3
"""Train 'Hey Nox' wake word using openwakeword's own preprocessor.

Feeds WAV files through openwakeword's Model.predict() pipeline
to extract features that exactly match what the model sees at inference.
Then trains a DNN classifier on those features.
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
ACAV_FEATURES = TRAINING_DIR / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
VALIDATION_FEATURES = TRAINING_DIR / "validation_set_features.npy"

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280


def load_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as f:
        raw = f.readframes(f.getnframes())
        return np.frombuffer(raw, dtype=np.int16)


def extract_features_via_oww(wav_dir: str, max_files: int = 0):
    """Extract features using openwakeword's actual preprocessor.

    Feeds audio chunk-by-chunk through the same pipeline that runs
    at inference time, ensuring training features exactly match.
    Returns list of [16, 96] feature arrays.
    """
    import openwakeword
    from openwakeword.model import Model

    # Load with a dummy pre-trained model just to get the preprocessor
    openwakeword.utils.download_models(["hey_jarvis"])
    oww = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

    files = sorted([f for f in os.listdir(wav_dir) if f.endswith(".wav")])
    if max_files > 0:
        files = files[:max_files]

    all_features = []

    for i, fname in enumerate(files):
        if i % 200 == 0:
            print(f"  Extracting: {i}/{len(files)}...", flush=True)

        audio = load_wav(os.path.join(wav_dir, fname))

        # Reset preprocessor state for each file
        oww.preprocessor.reset()
        oww.preprocessor.raw_data_remainder = np.empty(0)
        oww.preprocessor.accumulated_samples = 0

        # Feed audio chunk-by-chunk — this calls __call__ which
        # runs _streaming_features internally
        for start in range(0, len(audio) - CHUNK_SIZE + 1, CHUNK_SIZE):
            chunk = audio[start : start + CHUNK_SIZE]
            oww.preprocessor(chunk)

        # After processing all chunks, extract ALL 16-frame windows
        # "Hey Nox" could be at any position in the 2-second clip
        buf = oww.preprocessor.feature_buffer
        for w in range(0, buf.shape[0] - 16 + 1, 4):  # stride 4 (~320ms)
            all_features.append(buf[w : w + 16].copy())

    return all_features


class WakeWordDNN(nn.Module):
    def __init__(self, n_frames=16, embed_dim=96, hidden_size=64, n_layers=3):
        super().__init__()
        input_size = n_frames * embed_dim
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(start_dim=1))


def main():
    print("=" * 50)
    print("  Hey Nox — Wake Word Training v2")
    print("  (using openwakeword preprocessor)")
    print("=" * 50)
    print()

    positive_train_dir = str(OUTPUT_DIR / "positive_train")
    positive_test_dir = str(OUTPUT_DIR / "positive_test")

    print("[1/5] Extracting positive features via openwakeword preprocessor...")
    pos_train_list = extract_features_via_oww(positive_train_dir, max_files=2000)
    pos_train = np.array(pos_train_list, dtype=np.float32) if pos_train_list else np.zeros((0, 16, 96))
    print(f"  Train: {pos_train.shape}")

    pos_test_list = extract_features_via_oww(positive_test_dir, max_files=500)
    pos_test = np.array(pos_test_list, dtype=np.float32) if pos_test_list else np.zeros((0, 16, 96))
    print(f"  Test: {pos_test.shape}")

    print("[2/5] Loading negative features...")
    neg_all = np.load(str(ACAV_FEATURES), mmap_mode="r")
    n_neg = min(len(neg_all), len(pos_train) * 20)
    neg_idx = np.random.choice(len(neg_all), size=n_neg, replace=False)
    neg_train = neg_all[neg_idx].astype(np.float32)
    print(f"  Negative: {neg_train.shape}")

    val_raw = np.load(str(VALIDATION_FEATURES))
    # Group into windows of 16
    val_features = []
    for i in range(0, len(val_raw) - 16 + 1, 16):
        val_features.append(val_raw[i:i+16])
    val_features = np.array(val_features, dtype=np.float32)
    val_hours = len(val_raw) * 0.08 / 3600
    print(f"  Validation: {val_features.shape} (~{val_hours:.1f}h)")

    print("[3/5] Training...")
    X = np.vstack([pos_train, neg_train])
    y = np.concatenate([np.ones(len(pos_train)), np.zeros(len(neg_train))]).astype(np.float32)

    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y).unsqueeze(1)

    model = WakeWordDNN(n_frames=16, embed_dim=96, hidden_size=64, n_layers=3)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    batch_size = 512
    best_loss = float("inf")
    best_state = None

    for epoch in range(20):
        model.train()
        total_loss, n_batch = 0, 0
        for i in range(0, len(X_t), batch_size):
            bx, by = X_t[i:i+batch_size], y_t[i:i+batch_size]
            pred = model(bx)
            loss = loss_fn(pred, by)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batch += 1

        avg = total_loss / n_batch

        model.eval()
        with torch.no_grad():
            pos_acc = (model(torch.from_numpy(pos_test)).numpy().flatten() > 0.5).mean()
            val_preds = model(torch.from_numpy(val_features)).numpy().flatten()
            fp_rate = (val_preds > 0.5).sum() / val_hours

        print(f"  Epoch {epoch+1:2d}/20: loss={avg:.4f}  pos_acc={pos_acc:.3f}  fp/hr={fp_rate:.1f}")

        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    print("[4/5] Verifying on real recordings...")
    real_features = extract_features_via_oww(str(SAMPLES_DIR), max_files=50)
    if real_features:
        real_arr = np.array(real_features, dtype=np.float32)
        model.eval()
        with torch.no_grad():
            real_scores = model(torch.from_numpy(real_arr)).numpy().flatten()
        print(f"  Real recordings: {len(real_scores)} windows")
        print(f"  Scores: min={real_scores.min():.3f} max={real_scores.max():.3f} mean={real_scores.mean():.3f}")
        print(f"  Detection rate (>0.5): {(real_scores > 0.5).mean():.1%}")

    print("[5/5] Exporting ONNX...")
    output = PROJECT_DIR / "models" / "hey_nox.onnx"
    output.parent.mkdir(exist_ok=True)

    model.eval()
    dummy = torch.randn(1, 16, 96)
    torch.onnx.export(
        model, dummy, str(output),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    print(f"\n  Model: {output} ({output.stat().st_size / 1024:.1f} KB)")
    print("  Config: wake_word.model = 'hey_nox', threshold = 0.5")
    print("  Run: nox restart")


if __name__ == "__main__":
    main()
