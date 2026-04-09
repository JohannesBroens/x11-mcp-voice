"""Tests for wake word model and WakeWordDetector."""
from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Paths relative to project root
_PROJECT_ROOT = Path(__file__).parent.parent
_MODEL_PATH = _PROJECT_ROOT / "models" / "hey_nox.onnx"
_SAMPLES_DIR = _PROJECT_ROOT / "wake_word_samples" / "hey_nox"

_SAMPLE_RATE = 16000
_CHUNK_SAMPLES = 1280


# ---------------------------------------------------------------------------
# Test 1: custom model file exists on disk
# ---------------------------------------------------------------------------

def test_custom_model_file_exists():
    assert _MODEL_PATH.exists(), f"Model file not found: {_MODEL_PATH}"
    assert _MODEL_PATH.is_file(), f"Model path is not a file: {_MODEL_PATH}"
    assert _MODEL_PATH.stat().st_size > 0, "Model file is empty"


# ---------------------------------------------------------------------------
# Tests 2-4: real openwakeword integration (marked audio, skip when mocked)
# ---------------------------------------------------------------------------

@pytest.mark.audio
def test_custom_model_loads_in_openwakeword():
    """Load hey_nox.onnx via openwakeword and verify input shape [batch, 16, 96]."""
    oww = pytest.importorskip(
        "openwakeword",
        reason="openwakeword is mocked; skipping real model load test",
    )
    # If the import returned a MagicMock we're in the stubbed environment — skip.
    if isinstance(oww, MagicMock):
        pytest.skip("openwakeword is mocked; skipping real model load test")

    import openwakeword.utils
    from openwakeword.model import Model

    # Download hey_jarvis to initialise the shared mel-spectrogram preprocessor.
    openwakeword.utils.download_models(["hey_jarvis"])

    model = Model(wakeword_models=[str(_MODEL_PATH)], inference_framework="onnx")

    # openwakeword stores ONNX sessions in model.models list.
    assert len(model.models) > 0, "No models loaded"

    # Each ONNX session exposes input metadata via get_inputs().
    session = model.models[0]
    input_shape = session.get_inputs()[0].shape
    assert list(input_shape) == [1, 16, 96], (
        f"Unexpected input shape: {input_shape} — expected [1, 16, 96]"
    )


@pytest.mark.audio
def test_real_recording_scores_above_threshold():
    """Feed a real hey_nox WAV through the model; at least one chunk must score >= 0.5."""
    oww = pytest.importorskip(
        "openwakeword",
        reason="openwakeword is mocked; skipping real scoring test",
    )
    if isinstance(oww, MagicMock):
        pytest.skip("openwakeword is mocked; skipping real scoring test")

    wav_files = sorted(_SAMPLES_DIR.glob("*.wav"))
    assert wav_files, f"No WAV files found in {_SAMPLES_DIR}"

    import openwakeword.utils
    import wave
    from openwakeword.model import Model

    openwakeword.utils.download_models(["hey_jarvis"])
    model = Model(wakeword_models=[str(_MODEL_PATH)], inference_framework="onnx")

    wav_path = wav_files[0]
    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getframerate() == _SAMPLE_RATE, (
            f"Expected 16kHz, got {wf.getframerate()}Hz in {wav_path.name}"
        )
        assert wf.getnchannels() == 1, f"Expected mono, got {wf.getnchannels()} channels"
        raw = wf.readframes(wf.getnframes())

    audio = np.frombuffer(raw, dtype=np.int16)

    max_score = 0.0
    for start in range(0, len(audio) - _CHUNK_SAMPLES + 1, _CHUNK_SAMPLES):
        chunk = audio[start : start + _CHUNK_SAMPLES]
        prediction = model.predict(chunk)
        for score in prediction.values():
            if score > max_score:
                max_score = score

    assert max_score >= 0.5, (
        f"Max score for {wav_path.name} was {max_score:.3f} — expected >= 0.5"
    )


@pytest.mark.audio
def test_silence_scores_below_threshold():
    """2 seconds of silence fed through the model must never score >= 0.3."""
    oww = pytest.importorskip(
        "openwakeword",
        reason="openwakeword is mocked; skipping silence test",
    )
    if isinstance(oww, MagicMock):
        pytest.skip("openwakeword is mocked; skipping silence test")

    import openwakeword.utils
    from openwakeword.model import Model

    openwakeword.utils.download_models(["hey_jarvis"])
    model = Model(wakeword_models=[str(_MODEL_PATH)], inference_framework="onnx")

    silence = np.zeros(_SAMPLE_RATE * 2, dtype=np.int16)

    for start in range(0, len(silence) - _CHUNK_SAMPLES + 1, _CHUNK_SAMPLES):
        chunk = silence[start : start + _CHUNK_SAMPLES]
        prediction = model.predict(chunk)
        for score in prediction.values():
            assert score < 0.3, (
                f"Silence scored {score:.3f} at offset {start} — expected < 0.3"
            )


# ---------------------------------------------------------------------------
# Tests 5-6: WakeWordDetector unit tests (use mocked openwakeword from conftest)
# ---------------------------------------------------------------------------

def test_wake_word_detector_init():
    """WakeWordDetector stores model name and threshold at construction time."""
    from x11_mcp_voice.wake_word import WakeWordDetector

    detector = WakeWordDetector(model="hey_nox", threshold=0.5)
    assert detector._model_name == "hey_nox"
    assert detector._threshold == 0.5


def test_wake_word_detector_callback_on_detection():
    """on_wake callback fires when the mocked model returns a score >= threshold."""
    import sys
    from x11_mcp_voice.wake_word import WakeWordDetector

    callback_called = threading.Event()

    def _on_wake():
        callback_called.set()

    # Build a mock openwakeword Model that returns a high score on predict().
    mock_model_instance = MagicMock()
    mock_model_instance.predict.return_value = {"hey_nox": 0.95}
    mock_model_instance.reset = MagicMock()

    mock_model_cls = MagicMock(return_value=mock_model_instance)
    mock_oww_module = MagicMock()
    mock_oww_model_module = MagicMock()
    mock_oww_model_module.Model = mock_model_cls

    # Mock the audio stream so _run() gets one chunk then raises to stop the loop.
    fake_chunk = np.zeros((_CHUNK_SAMPLES, 1), dtype=np.int16)
    call_count = [0]

    def fake_read(_n):
        call_count[0] += 1
        if call_count[0] > 1:
            raise RuntimeError("stop")
        return fake_chunk, False

    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.read = fake_read

    # _run() imports openwakeword and openwakeword.model locally, so we patch
    # sys.modules directly (they are already MagicMock stubs from conftest.py).
    sys.modules["openwakeword"] = mock_oww_module
    sys.modules["openwakeword.model"] = mock_oww_model_module

    detector = WakeWordDetector(model="hey_nox", threshold=0.5)

    try:
        with patch("x11_mcp_voice.wake_word.sd.InputStream", return_value=mock_stream):
            # Make the custom .onnx path appear non-existent so the branch that
            # calls Model(wakeword_models=[model_name]) is exercised (either
            # branch ends up calling mock_model_cls, so both are fine).
            with patch.object(type(_MODEL_PATH), "exists", return_value=False):
                detector.start(on_wake=_on_wake)
                fired = callback_called.wait(timeout=3.0)
        detector.stop()
    finally:
        # Restore conftest stubs so other tests are unaffected.
        sys.modules["openwakeword"] = MagicMock()
        sys.modules["openwakeword.model"] = MagicMock()

    assert fired, "on_wake callback was not called after a high-score prediction"


# ---------------------------------------------------------------------------
# Test 7: State enum sanity check
# ---------------------------------------------------------------------------

def test_state_enum_has_all_expected_values():
    """All 7 daemon states exist with their Latin values."""
    from x11_mcp_voice.state import State

    assert State.IDLE.value == "somnus"
    assert State.WAKE.value == "excito"
    assert State.LISTENING.value == "ausculto"
    assert State.PROCESSING.value == "cogito"
    assert State.SPEAKING.value == "dico"
    assert State.CONTROLLING.value == "impero"
    assert State.ERROR.value == "erratum"
    assert len(State) == 7


# ---------------------------------------------------------------------------
# Tests 8-9: Config defaults and custom wake word loading
# ---------------------------------------------------------------------------

def test_config_wake_word_defaults():
    """Default wake word config is model='hey_jarvis', threshold=0.7."""
    from x11_mcp_voice.config import Config

    cfg = Config()
    assert cfg.wake_word.model == "hey_jarvis"
    assert cfg.wake_word.threshold == 0.7


def test_config_loads_custom_wake_word(tmp_path):
    """Config loaded from YAML with custom wake_word values reflects those values."""
    from x11_mcp_voice.config import load_config

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "wake_word:\n"
        "  model: hey_nox\n"
        "  threshold: 0.5\n"
    )
    cfg = load_config(str(config_file))
    assert cfg.wake_word.model == "hey_nox"
    assert cfg.wake_word.threshold == 0.5
    # Unrelated defaults are unchanged.
    assert cfg.stt.model == "nvidia/parakeet-tdt-0.6b-v2"
    assert cfg.tts.voice == "en_US-ryan-medium"
