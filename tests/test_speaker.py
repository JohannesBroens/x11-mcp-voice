"""Tests for the TTS speaker module."""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from x11_mcp_voice.speaker import (
    Speaker,
    KokoroBackend,
    PiperBackend,
    _KOKORO_DIR,
    _resolve_voice_path,
)


# ---------------------------------------------------------------------------
# Model file existence tests — these catch the "file not found" class of bugs
# ---------------------------------------------------------------------------


def test_kokoro_model_dir_exists():
    """Kokoro model directory should exist after install."""
    assert _KOKORO_DIR.exists(), (
        f"Kokoro model directory not found: {_KOKORO_DIR}. "
        "Run install.sh or download models manually."
    )


def test_kokoro_model_files_exist():
    """Both Kokoro model files must be present."""
    model = _KOKORO_DIR / "kokoro-v1.0.onnx"
    voices = _KOKORO_DIR / "voices-v1.0.bin"
    assert model.exists(), f"Missing: {model}"
    assert voices.exists(), f"Missing: {voices}"
    assert model.stat().st_size > 1_000_000, f"Model file too small: {model.stat().st_size}"
    assert voices.stat().st_size > 1_000_000, f"Voices file too small: {voices.stat().st_size}"


def test_kokoro_backend_raises_on_missing_files(tmp_path, monkeypatch):
    """KokoroBackend should raise FileNotFoundError with helpful message if models missing."""
    monkeypatch.setattr("x11_mcp_voice.speaker._KOKORO_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="Kokoro model files not found"):
        KokoroBackend(voice="af_heart")


# ---------------------------------------------------------------------------
# Speaker initialization tests
# ---------------------------------------------------------------------------


def test_speaker_init_kokoro_fallback_to_piper():
    """If Kokoro init fails, Speaker should fall back to piper without crashing."""
    with patch("x11_mcp_voice.speaker.KokoroBackend", side_effect=Exception("init failed")):
        speaker = Speaker(engine="kokoro", voice="af_heart")
        assert isinstance(speaker._backend, PiperBackend)


def test_speaker_init_piper_direct():
    """Speaker with engine='piper' should use PiperBackend."""
    speaker = Speaker(engine="piper", voice="en_US-ryan-medium")
    assert isinstance(speaker._backend, PiperBackend)


def test_speaker_init_default_engine():
    """Default engine should be kokoro."""
    # This tests that the config default matches what Speaker expects
    from x11_mcp_voice.config import TTSConfig
    assert TTSConfig().engine == "kokoro"


# ---------------------------------------------------------------------------
# Piper voice resolution tests
# ---------------------------------------------------------------------------


def test_resolve_voice_path_with_onnx_extension(tmp_path):
    """Full .onnx path should be returned as-is if it exists."""
    fake_voice = tmp_path / "test.onnx"
    fake_voice.touch()
    assert _resolve_voice_path(str(fake_voice)) == str(fake_voice)


def test_resolve_voice_path_by_name():
    """Voice name should resolve to ~/.local/share/piper-voices/{name}.onnx."""
    path = _resolve_voice_path("en_US-ryan-medium")
    assert path.endswith("en_US-ryan-medium.onnx")
    assert "piper-voices" in path


# ---------------------------------------------------------------------------
# Audio playback tests (mocked)
# ---------------------------------------------------------------------------


def test_speaker_speak_calls_backend(monkeypatch):
    """speak() should call backend.synthesize() and play audio."""
    import numpy as np

    mock_backend = MagicMock()
    mock_backend.synthesize.return_value = (np.zeros(1000, dtype=np.float32), 24000)

    speaker = Speaker.__new__(Speaker)
    speaker._backend = mock_backend
    speaker._stop_event = MagicMock()
    speaker._stop_event.is_set.return_value = False
    speaker._playing = False
    speaker._output_device = None

    with patch("x11_mcp_voice.speaker.sd") as mock_sd:
        mock_sd.wait = MagicMock()
        speaker.speak("hello")

    mock_backend.synthesize.assert_called_once_with("hello")


def test_speaker_speak_empty_text():
    """speak() with empty text should return immediately."""
    speaker = Speaker.__new__(Speaker)
    speaker._backend = MagicMock()
    speaker._stop_event = MagicMock()
    speaker._playing = False

    speaker.speak("")
    speaker._backend.synthesize.assert_not_called()

    speaker.speak("   ")
    speaker._backend.synthesize.assert_not_called()
