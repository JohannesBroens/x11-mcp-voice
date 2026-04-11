"""Tests for the speech-to-text transcriber."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from x11_mcp_voice.transcriber import Transcriber


def test_transcriber_init():
    """Transcriber initializes with model name and device."""
    t = Transcriber(model_name="nvidia/parakeet-tdt-0.6b-v2", device="cuda")
    assert t is not None


def test_transcribe_returns_string():
    """transcribe() should return a string."""
    t = Transcriber.__new__(Transcriber)
    t._model = MagicMock()
    t._model.transcribe.return_value = ["hello world"]

    audio = np.zeros(16000, dtype=np.float32)
    result = t.transcribe(audio)
    assert isinstance(result, str)
    assert result == "hello world"


def test_transcribe_empty_audio():
    """Empty audio should return empty string or handle gracefully."""
    t = Transcriber.__new__(Transcriber)
    t._model = MagicMock()
    t._model.transcribe.return_value = [""]

    audio = np.zeros(0, dtype=np.float32)
    result = t.transcribe(audio)
    assert isinstance(result, str)


def test_transcribe_strips_whitespace():
    """Transcribed text should be stripped of leading/trailing whitespace."""
    t = Transcriber.__new__(Transcriber)
    t._model = MagicMock()
    t._model.transcribe.return_value = ["  hello world  "]

    audio = np.zeros(16000, dtype=np.float32)
    result = t.transcribe(audio)
    assert result == "hello world"


def test_transcribe_handles_hypothesis_object():
    """NeMo models can return hypothesis objects with .text attribute."""
    t = Transcriber.__new__(Transcriber)
    t._model = MagicMock()
    hyp = MagicMock()
    hyp.text = "hypothesis one"
    t._model.transcribe.return_value = [hyp]

    audio = np.zeros(16000, dtype=np.float32)
    result = t.transcribe(audio)
    assert isinstance(result, str)
    assert result == "hypothesis one"


def test_transcribe_empty_hypotheses():
    """Empty hypotheses list should return empty string."""
    t = Transcriber.__new__(Transcriber)
    t._model = MagicMock()
    t._model.transcribe.return_value = []

    audio = np.zeros(16000, dtype=np.float32)
    result = t.transcribe(audio)
    assert result == ""


def test_transcribe_converts_dtype():
    """Non-float32 audio should be converted before transcription."""
    t = Transcriber.__new__(Transcriber)
    t._model = MagicMock()
    t._model.transcribe.return_value = ["converted"]

    audio = np.zeros(16000, dtype=np.int16)
    result = t.transcribe(audio)
    assert result == "converted"
    # Verify model received the audio (as float32)
    call_args = t._model.transcribe.call_args
    passed_audio = call_args[1]["audio"][0] if "audio" in call_args[1] else call_args[0][0][0]
    assert passed_audio.dtype == np.float32
