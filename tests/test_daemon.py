import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import numpy as np

from x11_mcp_voice.daemon import Daemon, State
from x11_mcp_voice.config import Config


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def daemon(config):
    with patch("x11_mcp_voice.daemon.WakeWordDetector"), \
         patch("x11_mcp_voice.daemon.Transcriber"), \
         patch("x11_mcp_voice.daemon.Speaker"), \
         patch("x11_mcp_voice.daemon.MediaController"), \
         patch("x11_mcp_voice.daemon.Agent"):
        d = Daemon(config)
    return d


def test_initial_state(daemon):
    assert daemon._state == State.IDLE


def test_on_wake_transitions_to_listening(daemon):
    daemon._state = State.IDLE
    daemon._media = MagicMock()
    daemon._media.pause.return_value = True
    daemon._loop = MagicMock()
    daemon._wake_event = asyncio.Event()

    # Simulate wake word callback (runs on wake word thread)
    daemon._on_wake_word()

    assert daemon._wake_event.is_set()


@pytest.mark.asyncio
async def test_process_records_and_transcribes(daemon):
    daemon._transcriber = MagicMock()
    daemon._transcriber.transcribe.return_value = "hello claude"
    daemon._agent = AsyncMock()
    daemon._agent.send = AsyncMock(return_value="Hi there!")
    daemon._speaker = MagicMock()

    # Simulate recorded audio
    audio = np.zeros(16000, dtype=np.float32)
    result = await daemon._process(audio)

    assert result == "Hi there!"
    daemon._transcriber.transcribe.assert_called_once()
    daemon._agent.send.assert_called_once_with("hello claude")


@pytest.mark.asyncio
async def test_process_empty_transcription_skips_agent(daemon):
    daemon._transcriber = MagicMock()
    daemon._transcriber.transcribe.return_value = ""
    daemon._agent = AsyncMock()

    audio = np.zeros(16000, dtype=np.float32)
    result = await daemon._process(audio)

    assert result is None
    daemon._agent.send.assert_not_called()
