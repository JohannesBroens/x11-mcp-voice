import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import numpy as np

from x11_mcp_voice.daemon import Daemon
from x11_mcp_voice.state import State
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
         patch("x11_mcp_voice.daemon.Agent"), \
         patch("x11_mcp_voice.daemon.StateServer"):
        d = Daemon(config)
    return d


def test_initial_state(daemon):
    assert daemon._state == State.IDLE


def test_daemon_has_state_server(daemon):
    assert hasattr(daemon, '_state_server')


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
async def test_process_text_sends_to_agent(daemon):
    daemon._agent = AsyncMock()
    daemon._agent.send = AsyncMock(return_value="Hi there!")

    result = await daemon._process_text("hello claude")

    assert result == "Hi there!"
    daemon._agent.send.assert_called_once_with("hello claude")


@pytest.mark.asyncio
async def test_process_text_error_returns_fallback(daemon):
    daemon._agent = AsyncMock()
    daemon._agent.send = AsyncMock(side_effect=RuntimeError("boom"))

    result = await daemon._process_text("hello claude")

    assert result == "I couldn't process that. Try again."
