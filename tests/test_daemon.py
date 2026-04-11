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
         patch("x11_mcp_voice.daemon.StateServer"), \
         patch("x11_mcp_voice.daemon.rotate"):
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

    assert "RuntimeError" in result


@pytest.mark.asyncio
async def test_handle_interaction_resumes_media_on_exception(daemon):
    """Media must resume even if _handle_interaction_inner raises an exception."""
    daemon._agent = MagicMock()
    daemon._agent.check_timeout = MagicMock()
    daemon._agent.reset = MagicMock()
    daemon._media = MagicMock()
    daemon._state_server = AsyncMock()
    daemon._config.media.auto_pause = True
    daemon._config.conversation.style = "auto"

    # Make the inner method raise
    async def boom():
        raise RuntimeError("test explosion")

    daemon._handle_interaction_inner = boom

    with pytest.raises(RuntimeError, match="test explosion"):
        await daemon._handle_interaction()

    # Media must have been resumed despite the exception
    daemon._media.resume.assert_called_once()


@pytest.mark.asyncio
async def test_handle_interaction_resets_walkie_talkie_on_exception(daemon):
    """In walkie_talkie mode, agent.reset() is called even on exception."""
    daemon._agent = MagicMock()
    daemon._agent.check_timeout = MagicMock()
    daemon._agent.reset = MagicMock()
    daemon._media = MagicMock()
    daemon._state_server = AsyncMock()
    daemon._config.media.auto_pause = False
    daemon._config.conversation.style = "walkie_talkie"

    async def boom():
        raise RuntimeError("test explosion")

    daemon._handle_interaction_inner = boom

    with pytest.raises(RuntimeError, match="test explosion"):
        await daemon._handle_interaction()

    daemon._agent.reset.assert_called_once()
