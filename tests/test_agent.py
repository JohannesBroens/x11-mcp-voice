import json
import pytest
from unittest.mock import AsyncMock, patch

from x11_mcp_voice.agent import Agent
from x11_mcp_voice.config import AgentConfig, ConversationConfig


@pytest.fixture
def agent_config():
    return AgentConfig(model="claude-sonnet-4-6")


@pytest.fixture
def conversation_config():
    return ConversationConfig(style="auto")


def test_agent_init(agent_config, conversation_config):
    agent = Agent(agent_config, conversation_config)
    assert agent._messages == []
    assert agent._model == "claude-sonnet-4-6"
    assert agent._session_id is None


def test_agent_reset(agent_config, conversation_config):
    agent = Agent(agent_config, conversation_config)
    agent._messages = [{"role": "user", "content": "hello"}]
    agent._session_id = "some-session-id"
    agent.reset()
    assert agent._messages == []
    assert agent._session_id is None


def _mock_subprocess(stdout_lines: list[str], returncode: int = 0):
    """Create a mock for asyncio.create_subprocess_exec."""
    stdout_data = "\n".join(stdout_lines).encode() + b"\n"

    mock_proc = AsyncMock()
    mock_proc.returncode = returncode
    mock_proc.wait = AsyncMock()

    # Simulate readline() returning one line at a time, then empty
    lines = [line.encode() + b"\n" for line in stdout_lines] + [b""]
    mock_proc.stdout.readline = AsyncMock(side_effect=lines)
    mock_proc.stderr.read = AsyncMock(return_value=b"")

    return mock_proc


@pytest.mark.asyncio
async def test_agent_send_text_response(agent_config, conversation_config):
    agent = Agent(agent_config, conversation_config)

    result_event = json.dumps({
        "type": "result",
        "subtype": "success",
        "result": "Hello! How can I help?",
        "session_id": "sess-123",
    })

    mock_proc = _mock_subprocess([result_event])

    with patch("x11_mcp_voice.agent.asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await agent.send("hello")

    assert result == "Hello! How can I help?"
    assert len(agent._messages) == 2  # user + assistant
    assert agent._session_id == "sess-123"  # captured for multi-turn


@pytest.mark.asyncio
async def test_agent_send_walkie_talkie_no_session(agent_config):
    """In walkie_talkie mode, session_id should not be stored."""
    conv_config = ConversationConfig(style="walkie_talkie")
    agent = Agent(agent_config, conv_config)

    result_event = json.dumps({
        "type": "result",
        "subtype": "success",
        "result": "Done.",
        "session_id": "sess-456",
    })

    mock_proc = _mock_subprocess([result_event])

    with patch("x11_mcp_voice.agent.asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await agent.send("open steam")

    assert result == "Done."
    assert agent._session_id is None  # walkie_talkie doesn't persist sessions


@pytest.mark.asyncio
async def test_agent_send_resumes_session(agent_config, conversation_config):
    """Second send() should pass --resume with the session_id."""
    agent = Agent(agent_config, conversation_config)
    agent._session_id = "existing-session"

    result_event = json.dumps({
        "type": "result",
        "subtype": "success",
        "result": "Yes, I can do that.",
        "session_id": "existing-session",
    })

    mock_proc = _mock_subprocess([result_event])

    with patch("x11_mcp_voice.agent.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        result = await agent.send("yes please")

    assert result == "Yes, I can do that."
    # Verify --resume was passed
    call_args = mock_exec.call_args[0]
    assert "--resume" in call_args
    assert "existing-session" in call_args


@pytest.mark.asyncio
async def test_agent_send_error_fallback(agent_config, conversation_config):
    """If Claude Code exits non-zero with no result, return error message."""
    agent = Agent(agent_config, conversation_config)

    mock_proc = _mock_subprocess([], returncode=1)
    mock_proc.stderr.read = AsyncMock(return_value=b"something went wrong")

    with patch("x11_mcp_voice.agent.asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await agent.send("do something")

    assert result == "I couldn't process that. Try again."
