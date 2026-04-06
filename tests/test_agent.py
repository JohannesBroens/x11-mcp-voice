import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from x11_mcp_voice.agent import Agent
from x11_mcp_voice.config import AgentConfig, ConversationConfig


@pytest.fixture
def agent_config():
    return AgentConfig(
        model="claude-sonnet-4-6",
        x11_mcp_command="/usr/bin/python",
        x11_mcp_args=["-m", "x11_mcp"],
    )


@pytest.fixture
def conversation_config():
    return ConversationConfig(style="auto")


def test_agent_init(agent_config, conversation_config):
    agent = Agent(agent_config, conversation_config)
    assert agent._messages == []
    assert agent._model == "claude-sonnet-4-6"


def test_agent_reset(agent_config, conversation_config):
    agent = Agent(agent_config, conversation_config)
    agent._messages = [{"role": "user", "content": "hello"}]
    agent.reset()
    assert agent._messages == []


@pytest.mark.asyncio
async def test_agent_send_text_response(agent_config, conversation_config):
    agent = Agent(agent_config, conversation_config)

    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = "Hello! How can I help?"

    mock_response = MagicMock()
    mock_response.content = [mock_text_block]
    mock_response.stop_reason = "end_turn"

    with patch.object(agent, "_client") as mock_client:
        mock_client.messages.create = MagicMock(return_value=mock_response)
        agent._tools = []  # No MCP tools for this test

        result = await agent.send("hello")

    assert result == "Hello! How can I help?"
    assert len(agent._messages) == 2  # user + assistant


@pytest.mark.asyncio
async def test_agent_send_with_tool_use(agent_config, conversation_config):
    agent = Agent(agent_config, conversation_config)
    agent._tools = []
    agent._mcp_session = AsyncMock()

    # First response: tool use
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "tool_123"
    tool_block.name = "screenshot"
    tool_block.input = {}

    response1 = MagicMock()
    response1.content = [tool_block]
    response1.stop_reason = "tool_use"

    # Tool result from MCP
    tool_result_content = MagicMock()
    tool_result_content.type = "text"
    tool_result_content.text = "screenshot taken"

    mcp_result = MagicMock()
    mcp_result.content = [tool_result_content]
    mcp_result.isError = False
    agent._mcp_session.call_tool = AsyncMock(return_value=mcp_result)

    # Second response: text
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "I took a screenshot."

    response2 = MagicMock()
    response2.content = [text_block]
    response2.stop_reason = "end_turn"

    with patch.object(agent, "_client") as mock_client:
        mock_client.messages.create = MagicMock(side_effect=[response1, response2])
        result = await agent.send("take a screenshot")

    assert result == "I took a screenshot."
    agent._mcp_session.call_tool.assert_called_once_with(name="screenshot", arguments={})
