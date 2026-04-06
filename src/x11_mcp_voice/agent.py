from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from x11_mcp_voice.config import AgentConfig, ConversationConfig

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a voice-activated desktop automation assistant running on a Linux system.
You have access to x11-mcp tools to control the desktop: take screenshots, click, type, manage windows, etc.

Desktop context:
- Dual monitors: primary monitor (automation target), secondary monitor (dev tools)
- Applications: desktop applications
- The user speaks to you via voice. Keep responses concise and natural for speech.

{style_instruction}

When executing desktop tasks:
1. Take a screenshot first to see what's on screen
2. Use accessibility tools or visual inspection to find UI elements
3. Click/type to interact, then verify with another screenshot
4. If something goes wrong, explain what happened and ask what to do next
"""

_STYLE_INSTRUCTIONS = {
    "auto": "If you need more information from the user, ask naturally.",
    "confirmations": (
        "When you need to ask the user a question, phrase it as a clear yes/no question. "
        "For example: 'Do you want me to continue?' or 'Should I open that app?'. "
        "Keep questions simple so they're easy to answer by voice."
    ),
    "walkie_talkie": "Each message from the user is a standalone command. Be direct and concise.",
}


class Agent:
    """Claude API client with x11-mcp MCP tools for desktop automation."""

    def __init__(self, agent_config: AgentConfig, conversation_config: ConversationConfig):
        self._model = agent_config.model
        self._agent_config = agent_config
        self._style = conversation_config.style
        self._client = anthropic.Anthropic()
        self._messages: list[dict[str, Any]] = []
        self._tools: list[dict[str, Any]] = []
        self._mcp_session = None  # Set during connect()

    async def connect(self) -> None:
        """Establish MCP connection to x11-mcp server and discover tools."""
        from mcp import ClientSession, StdioServerParameters, stdio_client

        server_params = StdioServerParameters(
            command=self._agent_config.x11_mcp_command,
            args=self._agent_config.x11_mcp_args,
            env={"DISPLAY": ":0"},
        )

        # Store the context managers so they stay alive
        self._stdio_cm = stdio_client(server_params)
        read_stream, write_stream = await self._stdio_cm.__aenter__()

        self._session_cm = ClientSession(read_stream, write_stream)
        self._mcp_session = await self._session_cm.__aenter__()
        await self._mcp_session.initialize()

        # Discover tools and convert to Anthropic format
        tools_result = await self._mcp_session.list_tools()
        self._tools = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            for tool in tools_result.tools
        ]
        log.info("Connected to x11-mcp, discovered %d tools", len(self._tools))

    async def disconnect(self) -> None:
        """Close MCP connection."""
        if self._mcp_session is not None:
            await self._session_cm.__aexit__(None, None, None)
            await self._stdio_cm.__aexit__(None, None, None)
            self._mcp_session = None
            log.info("Disconnected from x11-mcp")

    def reset(self) -> None:
        """Clear conversation history for a new interaction."""
        self._messages = []

    async def send(self, text: str) -> str:
        """Send user text to Claude, execute any tool calls, return final text response."""
        self._messages.append({"role": "user", "content": text})

        style_instruction = _STYLE_INSTRUCTIONS.get(self._style, _STYLE_INSTRUCTIONS["auto"])
        system = _SYSTEM_PROMPT.format(style_instruction=style_instruction)

        while True:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "max_tokens": 4096,
                "system": system,
                "messages": self._messages,
            }
            if self._tools:
                kwargs["tools"] = self._tools

            response = self._client.messages.create(**kwargs)

            # Append assistant response to history
            self._messages.append({"role": "assistant", "content": response.content})

            # Check for tool use
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                break

            # Execute tools and collect results
            tool_results = []
            for tool_use in tool_uses:
                log.info("Calling tool: %s(%s)", tool_use.name, json.dumps(tool_use.input)[:200])
                try:
                    mcp_result = await self._mcp_session.call_tool(
                        name=tool_use.name,
                        arguments=tool_use.input,
                    )
                    # Convert MCP result to string for Anthropic API
                    result_text = "\n".join(
                        c.text for c in mcp_result.content if hasattr(c, "text")
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": result_text,
                        "is_error": mcp_result.isError,
                    })
                except Exception as e:
                    log.error("Tool %s failed: %s", tool_use.name, e)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": f"Error: {e}",
                        "is_error": True,
                    })

            self._messages.append({"role": "user", "content": tool_results})

        # Extract final text from response
        text_parts = [b.text for b in response.content if b.type == "text"]
        return " ".join(text_parts)
