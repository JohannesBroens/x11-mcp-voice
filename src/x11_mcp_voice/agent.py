from __future__ import annotations

import asyncio
import json
import logging

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
    """Claude Code subprocess wrapper for desktop automation.

    Uses `claude --print` with stream-json to send commands to Claude Code,
    which already has x11-mcp tools configured. No separate API key needed.
    """

    def __init__(self, agent_config: AgentConfig, conversation_config: ConversationConfig):
        self._model = agent_config.model
        self._style = conversation_config.style
        self._session_id: str | None = None
        self._messages: list[dict] = []

        style_instruction = _STYLE_INSTRUCTIONS.get(self._style, _STYLE_INSTRUCTIONS["auto"])
        self._system = _SYSTEM_PROMPT.format(style_instruction=style_instruction)

    async def connect(self) -> None:
        """No-op. Claude Code manages its own MCP connections."""
        log.info("Agent ready (using Claude Code subprocess)")

    async def disconnect(self) -> None:
        """No-op. Claude Code cleans up on process exit."""
        log.info("Agent disconnected")

    def reset(self) -> None:
        """Clear conversation state so next send() starts a fresh session."""
        self._session_id = None
        self._messages = []

    async def send(self, text: str) -> str:
        """Send text to Claude Code via subprocess, return response text.

        Spawns `claude --print` with stream-json, pipes the user message,
        parses the JSON response stream, and returns the final text.
        """
        self._messages.append({"role": "user", "content": text})

        cmd = [
            "claude", "--print",
            "--output-format", "stream-json",
            "--model", self._model,
            "--system-prompt", self._system,
            "--allowedTools", "mcp__x11-mcp__*",  # auto-approve all x11-mcp tools
        ]

        if self._session_id is not None:
            cmd.extend(["--resume", self._session_id])
        else:
            cmd.append("--no-session-persistence")

        # User message as positional argument
        cmd.append(text)

        log.debug("Spawning: %s", " ".join(cmd[:6]) + "...")

        # Strip all ANTHROPIC_* env vars so Claude Code uses OAuth
        # (Max subscription) instead of trying API key auth
        import os
        stripped = [k for k in os.environ if k.startswith("ANTHROPIC_")]
        env = {k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC_")}
        if stripped:
            log.info("Stripped env vars for OAuth: %s", stripped)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        result_text = ""
        session_id = None

        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            line = line.decode().strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")

            if event_type == "assistant":
                message = event.get("message", {})
                for block in message.get("content", []):
                    if block.get("type") == "text":
                        log.debug("Claude: %s", block["text"][:200])

            elif event_type == "result":
                result_text = event.get("result", "")
                session_id = event.get("session_id")
                log.info("Response: %s", result_text[:200])

        await proc.wait()

        if proc.returncode != 0:
            stderr = (await proc.stderr.read()).decode().strip()
            log.error("Claude Code exited with %d: %s", proc.returncode, stderr[:500])
            if not result_text:
                result_text = "I couldn't process that. Try again."

        if session_id and self._style != "walkie_talkie":
            self._session_id = session_id

        self._messages.append({"role": "assistant", "content": result_text})
        return result_text
