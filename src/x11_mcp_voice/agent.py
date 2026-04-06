from __future__ import annotations

import asyncio
import json
import logging
import re

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


def _clean_for_speech(text: str) -> str:
    """Strip markdown formatting and emoji so TTS reads natural text."""
    # Remove markdown bold/italic
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
    # Remove markdown links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove backticks
    text = text.replace('`', '')
    # Remove emoji (Unicode emoji ranges)
    text = re.sub(
        r'[\U0001F300-\U0001FAFF\U00002702-\U000027B0\U0000FE00-\U0000FE0F\U0000200D]',
        '', text,
    )
    # Collapse multiple spaces
    text = re.sub(r'  +', ' ', text)
    return text.strip()


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
            "--permission-mode", "bypassPermissions",
        ]

        # Each invocation is a fresh session. --resume with --no-session-persistence
        # doesn't work reliably, so we always start fresh. Multi-turn context is
        # provided via the system prompt instead.
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
            limit=10 * 1024 * 1024,  # 10MB buffer — screenshots are large base64 JSON
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
            event_subtype = event.get("subtype", "")

            if event_type == "assistant":
                message = event.get("message", {})
                for block in message.get("content", []):
                    if block.get("type") == "text":
                        log.debug("Claude: %s", block["text"][:200])

            elif event_type == "result":
                result_text = event.get("result", "")
                session_id = event.get("session_id")
                is_error = event.get("is_error", False)
                if is_error:
                    log.error("Claude error result: %s", result_text[:500])
                else:
                    log.info("Response: %s", result_text[:200])

            elif event_type == "error":
                log.error("Claude stream error: %s", str(event)[:500])

            elif event_type != "system":
                log.debug("Event: type=%s subtype=%s", event_type, event_subtype)

        stderr_data = await proc.stderr.read()
        await proc.wait()

        if proc.returncode != 0:
            stderr = stderr_data.decode().strip()
            log.error("Claude Code exited with %d: stderr=%s", proc.returncode, stderr[:500] if stderr else "(empty)")
            if not result_text:
                result_text = "I couldn't process that. Try again."

        if session_id and self._style != "walkie_talkie":
            self._session_id = session_id

        self._messages.append({"role": "assistant", "content": result_text})
        return _clean_for_speech(result_text)
