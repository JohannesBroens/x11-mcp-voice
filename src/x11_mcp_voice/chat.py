"""Nox terminal chat TUI.

Connects to the daemon's Unix socket and displays a live Rich-based
terminal UI showing the current state, ASCII face, and conversation
transcript.

Run as: python -m x11_mcp_voice.chat
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from x11_mcp_voice.config import load_config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ASCII faces — 7 states, 5 lines each, true hexagonal shape
# ---------------------------------------------------------------------------

# Every line is EXACTLY 16 chars. Pure ASCII only (no wide Unicode).
# Hex geometry via / and \ diagonal chars:
#   line 0: 4-space indent, 8-char flat top     "    ________    "
#   line 1: 3-space indent, / forehead \         "   / glyph  \   "
#   line 2: 2-space indent, / eyes     \         "  /  L    R  \  "
#   line 3: 2-space indent, \ mouth    /         "  \   mouth  /  "
#   line 4: 3-space indent, \ base     /         "   \________/   "

FACES: dict[str, list[str]] = {
    "somnus": [
        "    ________    ",
        "   /  z  z  \\   ",
        "  /  -    -  \\  ",
        "  \\    --    /  ",
        "   \\________/   ",
    ],
    "excito": [
        "    ________    ",
        "   /   !!   \\   ",
        "  /  O    O  \\  ",
        "  \\    ()    /  ",
        "   \\________/   ",
    ],
    "ausculto": [
        "    ________    ",
        "   /  [*]   \\   ",
        "  /  o    o  \\  ",
        "  \\    vv    /  ",
        "   \\________/   ",
    ],
    "cogito": [
        "    ________    ",
        "   /  ...   \\   ",
        "  /  o    o  \\  ",
        "  \\    ~~    /  ",
        "   \\________/   ",
    ],
    "dico": [
        "    ________    ",
        "   /   ~*   \\   ",
        "  /  #    #  \\  ",
        "  \\    <>    /  ",
        "   \\________/   ",
    ],
    "impero": [
        "    ________    ",
        "   /   >>   \\   ",
        "  /  =    =  \\  ",
        "  \\    __    /  ",
        "   \\________/   ",
    ],
    "erratum": [
        "    ________    ",
        "   /   /!   \\   ",
        "  /  x    x  \\  ",
        "  \\    __    /  ",
        "   \\________/   ",
    ],
}

# Alternate frames for animated states (same 16-char alignment)
FACES_ALT: dict[str, list[str]] = {
    "ausculto": [
        "    ________    ",
        "   /  [*]   \\   ",
        "  /  O    O  \\  ",
        "  \\    ^^    /  ",
        "   \\________/   ",
    ],
    "cogito": [
        "    ________    ",
        "   /   ..   \\   ",
        "  /  o    o  \\  ",
        "  \\    --    /  ",
        "   \\________/   ",
    ],
    "dico": [
        "    ________    ",
        "   /   ~*   \\   ",
        "  /  #    #  \\  ",
        "  \\    ()    /  ",
        "   \\________/   ",
    ],
}

_ANIMATED_STATES = set(FACES_ALT.keys())

# ---------------------------------------------------------------------------
# Colors and labels per state
# ---------------------------------------------------------------------------

STATE_COLORS: dict[str, str] = {
    "somnus": "orange3",
    "excito": "bold yellow",
    "ausculto": "bold dodger_blue1",
    "cogito": "bold medium_purple",
    "dico": "bold green3",
    "impero": "bold orange_red1",
    "erratum": "bold red",
}

STATE_LABELS: dict[str, str] = {
    "somnus": "Sleeping",
    "excito": "Waking",
    "ausculto": "Listening",
    "cogito": "Thinking",
    "dico": "Speaking",
    "impero": "Controlling",
    "erratum": "Error",
}


# ---------------------------------------------------------------------------
# NoxChat TUI
# ---------------------------------------------------------------------------

class NoxChat:
    """Rich-based terminal UI for monitoring the Nox voice daemon."""

    def __init__(self, socket_path: str):
        self._socket_path = socket_path
        self._current_state = "somnus"
        self._connected = False
        self._messages: list[dict] = []
        self._max_messages = 50
        self._frame = 0

    def _on_message(self, msg: dict) -> None:
        """Handle an incoming state message from the daemon socket."""
        self._current_state = msg.get("state", self._current_state)

        user_text = msg.get("user_text")
        if user_text is not None:
            self._messages.append({
                "role": "user",
                "text": user_text,
                "time": datetime.now().strftime("%H:%M:%S"),
            })

        assistant_text = msg.get("assistant_text")
        if assistant_text is not None:
            self._messages.append({
                "role": "assistant",
                "text": assistant_text,
                "time": datetime.now().strftime("%H:%M:%S"),
            })

        # Trim to max
        if len(self._messages) > self._max_messages:
            self._messages = self._messages[-self._max_messages:]

    def _render_face(self) -> Text:
        """Return the colored ASCII face for the current state and frame."""
        state = self._current_state
        color = STATE_COLORS.get(state, "white")

        # Animate: toggle between primary and alt every 2 frames
        if state in _ANIMATED_STATES and self._frame % 4 >= 2:
            lines = FACES_ALT[state]
        else:
            lines = FACES.get(state, FACES["erratum"])

        face_text = Text()
        for i, line in enumerate(lines):
            face_text.append(line, style=color)
            if i < len(lines) - 1:
                face_text.append("\n")
        return face_text

    def _render_header(self) -> Table:
        """Render the header: face on the left, state info on the right."""
        state = self._current_state
        color = STATE_COLORS.get(state, "white")
        label = STATE_LABELS.get(state, state)

        table = Table.grid(padding=(0, 3))
        table.add_column(justify="left", width=20)
        table.add_column(justify="left")

        face = self._render_face()

        info = Text()
        info.append("N O X\n", style=f"bold {color}")
        info.append(f"{state}", style=color)
        info.append(f" {chr(0x00B7)} {label}\n", style="dim")
        if self._connected:
            info.append("* Connected\n", style="green")
        else:
            info.append("* Disconnected\n", style="red")
        info.append(datetime.now().strftime("%H:%M:%S"), style="dim")

        table.add_row(face, info)
        return table

    def _render_transcript(self) -> Panel:
        """Render the scrolling conversation transcript."""
        text = Text()

        if not self._messages:
            text.append("  Waiting for conversation...", style="dim italic")
        else:
            for i, msg in enumerate(self._messages):
                ts = msg["time"]
                if msg["role"] == "user":
                    text.append(f"  {ts}  ", style="dim")
                    text.append("You\n", style="bold cyan")
                    text.append(f"  {msg['text']}\n", style="white")
                else:
                    text.append(f"  {ts}  ", style="dim")
                    text.append("Nox\n", style="bold green")
                    text.append(f"  {msg['text']}\n", style="white")

                if i < len(self._messages) - 1:
                    text.append("\n")

        return Panel(text, border_style="dim", padding=(1, 1))

    def _render(self) -> Group:
        """Combine all UI elements into a single renderable group."""
        self._frame += 1

        header = self._render_header()
        transcript = self._render_transcript()
        footer = Text("Ctrl+C to exit", style="dim", justify="center")

        return Group(
            Panel(header, border_style="dim", padding=(1, 1)),
            transcript,
            footer,
        )

    async def _listen_loop(self) -> None:
        """Async socket client that listens for state updates and reconnects on failure."""
        while True:
            try:
                reader, writer = await asyncio.open_unix_connection(self._socket_path)
                self._connected = True
                log.debug("Connected to daemon at %s", self._socket_path)

                while True:
                    line = await reader.readline()
                    if not line:
                        break
                    try:
                        msg = json.loads(line)
                        self._on_message(msg)
                    except json.JSONDecodeError:
                        continue

            except (ConnectionRefusedError, FileNotFoundError, OSError):
                pass

            self._connected = False
            self._current_state = "erratum"
            await asyncio.sleep(3)

    async def run(self) -> None:
        """Start the listener task and run the Rich Live display."""
        # Start socket listener in background
        listener = asyncio.create_task(self._listen_loop())

        try:
            with Live(
                self._render(),
                refresh_per_second=2,
                screen=True,
            ) as live:
                while True:
                    live.update(self._render())
                    await asyncio.sleep(0.5)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            listener.cancel()
            try:
                await listener
            except asyncio.CancelledError:
                pass


def main() -> None:
    config = load_config()
    socket_path = config.service.socket_path or f"/run/user/{os.getuid()}/nox.sock"
    chat = NoxChat(socket_path)
    asyncio.run(chat.run())


if __name__ == "__main__":
    main()
