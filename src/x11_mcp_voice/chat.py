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
# ASCII faces — 7 states, multi-frame animations, true hexagonal shape
# ---------------------------------------------------------------------------

# Every line is EXACTLY 16 chars. Pure ASCII only (no wide Unicode).
# Hex geometry via / and \ diagonal chars:
#   line 0: "    ________    "  (top flat edge, 4-indent)
#   line 1: "   / glyph  \   "  (forehead, 3-indent)
#   line 2: "  /  L    R  \  "  (eyes, 2-indent — widest)
#   line 3: "  \   mouth  /  "  (mouth, 2-indent — widest)
#   line 4: "   \________/   "  (base, 3-indent)
#
# FACE_FRAMES[state] = list of (color, lines) tuples.
# The renderer cycles through frames using frame_index % len(frames).

FACE_FRAMES: dict[str, list[tuple[str, list[str]]]] = {
    # Somnus: 4-frame breathing cycle — z's drift, eyes open/close
    "somnus": [
        ("dim orange3", [
            "    ________    ",
            "   /        \\   ",
            "  /  u    u  \\  ",
            "  \\    -     /  ",
            "   \\________/   ",
        ]),
        ("dim orange3", [
            "    ________    ",
            "   /   z    \\   ",
            "  /  u    u  \\  ",
            "  \\    .     /  ",
            "   \\________/   ",
        ]),
        ("orange3", [
            "    ________    ",
            "   /  z  z  \\   ",
            "  /  -    -  \\  ",
            "  \\    -     /  ",
            "   \\________/   ",
        ]),
        ("dim orange3", [
            "    ________    ",
            "   /     z  \\   ",
            "  /  u    u  \\  ",
            "  \\    .     /  ",
            "   \\________/   ",
        ]),
    ],
    # Excito: 2-frame startle wobble — mouth flips, ! flickers
    "excito": [
        ("bold yellow", [
            "    ________    ",
            "   /   !!   \\   ",
            "  /  @    @  \\  ",
            "  \\    oO    /  ",
            "   \\________/   ",
        ]),
        ("bold yellow", [
            "    ________    ",
            "   /   !    \\   ",
            "  /  @    @  \\  ",
            "  \\    Oo    /  ",
            "   \\________/   ",
        ]),
    ],
    # Ausculto: 2-frame listen pulse — eyes widen, mouth shifts
    "ausculto": [
        ("bold dodger_blue1", [
            "    ________    ",
            "   /  [*]   \\   ",
            "  /  ^    ^  \\  ",
            "  \\    w     /  ",
            "   \\________/   ",
        ]),
        ("bold dodger_blue1", [
            "    ________    ",
            "   /  [*]   \\   ",
            "  /  o    o  \\  ",
            "  \\    u     /  ",
            "   \\________/   ",
        ]),
    ],
    # Cogito: 4-frame think cycle — squint swaps, dots shift, mouth changes
    "cogito": [
        ("bold medium_purple", [
            "    ________    ",
            "   /  . .   \\   ",
            "  /  o    -  \\  ",
            "  \\    ~     /  ",
            "   \\________/   ",
        ]),
        ("medium_purple", [
            "    ________    ",
            "   /   ..   \\   ",
            "  /  o    -  \\  ",
            "  \\    =     /  ",
            "   \\________/   ",
        ]),
        ("bold medium_purple", [
            "    ________    ",
            "   /    . . \\   ",
            "  /  -    o  \\  ",
            "  \\    ~     /  ",
            "   \\________/   ",
        ]),
        ("medium_purple", [
            "    ________    ",
            "   /   ..   \\   ",
            "  /  -    o  \\  ",
            "  \\    =     /  ",
            "   \\________/   ",
        ]),
    ],
    # Dico: 3-frame talk cycle — mouth shape changes, note bounces
    "dico": [
        ("bold green3", [
            "    ________    ",
            "   /   ~*   \\   ",
            "  /  ^    ^  \\  ",
            "  \\    D     /  ",
            "   \\________/   ",
        ]),
        ("bold green3", [
            "    ________    ",
            "   /    *~  \\   ",
            "  /  ^    ^  \\  ",
            "  \\    O     /  ",
            "   \\________/   ",
        ]),
        ("green3", [
            "    ________    ",
            "   /   ~*   \\   ",
            "  /  ^    ^  \\  ",
            "  \\    o     /  ",
            "   \\________/   ",
        ]),
    ],
    # Impero: 2-frame menace pulse — cursor twitches, smirk shifts
    "impero": [
        ("bold orange_red1", [
            "    ________    ",
            "   /   >>   \\   ",
            "  /  >    <  \\  ",
            "  \\    J     /  ",
            "   \\________/   ",
        ]),
        ("orange_red1", [
            "    ________    ",
            "   /   }>   \\   ",
            "  /  >    <  \\  ",
            "  \\    j     /  ",
            "   \\________/   ",
        ]),
    ],
    # Erratum: 2-frame distress — eyes pulse, mouth wobbles
    "erratum": [
        ("bold red", [
            "    ________    ",
            "   /   /!   \\   ",
            "  /  x    x  \\  ",
            "  \\    n     /  ",
            "   \\________/   ",
        ]),
        ("red", [
            "    ________    ",
            "   /   /!   \\   ",
            "  /  X    X  \\  ",
            "  \\    v     /  ",
            "   \\________/   ",
        ]),
    ],
}

# Legacy aliases for tests
FACES: dict[str, list[str]] = {k: v[0][1] for k, v in FACE_FRAMES.items()}
FACES_ALT: dict[str, list[str]] = {
    k: v[1][1] for k, v in FACE_FRAMES.items() if len(v) > 1
}

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
        """Return the colored ASCII face for the current state and animation frame."""
        state = self._current_state
        frames = FACE_FRAMES.get(state, FACE_FRAMES["erratum"])

        # Hold each frame for 6 ticks (~1.5s at 4 FPS) before advancing
        frame_idx = (self._frame // 6) % len(frames)
        color, lines = frames[frame_idx]

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

    def _render_transcript(self, max_lines: int = 0) -> Panel:
        """Render the scrolling conversation transcript.

        Shows only the most recent messages that fit in max_lines.
        This gives auto-scroll-to-bottom behavior.
        """
        text = Text()

        if not self._messages:
            text.append("  Waiting for conversation...", style="dim italic")
        else:
            # Estimate lines per message: 2 (header + text) + 1 blank + wrapping
            # Show messages from the end until we run out of space
            messages = self._messages
            if max_lines > 0:
                # Rough estimate: each message takes ~3-4 lines (header, text, blank)
                # plus long messages wrap. Be conservative.
                visible = []
                lines_used = 0
                for msg in reversed(messages):
                    # Estimate: 1 header + ceil(text_len/70) text lines + 1 blank
                    msg_lines = 2 + len(msg["text"]) // 70 + 1
                    if lines_used + msg_lines > max_lines and visible:
                        break
                    visible.append(msg)
                    lines_used += msg_lines
                messages = list(reversed(visible))

            for i, msg in enumerate(messages):
                ts = msg["time"]
                if msg["role"] == "user":
                    text.append(f"  {ts}  ", style="dim")
                    text.append("You\n", style="bold cyan")
                    text.append(f"  {msg['text']}\n", style="white")
                else:
                    text.append(f"  {ts}  ", style="dim")
                    text.append("Nox\n", style="bold green")
                    text.append(f"  {msg['text']}\n", style="white")

                if i < len(messages) - 1:
                    text.append("\n")

        return Panel(text, border_style="dim", padding=(1, 1))

    def _render(self, console_height: int = 40) -> Group:
        """Combine all UI elements into a single renderable group."""
        self._frame += 1

        header = self._render_header()
        # Header panel: 5 face lines + 2 padding + 2 border = ~9 lines
        # Footer: 1 line. Transcript gets the rest.
        transcript_lines = max(console_height - 12, 5)
        transcript = self._render_transcript(max_lines=transcript_lines)
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
        from rich.console import Console
        console = Console()

        # Start socket listener in background
        listener = asyncio.create_task(self._listen_loop())

        try:
            with Live(
                self._render(console.height),
                console=console,
                refresh_per_second=4,
                screen=True,
            ) as live:
                while True:
                    live.update(self._render(console.height))
                    await asyncio.sleep(0.25)
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
