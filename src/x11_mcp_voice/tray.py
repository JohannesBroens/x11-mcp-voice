"""Nox system tray indicator.

Connects to the daemon's Unix socket and displays live state as a tray icon.
Run as: python -m x11_mcp_voice.tray
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import threading
from pathlib import Path

from x11_mcp_voice.config import load_config
from x11_mcp_voice.state import State

log = logging.getLogger(__name__)

_ACTIVE_STATES = {"ausculto", "cogito", "dico", "impero", "excito"}
_VALID_STATES = {s.value for s in State}


class NoxTray:
    """System tray indicator that reflects daemon state."""

    def __init__(self, socket_path: str):
        self._socket_path = socket_path
        self._icons_dir = str(Path(__file__).parent / "icons")
        self._current_state = "erratum"
        self._tray = None
        self._running = True

    def _icon_filename(self, state: str) -> str:
        if state not in _VALID_STATES:
            state = "erratum"
        return f"nox-{state}.png"

    def _tooltip(self, state: str) -> str:
        suffix = "..." if state in _ACTIVE_STATES else ""
        return f"Nox \u2014 {state}{suffix}"

    def _load_icon(self, state: str):
        from PIL import Image
        icon_path = os.path.join(self._icons_dir, self._icon_filename(state))
        if not os.path.exists(icon_path):
            icon_path = os.path.join(self._icons_dir, "nox-erratum.png")
        return Image.open(icon_path)

    def _update_icon(self, state: str) -> None:
        self._current_state = state
        if self._tray:
            self._tray.icon = self._load_icon(state)
            self._tray.title = self._tooltip(state)

    def _on_start(self, tray_icon) -> None:
        thread = threading.Thread(target=self._socket_loop, daemon=True)
        thread.start()

    def _socket_loop(self) -> None:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self._listen())

    async def _listen(self) -> None:
        while self._running:
            try:
                reader, writer = await asyncio.open_unix_connection(self._socket_path)
                log.info("Connected to daemon at %s", self._socket_path)
                while self._running:
                    line = await reader.readline()
                    if not line:
                        break
                    try:
                        msg = json.loads(line)
                        state = msg.get("state", "erratum")
                        self._update_icon(state)
                    except json.JSONDecodeError:
                        continue
            except (ConnectionRefusedError, FileNotFoundError, OSError):
                self._update_icon("erratum")
                await asyncio.sleep(5)

    def _systemctl(self, action: str) -> None:
        subprocess.run(["systemctl", "--user", action, "nox-daemon"], capture_output=True)

    def _open_log(self) -> None:
        log_file = Path.home() / ".local" / "log" / "nox" / "daemon.log"
        if log_file.exists():
            subprocess.Popen(["xdg-open", str(log_file)])

    def run(self) -> None:
        import pystray
        menu = pystray.Menu(
            pystray.MenuItem(lambda item: f"\u25cf {self._current_state}", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Start", lambda: self._systemctl("start")),
            pystray.MenuItem("Stop", lambda: self._systemctl("stop")),
            pystray.MenuItem("Restart", lambda: self._systemctl("restart")),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Open Log", lambda: self._open_log()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit Tray", lambda: self._quit()),
        )
        self._tray = pystray.Icon(
            name="nox", icon=self._load_icon("erratum"),
            title=self._tooltip("erratum"), menu=menu,
        )
        self._tray.run(setup=self._on_start)

    def _quit(self) -> None:
        self._running = False
        if self._tray:
            self._tray.stop()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s nox-tray %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    config = load_config()
    socket_path = config.service.socket_path
    if socket_path is None:
        socket_path = f"/run/user/{os.getuid()}/nox.sock"
    tray = NoxTray(socket_path)
    tray.run()


if __name__ == "__main__":
    main()
