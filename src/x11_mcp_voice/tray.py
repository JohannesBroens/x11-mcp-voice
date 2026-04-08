"""Nox system tray indicator.

Connects to the daemon's Unix socket and displays live state as a tray icon.
Uses AyatanaAppIndicator3 directly for reliable GNOME Shell support.
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

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("AyatanaAppIndicator3", "0.1")
from gi.repository import GLib, Gtk, AyatanaAppIndicator3 as AppIndicator

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
        self._running = True

        # AppIndicator uses icon names (no extension) from a theme path
        self._indicator = AppIndicator.Indicator.new(
            "nox",
            "nox-erratum",
            AppIndicator.IndicatorCategory.APPLICATION_STATUS,
        )
        self._indicator.set_icon_theme_path(self._icons_dir)
        self._indicator.set_status(AppIndicator.IndicatorStatus.ACTIVE)
        self._indicator.set_title("Nox - erratum")
        self._indicator.set_menu(self._build_menu())

    def _icon_name(self, state: str) -> str:
        """Return icon name (no extension) for the given state."""
        if state not in _VALID_STATES:
            state = "erratum"
        return f"nox-{state}"

    def _icon_filename(self, state: str) -> str:
        """Return icon filename for tests."""
        if state not in _VALID_STATES:
            state = "erratum"
        return f"nox-{state}.png"

    def _tooltip(self, state: str) -> str:
        suffix = "..." if state in _ACTIVE_STATES else ""
        return f"Nox - {state}{suffix}"

    def _update_icon(self, state: str) -> None:
        """Update the tray icon. Must be called from the GTK main thread."""
        self._current_state = state
        icon_name = self._icon_name(state)
        title = self._tooltip(state)

        def _do_update():
            self._indicator.set_icon_full(icon_name, title)
            self._indicator.set_title(title)
            self._status_item.set_label(f"● {state}")

        GLib.idle_add(_do_update)

    def _build_menu(self) -> Gtk.Menu:
        menu = Gtk.Menu()

        self._status_item = Gtk.MenuItem(label=f"● {self._current_state}")
        self._status_item.set_sensitive(False)
        menu.append(self._status_item)

        menu.append(Gtk.SeparatorMenuItem())

        for label, action in [
            ("Start", "start"),
            ("Stop", "stop"),
            ("Restart", "restart"),
        ]:
            item = Gtk.MenuItem(label=label)
            item.connect("activate", lambda _, a=action: self._systemctl(a))
            menu.append(item)

        menu.append(Gtk.SeparatorMenuItem())

        open_chat_item = Gtk.MenuItem(label="Open Chat")
        open_chat_item.connect("activate", lambda _: self._open_chat())
        menu.append(open_chat_item)

        self._chat_toggle = Gtk.CheckMenuItem(label="Chat on Login")
        self._chat_toggle.set_active(self._chat_desktop_exists())
        self._chat_toggle.connect("toggled", self._on_chat_toggle)
        menu.append(self._chat_toggle)

        menu.append(Gtk.SeparatorMenuItem())

        log_item = Gtk.MenuItem(label="Open Log")
        log_item.connect("activate", lambda _: self._open_log())
        menu.append(log_item)

        menu.append(Gtk.SeparatorMenuItem())

        quit_item = Gtk.MenuItem(label="Quit Tray")
        quit_item.connect("activate", lambda _: self._quit())
        menu.append(quit_item)

        menu.show_all()
        return menu

    def _systemctl(self, action: str) -> None:
        subprocess.run(
            ["systemctl", "--user", action, "nox-daemon"], capture_output=True
        )

    def _open_chat(self) -> None:
        """Open a new terminal with nox chat."""
        nox_bin = Path.home() / ".local" / "bin" / "nox"
        subprocess.Popen(
            ["gnome-terminal", "--title=Nox Chat", "--", str(nox_bin), "chat"]
        )

    def _chat_desktop_path(self) -> Path:
        return Path.home() / ".config" / "autostart" / "nox-chat.desktop"

    def _chat_desktop_exists(self) -> bool:
        return self._chat_desktop_path().exists()

    def _on_chat_toggle(self, widget) -> None:
        """Toggle chat-on-login autostart."""
        if widget.get_active():
            # Enable: run nox install-chat-desktop via subprocess
            subprocess.run(
                [str(Path.home() / ".local" / "bin" / "nox"), "install"],
                capture_output=True,
            )
        else:
            # Disable: remove the .desktop file
            desktop = self._chat_desktop_path()
            if desktop.exists():
                desktop.unlink()
        log.info("Chat on login: %s", "enabled" if widget.get_active() else "disabled")

    def _open_log(self) -> None:
        log_file = Path.home() / ".local" / "log" / "nox" / "daemon.log"
        if log_file.exists():
            subprocess.Popen(["xdg-open", str(log_file)])

    def _quit(self) -> None:
        self._running = False
        Gtk.main_quit()

    def _socket_loop(self) -> None:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self._listen())

    async def _listen(self) -> None:
        while self._running:
            try:
                reader, writer = await asyncio.open_unix_connection(
                    self._socket_path
                )
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

    def run(self) -> None:
        """Start socket listener thread and run GTK main loop."""
        thread = threading.Thread(target=self._socket_loop, daemon=True)
        thread.start()
        Gtk.main()


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
