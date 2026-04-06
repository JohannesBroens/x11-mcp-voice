from __future__ import annotations

import logging
import subprocess

log = logging.getLogger(__name__)


class MediaController:
    """Controls media playback via playerctl (MPRIS protocol)."""

    def __init__(self, player: str | None = None):
        self._player = player
        self._we_paused = False
        self.available = True

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        cmd = ["playerctl"]
        if self._player:
            cmd.extend(["--player", self._player])
        cmd.extend(args)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5)

    def is_playing(self) -> bool:
        if not self.available:
            return False
        try:
            result = self._run("status")
            return result.stdout.strip() == "Playing"
        except FileNotFoundError:
            log.warning("playerctl not found — media control disabled")
            self.available = False
            return False
        except subprocess.TimeoutExpired:
            return False

    def pause(self) -> bool:
        """Pause media if playing. Returns True if we paused it."""
        if not self.available:
            return False
        if not self.is_playing():
            self._we_paused = False
            return False
        try:
            self._run("pause")
            self._we_paused = True
            return True
        except FileNotFoundError:
            log.warning("playerctl not found — media control disabled")
            self.available = False
            return False

    def resume(self) -> None:
        """Resume media only if we paused it."""
        if not self._we_paused or not self.available:
            return
        try:
            self._run("play")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            log.warning("playerctl resume failed")
        self._we_paused = False
