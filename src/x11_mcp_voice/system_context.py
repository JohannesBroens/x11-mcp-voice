"""Auto-detect system context for the agent's system prompt."""
import logging
import os
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def detect_system_context() -> str:
    """Gather desktop environment info that helps Nox operate the desktop."""
    ctx = []

    def _run(cmd: str) -> str:
        try:
            r = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=5)
            return r.stdout.strip()
        except Exception:
            return ""

    # Desktop environment
    de = os.environ.get("XDG_CURRENT_DESKTOP", "unknown")
    session = os.environ.get("XDG_SESSION_TYPE", "unknown")
    ctx.append(f"Desktop: {de} ({session})")

    # Monitors
    monitors = _run("xrandr --listmonitors")
    if monitors:
        lines = [l.strip() for l in monitors.split("\n") if "+" in l]
        for l in lines:
            ctx.append(f"Monitor: {l}")

    # Default browser
    browser = _run("xdg-settings get default-web-browser")
    if browser:
        ctx.append(f"Default browser: {browser.replace('.desktop', '')}")

    # Keyboard layout
    kb = _run("setxkbmap -query")
    for line in kb.split("\n"):
        if "layout" in line:
            ctx.append(f"Keyboard {line.strip()}")

    # Locale and timezone
    ctx.append(f"Locale: {os.environ.get('LANG', 'unknown')}")
    tz = _run("timedatectl show --property=Timezone --value")
    if tz:
        ctx.append(f"Timezone: {tz}")

    # Terminal
    try:
        term = os.path.basename(os.path.realpath("/usr/bin/x-terminal-emulator"))
        ctx.append(f"Terminal: {term}")
    except OSError:
        pass

    # File manager
    fm = _run("xdg-mime query default inode/directory")
    if fm:
        ctx.append(f"File manager: {fm.replace('.desktop', '')}")

    # Installed apps (useful for Nox to know what's available)
    apps = []
    for app in ["firefox", "brave-browser", "google-chrome", "code", "discord",
                "steam", "spotify", "youtube-music", "vlc", "gimp", "blender"]:
        if _run(f"which {app}"):
            apps.append(app)
    if apps:
        ctx.append(f"Installed apps: {', '.join(apps)}")

    return "\n".join(f"- {line}" for line in ctx)
