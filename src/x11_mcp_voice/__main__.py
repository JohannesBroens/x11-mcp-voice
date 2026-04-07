"""Entry point for x11-mcp-voice daemon.

Usage: python -m x11_mcp_voice [--config path/to/config.yaml]
"""
from __future__ import annotations

import argparse
import asyncio
import logging

from x11_mcp_voice.config import load_config
from x11_mcp_voice.daemon import Daemon


def main() -> None:
    parser = argparse.ArgumentParser(description="Nox — voice-activated desktop automation")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    config = load_config(args.config)

    # Log directory
    from pathlib import Path
    from logging.handlers import RotatingFileHandler

    log_dir = Path.home() / ".local" / "log" / "nox"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = config.service.log_file or str(log_dir / "daemon.log")

    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s nox [%(nox_state)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        defaults={"nox_state": "somnus"},
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=3,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    logging.getLogger(__name__).info(
        "starting daemon (wake_word=%s, stt=%s, style=%s)",
        config.wake_word.model,
        config.stt.model,
        config.conversation.style,
    )

    daemon = Daemon(config)
    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
