"""Entry point for x11-mcp-voice daemon.

Usage: python -m x11_mcp_voice [--config path/to/config.yaml]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from x11_mcp_voice.config import load_config
from x11_mcp_voice.daemon import Daemon


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice-activated desktop automation daemon")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)
    logging.getLogger(__name__).info(
        "Starting daemon (wake_word=%s, stt=%s, style=%s)",
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
