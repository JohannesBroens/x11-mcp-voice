"""Interactive onboarding wizard for Nox.

Run as: nox setup
"""
from __future__ import annotations

import os
import sys
import time
import yaml
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.text import Text

console = Console()

CONFIG_PATH = Path.home() / ".config" / "x11-mcp-voice" / "config.yaml"
CONTEXT_PATH = Path.home() / ".config" / "nox" / "user-context.txt"

VOICE_SAMPLES = {
    "af_heart": "Hi, I'm Nox using the Heart voice. How do I sound?",
    "af_bella": "Hi, I'm Nox using the Bella voice. How do I sound?",
    "bf_emma": "Hi, I'm Nox using the Emma voice. How do I sound?",
}


def _speak(text: str, voice: str = "af_heart", engine: str = "kokoro") -> None:
    """Speak text using the TTS engine."""
    try:
        from x11_mcp_voice.speaker import Speaker
        speaker = Speaker(engine=engine, voice=voice)
        speaker.speak(text)
    except Exception as e:
        console.print(f"[dim](TTS unavailable: {e})[/]")


def _save_config(choices: dict, config_path: Path | None = None) -> None:
    """Save onboarding choices to config.yaml."""
    path = config_path or CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    config = {}
    if path.exists():
        with open(path) as f:
            config = yaml.safe_load(f) or {}

    config.setdefault("tts", {})
    config["tts"]["engine"] = "kokoro"
    config["tts"]["voice"] = choices.get("voice", "af_heart")

    config.setdefault("conversation", {})
    config["conversation"]["style"] = choices.get("style", "auto")

    config.setdefault("wake_word", {})
    config["wake_word"]["model"] = "hey_nox"
    config["wake_word"]["threshold"] = 0.5

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    console.print(f"[green]Config saved to {path}[/]")


def _save_context(choices: dict, context_path: Path | None = None) -> None:
    """Create user-context.txt with gathered info."""
    path = context_path or CONTEXT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Nox User Context", "# Edit this file to customize Nox's behavior", ""]

    name = choices.get("name")
    if name:
        lines.append(f"- My name is {name}")

    lines.append(f"- I prefer the {choices.get('voice', 'af_heart')} voice")
    lines.append(f"- Conversation style: {choices.get('style', 'auto')}")
    lines.append("")
    lines.append("# Add your own preferences below:")
    lines.append("# - When I say 'play music', use the YouTube Music desktop app")
    lines.append("# - I prefer concise answers")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    console.print(f"[green]Context saved to {path}[/]")


def main() -> None:
    console.print()
    console.print(Panel(
        Text("Nox Setup Wizard", style="bold orange3", justify="center"),
        border_style="orange3",
        padding=(1, 2),
    ))
    console.print()

    choices = {}

    # Step 1: Greeting
    console.print("[bold]Step 1:[/] Introduction")
    console.print("[dim]Nox will speak to you...[/]")
    _speak("Hi! I'm Nox, your voice assistant. Let me help you get set up.")
    console.print()

    # Step 2: Voice selection
    console.print("[bold]Step 2:[/] Choose a voice")
    console.print("[dim]I'll play three voice samples. Pick the one you like best.[/]")
    console.print()

    for i, (voice_id, sample_text) in enumerate(VOICE_SAMPLES.items(), 1):
        console.print(f"  Playing voice {i}: [cyan]{voice_id}[/]")
        _speak(sample_text, voice=voice_id)
        time.sleep(0.5)

    voice_choice = Prompt.ask(
        "\nWhich voice do you prefer?",
        choices=["1", "2", "3"],
        default="1",
    )
    voice_map = {"1": "af_heart", "2": "af_bella", "3": "bf_emma"}
    choices["voice"] = voice_map[voice_choice]
    console.print(f"[green]Selected: {choices['voice']}[/]")
    console.print()

    # Step 3: Conversation style
    console.print("[bold]Step 3:[/] Conversation style")
    console.print("  [cyan]auto[/] — Nox can ask follow-up questions and have a conversation")
    console.print("  [cyan]walkie_talkie[/] — One command at a time, no follow-ups")

    style = Prompt.ask(
        "\nWhich style?",
        choices=["auto", "walkie_talkie"],
        default="auto",
    )
    choices["style"] = style
    console.print()

    # Step 4: User's name
    console.print("[bold]Step 4:[/] What should Nox call you?")
    name = Prompt.ask("Your name", default="")
    if name.strip():
        choices["name"] = name.strip()
        _speak(f"Nice to meet you, {name}!", voice=choices["voice"])
    console.print()

    # Step 5: System context
    from x11_mcp_voice.system_context import detect_system_context
    ctx = detect_system_context()
    console.print("[bold]Step 5:[/] Your system")
    console.print(ctx)
    console.print()

    # Step 6: Save config
    console.print("[bold]Step 6:[/] Saving your preferences...")
    _save_config(choices)
    _save_context(choices)
    console.print()

    # Step 7: Ready
    console.print(Panel(
        Text("Setup complete!", style="bold green", justify="center"),
        border_style="green",
        padding=(1, 2),
    ))
    console.print()
    console.print("  [bold]nox[/]          — Start the daemon + tray")
    console.print("  [bold]nox chat[/]     — Open the chat TUI")
    console.print("  [bold]nox context[/]  — Edit your preferences")
    console.print("  Say [bold cyan]Hey Nox[/] to wake me up!")
    console.print()

    _speak("All set! Say Hey Nox whenever you need me.", voice=choices["voice"])


if __name__ == "__main__":
    main()
