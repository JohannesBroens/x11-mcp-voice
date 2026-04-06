from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class WakeWordConfig:
    model: str = "hey_jarvis"
    threshold: float = 0.7


@dataclass
class STTConfig:
    model: str = "nvidia/parakeet-tdt-0.6b-v2"
    device: str = "cuda"


@dataclass
class TTSConfig:
    voice: str = "en_US-ryan-medium"
    speed: float = 1.0


@dataclass
class MediaConfig:
    auto_pause: bool = True
    player: str | None = None


@dataclass
class AgentConfig:
    model: str = "claude-sonnet-4-6"
    x11_mcp_command: str = "~/Documents/git/x11-mcp/.venv/bin/python"
    x11_mcp_args: list[str] = field(default_factory=lambda: ["-m", "x11_mcp"])


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    silence_threshold_ms: int = 500
    max_recording_s: int = 30


@dataclass
class ConversationConfig:
    style: str = "auto"
    followup_timeout_s: float = 3.0


@dataclass
class Config:
    wake_word: WakeWordConfig = field(default_factory=WakeWordConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    media: MediaConfig = field(default_factory=MediaConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)


_SECTION_CLASSES = {
    "wake_word": WakeWordConfig,
    "stt": STTConfig,
    "tts": TTSConfig,
    "media": MediaConfig,
    "agent": AgentConfig,
    "audio": AudioConfig,
    "conversation": ConversationConfig,
}


def _merge_section(section_cls, overrides: dict):
    """Create a config section dataclass, applying only known fields from overrides."""
    known = {f.name for f in fields(section_cls)}
    filtered = {k: v for k, v in overrides.items() if k in known}
    return section_cls(**filtered)


def load_config(path: str | None = None) -> Config:
    """Load config from YAML file, merging with defaults.

    If path is None, searches ./config.yaml then ~/.config/x11-mcp-voice/config.yaml.
    Returns default Config if no file found.
    """
    if path is None:
        candidates = [
            Path("config.yaml"),
            Path.home() / ".config" / "x11-mcp-voice" / "config.yaml",
        ]
        for candidate in candidates:
            if candidate.is_file():
                path = str(candidate)
                break

    if path is None or not Path(path).is_file():
        return Config()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    sections = {}
    for name, cls in _SECTION_CLASSES.items():
        if name in raw and isinstance(raw[name], dict):
            sections[name] = _merge_section(cls, raw[name])
        else:
            sections[name] = cls()

    return Config(**sections)
