from __future__ import annotations

import enum


class State(enum.Enum):
    """Daemon states with Latin names for logs and IPC."""
    IDLE = "somnus"
    WAKE = "excito"
    LISTENING = "ausculto"
    PROCESSING = "cogito"
    SPEAKING = "dico"
    CONTROLLING = "impero"
    ERROR = "erratum"
