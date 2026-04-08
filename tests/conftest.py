"""
Stub out heavy native dependencies that are not installed in the test environment.
This must run before any test module is imported, so it lives here in conftest.py.
"""
import sys
from unittest.mock import MagicMock

_STUB_MODULES = [
    "sounddevice",
    "openwakeword",
    "openwakeword.model",
    "nemo",
    "nemo.collections",
    "nemo.collections.asr",
    "piper",
    "torch",
    "pystray",
    "PIL",
    "PIL.Image",
    "gi",
    "gi.repository",
]

for _mod in _STUB_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# gi.require_version needs to be a no-op
sys.modules["gi"].require_version = MagicMock()
