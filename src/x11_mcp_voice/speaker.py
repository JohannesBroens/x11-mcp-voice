from __future__ import annotations

import io
import logging
import threading
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

# Default location where install.sh downloads piper voice models
_VOICES_DIR = Path.home() / ".local" / "share" / "piper-voices"


def _resolve_voice_path(voice: str) -> str:
    """Resolve a voice name or path to an .onnx model file path.

    Accepts either:
    - A full path to an .onnx file (returned as-is)
    - A voice name like "en_US-ryan-medium" (resolved to ~/.local/share/piper-voices/)
    """
    path = Path(voice)
    if path.suffix == ".onnx" and path.exists():
        return str(path)

    # Try the standard voices directory
    candidate = _VOICES_DIR / f"{voice}.onnx"
    if candidate.exists():
        return str(candidate)

    # Fall through — let PiperVoice.load() raise a clear error
    return str(candidate)


class TTSBackend:
    """Base class for TTS backends."""

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class KokoroBackend(TTSBackend):
    """Kokoro-82M neural TTS. Models auto-download on first run (~350MB)."""

    def __init__(self, voice: str = "af_heart", speed: float = 1.0):
        from kokoro_onnx import Kokoro

        self._kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
        self._voice = voice
        self._speed = speed
        log.info("Kokoro TTS initialized (voice=%s)", voice)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        samples, sample_rate = self._kokoro.create(
            text, voice=self._voice, speed=self._speed, lang="en-us"
        )
        return samples.astype(np.float32), sample_rate

    def close(self) -> None:
        self._kokoro = None


class PiperBackend(TTSBackend):
    """Piper TTS fallback (lightweight, robotic)."""

    def __init__(self, voice: str = "en_US-ryan-medium", speed: float = 1.0):
        self._voice_path = _resolve_voice_path(voice)
        self._speed = speed
        log.info("Piper TTS initialized (voice=%s)", voice)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        from piper import PiperVoice

        voice = PiperVoice.load(self._voice_path)
        wav_buffer = io.BytesIO()

        wav_file = wave.open(wav_buffer, "wb")
        voice.synthesize_wav(text, wav_file)
        wav_file.close()

        wav_buffer.seek(0)
        with wave.open(wav_buffer, "rb") as reader:
            sample_rate = reader.getframerate()
            raw = reader.readframes(reader.getnframes())
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        return audio, sample_rate


class Speaker:
    """Text-to-speech with pluggable backend (Kokoro or Piper)."""

    def __init__(self, engine: str = "kokoro", voice: str = "af_heart", speed: float = 1.0):
        if engine == "kokoro":
            try:
                self._backend = KokoroBackend(voice, speed)
            except Exception:
                log.warning("Kokoro init failed, falling back to piper")
                self._backend = PiperBackend("en_US-ryan-medium", speed)
        else:
            self._backend = PiperBackend(voice, speed)

        self._stop_event = threading.Event()
        self._playing = False

    def speak(self, text: str) -> None:
        """Synthesize and play text. Blocks until playback completes or stop() is called."""
        if not text.strip():
            return

        self._stop_event.clear()
        self._playing = True

        try:
            audio_data, sample_rate = self._backend.synthesize(text)
            self._play(audio_data, sample_rate)
        except Exception:
            log.exception("TTS synthesis/playback failed")
            log.warning("Unspeakable text: %s", text[:200])
        finally:
            self._playing = False

    def stop(self) -> None:
        """Interrupt current playback immediately."""
        self._stop_event.set()
        sd.stop()

    def _play(self, audio: np.ndarray, sample_rate: int) -> None:
        """Play audio through speakers. Respects stop_event."""
        if self._stop_event.is_set():
            return

        # Play in chunks so we can check stop_event
        chunk_size = sample_rate  # 1 second chunks
        for i in range(0, len(audio), chunk_size):
            if self._stop_event.is_set():
                return
            chunk = audio[i : i + chunk_size]
            sd.play(chunk, samplerate=sample_rate)
            sd.wait()
