from __future__ import annotations

import io
import logging
import threading
import wave

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)


class Speaker:
    """Text-to-speech via piper-tts with sounddevice playback."""

    def __init__(self, voice: str = "en_US-ryan-medium", speed: float = 1.0):
        self._voice = voice
        self._speed = speed
        self._stop_event = threading.Event()
        self._playing = False
        log.info("TTS initialized with voice=%s speed=%.1f", voice, speed)

    def speak(self, text: str) -> None:
        """Synthesize and play text. Blocks until playback completes or stop() is called."""
        if not text.strip():
            return

        self._stop_event.clear()
        self._playing = True

        try:
            audio_data, sample_rate = self._synthesize(text)
            self._play(audio_data, sample_rate)
        except Exception:
            log.exception("TTS failed, falling back to console output")
            print(f"[TTS] {text}")
        finally:
            self._playing = False

    def stop(self) -> None:
        """Interrupt current playback immediately."""
        self._stop_event.set()
        sd.stop()

    def _synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Run piper-tts synthesis, return (audio_array, sample_rate)."""
        from piper import PiperVoice

        voice = PiperVoice.load(self._voice)
        wav_buffer = io.BytesIO()

        with wave.open(wav_buffer, "wb") as wav_file:
            voice.synthesize(text, wav_file)

        wav_buffer.seek(0)
        with wave.open(wav_buffer, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            raw = wav_file.readframes(n_frames)
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        return audio, sample_rate

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
