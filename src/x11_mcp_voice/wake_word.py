from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

# openwakeword expects 16kHz mono int16 in 1280-sample chunks (80ms)
_CHUNK_SAMPLES = 1280
_SAMPLE_RATE = 16000


class WakeWordDetector:
    """Always-on wake word detection via openwakeword on a background thread."""

    def __init__(self, model: str = "hey_jarvis", threshold: float = 0.7):
        self._model_name = model
        self._threshold = threshold
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._on_wake: Callable[[], None] | None = None
        self._audio_callback: Callable[[np.ndarray], None] | None = None

    def start(
        self,
        on_wake: Callable[[], None],
        audio_callback: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        """Start wake word detection on a background thread.

        Args:
            on_wake: Called when wake word is detected.
            audio_callback: If provided, receives every audio chunk (for recording buffer).
        """
        self._on_wake = on_wake
        self._audio_callback = audio_callback
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="wake-word")
        self._thread.start()
        log.info("Wake word detector started (model=%s, threshold=%.2f)", self._model_name, self._threshold)

    def stop(self) -> None:
        """Stop the background thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        log.info("Wake word detector stopped")

    def _run(self) -> None:
        """Background thread: stream mic audio, detect wake word."""
        import openwakeword
        from openwakeword.model import Model

        # Download/load model
        openwakeword.utils.download_models([self._model_name])
        oww_model = Model(wakeword_models=[self._model_name])

        try:
            with sd.InputStream(
                samplerate=_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=_CHUNK_SAMPLES,
            ) as stream:
                while not self._stop_event.is_set():
                    audio_chunk, overflowed = stream.read(_CHUNK_SAMPLES)
                    if overflowed:
                        log.debug("Audio buffer overflow in wake word thread")

                    # audio_chunk is (1280, 1) int16 — flatten for openwakeword
                    chunk_flat = audio_chunk.flatten()

                    # Also send float32 version to audio_callback for recording
                    if self._audio_callback is not None:
                        float_chunk = chunk_flat.astype(np.float32) / 32768.0
                        self._audio_callback(float_chunk)

                    # Feed to wake word model
                    prediction = oww_model.predict(chunk_flat)

                    for model_name, score in prediction.items():
                        if score >= self._threshold:
                            log.info("Wake word detected: %s (score=%.3f)", model_name, score)
                            oww_model.reset()
                            if self._on_wake:
                                self._on_wake()
                            break

        except Exception:
            log.exception("Wake word thread crashed")
