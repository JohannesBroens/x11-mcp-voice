from __future__ import annotations

import asyncio
import enum
import logging
import signal
import time

import numpy as np

from x11_mcp_voice.agent import Agent
from x11_mcp_voice.config import Config
from x11_mcp_voice.media_control import MediaController
from x11_mcp_voice.speaker import Speaker
from x11_mcp_voice.transcriber import Transcriber
from x11_mcp_voice.wake_word import WakeWordDetector

log = logging.getLogger(__name__)


class State(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


class Daemon:
    """Voice daemon state machine: IDLE -> LISTENING -> PROCESSING -> SPEAKING -> IDLE."""

    def __init__(self, config: Config):
        self._config = config
        self._state = State.IDLE
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake_event = asyncio.Event()

        # Audio recording buffer
        self._audio_buffer: list[np.ndarray] = []
        self._recording = False

        # Components
        self._wake_detector = WakeWordDetector(
            model=config.wake_word.model,
            threshold=config.wake_word.threshold,
        )
        self._transcriber = Transcriber(
            model_name=config.stt.model,
            device=config.stt.device,
        )
        self._speaker = Speaker(
            voice=config.tts.voice,
            speed=config.tts.speed,
        )
        self._media = MediaController(player=config.media.player)
        self._agent = Agent(config.agent, config.conversation)

    async def run(self) -> None:
        """Main entry point. Runs the state machine forever until interrupted."""
        self._loop = asyncio.get_running_loop()

        # Handle shutdown signals
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._loop.add_signal_handler(sig, self._shutdown)

        # Connect to x11-mcp
        await self._agent.connect()

        # Start wake word detection
        self._wake_detector.start(
            on_wake=self._on_wake_word,
            audio_callback=self._on_audio_chunk,
        )

        log.info("Daemon started — listening for wake word")

        try:
            while True:
                await self._wake_event.wait()
                self._wake_event.clear()
                await self._handle_interaction()
        except asyncio.CancelledError:
            pass
        finally:
            await self._cleanup()

    def _on_wake_word(self) -> None:
        """Called from wake word thread when wake word detected.

        Uses call_soon_threadsafe when a real asyncio event loop is available
        to safely schedule the event set from a background thread. Falls back
        to direct set otherwise (e.g. in tests with a mock loop).
        """
        if isinstance(self._loop, asyncio.AbstractEventLoop):
            self._loop.call_soon_threadsafe(self._wake_event.set)
        else:
            self._wake_event.set()

    def _on_audio_chunk(self, chunk: np.ndarray) -> None:
        """Called from wake word thread for every audio chunk."""
        if self._recording:
            self._audio_buffer.append(chunk.copy())

    async def _handle_interaction(self) -> None:
        """Handle a complete interaction: listen -> process -> speak, with follow-up loop."""
        self._state = State.LISTENING
        log.info("State: LISTENING")

        # Pause media
        if self._config.media.auto_pause:
            self._media.pause()

        is_followup = False
        while True:
            # Record audio
            audio = await self._record()
            if audio is None or len(audio) < self._config.audio.sample_rate * 0.3:
                # Too short — probably noise
                if is_followup:
                    break  # No follow-up, end interaction
                log.debug("Recording too short, returning to idle")
                break

            # Process: transcribe + send to Claude
            self._state = State.PROCESSING
            log.info("State: PROCESSING")
            response = await self._process(audio)

            if response is None:
                break

            # Pause media before speaking so TTS is audible
            if self._config.media.auto_pause:
                self._media.pause()

            # Speak response
            self._state = State.SPEAKING
            log.info("State: SPEAKING (%d chars)", len(response))
            await asyncio.get_event_loop().run_in_executor(None, self._speaker.speak, response)

            # Check if we should listen for follow-up
            if self._config.conversation.style == "walkie_talkie":
                break

            # Pause media during follow-up listening so mic doesn't pick up music
            if self._config.media.auto_pause:
                self._media.pause()

            # Wait for follow-up speech (VAD-based)
            self._state = State.LISTENING
            log.info("State: LISTENING (follow-up window)")
            is_followup = True
            has_speech = await self._wait_for_speech(self._config.conversation.followup_timeout_s)
            if not has_speech:
                break

        # End of interaction
        self._state = State.IDLE
        log.info("State: IDLE")
        if self._config.media.auto_pause:
            self._media.resume()

        # Reset conversation in walkie_talkie mode
        if self._config.conversation.style == "walkie_talkie":
            self._agent.reset()

    async def _record(self) -> np.ndarray | None:
        """Record audio until silence detected. Returns audio buffer as numpy array."""
        self._audio_buffer = []
        self._recording = True

        silence_frames = 0
        frames_per_chunk = 1280  # 80ms at 16kHz
        silence_threshold = int(
            self._config.audio.silence_threshold_ms / 80  # chunks of silence
        )
        max_chunks = int(
            self._config.audio.max_recording_s * self._config.audio.sample_rate / frames_per_chunk
        )
        speech_started = False

        # Load silero VAD
        vad_model, vad_utils = _load_silero_vad()

        chunk_count = 0
        while chunk_count < max_chunks:
            # Wait for audio chunks to arrive from the wake word thread
            await asyncio.sleep(0.08)  # ~80ms, one chunk period
            chunk_count += 1

            if not self._audio_buffer:
                continue

            # Check latest chunk for speech
            latest = self._audio_buffer[-1]
            is_speech = _check_vad(vad_model, latest)

            if is_speech:
                speech_started = True
                silence_frames = 0
            elif speech_started:
                silence_frames += 1
                if silence_frames >= silence_threshold:
                    break

        self._recording = False

        if not self._audio_buffer:
            return None

        return np.concatenate(self._audio_buffer)

    async def _process(self, audio: np.ndarray) -> str | None:
        """Transcribe audio and send to Claude agent. Returns response text or None."""
        text = self._transcriber.transcribe(audio)
        log.info("Transcribed: %s", text)

        if not text.strip():
            return None

        try:
            response = await self._agent.send(text)
            log.info("Claude response: %s", response[:200])
            return response
        except Exception:
            log.exception("Agent failed")
            return "I couldn't process that. Try again."

    async def _wait_for_speech(self, timeout: float) -> bool:
        """Wait for speech within timeout. Returns True if speech detected."""
        vad_model, _ = _load_silero_vad()
        self._audio_buffer = []
        self._recording = True

        deadline = time.monotonic() + timeout
        detected = False

        while time.monotonic() < deadline:
            await asyncio.sleep(0.08)
            if self._audio_buffer:
                latest = self._audio_buffer[-1]
                if _check_vad(vad_model, latest):
                    detected = True
                    break

        if not detected:
            self._recording = False
        # If detected, leave recording=True so _record() continues capturing

        return detected

    def _shutdown(self) -> None:
        """Signal handler for graceful shutdown."""
        log.info("Shutdown requested")
        for task in asyncio.all_tasks(self._loop):
            task.cancel()

    async def _cleanup(self) -> None:
        """Clean up all resources."""
        self._wake_detector.stop()
        await self._agent.disconnect()
        log.info("Daemon stopped")


# Silero VAD singleton
_silero_vad = None


def _load_silero_vad():
    """Load silero VAD model (cached singleton)."""
    global _silero_vad
    if _silero_vad is None:
        import torch
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        _silero_vad = (model, utils)
    return _silero_vad


def _check_vad(model, audio_chunk: np.ndarray) -> bool:
    """Check if audio chunk contains speech using silero VAD."""
    import torch
    tensor = torch.from_numpy(audio_chunk).float()
    if tensor.dim() > 1:
        tensor = tensor.squeeze()
    # Silero VAD requires exactly 512 samples at 16kHz.
    # Our chunks are 1280 samples (80ms) — check the last 512.
    if tensor.shape[0] > 512:
        tensor = tensor[-512:]
    speech_prob = model(tensor, 16000).item()
    return speech_prob > 0.5
