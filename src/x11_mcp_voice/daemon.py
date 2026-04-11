from __future__ import annotations

import asyncio
import logging
import signal
import time

import numpy as np

from x11_mcp_voice.agent import Agent
from x11_mcp_voice.config import Config
from x11_mcp_voice.media_control import MediaController
from x11_mcp_voice.speaker import Speaker
from x11_mcp_voice.state import InputServer, State, StateServer
from x11_mcp_voice.transcriber import Transcriber
from x11_mcp_voice.transcript import rotate, save_message
from x11_mcp_voice.wake_word import WakeWordDetector

log = logging.getLogger(__name__)


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

        # State broadcasting
        socket_path = config.service.socket_path
        if socket_path is None:
            import os
            socket_path = f"/run/user/{os.getuid()}/nox.sock"
        self._state_server = StateServer(socket_path)

        # Text input from chat TUI
        input_socket = socket_path.replace("nox.sock", "nox-input.sock")
        self._input_server = InputServer(input_socket)
        self._text_input_queue: asyncio.Queue[str] = asyncio.Queue()

        # Components
        self._wake_detector = WakeWordDetector(
            model=config.wake_word.model,
            threshold=config.wake_word.threshold,
            input_device=config.audio.input_device,
        )
        self._transcriber = Transcriber(
            model_name=config.stt.model,
            device=config.stt.device,
        )
        self._speaker = Speaker(
            engine=config.tts.engine,
            voice=config.tts.voice,
            speed=config.tts.speed,
            output_device=config.audio.output_device,
        )
        self._media = MediaController(player=config.media.player)
        self._agent = Agent(config.agent, config.conversation,
                            state_callback=self._on_agent_tool_use)

    async def run(self) -> None:
        """Main entry point. Runs the state machine forever until interrupted."""
        self._loop = asyncio.get_running_loop()

        # Handle shutdown signals
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._loop.add_signal_handler(sig, self._shutdown)

        # Rotate old transcript entries
        rotate(keep_days=7)

        # Connect to x11-mcp
        await self._agent.connect()
        await self._state_server.start()
        await self._input_server.start(self._on_text_input)

        # Start wake word detection
        self._wake_detector.start(
            on_wake=self._on_wake_word,
            audio_callback=self._on_audio_chunk,
        )

        log.info("Daemon started — listening for wake word")

        try:
            while True:
                # Wait for either wake word or text input
                wake_task = asyncio.create_task(self._wake_event.wait())
                text_task = asyncio.create_task(self._text_input_queue.get())

                done, pending = await asyncio.wait(
                    {wake_task, text_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for t in pending:
                    t.cancel()

                if text_task in done:
                    text = text_task.result()
                    await self._handle_text_interaction(text)
                else:
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

        If the daemon is currently speaking or controlling the desktop,
        the wake word acts as an interrupt — TTS is stopped immediately
        so the user can issue a new command.
        """
        if isinstance(self._loop, asyncio.AbstractEventLoop):
            self._loop.call_soon_threadsafe(self._wake_event.set)
        else:
            self._wake_event.set()

        # Interrupt TTS if speaking
        if self._state in (State.SPEAKING, State.CONTROLLING):
            self._speaker.stop()
            log.info("Wake word interrupt — stopping current action")

    async def _on_text_input(self, text: str) -> None:
        """Called when text input arrives from chat TUI via InputServer."""
        await self._text_input_queue.put(text)

    async def _handle_text_interaction(self, text: str) -> None:
        """Handle a text input interaction (skip wake word + recording)."""
        try:
            self._agent.check_timeout(self._config.conversation.session_timeout_s)
            save_message("user", text)
            await self._set_state(State.PROCESSING, user_text=text)
            response = await self._process_text(text)
            if response:
                save_message("assistant", response)
                if self._config.media.auto_pause:
                    self._media.pause()
                await self._set_state(State.SPEAKING, assistant_text=response)
                await asyncio.get_event_loop().run_in_executor(
                    None, self._speaker.speak, response
                )
        except Exception:
            log.exception("Text interaction failed")
        finally:
            await self._set_state(State.IDLE)
            if self._config.media.auto_pause:
                self._media.resume()

    def _on_audio_chunk(self, chunk: np.ndarray) -> None:
        """Called from wake word thread for every audio chunk."""
        if self._recording:
            self._audio_buffer.append(chunk.copy())

    async def _set_state(
        self,
        state: State,
        detail: str | None = None,
        user_text: str | None = None,
        assistant_text: str | None = None,
    ) -> None:
        """Update state and broadcast via socket."""
        self._state = state
        log.info("%s", detail or state.value, extra={"nox_state": state.value})
        await self._state_server.set_state(
            state, detail, user_text=user_text, assistant_text=assistant_text
        )

    async def _on_agent_tool_use(self, event: dict) -> None:
        """Called from agent when x11-mcp tool is used."""
        tool_name = event.get("tool", "unknown")
        await self._set_state(State.CONTROLLING, detail=f"x11-mcp: {tool_name}")
        await self._state_server.broadcast_event(event)

    async def _handle_interaction(self) -> None:
        """Handle a complete interaction: listen -> process -> speak, with follow-up loop.

        Wraps the inner logic in try/finally to guarantee media resumes
        and state returns to IDLE even if an exception occurs.
        """
        try:
            await self._handle_interaction_inner()
        finally:
            # ALWAYS resume media, even if an exception occurred
            await self._set_state(State.IDLE)
            if self._config.media.auto_pause:
                self._media.resume()
            if self._config.conversation.style == "walkie_talkie":
                self._agent.reset()

    async def _handle_interaction_inner(self) -> None:
        """Core interaction logic: listen -> process -> speak, with follow-up loop."""
        # Reset stale session before starting
        self._agent.check_timeout(self._config.conversation.session_timeout_s)

        await self._set_state(State.WAKE)
        await self._set_state(State.LISTENING)

        # Pause media — wait briefly for playerctl to propagate over D-Bus
        # so the mic doesn't pick up lingering audio
        if self._config.media.auto_pause:
            self._media.pause()
            await asyncio.sleep(0.3)

        is_followup = False
        while True:
            # Record audio
            audio = await self._record()
            if audio is None or len(audio) < self._config.audio.sample_rate * self._config.audio.min_speech_s:
                # Too short — probably noise
                if is_followup:
                    break  # No follow-up, end interaction
                log.debug("Recording too short, returning to idle")
                break

            # Transcribe audio
            text = self._transcriber.transcribe(audio)
            log.info("Transcribed: %s", text)
            if not text.strip():
                break

            # Proofread mode: show transcription, wait for edit or timeout
            if self._config.conversation.proofread:
                await self._state_server.broadcast_event({
                    "type": "proofread", "text": text,
                })
                try:
                    edited = await asyncio.wait_for(
                        self._text_input_queue.get(), timeout=10.0,
                    )
                    text = edited  # use the edited version
                    log.info("Proofread edited: %s", text)
                except asyncio.TimeoutError:
                    log.debug("Proofread timeout — using original transcription")

            # Persist user message
            save_message("user", text)

            # Process: send to Claude
            await self._set_state(State.PROCESSING, user_text=text)
            response = await self._process_text(text)

            if response is None:
                break

            # Persist assistant response
            save_message("assistant", response)

            # Pause media before speaking so TTS is audible
            if self._config.media.auto_pause:
                self._media.pause()

            # Speak response
            await self._set_state(State.SPEAKING, detail=f"{len(response)} chars", assistant_text=response)
            await asyncio.get_event_loop().run_in_executor(None, self._speaker.speak, response)

            # Check if we should listen for follow-up
            if self._config.conversation.style == "walkie_talkie":
                break

            # Resume media during follow-up window — the user has to speak
            # over the music to trigger a follow-up, which filters out
            # ambient noise, keyboard sounds, and accidental vocalizations.
            # Media gets paused again at the top of the loop if they do speak.
            if self._config.media.auto_pause:
                self._media.resume()

            # Wait for follow-up speech (VAD-based)
            # Use longer timeout when Claude asked a question
            if response.rstrip().endswith("?"):
                timeout = self._config.conversation.followup_timeout_question_s
            else:
                timeout = self._config.conversation.followup_timeout_statement_s

            await self._set_state(State.LISTENING, detail="follow-up window")
            is_followup = True
            has_speech = await self._wait_for_speech(timeout)
            if not has_speech:
                break

            # User is speaking — pause media again for clean recording
            if self._config.media.auto_pause:
                self._media.pause()
                await asyncio.sleep(0.3)

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
            is_speech = _check_vad(vad_model, latest, self._config.audio.vad_threshold)

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

    async def _process_text(self, text: str) -> str | None:
        """Send transcribed text to Claude agent. Returns response text or None."""
        try:
            response = await self._agent.send(text)
            log.info("Claude response: %s", response[:200])
            return response
        except FileNotFoundError:
            log.exception("Agent failed — claude binary not found")
            return "Claude Code is not installed or not on the path. Check the logs."
        except Exception as exc:
            log.exception("Agent failed")
            # Give the user the actual error type so they know what went wrong
            err_name = type(exc).__name__
            return f"Something went wrong: {err_name}. Check the logs for details."

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
                if _check_vad(vad_model, latest, self._config.audio.vad_threshold):
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
        await self._input_server.stop()
        await self._state_server.stop()
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


def _check_vad(model, audio_chunk: np.ndarray, threshold: float = 0.6) -> bool:
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
    return speech_prob > threshold
