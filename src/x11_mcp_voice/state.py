from __future__ import annotations

import asyncio
import enum
import json
import logging
import time

log = logging.getLogger(__name__)


class State(enum.Enum):
    """Daemon states with Latin names for logs and IPC."""
    IDLE = "somnus"
    WAKE = "excito"
    LISTENING = "ausculto"
    PROCESSING = "cogito"
    SPEAKING = "dico"
    CONTROLLING = "impero"
    ERROR = "erratum"


class StateServer:
    """Async Unix socket server that broadcasts state changes to connected clients."""

    def __init__(self, socket_path: str):
        self._socket_path = socket_path
        self._state = State.IDLE
        self._detail: str | None = None
        self._user_text: str | None = None
        self._assistant_text: str | None = None
        self._clients: list[asyncio.StreamWriter] = []
        self._server: asyncio.AbstractServer | None = None

    @property
    def state(self) -> State:
        return self._state

    async def start(self) -> None:
        import os
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)
        self._server = await asyncio.start_unix_server(
            self._handle_client, path=self._socket_path
        )
        log.info("StateServer listening on %s", self._socket_path)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
        for writer in list(self._clients):
            try:
                writer.close()
            except Exception:
                pass
        self._clients.clear()
        if self._server:
            await self._server.wait_closed()
        import os
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)
        log.info("StateServer stopped")

    async def set_state(
        self,
        state: State,
        detail: str | None = None,
        user_text: str | None = None,
        assistant_text: str | None = None,
    ) -> None:
        self._state = state
        self._detail = detail
        self._user_text = user_text if user_text is not None else self._user_text
        self._assistant_text = assistant_text if assistant_text is not None else self._assistant_text
        msg = self._make_message(user_text=user_text, assistant_text=assistant_text)
        await self._broadcast(msg)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._clients.append(writer)
        log.debug("Client connected (%d total)", len(self._clients))
        # Yield once so the client's readline() starts waiting before we
        # write; this ensures the selector wakes it up on data arrival.
        await asyncio.sleep(0)
        msg = self._make_message()
        try:
            writer.write(msg)
            await writer.drain()
        except (ConnectionError, BrokenPipeError):
            if writer in self._clients:
                self._clients.remove(writer)

    def _make_message(
        self,
        user_text: str | None = None,
        assistant_text: str | None = None,
    ) -> bytes:
        data: dict = {"state": self._state.value, "timestamp": int(time.time())}
        if self._detail:
            data["detail"] = self._detail
        if user_text is not None:
            data["user_text"] = user_text
        if assistant_text is not None:
            data["assistant_text"] = assistant_text
        return json.dumps(data).encode() + b"\n"

    async def broadcast_event(self, event: dict) -> None:
        """Broadcast a generic event (tool use, thinking, etc.) to clients."""
        import time as _time
        event.setdefault("timestamp", int(_time.time()))
        msg = json.dumps(event).encode() + b"\n"
        await self._broadcast(msg)

    async def _broadcast(self, msg: bytes) -> None:
        dead: list[asyncio.StreamWriter] = []
        for writer in self._clients:
            try:
                writer.write(msg)
                await writer.drain()
            except (ConnectionError, BrokenPipeError):
                dead.append(writer)
        for writer in dead:
            self._clients.remove(writer)
