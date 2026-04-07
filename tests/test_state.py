from x11_mcp_voice.state import State


def test_state_values_are_latin():
    assert State.IDLE.value == "somnus"
    assert State.WAKE.value == "excito"
    assert State.LISTENING.value == "ausculto"
    assert State.PROCESSING.value == "cogito"
    assert State.SPEAKING.value == "dico"
    assert State.CONTROLLING.value == "impero"
    assert State.ERROR.value == "erratum"


def test_state_from_value():
    assert State("somnus") is State.IDLE
    assert State("impero") is State.CONTROLLING


import asyncio
import json
import os
import pytest

from x11_mcp_voice.state import State, StateServer


@pytest.mark.asyncio
async def test_state_server_start_stop(tmp_path):
    sock_path = str(tmp_path / "nox.sock")
    server = StateServer(sock_path)
    await server.start()
    assert os.path.exists(sock_path)
    await server.stop()


@pytest.mark.asyncio
async def test_state_server_broadcasts_state(tmp_path):
    sock_path = str(tmp_path / "nox.sock")
    server = StateServer(sock_path)
    await server.start()

    reader, writer = await asyncio.open_unix_connection(sock_path)
    line = await asyncio.wait_for(reader.readline(), timeout=2.0)
    msg = json.loads(line)
    assert msg["state"] == "somnus"

    await server.set_state(State.LISTENING)
    line = await asyncio.wait_for(reader.readline(), timeout=2.0)
    msg = json.loads(line)
    assert msg["state"] == "ausculto"

    writer.close()
    await writer.wait_closed()
    await server.stop()


@pytest.mark.asyncio
async def test_state_server_sends_detail(tmp_path):
    sock_path = str(tmp_path / "nox.sock")
    server = StateServer(sock_path)
    await server.start()

    reader, writer = await asyncio.open_unix_connection(sock_path)
    await reader.readline()

    await server.set_state(State.PROCESSING, detail="transcribed: hello")
    line = await asyncio.wait_for(reader.readline(), timeout=2.0)
    msg = json.loads(line)
    assert msg["state"] == "cogito"
    assert msg["detail"] == "transcribed: hello"

    writer.close()
    await writer.wait_closed()
    await server.stop()


@pytest.mark.asyncio
async def test_state_server_handles_client_disconnect(tmp_path):
    sock_path = str(tmp_path / "nox.sock")
    server = StateServer(sock_path)
    await server.start()

    reader, writer = await asyncio.open_unix_connection(sock_path)
    await reader.readline()

    writer.close()
    await writer.wait_closed()

    await server.set_state(State.SPEAKING)
    await server.stop()
