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


@pytest.mark.asyncio
async def test_state_server_broadcasts_user_text(tmp_path):
    sock_path = str(tmp_path / "nox.sock")
    server = StateServer(sock_path)
    await server.start()
    reader, writer = await asyncio.open_unix_connection(sock_path)
    await reader.readline()  # consume initial state
    await server.set_state(State.PROCESSING, user_text="hello world")
    line = await asyncio.wait_for(reader.readline(), timeout=2.0)
    msg = json.loads(line)
    assert msg["state"] == "cogito"
    assert msg["user_text"] == "hello world"
    assert "assistant_text" not in msg
    writer.close()
    await writer.wait_closed()
    await server.stop()


@pytest.mark.asyncio
async def test_input_server_start_stop(tmp_path):
    from x11_mcp_voice.state import InputServer
    sock_path = str(tmp_path / "nox-input.sock")
    server = InputServer(sock_path)
    await server.start(callback=lambda text: None)
    assert os.path.exists(sock_path)
    await server.stop()


@pytest.mark.asyncio
async def test_input_server_receives_text(tmp_path):
    from x11_mcp_voice.state import InputServer
    import socket as sock
    sock_path = str(tmp_path / "nox-input.sock")
    received = []

    async def on_text(text):
        received.append(text)

    server = InputServer(sock_path)
    await server.start(callback=on_text)

    # Send text via raw socket
    s = sock.socket(sock.AF_UNIX, sock.SOCK_STREAM)
    s.connect(sock_path)
    s.sendall(b"hello world\n")
    s.close()

    # Give the server time to process
    await asyncio.sleep(0.1)

    assert received == ["hello world"]
    await server.stop()


@pytest.mark.asyncio
async def test_input_server_ignores_blank_lines(tmp_path):
    from x11_mcp_voice.state import InputServer
    import socket as sock
    sock_path = str(tmp_path / "nox-input.sock")
    received = []

    async def on_text(text):
        received.append(text)

    server = InputServer(sock_path)
    await server.start(callback=on_text)

    s = sock.socket(sock.AF_UNIX, sock.SOCK_STREAM)
    s.connect(sock_path)
    s.sendall(b"\n\nactual text\n\n")
    s.close()

    await asyncio.sleep(0.1)
    assert received == ["actual text"]
    await server.stop()


@pytest.mark.asyncio
async def test_state_server_broadcasts_assistant_text(tmp_path):
    sock_path = str(tmp_path / "nox.sock")
    server = StateServer(sock_path)
    await server.start()
    reader, writer = await asyncio.open_unix_connection(sock_path)
    await reader.readline()  # consume initial state
    await server.set_state(State.SPEAKING, assistant_text="I opened Firefox")
    line = await asyncio.wait_for(reader.readline(), timeout=2.0)
    msg = json.loads(line)
    assert msg["state"] == "dico"
    assert msg["assistant_text"] == "I opened Firefox"
    assert "user_text" not in msg
    writer.close()
    await writer.wait_closed()
    await server.stop()
