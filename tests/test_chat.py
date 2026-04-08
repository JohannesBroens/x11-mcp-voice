from x11_mcp_voice.state import State


def test_chat_module_imports():
    from x11_mcp_voice.chat import NoxChat, FACES, STATE_COLORS, STATE_LABELS
    assert NoxChat is not None


def test_faces_exist_for_all_states():
    from x11_mcp_voice.chat import FACES
    for s in State:
        assert s.value in FACES, f"Missing face for {s.value}"


def test_colors_exist_for_all_states():
    from x11_mcp_voice.chat import STATE_COLORS
    for s in State:
        assert s.value in STATE_COLORS


def test_labels_exist_for_all_states():
    from x11_mcp_voice.chat import STATE_LABELS
    for s in State:
        assert s.value in STATE_LABELS


def test_on_message_updates_state():
    from x11_mcp_voice.chat import NoxChat
    chat = NoxChat.__new__(NoxChat)
    chat._current_state = "somnus"
    chat._messages = []
    chat._connected = True
    chat._max_messages = 50
    chat._on_message({"state": "cogito", "timestamp": 123})
    assert chat._current_state == "cogito"


def test_on_message_appends_user_text():
    from x11_mcp_voice.chat import NoxChat
    chat = NoxChat.__new__(NoxChat)
    chat._current_state = "somnus"
    chat._messages = []
    chat._connected = True
    chat._max_messages = 50
    chat._on_message({"state": "cogito", "timestamp": 123, "user_text": "open firefox"})
    assert len(chat._messages) == 1
    assert chat._messages[0]["role"] == "user"
    assert chat._messages[0]["text"] == "open firefox"


def test_on_message_appends_assistant_text():
    from x11_mcp_voice.chat import NoxChat
    chat = NoxChat.__new__(NoxChat)
    chat._current_state = "somnus"
    chat._messages = []
    chat._connected = True
    chat._max_messages = 50
    chat._on_message({"state": "dico", "timestamp": 123, "assistant_text": "Opening Firefox now"})
    assert len(chat._messages) == 1
    assert chat._messages[0]["role"] == "assistant"
    assert chat._messages[0]["text"] == "Opening Firefox now"


def test_on_message_respects_max_messages():
    from x11_mcp_voice.chat import NoxChat
    chat = NoxChat.__new__(NoxChat)
    chat._current_state = "somnus"
    chat._messages = []
    chat._connected = True
    chat._max_messages = 3
    for i in range(5):
        chat._on_message({"state": "cogito", "timestamp": i, "user_text": f"msg {i}"})
    assert len(chat._messages) == 3
    assert chat._messages[0]["text"] == "msg 2"


def test_faces_have_five_lines():
    from x11_mcp_voice.chat import FACES
    for state, lines in FACES.items():
        assert len(lines) == 5, f"Face for {state} has {len(lines)} lines, expected 5"


def test_faces_alt_subset_of_faces():
    from x11_mcp_voice.chat import FACES, FACES_ALT
    for state in FACES_ALT:
        assert state in FACES, f"FACES_ALT has {state} but FACES does not"
