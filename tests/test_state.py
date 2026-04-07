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
