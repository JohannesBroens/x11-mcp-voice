import pytest
from unittest.mock import patch, MagicMock
from x11_mcp_voice.state import State


def test_tray_module_imports():
    from x11_mcp_voice import tray
    assert hasattr(tray, 'NoxTray')


def test_tray_state_to_icon_name():
    from x11_mcp_voice.tray import NoxTray
    tray_obj = NoxTray.__new__(NoxTray)
    tray_obj._icons_dir = "/fake/icons"
    assert tray_obj._icon_filename("somnus") == "nox-somnus.png"
    assert tray_obj._icon_filename("impero") == "nox-impero.png"
    assert tray_obj._icon_filename("unknown") == "nox-erratum.png"


def test_tray_tooltip_format():
    from x11_mcp_voice.tray import NoxTray
    tray_obj = NoxTray.__new__(NoxTray)
    assert tray_obj._tooltip("somnus") == "Nox - somnus"
    assert tray_obj._tooltip("cogito") == "Nox - cogito..."


_ACTIVE_STATES = {"ausculto", "cogito", "dico", "impero", "excito"}


def test_tray_tooltip_ellipsis_for_active_states():
    from x11_mcp_voice.tray import NoxTray
    tray_obj = NoxTray.__new__(NoxTray)
    for state_val in _ACTIVE_STATES:
        assert tray_obj._tooltip(state_val).endswith("..."), f"{state_val} should have ellipsis"
    assert not tray_obj._tooltip("somnus").endswith("...")
    assert not tray_obj._tooltip("erratum").endswith("...")
