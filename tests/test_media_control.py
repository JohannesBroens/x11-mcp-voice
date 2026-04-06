from unittest.mock import patch, MagicMock
import subprocess

from x11_mcp_voice.media_control import MediaController


def test_pause_when_playing():
    ctrl = MediaController()
    with patch("x11_mcp_voice.media_control.subprocess.run") as mock_run:
        mock_run.side_effect = [
            MagicMock(stdout="Playing\n", returncode=0),
            MagicMock(returncode=0),
        ]
        paused = ctrl.pause()
        assert paused is True
        assert ctrl._we_paused is True
        assert mock_run.call_count == 2


def test_pause_when_already_paused():
    ctrl = MediaController()
    with patch("x11_mcp_voice.media_control.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="Paused\n", returncode=0)
        paused = ctrl.pause()
        assert paused is False
        assert ctrl._we_paused is False


def test_resume_after_we_paused():
    ctrl = MediaController()
    ctrl._we_paused = True
    with patch("x11_mcp_voice.media_control.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        ctrl.resume()
        mock_run.assert_called_once()
        assert ctrl._we_paused is False


def test_resume_when_we_did_not_pause():
    ctrl = MediaController()
    ctrl._we_paused = False
    with patch("x11_mcp_voice.media_control.subprocess.run") as mock_run:
        ctrl.resume()
        mock_run.assert_not_called()


def test_pause_playerctl_not_found():
    ctrl = MediaController()
    with patch("x11_mcp_voice.media_control.subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError("playerctl not found")
        paused = ctrl.pause()
        assert paused is False
        assert ctrl.available is False


def test_is_playing():
    ctrl = MediaController()
    with patch("x11_mcp_voice.media_control.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="Playing\n", returncode=0)
        assert ctrl.is_playing() is True

        mock_run.return_value = MagicMock(stdout="Paused\n", returncode=0)
        assert ctrl.is_playing() is False
