"""Tests for the onboarding wizard."""
import yaml


def test_onboard_module_imports():
    """Verify onboard module can be imported."""
    from x11_mcp_voice import onboard

    assert hasattr(onboard, "main")
    assert hasattr(onboard, "_save_config")
    assert hasattr(onboard, "_save_context")
    assert hasattr(onboard, "VOICE_SAMPLES")


def test_save_config_creates_file(tmp_path):
    """_save_config should create a valid YAML config file."""
    from x11_mcp_voice.onboard import _save_config

    config_file = tmp_path / "config.yaml"
    choices = {"voice": "af_bella", "style": "walkie_talkie"}

    _save_config(choices, config_path=config_file)

    assert config_file.exists()
    with open(config_file) as f:
        data = yaml.safe_load(f)

    assert data["tts"]["voice"] == "af_bella"
    assert data["tts"]["engine"] == "kokoro"
    assert data["conversation"]["style"] == "walkie_talkie"
    assert data["wake_word"]["model"] == "hey_nox"
    assert data["wake_word"]["threshold"] == 0.5


def test_save_config_merges_existing(tmp_path):
    """_save_config should preserve existing config keys."""
    from x11_mcp_voice.onboard import _save_config

    config_file = tmp_path / "config.yaml"
    # Write pre-existing config
    with open(config_file, "w") as f:
        yaml.dump({"audio": {"sample_rate": 16000}}, f)

    _save_config({"voice": "bf_emma", "style": "auto"}, config_path=config_file)

    with open(config_file) as f:
        data = yaml.safe_load(f)

    # New keys present
    assert data["tts"]["voice"] == "bf_emma"
    # Old keys preserved
    assert data["audio"]["sample_rate"] == 16000


def test_save_context_creates_file(tmp_path):
    """_save_context should create a user-context.txt file."""
    from x11_mcp_voice.onboard import _save_context

    context_file = tmp_path / "user-context.txt"
    choices = {"name": "Alice", "voice": "af_bella", "style": "auto"}

    _save_context(choices, context_path=context_file)

    assert context_file.exists()
    content = context_file.read_text()
    assert "My name is Alice" in content
    assert "af_bella" in content
    assert "auto" in content


def test_save_context_without_name(tmp_path):
    """_save_context should work when no name is provided."""
    from x11_mcp_voice.onboard import _save_context

    context_file = tmp_path / "user-context.txt"
    choices = {"voice": "af_heart", "style": "walkie_talkie"}

    _save_context(choices, context_path=context_file)

    content = context_file.read_text()
    assert "My name is" not in content
    assert "af_heart" in content
    assert "walkie_talkie" in content
