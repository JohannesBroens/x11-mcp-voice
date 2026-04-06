from x11_mcp_voice.config import Config, load_config


def test_default_config():
    cfg = Config()
    assert cfg.wake_word.model == "hey_jarvis"
    assert cfg.wake_word.threshold == 0.7
    assert cfg.stt.model == "nvidia/parakeet-tdt-0.6b-v2"
    assert cfg.stt.device == "cuda"
    assert cfg.tts.voice == "en_US-ryan-medium"
    assert cfg.tts.speed == 1.0
    assert cfg.media.auto_pause is True
    assert cfg.media.player is None
    assert cfg.agent.model == "claude-sonnet-4-6"
    assert cfg.audio.sample_rate == 16000
    assert cfg.audio.silence_threshold_ms == 500
    assert cfg.audio.max_recording_s == 30
    assert cfg.conversation.style == "auto"
    assert cfg.conversation.followup_timeout_s == 3.0


def test_load_config_from_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "wake_word:\n"
        "  model: custom_wake\n"
        "  threshold: 0.9\n"
        "conversation:\n"
        "  style: walkie_talkie\n"
    )
    cfg = load_config(str(config_file))
    assert cfg.wake_word.model == "custom_wake"
    assert cfg.wake_word.threshold == 0.9
    assert cfg.conversation.style == "walkie_talkie"
    # Unspecified fields keep defaults
    assert cfg.stt.model == "nvidia/parakeet-tdt-0.6b-v2"
    assert cfg.tts.voice == "en_US-ryan-medium"


def test_load_config_no_file():
    cfg = load_config(None)
    assert cfg.wake_word.model == "hey_jarvis"


def test_load_config_partial_section(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("audio:\n  sample_rate: 44100\n")
    cfg = load_config(str(config_file))
    assert cfg.audio.sample_rate == 44100
    # Other audio fields keep defaults
    assert cfg.audio.silence_threshold_ms == 500
    assert cfg.audio.max_recording_s == 30
