import json
from datetime import datetime, timedelta

from x11_mcp_voice import transcript


def test_save_and_load(tmp_path, monkeypatch):
    monkeypatch.setattr(transcript, "TRANSCRIPT_DIR", tmp_path)
    monkeypatch.setattr(transcript, "TRANSCRIPT_FILE", tmp_path / "transcript.jsonl")

    transcript.save_message("user", "Hello")
    transcript.save_message("assistant", "Hi there")
    transcript.save_message("user", "What time is it?")

    messages = transcript.load_recent()
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[0]["text"] == "Hello"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["text"] == "Hi there"
    assert messages[2]["role"] == "user"
    assert messages[2]["text"] == "What time is it?"
    # Verify time and date fields exist
    assert "time" in messages[0]
    assert "date" in messages[0]


def test_load_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(transcript, "TRANSCRIPT_FILE", tmp_path / "nonexistent.jsonl")

    messages = transcript.load_recent()
    assert messages == []


def test_rotate(tmp_path, monkeypatch):
    monkeypatch.setattr(transcript, "TRANSCRIPT_DIR", tmp_path)
    monkeypatch.setattr(transcript, "TRANSCRIPT_FILE", tmp_path / "transcript.jsonl")

    today = datetime.now().strftime("%Y-%m-%d")
    old_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    # Write messages with explicit dates
    tf = tmp_path / "transcript.jsonl"
    with open(tf, "w") as f:
        f.write(json.dumps({"role": "user", "text": "old msg", "time": "10:00:00", "date": old_date}) + "\n")
        f.write(json.dumps({"role": "assistant", "text": "old reply", "time": "10:00:01", "date": old_date}) + "\n")
        f.write(json.dumps({"role": "user", "text": "recent msg", "time": "12:00:00", "date": today}) + "\n")

    transcript.rotate(keep_days=7)

    messages = transcript.load_recent()
    assert len(messages) == 1
    assert messages[0]["text"] == "recent msg"


def test_max_messages(tmp_path, monkeypatch):
    monkeypatch.setattr(transcript, "TRANSCRIPT_DIR", tmp_path)
    monkeypatch.setattr(transcript, "TRANSCRIPT_FILE", tmp_path / "transcript.jsonl")

    for i in range(100):
        transcript.save_message("user", f"message {i}")

    messages = transcript.load_recent(max_messages=10)
    assert len(messages) == 10
    assert messages[0]["text"] == "message 90"
    assert messages[-1]["text"] == "message 99"
