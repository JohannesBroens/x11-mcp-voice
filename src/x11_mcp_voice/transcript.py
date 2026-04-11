"""Transcript persistence — saves conversation to disk as JSONL."""
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

TRANSCRIPT_DIR = Path.home() / ".local" / "share" / "nox"
TRANSCRIPT_FILE = TRANSCRIPT_DIR / "transcript.jsonl"


def save_message(role: str, text: str) -> None:
    """Append a message to the transcript file."""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "role": role,
        "text": text,
        "time": datetime.now().strftime("%H:%M:%S"),
        "date": datetime.now().strftime("%Y-%m-%d"),
    }
    with open(TRANSCRIPT_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_recent(max_messages: int = 50) -> list[dict]:
    """Load the most recent messages from the transcript file."""
    if not TRANSCRIPT_FILE.exists():
        return []
    messages = []
    with open(TRANSCRIPT_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return messages[-max_messages:]


def rotate(keep_days: int = 7) -> None:
    """Remove transcript entries older than keep_days."""
    if not TRANSCRIPT_FILE.exists():
        return
    cutoff = (datetime.now() - timedelta(days=keep_days)).strftime("%Y-%m-%d")
    kept = []
    with open(TRANSCRIPT_FILE) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get("date", "9999") >= cutoff:
                    kept.append(line)
            except json.JSONDecodeError:
                continue
    with open(TRANSCRIPT_FILE, "w") as f:
        f.writelines(kept)
