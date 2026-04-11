"""Tests for _clean_for_speech() text cleaning."""
from x11_mcp_voice.agent import _clean_for_speech


def test_strip_bold():
    assert _clean_for_speech("**bold text**") == "bold text"
    assert _clean_for_speech("*italic*") == "italic"
    assert _clean_for_speech("***both***") == "both"


def test_strip_markdown_links():
    assert _clean_for_speech("[click here](https://example.com)") == "click here"


def test_strip_standalone_urls():
    assert _clean_for_speech("Visit https://example.com/path for info") == "Visit for info"
    assert _clean_for_speech("See http://x.com") == "See"


def test_strip_code_blocks():
    assert "print" not in _clean_for_speech("```python\nprint('hi')\n```")


def test_strip_backticks():
    assert _clean_for_speech("Use `pip install`") == "Use pip install"


def test_convert_subreddit():
    result = _clean_for_speech("Check r/rav4prime for info")
    assert "the rav4prime subreddit" in result
    assert "r/" not in result


def test_strip_markdown_headers():
    assert _clean_for_speech("# Title\nBody") == "Title\nBody"
    assert _clean_for_speech("### Heading") == "Heading"


def test_strip_list_bullets():
    assert _clean_for_speech("- item one\n* item two") == "item one\nitem two"


def test_strip_emoji():
    result = _clean_for_speech("Hello \U0001f389 World \U0001f680")
    assert "\U0001f389" not in result
    assert "\U0001f680" not in result
    assert "Hello" in result


def test_collapse_spaces():
    assert _clean_for_speech("too    many   spaces") == "too many spaces"


def test_collapse_blank_lines():
    result = _clean_for_speech("line one\n\n\n\n\nline two")
    assert "\n\n\n" not in result


def test_passthrough_normal_text():
    text = "I opened Firefox and navigated to YouTube."
    assert _clean_for_speech(text) == text


def test_combined_cleanup():
    text = "**Check** [this link](https://x.com) and visit r/linux for `tips`"
    result = _clean_for_speech(text)
    assert "**" not in result
    assert "https" not in result
    assert "`" not in result
    assert "the linux subreddit" in result
