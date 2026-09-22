"""Credential files must be parsed, never executed.

Sourcing one that holds a bare value makes the shell try to run it, and the secret lands in
the error output. That happened to a Hugging Face token in this project; it had to be
revoked. These tests pin the parser that replaced it.
"""
import pytest

pytest.importorskip("magezero", reason="engine dependency not installed")

from draftzero.secrets import load, redact  # noqa: E402


def test_parses_export_style(tmp_path):
    f = tmp_path / "e"; f.write_text("export HF_TOKEN=hf_abc\nexport HF_REPO=a/b\n"); f.chmod(0o600)
    assert load(f, export=False) == {"HF_TOKEN": "hf_abc", "HF_REPO": "a/b"}


def test_parses_plain_key_value(tmp_path):
    f = tmp_path / "e"; f.write_text("HF_TOKEN=hf_abc\n"); f.chmod(0o600)
    assert load(f, export=False) == {"HF_TOKEN": "hf_abc"}


def test_parses_a_bare_value(tmp_path):
    """The shape that caused the leak: a pasted token with no key."""
    f = tmp_path / "e"; f.write_text("hf_abc123\n"); f.chmod(0o600)
    assert load(f, default_key="HF_TOKEN", export=False) == {"HF_TOKEN": "hf_abc123"}


def test_ignores_comments_and_blanks(tmp_path):
    f = tmp_path / "e"; f.write_text("# a note\n\nHF_TOKEN=x\n\n"); f.chmod(0o600)
    assert load(f, export=False) == {"HF_TOKEN": "x"}


def test_strips_quotes(tmp_path):
    f = tmp_path / "e"; f.write_text('HF_TOKEN="hf_q"\n'); f.chmod(0o600)
    assert load(f, export=False)["HF_TOKEN"] == "hf_q"


def test_missing_file_is_empty_not_an_error(tmp_path):
    assert load(tmp_path / "nope", export=False) == {}


def test_redact_never_returns_the_whole_value():
    assert redact("hf_supersecretvalue") == "hf_s...<19 chars>"
    assert "supersecret" not in redact("hf_supersecretvalue")
    assert redact("") == "<empty>"
