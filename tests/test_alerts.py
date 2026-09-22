"""Alerting must never break a run, and push must never be taken down by e-mail config."""
import json

import pytest

pytest.importorskip("magezero", reason="engine dependency not installed")

from draftzero import alerts  # noqa: E402


def test_writes_a_record_even_with_no_transport(tmp_path, monkeypatch):
    monkeypatch.delenv("DZ_ALERT_TOPIC", raising=False)
    assert alerts.send(alerts.CRASH, "trainer died", "traceback here", tmp_path) is False
    rec = json.loads((tmp_path / "alerts.jsonl").read_text().splitlines()[0])
    assert rec["event"] == "crash"
    assert rec["summary"] == "trainer died"


def test_never_raises_on_an_unwritable_run_dir(monkeypatch):
    monkeypatch.delenv("DZ_ALERT_TOPIC", raising=False)
    # an alert failing must not be able to kill the run it is reporting on
    assert alerts.send(alerts.CRASH, "x", "y", "/nonexistent/path/xyz") is False


def test_email_rejection_falls_back_to_push(monkeypatch):
    """ntfy answers 400 for anonymous e-mail. That must not take push down with it."""
    import urllib.error
    calls = []

    class FakeResp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=0):
        calls.append(dict(req.headers))
        if any(k.lower() == "email" for k in req.headers):
            raise urllib.error.HTTPError(req.full_url, 400, "anonymous email", {}, None)
        return FakeResp()

    monkeypatch.setenv("DZ_ALERT_TOPIC", "t")
    monkeypatch.setenv("DZ_ALERT_EMAIL", "someone@example.com")
    monkeypatch.setattr(alerts.urllib.request, "urlopen", fake_urlopen)

    assert alerts.send(alerts.CRASH, "boom") is True
    assert len(calls) == 2, "should try with e-mail, then without"
    assert any(k.lower() == "email" for k in calls[0])
    assert not any(k.lower() == "email" for k in calls[1])


def test_every_event_has_a_priority_and_tag():
    for ev in (alerts.CRASH, alerts.RESTART, alerts.RESTART_FAILED, alerts.STALLED,
               alerts.BUDGET, alerts.TERMINATING, alerts.SYNC_FAILED, alerts.DONE):
        assert ev in alerts.PRIORITY and ev in alerts.TAG
