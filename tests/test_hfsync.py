"""HF push is the ONLY durable copy on a volume-less pod, so its failure modes matter.

The safety property under test: terminate must never fire on a push we only think happened.
push() never raises (a failed upload must not kill the run), and has_checkpoint() confirms
the weights are actually in the repo, not merely uploaded from our side.
"""
import pytest

pytest.importorskip("magezero", reason="engine dependency not installed")

from draftzero import hfsync  # noqa: E402


def test_configured_reflects_env(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_REPO", raising=False)
    assert hfsync.configured() is False
    monkeypatch.setenv("HF_TOKEN", "hf_x"); monkeypatch.setenv("HF_REPO", "u/r")
    assert hfsync.configured() is True


def test_push_is_noop_without_config(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    ok, detail = hfsync.push(tmp_path, tmp_path)
    assert ok is False and "not set" in detail


def test_push_reports_nothing_to_push(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_x"); monkeypatch.setenv("HF_REPO", "u/r")
    (tmp_path / "run.json").unlink(missing_ok=True)
    ok, detail = hfsync.push(tmp_path, tmp_path / "models")
    assert ok is False and "nothing to push" in detail


def test_push_uploads_checkpoints_and_never_raises(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "2026-01-01_00-00-00"; run.mkdir(parents=True)
    models = tmp_path / "models" / "M" / "ver1"; models.mkdir(parents=True)
    (models / "gen0.pt.gz").write_bytes(b"x" * 100)
    (run / "metrics.jsonl").write_text("{}\n")

    calls = []
    class FakeApi:
        def __init__(self, token=None): pass
        def upload_file(self, path_or_fileobj, path_in_repo, **kw):
            calls.append(path_in_repo)
            if "metrics" in path_in_repo:
                raise RuntimeError("network blip")   # one file fails
    monkeypatch.setenv("HF_TOKEN", "hf_x"); monkeypatch.setenv("HF_REPO", "u/r")
    monkeypatch.setattr(hfsync, "_api", lambda: (FakeApi(), "u/r"))

    ok, detail = hfsync.push(run, models, gen=0)
    # the checkpoint went; one artifact failed -> not "all ok", but it did not raise
    assert any("gen0.pt.gz" in c for c in calls)
    assert "pushed" in detail
    assert ok is False   # a partial failure must not read as fully safe


def test_has_checkpoint_requires_a_pt_in_the_repo(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_x"); monkeypatch.setenv("HF_REPO", "u/r")

    class FakeApi:
        def __init__(self, token=None): pass
        def list_repo_files(self, **kw):
            return ["run/models/M/ver1/gen0.pt.gz", "run/metrics.jsonl"]
    monkeypatch.setattr(hfsync, "_api", lambda: (FakeApi(), "u/r"))
    ok, detail = hfsync.has_checkpoint("run")
    assert ok is True and "1 checkpoints" in detail

    class EmptyApi(FakeApi):
        def list_repo_files(self, **kw): return ["run/metrics.jsonl"]
    monkeypatch.setattr(hfsync, "_api", lambda: (EmptyApi(), "u/r"))
    ok, detail = hfsync.has_checkpoint("run")
    assert ok is False   # artifacts present but NO checkpoint -> not safe to terminate


def test_push_logs_uploads_one_archive_with_every_log(tmp_path, monkeypatch):
    import tarfile
    monkeypatch.setenv("HF_TOKEN", "hf_x"); monkeypatch.setenv("HF_REPO", "u/r")
    run = tmp_path / "runs" / "r1"; run.mkdir(parents=True)
    (run / "jvm_gen1_00_self.log").write_text("game log")
    (run / "games.jsonl").write_text("{}")          # not a log: pushed elsewhere
    logs = tmp_path / "logs"; logs.mkdir()
    (logs / "loop.log").write_text("loop")
    seen = {}

    class FakeApi:
        def upload_file(self, path_or_fileobj, path_in_repo, **kw):
            with tarfile.open(path_or_fileobj) as t:
                seen["names"] = sorted(t.getnames())
            seen["remote"] = path_in_repo
    monkeypatch.setattr(hfsync, "_api", lambda: (FakeApi(), "u/r"))

    ok, detail = hfsync.push_logs(run, extra=(logs,))
    assert ok and seen["remote"] == "r1/logs.tar.gz"
    assert seen["names"] == ["r1/jvm_gen1_00_self.log", "r1/logs/loop.log"]
