"""hf:// model URIs (configs' player.init, `dz models pull`, published runs): how draftzero.hub reads them. The
Hugging Face download functions are faked; nothing touches the network."""
from __future__ import annotations

from pathlib import Path

import pytest

from draftzero import hub


@pytest.mark.parametrize("uri, want", [
    ("hf://danbrooks/draftzero-checkpoints/exp4/stage3/best_policy.pt.gz",
     hub.HubRef("danbrooks/draftzero-checkpoints", "exp4/stage3/best_policy.pt.gz")),
    ("hf://danbrooks/draftzero-e1@v0042/model.dzgw", hub.HubRef("danbrooks/draftzero-e1", "model.dzgw", "v0042")),
    ("hf://danbrooks/draftzero-e1@refs%2Fpr%2F3/a/b.pt", hub.HubRef("danbrooks/draftzero-e1", "a/b.pt", "refs/pr/3")),
    ("hf://danbrooks/draftzero-e1@v0042", hub.HubRef("danbrooks/draftzero-e1", None, "v0042")),
    ("hf://danbrooks/draftzero-e1", hub.HubRef("danbrooks/draftzero-e1", None)),
    ("hf://datasets/danbrooks/draftzero-games/g.jsonl",
     hub.HubRef("danbrooks/draftzero-games", "g.jsonl", None, "dataset")),
])
def test_parse(uri, want):
    assert hub.parse(uri) == want


@pytest.mark.parametrize("uri", ["hf://danbrooks", "hf:///repo/x", "hf://buckets/danbrooks/dz-e1",
                                 "danbrooks/repo/x", "hf://o/@v1/x"])
def test_parse_rejects(uri):
    with pytest.raises(ValueError):
        hub.parse(uri)


def test_local_paths_pass_through(tmp_path):
    assert hub.resolve(tmp_path / "m.pt") == tmp_path / "m.pt"
    assert hub.resolve("runs/x/model.pt") == Path("runs/x/model.pt")
    assert not hub.is_hub("runs/x") and hub.is_hub("hf://a/b/c")


def test_resolve_downloads_through_the_hub(monkeypatch, tmp_path):
    import huggingface_hub
    calls = []

    def fake_file(repo_id, filename, **kw):
        calls.append(("file", repo_id, filename, kw))
        return str(tmp_path / filename)

    def fake_snapshot(repo_id, **kw):
        calls.append(("snapshot", repo_id, kw))
        return str(tmp_path / "snap")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_file)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    assert hub.resolve("hf://danbrooks/dz-e1@v0003/player/model.pt") == tmp_path / "player/model.pt"
    assert hub.resolve("hf://datasets/danbrooks/games@main") == tmp_path / "snap"
    assert calls[0] == ("file", "danbrooks/dz-e1", "player/model.pt",
                        {"repo_type": "model", "revision": "v0003", "cache_dir": None, "token": None})
    assert calls[1] == ("snapshot", "danbrooks/games",
                        {"repo_type": "dataset", "revision": "main", "cache_dir": None, "token": None})
