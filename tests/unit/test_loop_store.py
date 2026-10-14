"""The loop's store (draftzero.loop.store): a local folder, and a Hugging Face Storage Bucket through a fake HfApi
(no network, no real bucket)."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from draftzero.loop import store as S
from draftzero.loop.store import HFBucketStore, LocalStore, open_store


def test_local_store_round_trips_and_lists(tmp_path):
    s = open_store(tmp_path / "st")
    assert isinstance(s, LocalStore) and s.read_json("control.json") is None and not s.exists("control.json")
    s.write_json("control.json", {"version": 3})
    src = tmp_path / "w.bin"
    src.write_bytes(b"abc")
    s.put_many({"versions/player/v0003/model.pt": src, "shards/a/000001/games.jsonl": b"data",
                "shards/ab/000001/games.jsonl": b"x"})
    assert s.read_json("control.json") == {"version": 3} and s.exists("control.json")
    assert s.list("shards/a/") == ["shards/a/000001/games.jsonl"]
    assert s.list("shards/a") == ["shards/a/000001/games.jsonl"]            # a folder, not a name prefix
    assert s.list("shards/") == ["shards/a/000001/games.jsonl", "shards/ab/000001/games.jsonl"]
    assert s.list("control.json") == ["control.json"] and s.list("nothing/") == []
    assert s.get("versions/player/v0003/model.pt", tmp_path / "out" / "w").read_bytes() == b"abc"
    s.delete(["versions/player/v0003/model.pt"])
    assert s.list("versions/") == [] and s.get("versions/player/v0003/model.pt", tmp_path / "o2") is None
    with pytest.raises(ValueError):
        s.put("../escape", b"x")
    s.ensure()


def test_local_store_list_skips_dot_files(tmp_path):
    """rsync stages a pulled file as .<name>.XXXXXX next to it, and the relay stages in .relay-tmp/: neither is a
    store file (a reader would take a half-written meta.json for a complete shard)."""
    s = LocalStore(tmp_path / "st")
    s.put_many({"shards/a/000001/games.jsonl": b"g", "heartbeats/a.json": b"{}"})
    for p in ("shards/a/000001/.meta.json.Ab12Cd", "heartbeats/.a.json.x", ".relay-tmp/shards/a/000002/meta.json",
              "shards/.hidden/000001/meta.json", "shards/a/000001/.tmp-123"):
        (tmp_path / "st" / p).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "st" / p).write_bytes(b"partial")
    assert s.list("shards/") == ["shards/a/000001/games.jsonl"]
    assert s.list("") == ["heartbeats/a.json", "shards/a/000001/games.jsonl"]


# ------------------------------------------------------------------------------------------------ a fake bucket

@dataclasses.dataclass
class BucketFile:
    path: str
    size: int
    type: str = "file"


@dataclasses.dataclass
class BucketFolder:
    path: str
    type: str = "directory"


class EntryNotFoundError(Exception):
    pass


class FakeApi:
    """The HfApi bucket methods the store uses, over a dict; records every call."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.calls: list[tuple] = []
        self.created = None
        self.fail_next = 0

    def _maybe_fail(self):
        if self.fail_next:
            self.fail_next -= 1
            raise ConnectionError("flaky network")

    def batch_bucket_files(self, bucket_id, *, add=None, copy=None, delete=None, token=None):
        self.calls.append(("batch", bucket_id, [d for _, d in add or []], list(delete or [])))
        self._maybe_fail()
        for src, dest in add or []:
            self.files[dest] = bytes(src) if isinstance(src, (bytes, bytearray)) else Path(src).read_bytes()
        for p in delete or []:
            self.files.pop(p, None)

    def list_bucket_tree(self, bucket_id, prefix=None, *, recursive=None, token=None):
        self.calls.append(("list", bucket_id, prefix, recursive))
        assert recursive is True
        folders = {p.rsplit("/", 1)[0] for p in self.files if "/" in p}
        out = [BucketFile(p, len(b)) for p, b in self.files.items() if p.startswith(prefix or "")]
        return out + [BucketFolder(f) for f in folders if f.startswith(prefix or "")]

    def download_bucket_files(self, bucket_id, files, *, raise_on_missing_files=False, token=None):
        self.calls.append(("download", bucket_id, [s for s, _ in files]))
        missing = [s for s, _ in files if s not in self.files]
        if missing and raise_on_missing_files:
            raise EntryNotFoundError(", ".join(missing))
        for src, dest in files:
            if src in self.files:
                Path(dest).write_bytes(self.files[src])

    def get_bucket_paths_info(self, bucket_id, paths, *, token=None):
        self.calls.append(("info", bucket_id, list(paths)))
        return [BucketFile(p, len(self.files[p])) for p in paths if p in self.files]

    def create_bucket(self, bucket_id, *, private=None, exist_ok=False, token=None, **kw):
        self.calls.append(("create", bucket_id, private, exist_ok))
        self.created = (bucket_id, private, exist_ok)


def test_bucket_store_put_get_list_delete(tmp_path):
    api = FakeApi()
    s = HFBucketStore("dan/dz-smoke", api=api)
    s.ensure()
    assert api.created == ("dan/dz-smoke", True, True)
    src = tmp_path / "corpus.jsonl.gz"
    src.write_bytes(b"corpus")
    # a shard, its heartbeat and metrics: ONE batch call, in the dict's order (meta.json last)
    s.put_many({"shards/a/000001/corpus.jsonl.gz": src, "heartbeats/a.json": b"{}",
                "shards/a/000001/meta.json": b'{"games": 2}'})
    batches = [c for c in api.calls if c[0] == "batch"]
    assert len(batches) == 1 and batches[0][2][-1] == "shards/a/000001/meta.json"
    assert s.read_json("shards/a/000001/meta.json") == {"games": 2}
    assert s.get("shards/a/000001/corpus.jsonl.gz", tmp_path / "x" / "c.gz").read_bytes() == b"corpus"
    assert s.get("shards/a/000001/nope", tmp_path / "x" / "n") is None and s.read_bytes("nope.json") is None
    assert not list((tmp_path / "x").glob(".tmp-*"))                     # a failed get leaves nothing behind
    s.put("shards/ab/000001/meta.json", b"{}")
    assert s.list("shards/a/") == ["shards/a/000001/corpus.jsonl.gz", "shards/a/000001/meta.json"]
    assert s.list("shards/a") == s.list("shards/a/")                     # folders are filtered out, names exact
    assert s.exists("heartbeats/a.json") and not s.exists("heartbeats/b.json")
    s.delete(["heartbeats/a.json"])
    assert not s.exists("heartbeats/a.json") and s.list("heartbeats/") == []
    s.delete([])                                                         # nothing to do: no call
    assert [c for c in api.calls if c[0] == "batch"][-1][3] == ["heartbeats/a.json"]
    assert "dan/dz-smoke" in s.describe()


def test_bucket_store_retries_transient_errors(tmp_path):
    api = FakeApi()
    s = HFBucketStore("dan/b", api=api, backoff_s=0)
    api.fail_next = 2
    s.put("a.json", b"1")
    assert api.files["a.json"] == b"1"
    api.fail_next = 5
    with pytest.raises(ConnectionError):
        s.put("b.json", b"2")


def test_only_network_trouble_429_and_5xx_are_retried():
    """HfHubHTTPError subclasses OSError (through httpx2.HTTPError): its status must decide, so a 401/403/404/400/422
    fails at once instead of four times with 12 s of pauses."""
    httpx2 = pytest.importorskip("httpx2")
    from huggingface_hub.errors import HfHubHTTPError

    def http(code):
        request = httpx2.Request("POST", "https://x")
        return HfHubHTTPError(f"HTTP {code}", response=httpx2.Response(code, request=request))
    for code in (400, 401, 403, 404, 422):
        assert not S._transient(http(code)), code
    for code in (408, 429, 500, 502, 503, 504):
        assert S._transient(http(code)), code
    assert S._transient(httpx2.ConnectError("refused")) and S._transient(httpx2.ReadTimeout("slow"))
    assert S._transient(ConnectionError("reset")) and S._transient(TimeoutError()) and S._transient(OSError("net"))
    assert not S._transient(FileNotFoundError("x")) and not S._transient(ValueError("bad path"))

    class Api(FakeApi):
        def batch_bucket_files(self, bucket_id, **kw):
            self.calls.append(("batch",))
            raise http(403)
    api = Api()
    with pytest.raises(HfHubHTTPError):
        HFBucketStore("dan/b", api=api, backoff_s=0).put("a.json", b"1")
    assert len(api.calls) == 1


def test_open_store_specs(monkeypatch, tmp_path):
    made = {}

    class Api:
        def __init__(self, token=None):
            made["token"] = token
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setenv("DZ_STORE_TOKEN", "hf_bucket_only")
    s = open_store("hf://buckets/danbrooks/dz-e1")
    assert isinstance(s, HFBucketStore) and s.bucket == "danbrooks/dz-e1" and made["token"] == "hf_bucket_only"
    with pytest.raises(ValueError, match="Storage Bucket"):
        open_store("hf://danbrooks/draftzero-selfplay-games")
    with pytest.raises(ValueError, match="relay"):
        open_store("ssh://r1/runs/x")
    with pytest.raises(ValueError, match="owner/name"):
        HFBucketStore("just-a-name", api=FakeApi())
    assert isinstance(open_store(tmp_path / "x"), LocalStore)
    assert S.dumps({"b": 1, "a": 2}).decode().index('"a"') < S.dumps({"b": 1, "a": 2}).decode().index('"b"')
