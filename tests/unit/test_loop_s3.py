"""draftzero.loop.s3: Signature Version 4 against AWS's published examples, the runpodvol:// spec, the mount lookup,
and S3Store against a fake S3 server that answers the way RunPod's does (folder keys, dot-files, 502s, pages).
No network."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import urllib.parse
from pathlib import Path

import pytest

from draftzero.loop import s3
from draftzero.loop.store import LocalStore, open_store

AK, SK = "AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
WHEN = dt.datetime(2013, 5, 24, tzinfo=dt.timezone.utc)
HOST = "https://examplebucket.s3.amazonaws.com"


def _sig(h: dict) -> str:
    return h["Authorization"].rsplit("Signature=", 1)[1]


# ------------------------------------------------------------------------------------------------ signing
# AWS, "Signature Calculations for the Authorization Header: Transferring Payload in a Single Chunk", examples.

def test_sigv4_get_object_example():
    h = s3.sigv4_headers("GET", f"{HOST}/test.txt", {"Range": "bytes=0-9"}, s3.EMPTY_SHA256, region="us-east-1",
                         access_key=AK, secret_key=SK, now=WHEN)
    assert "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date" in h["Authorization"]
    assert "Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request" in h["Authorization"]
    assert _sig(h) == "f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"
    assert h["x-amz-date"] == "20130524T000000Z"


def test_sigv4_put_object_example():
    body = b"Welcome to Amazon S3."
    sha = hashlib.sha256(body).hexdigest()
    assert sha == "44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072"
    h = s3.sigv4_headers("PUT", f"{HOST}/test%24file.text",
                         {"Date": "Fri, 24 May 2013 00:00:00 GMT", "x-amz-storage-class": "REDUCED_REDUNDANCY"},
                         sha, region="us-east-1", access_key=AK, secret_key=SK, now=WHEN)
    assert "SignedHeaders=date;host;x-amz-content-sha256;x-amz-date;x-amz-storage-class" in h["Authorization"]
    assert _sig(h) == "98ad721746da40c64f1a55b78f14c238d841ea1380cd77a1b5971af0ece108bd"


def test_sigv4_query_examples():
    h = s3.sigv4_headers("GET", f"{HOST}/?lifecycle", {}, s3.EMPTY_SHA256, region="us-east-1", access_key=AK,
                         secret_key=SK, now=WHEN)
    assert _sig(h) == "fea454ca298b7da1c68078a5d1bdbfbbe0d65c699e0f91ac7a200a0136783543"
    h = s3.sigv4_headers("GET", f"{HOST}/?max-keys=2&prefix=J", {}, s3.EMPTY_SHA256, region="us-east-1",
                         access_key=AK, secret_key=SK, now=WHEN)
    assert _sig(h) == "34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7"


# ------------------------------------------------------------------------------------------------ specs and mounts

def test_parse_spec_and_endpoint():
    assert s3.parse_spec("runpodvol://5njr2ebemc@EUR-IS-1/runs/e1/store/") == ("5njr2ebemc", "EUR-IS-1", "runs/e1/store")
    for bad in ("runpodvol://vol/runs/x", "runpodvol://vol@DC", "runpodvol://@DC/x", "hf://buckets/a/b"):
        with pytest.raises(ValueError):
            s3.parse_spec(bad)
    assert s3.endpoint_for("EUR-IS-1") == "https://s3api-eur-is-1.runpod.io"


MOUNTINFO = ("22 1 0:20 / / rw,relatime - overlay overlay rw\n"
             "631 22 0:56 /networkvolumes/5njr2ebemc /workspace rw,nosuid,nodev,relatime - fuse "
             "mfs#eur-is-1-2.runpod.net:9421 rw,user_id=0,group_id=0,allow_other\n")


def test_a_pod_opens_its_mount_and_any_other_machine_s3(tmp_path, monkeypatch):
    mi = tmp_path / "mountinfo"
    mi.write_text(MOUNTINFO)
    assert s3.volume_mount("5njr2ebemc", str(mi)) == Path("/workspace")
    assert s3.volume_mount("othervol00", str(mi)) is None and s3.volume_mount("x", str(tmp_path / "none")) is None
    mi.write_text(MOUNTINFO.replace("/workspace", str(tmp_path / "ws")))
    st = s3.open_volume_store("runpodvol://5njr2ebemc@EUR-IS-1/runs/x/store", mountinfo=str(mi))
    assert isinstance(st, LocalStore) and st.root == (tmp_path / "ws" / "runs/x/store").resolve()
    monkeypatch.setenv("DZ_S3_ACCESS_KEY_ID", "user_abc")
    monkeypatch.setenv("DZ_S3_SECRET_ACCESS_KEY", "rps_" + "s" * 40)
    remote = s3.open_volume_store("runpodvol://othervol00@EUR-IS-1/runs/x/store", mountinfo=str(mi),
                                  http=lambda *a: (200, {}, b""))
    assert isinstance(remote, s3.S3Store) and remote.remote and remote.prefix == "runs/x/store"
    assert remote.endpoint == "https://s3api-eur-is-1.runpod.io" and remote.region == "EUR-IS-1"
    assert "rps_" not in repr(remote)
    assert isinstance(open_store("runpodvol://othervol00@EUR-IS-1/runs/y/store"), s3.S3Store)


def test_credentials_come_from_the_file_unexported(tmp_path, monkeypatch):
    for k in ("DZ_S3_ACCESS_KEY_ID", "DZ_S3_SECRET_ACCESS_KEY", "RUNPOD_S3_ACCESS_KEY_ID"):
        monkeypatch.delenv(k, raising=False)
    f = tmp_path / "s3_env"
    f.write_text("RUNPOD_S3_ACCESS_KEY_ID=user_123\nexport RUNPOD_S3_SECRET_ACCESS_KEY=rps_secret\n")
    f.chmod(0o600)
    assert s3.credentials(str(f)) == ("user_123", "rps_secret")
    import os
    assert "RUNPOD_S3_ACCESS_KEY_ID" not in os.environ
    with pytest.raises(s3.S3Error, match="S3 API Keys"):
        s3.credentials(str(tmp_path / "missing"))

    def unreadable(*a, **k):                          # /root/.runpod/s3_env for a user who isn't root (CI)
        raise PermissionError(13, "Permission denied")
    monkeypatch.setattr(s3.secrets, "load", unreadable)
    monkeypatch.setenv("DZ_S3_ACCESS_KEY_ID", "user_env")
    monkeypatch.setenv("DZ_S3_SECRET_ACCESS_KEY", "rps_env")
    assert s3.credentials() == ("user_env", "rps_env")


# ------------------------------------------------------------------------------------------------ a fake S3

class FakeS3:
    """Path-style S3 over a dict, as RunPod's answers it: folder keys in listings (empty directories of files written
    through the mount), dot-files, scripted failures, small pages."""

    def __init__(self, bucket="vol1", page=3):
        self.bucket, self.page, self.objects, self.calls, self.fail = bucket, page, {}, [], []

    def __call__(self, method, url, body, headers):
        self.calls.append((method, url, headers))
        assert headers["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=")
        assert headers["x-amz-content-sha256"] == hashlib.sha256(body or b"").hexdigest()
        if self.fail:
            f = self.fail.pop(0)
            if isinstance(f, Exception):
                raise f
            return f, {}, b"<Error><Code>BadGateway</Code></Error>"
        u = urllib.parse.urlsplit(url)
        parts = u.path.lstrip("/").split("/", 1)
        if parts[0] != self.bucket:
            return 404, {}, b"<Error><Code>NoSuchBucket</Code></Error>"
        key = urllib.parse.unquote(parts[1]) if len(parts) > 1 else ""
        if method == "PUT":
            self.objects[key] = body
            return 200, {}, b""
        if method == "DELETE":
            self.objects.pop(key, None)
            return 204, {}, b""
        if method == "HEAD":
            return (200 if (not key or key in self.objects) else 404), {}, b""
        if key:
            return (200, {}, self.objects[key]) if key in self.objects else (404, {}, b"<Error><Code>NoSuchKey</Code></Error>")
        q = dict(urllib.parse.parse_qsl(u.query))
        keys = sorted(k for k in self.objects if k.startswith(q.get("prefix", "")))
        start = int(q.get("continuation-token", "0"))
        chunk = keys[start:start + self.page]
        more = start + self.page < len(keys)
        xml = "".join(f"<Contents><Key>{k}</Key><Size>1</Size></Contents>" for k in chunk)
        tail = f"<IsTruncated>true</IsTruncated><NextContinuationToken>{start + self.page}</NextContinuationToken>" \
            if more else "<IsTruncated>false</IsTruncated>"
        return 200, {}, (f'<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                         f"<Name>{self.bucket}</Name>{xml}{tail}</ListBucketResult>").encode()


def _store(fake, **kw):
    return s3.S3Store("vol1", "runs/x/store", region="EUR-IS-1", access_key="user_abc", secret_key="rps_" + "k" * 40,
                      http=fake, sleep=lambda s: None, **kw)


def test_s3_store_round_trips_lists_and_deletes(tmp_path):
    fake = FakeS3()
    st = _store(fake)
    st.ensure()
    src = tmp_path / "w.bin"
    src.write_bytes(b"abc")
    st.put_many({"shards/a/000001/games.jsonl": src, "shards/a/000001/meta.json": b"{}",
                 "shards/ab/000001/meta.json": b"{}"})
    st.write_json("control.json", {"version": 3})
    assert fake.objects["runs/x/store/control.json"] and st.read_json("control.json") == {"version": 3}
    # what a pod's writes look like through RunPod's S3: folder keys, temp files, its own upload area
    for k in ("runs/x/store/shards/", "runs/x/store/shards/a/", "runs/x/store/shards/a/000002/.tmp-abc123",
              "runs/x/store/.s3compat_uploads/x", "runs/x/other/control.json", "runs/x/storeX/control.json"):
        fake.objects[k] = b""
    assert st.list("shards/a/") == ["shards/a/000001/games.jsonl", "shards/a/000001/meta.json"]
    assert st.list("shards/a") == ["shards/a/000001/games.jsonl", "shards/a/000001/meta.json"]
    assert st.list("") == ["control.json", "shards/a/000001/games.jsonl", "shards/a/000001/meta.json",
                           "shards/ab/000001/meta.json"]                       # 3 pages of 3 keys
    assert st.exists("control.json") and not st.exists("nothing.json") and st.read_bytes("nothing.json") is None
    assert st.get("shards/a/000001/games.jsonl", tmp_path / "o" / "g").read_bytes() == b"abc"
    assert st.get("nothing", tmp_path / "o" / "n") is None and not (tmp_path / "o" / "n").exists()
    st.delete(["shards/a/000001/games.jsonl", "never/was.json"])
    assert "runs/x/store/shards/a/000001/games.jsonl" not in fake.objects
    with pytest.raises(ValueError):
        st.put("../escape", b"x")


def test_s3_store_retries_only_transient_errors_and_keeps_the_key_secret():
    fake = FakeS3()
    st = _store(fake)
    fake.fail = [502, ConnectionResetError("reset"), 503]
    st.write_json("control.json", {"a": 1})                       # three failures, then it goes through
    assert st.read_json("control.json") == {"a": 1}
    fake.fail = [403]
    with pytest.raises(s3.S3Error, match="HTTP 403") as e:
        st.read_bytes("control.json")
    assert "rps_" + "k" * 40 not in str(e.value)
    fake.fail = [500] * 10
    with pytest.raises(s3.S3Error, match="HTTP 500"):
        st.read_bytes("control.json")
    assert sum(1 for c in fake.calls if c[0] == "GET") >= 5
    fake.fail = []
    with pytest.raises(s3.S3Error, match="no network volume"):
        s3.S3Store("vol9", "x", region="EUR-IS-1", access_key="a", secret_key="b", http=fake).ensure()


def test_a_loop_runs_split_between_a_pod_mount_and_s3(tmp_path):
    """The volume seen two ways: the controller and the learner through S3 (d1), an actor through its mount (a pod).
    Here the "mount" is the fake server's own folder view, so both sides see the same files."""
    from test_loop import FakeGorge, Run, make_cfg      # noqa: F401 - the in-process loop harness (tests/unit)

    class FolderS3(FakeS3):
        """A fake S3 whose objects live in a folder: what the pods write through their mount, d1 reads through S3."""

        def __init__(self, root: Path):
            super().__init__(page=1000)
            self.root = root

        def __call__(self, method, url, body, headers):
            self.objects = {p.relative_to(self.root).as_posix(): p.read_bytes()
                            for p in self.root.rglob("*") if p.is_file()}
            before = dict(self.objects)
            out = super().__call__(method, url, body, headers)
            for k, v in self.objects.items():
                if before.get(k) != v:
                    (self.root / k).parent.mkdir(parents=True, exist_ok=True)
                    (self.root / k).write_bytes(v)
            for k in set(before) - set(self.objects):
                (self.root / k).unlink()
            return out

    vol = tmp_path / "volume"
    vol.mkdir()
    s3_side = s3.S3Store("vol1", "runs/x/store", region="EUR-IS-1", access_key="a", secret_key="b",
                         http=FolderS3(vol), sleep=lambda s: None)
    pod_side = LocalStore(vol / "runs/x/store")
    cfg = make_cfg(tmp_path)
    r = Run(tmp_path, cfg, store=s3_side)                # controller, learner, evaluator over S3
    from draftzero.loop.actor import Actor
    pod_actor = Actor(pod_side, cfg, "pod1", tmp_path / "work" / "pod1", gorge=r.gorge, log=lambda *a: None)
    r.controller.tick()
    r.learner.step()                                     # v0, written through S3
    r.controller.tick()
    assert pod_side.read_json("versions/player/latest.json")["version"] == 0
    pod_actor.step()
    pod_actor.step()                                     # shards written through the mount
    assert r.learner.step() == "published"               # ... read through S3
    assert json.loads(pod_side.read_bytes("versions/player/latest.json"))["version"] == 1
