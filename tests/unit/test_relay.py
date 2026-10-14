"""The relay between a run's store and actor hosts' stores, with real rsync between two local directories (the
transport is pluggable: a host without "ssh" is a local path), plus the SSH command lines it builds."""
import json
import shutil
from pathlib import Path

import pytest

from draftzero.loop import relay as rl
from draftzero.loop.relay import Host, Relay, Transport
from draftzero.workers.base import Result, local_run
from draftzero.workers.ssh import SSHTarget

pytestmark = pytest.mark.skipif(shutil.which("rsync") is None, reason="needs rsync")


def _w(p: Path, text: str = "x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def _shard(store: Path, actor: str, seq: int, complete: bool = True) -> Path:
    d = store / "shards" / actor / f"{seq:06d}"
    _w(d / "corpus.jsonl.gz", f"corpus {actor} {seq}")
    _w(d / "games.jsonl", '{"game": 0}\n')
    if complete:
        _w(d / "meta.json", json.dumps({"actor": actor, "seq": seq}))
    return d


def _version(store: Path, n: int, latest: bool = True) -> None:
    d = store / "versions" / "player" / f"v{n:04d}"
    _w(d / "model.pt", f"pt {n}")
    _w(d / "model.dzgw", f"dzgw {n}")
    _w(d / "meta.json", json.dumps({"version": n}))
    if latest:
        _w(store / "versions" / "player" / "latest.json", json.dumps({"version": n, "path": f"versions/player/v{n:04d}"}))


@pytest.fixture
def world(tmp_path):
    """A run store with three versions, a config and control; one host store with shards and an old version."""
    local = tmp_path / "runs" / "smoke"
    _w(local / "config.yml", "run: smoke\n")
    _w(local / "control.json", json.dumps({"format": "dz-control/1", "version": {"player": 3}, "stop": False}))
    for n in (1, 2, 3):
        _version(local, n)
    root = tmp_path / "pod1"
    remote = root / "store"
    _shard(remote, "pod1-a", 1)
    _shard(remote, "pod1-a", 2, complete=False)          # still being written: meta.json comes last
    _shard(remote, "pod1-b", 1)
    _w(remote / "heartbeats" / "pod1-a.json", json.dumps({"machine": "pod1-a", "seq": 2}))
    _w(remote / "metrics" / "actor" / "pod1-a" / "000001.jsonl", '{"name": "throughput.games_per_hour"}\n')
    _version(remote, 0)                                     # stale: deleted by the push
    _w(remote / "versions" / "player" / "latest.json", json.dumps({"version": 0}))
    return local, root


def test_one_sync_pulls_complete_shards_and_pushes_versions(world):
    local, root = world
    remote = root / "store"
    r = Relay(local, [{"name": "pod1", "root": str(root)}], log=lambda m: None)
    st = r.sync_once()["pod1"]
    assert st["ok"] and st["pulled"] == 2 and st["versions"] == ["player/v0003", "player/v0002"]
    assert st["pushed"] == ["config.yml", "control.json"]
    # pull: complete shards whole, the incomplete one not at all
    for actor, seq in (("pod1-a", 1), ("pod1-b", 1)):
        d = local / "shards" / actor / f"{seq:06d}"
        assert sorted(p.name for p in d.iterdir()) == ["corpus.jsonl.gz", "games.jsonl", "meta.json"]
        assert (d / "corpus.jsonl.gz").read_text() == f"corpus {actor} {seq}"
    assert not (local / "shards" / "pod1-a" / "000002").exists()
    assert json.loads((local / "heartbeats" / "pod1-a.json").read_text())["seq"] == 2
    assert (local / "metrics" / "actor" / "pod1-a" / "000001.jsonl").exists()
    # push: latest + previous, the stale version gone, the pointers current, the host's shards untouched
    vp = remote / "versions" / "player"
    assert sorted(p.name for p in vp.iterdir()) == ["latest.json", "v0002", "v0003"]
    assert (vp / "v0003" / "model.dzgw").read_text() == "dzgw 3"
    assert json.loads((vp / "latest.json").read_text())["version"] == 3
    assert (remote / "config.yml").read_text() == "run: smoke\n"
    assert json.loads((remote / "control.json").read_text())["version"] == {"player": 3}
    assert (remote / "shards" / "pod1-a" / "000002" / "games.jsonl").exists()
    # no rsync debris in the store, and the status file
    assert not list((local / rl.TMP).rglob("*"))
    status = json.loads((local / rl.STATUS).read_text())
    assert status["format"] == "dz-relay/1" and status["hosts"]["pod1"]["pulled"] == 2


def test_later_syncs_bring_new_shards_and_versions(world):
    local, root = world
    remote = root / "store"
    r = Relay(local, [Host("pod1", str(root))], log=lambda m: None)
    r.sync_once()
    _w(remote / "shards" / "pod1-a" / "000002" / "meta.json", "{}")       # the actor finished the shard
    _w(remote / "heartbeats" / "pod1-a.json", json.dumps({"seq": 3}))
    _version(local, 4)                                                      # the learner published v4
    (local / "control.json").write_text(json.dumps({"version": {"player": 4}}))
    st = r.sync_once()["pod1"]
    assert st["pulled"] == 1 and st["versions"] == ["player/v0004", "player/v0003"]
    assert (local / "shards" / "pod1-a" / "000002" / "meta.json").exists()
    assert json.loads((local / "heartbeats" / "pod1-a.json").read_text())["seq"] == 3
    vp = remote / "versions" / "player"
    assert sorted(p.name for p in vp.iterdir()) == ["latest.json", "v0003", "v0004"]
    assert json.loads((remote / "control.json").read_text())["version"] == {"player": 4}
    assert r.sync_once()["pod1"]["pulled"] == 0                             # nothing new: nothing pulled


def test_a_shard_here_already_is_not_pulled_again(world):
    local, root = world
    _shard(local, "pod1-a", 1)
    (local / "shards" / "pod1-a" / "000001" / "corpus.jsonl.gz").write_text("ours")
    assert Relay(local, [Host("pod1", str(root))], log=lambda m: None).sync_once()["pod1"]["pulled"] == 1
    assert (local / "shards" / "pod1-a" / "000001" / "corpus.jsonl.gz").read_text() == "ours"


def test_meta_json_is_copied_after_the_rest(world):
    local, root = world
    calls = []

    def runner(argv, **kw):
        calls.append(argv)
        return local_run(argv, **kw)
    Relay(local, [Host("pod1", str(root))], transport=lambda h: Transport(runner=runner), log=lambda m: None).sync_once()
    pulls = [a for a in calls if any(x.startswith("--files-from=") for x in a)]
    assert len(pulls) == 2
    assert "--exclude=meta.json" in pulls[0] and "--exclude=meta.json" not in pulls[1]
    assert Path(next(x for x in pulls[1] if x.startswith("--files-from="))[13:]).name == "meta"
    # and the pushes: versions, latest.json, then config + control
    pushes = [a for a in calls if a[-1].endswith("/store/") and "--list-only" not in a]
    assert "--delete-excluded" in pushes[0] and pushes[1][-2].endswith("latest.json")
    assert pushes[2][-3:-1] == [str(local.resolve() / "config.yml"), str(local.resolve() / "control.json")]


def test_a_new_host_gets_its_store_and_an_empty_one_pulls_nothing(tmp_path):
    local = tmp_path / "store"
    _w(local / "config.yml", "run: x\n")
    _w(local / "control.json", "{}")
    root = tmp_path / "fresh-pod"                     # nothing there yet: no store, no shards, no heartbeats
    st = Relay(local, [Host("p", str(root))], log=lambda m: None).sync_once()["p"]
    assert st == {**st, "ok": True, "pulled": 0, "versions": [], "pushed": ["config.yml", "control.json"]}
    assert (root / "store" / "config.yml").exists() and not (root / "store" / "versions").exists()


def test_versions_without_latest_json_need_meta_json(tmp_path):
    local = tmp_path / "store"
    _version(local, 1, latest=False)
    _version(local, 2, latest=False)
    (local / "versions" / "player" / "v0002" / "meta.json").unlink()      # v2 still being published
    r = Relay(local, [Host("p", str(tmp_path / "h"))], log=lambda m: None, keep_versions=2)
    assert r.keep(local / "versions" / "player") == ["v0001"]
    _version(local, 3)
    assert r.keep(local / "versions" / "player") == ["v0003", "v0002"]
    assert Relay(local, [], keep_versions=1).keep(local / "versions" / "player") == ["v0003"]


def test_one_failing_host_does_not_stop_the_others(world, tmp_path):
    local, root = world

    class Broken(Transport):
        def rsync(self, args, *, mkdir=None):
            return Result(255, "", "ssh: connect to host 1.2.3.4 port 40066: Connection timed out")

    r = Relay(local, [Host("dead", "/nowhere", "root@1.2.3.4 -p 40066"), Host("pod1", str(root))],
              transport=lambda h: Broken() if h.ssh else Transport(), log=lambda m: None)
    res = r.sync_once()
    assert not res["dead"]["ok"] and "timed out" in res["dead"]["error"]
    assert res["pod1"]["ok"] and res["pod1"]["pulled"] == 2
    assert not json.loads((local / rl.STATUS).read_text())["hosts"]["dead"]["ok"]


def test_run_stops_one_sync_after_the_run_does(world):
    local, root = world
    r = Relay(local, [Host("pod1", str(root))], log=lambda m: None)
    sleeps = []
    r.run(5, iterations=3, sleep=sleeps.append)
    assert sleeps == [5, 5]
    (local / "control.json").write_text(json.dumps({"stop": True}))
    sleeps.clear()
    _shard(root / "store", "pod1-a", 7)
    r.run(5, iterations=10, sleep=sleeps.append)            # the cap only guards the test against a loop
    assert sleeps == [5] and (local / "shards" / "pod1-a" / "000007" / "meta.json").exists()
    assert json.loads((root / "store" / "control.json").read_text())["stop"] is True


def test_ssh_transport_command_lines():
    calls = []

    def runner(argv, **kw):
        calls.append(argv)
        return Result(0, "")
    h = Host("pod1", "/root/dz/run", "root@1.2.3.4 -p 40066 -i /k -o StrictHostKeyChecking=no")
    t = rl.default_transport(h)
    assert t.target.strict is False and t.target.port == 40066 and "ControlMaster=auto" in t.target.options
    t.runner = runner
    t.rsync(["--list-only", t.remote("/root/dz/run/store/shards/")])
    t.rsync(["a", t.remote("/root/dz/run/store/")], mkdir="/root/dz/run/store")
    ls, push = calls
    e = ls[ls.index("-e") + 1]
    assert e.startswith("ssh ") and "-p 40066" in e and "-i /k" in e and "BatchMode=yes" in e
    assert ls[-1] == "root@1.2.3.4:/root/dz/run/store/shards/" and "--rsync-path" not in ls
    assert push[push.index("--rsync-path") + 1] == "mkdir -p /root/dz/run/store && rsync"
    assert "-a" not in push and "--no-o" in push


def test_complete_shards_parses_rsync_listings():
    listing = """drwxr-xr-x          4,096 2026/10/14 15:48:44 .
drwxr-xr-x          4,096 2026/10/14 15:48:44 a1
-rw-r--r--              2 2026/10/14 15:48:44 a1/000001/meta.json
drwxr-xr-x          4,096 2026/10/14 15:48:44 a1/000002
-rw-r--r--          1,234 2026/10/14 15:48:44 a2/000001/meta.json
-rw-r--r--              2 2026/10/14 15:48:44 a2/meta.json
-rw-r--r--              2 2026/10/14 15:48:44 a3/000001/sub/meta.json
"""
    assert rl.complete_shards(listing) == {"a1/000001", "a2/000001"}


def test_the_store_root_is_the_folder_given(tmp_path):
    """A Path has a .root attribute ("/"): the relay must use the folder itself, and never a filesystem root."""
    assert Relay(tmp_path / "s", []).root == (tmp_path / "s").resolve()
    assert Relay(str(tmp_path / "s"), []).root == (tmp_path / "s").resolve()

    class Store:
        root = tmp_path / "t"
    assert Relay(Store(), []).root == (tmp_path / "t").resolve()
    with pytest.raises(ValueError, match="filesystem root"):
        Relay("/", [])


def test_hosts_are_checked():
    with pytest.raises(ValueError):
        Host.of({"name": "x"})
    with pytest.raises(ValueError):
        Host.of({"name": "x", "root": "/r", "port": 22})
    with pytest.raises(ValueError):
        Relay("/tmp/never-used-relay-store-test", [Host("a", "/r"), Host("a", "/s")])
    assert Host.of({"name": "x", "root": "/r/"}).store == "/r/store"


class FolderStore:
    """The contract's Store API over a folder, standing in for a Hugging Face bucket upstream."""

    def __init__(self, root: Path):
        self.root, self.puts = Path(root), []

    def read_bytes(self, path):
        p = self.root / path
        return p.read_bytes() if p.is_file() else None

    def list(self, prefix):
        base = self.root / prefix
        return sorted(str(p.relative_to(self.root)) for p in base.rglob("*") if p.is_file()) if base.exists() else []

    def get(self, path, dest):
        if not (self.root / path).is_file():
            return None
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.root / path, dest)
        return Path(dest)

    def put_many(self, files, message=""):
        self.puts.append(list(files))
        for path, src in files.items():
            _w(self.root / path, Path(src).read_text())


def test_upstream_store_mirrors_down_and_uploads_what_came_in(world, tmp_path):
    run, root = world                       # the world's run folder plays the upstream store here
    up = FolderStore(run)
    mirror = tmp_path / "relay-mirror"
    r = Relay(mirror, [Host("pod1", str(root))], upstream=up, log=lambda m: None)
    st = r.sync_once()["pod1"]
    assert r.upstream_status["ok"] and r.upstream_status["down"]["got"][:2] == ["config.yml", "control.json"]
    assert st["versions"] == ["player/v0003", "player/v0002"]
    vp = root / "store" / "versions" / "player"
    assert sorted(p.name for p in vp.iterdir()) == ["latest.json", "v0002", "v0003"]
    assert (root / "store" / "control.json").read_bytes() == (run / "control.json").read_bytes()
    # up: both complete shards, meta.json last in each batch; heartbeats and metrics
    shard_puts = [p for p in up.puts if p[0].startswith("shards/")]
    assert len(shard_puts) == 2 and all(p[-1].endswith("/meta.json") for p in shard_puts)
    assert (run / "shards" / "pod1-a" / "000001" / "corpus.jsonl.gz").read_text() == "corpus pod1-a 1"
    assert json.loads((run / "heartbeats" / "pod1-a.json").read_text())["seq"] == 2
    assert (run / "metrics" / "actor" / "pod1-a" / "000001.jsonl").exists()
    assert r.upstream_status["up"] == {"ok": True, "shards": 2, "files": 2}
    # nothing new: nothing uploaded; a changed heartbeat goes up again
    n = len(up.puts)
    r.sync_once()
    assert len(up.puts) == n
    _w(root / "store" / "heartbeats" / "pod1-a.json", json.dumps({"seq": 9}))
    r.sync_once()
    assert up.puts[-1] == ["heartbeats/pod1-a.json"] and r.upstream_status["up"]["files"] == 1
    status = json.loads((mirror / rl.STATUS).read_text())
    assert status["upstream"]["ok"]


def test_an_unreachable_upstream_does_not_stop_the_hosts(world, tmp_path):
    run, root = world

    class Down(FolderStore):
        def read_bytes(self, path):
            raise ConnectionError("hub unreachable")

        def put_many(self, files, message=""):
            raise ConnectionError("hub unreachable")
    r = Relay(tmp_path / "m", [Host("pod1", str(root))], upstream=Down(run), log=lambda m: None)
    res = r.sync_once()
    assert res["pod1"]["ok"] and res["pod1"]["pulled"] == 2
    assert not r.upstream_status["ok"] and "unreachable" in r.upstream_status["down"]["error"]
    # the shards wait in the mirror: the next sync with the store back uploads them
    r.upstream = FolderStore(run)
    r.sync_once()
    assert r.upstream_status["ok"] and r.upstream_status["up"]["shards"] == 2


def test_cli_once(world, capsys):
    local, root = world
    assert rl.main(["--store", str(local), "--host", "pod1", str(root), "--once"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["pod1"]["pulled"] == 2


def test_missing_remote_dir_over_ssh_is_not_an_error():
    # A pod before its actor's first shard: the sender cannot cd into shards/ and the receiver sees a broken pipe.
    err = ('rsync: [sender] change_dir "/root/dz/run/store/shards" failed: No such file or directory (2)\n'
           "rsync error: some files/attrs were not transferred (see previous errors) (code 23) at main.c(1874)\n"
           "rsync: [Receiver] write error: Broken pipe (32)\n")
    assert rl._missing_only(Result(23, "", err))
    assert not rl._missing_only(Result(23, "", err + "rsync: [sender] read errors mapping \"x\": Input/output error (5)\n"))
    assert not rl._missing_only(Result(12, "", err))
