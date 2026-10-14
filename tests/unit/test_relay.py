"""The relay between a run's store and actor hosts' stores, with real rsync between two local directories (the
transport is pluggable: a host without "ssh" is a local path), plus the SSH command lines it builds. The player's
shards and versions first, then the deck builder's (builder/shards/, versions/builder/)."""
import json
import os
import shutil
from pathlib import Path

import pytest

from draftzero.loop import builder as bld
from draftzero.loop import control as ctl
from draftzero.loop import relay as rl
from draftzero.loop.relay import Host, Relay, Transport
from draftzero.loop.store import LocalStore
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
    assert "--delete-excluded" in pushes[0] and pushes[1][-2].endswith("/./versions/player/latest.json")
    assert [Path(p).name for p in pushes[2][-3:-1]] == ["config.yml", "control.json"]


def test_push_sends_the_pointers_as_they_were_before_the_versions(world):
    """The learner publishes v4 and the controller names it in control.json while the versions are in flight: the
    host must get the latest.json and control.json read before the push (v3), never a pointer to a version it
    lacks; the next sync brings v4."""
    local, root = world
    remote = root / "store"
    bumped = []

    def runner(argv, **kw):
        r = local_run(argv, **kw)
        if "--delete-excluded" in argv and not bumped:          # right after the first versions rsync
            _version(local, 4)
            (local / "control.json").write_text(json.dumps({"format": "dz-control/1", "version": {"player": 4}}))
            bumped.append(1)
        return r
    r = Relay(local, [Host("pod1", str(root))], transport=lambda h: Transport(runner=runner), log=lambda m: None)
    st = r.sync_once()["pod1"]
    assert st["ok"] and st["versions"] == ["player/v0003", "player/v0002"]
    vp = remote / "versions" / "player"
    named = json.loads((remote / "control.json").read_text())["version"]["player"]
    assert named == 3 and json.loads((vp / "latest.json").read_text())["version"] == 3
    assert (vp / f"v{named:04d}" / "meta.json").exists()
    assert not list((local / rl.TMP).rglob("*"))
    r.sync_once()
    assert json.loads((remote / "control.json").read_text())["version"]["player"] == 4
    assert (vp / "v0004" / "meta.json").exists()


def test_push_keeps_the_version_control_json_names(world):
    """The controller lags the learner: control.json names v1 while latest.json is v3. A host keeps v1 too."""
    local, root = world
    (local / "control.json").write_text(json.dumps({"format": "dz-control/1", "version": {"player": 1}}))
    st = Relay(local, [Host("pod1", str(root))], log=lambda m: None).sync_once()["pod1"]
    assert st["versions"] == ["player/v0003", "player/v0002", "player/v0001"]
    vp = root / "store" / "versions" / "player"
    assert sorted(p.name for p in vp.iterdir()) == ["latest.json", "v0001", "v0002", "v0003"]
    assert json.loads((root / "store" / "control.json").read_text())["version"]["player"] == 1
    # a control.json naming a version this store lacks stays here; the host keeps the one it has
    (local / "control.json").write_text(json.dumps({"format": "dz-control/1", "version": {"player": 7}}))
    st = Relay(local, [Host("pod1", str(root))], log=lambda m: None).sync_once()["pod1"]
    assert st["ok"] and st["pushed"] == ["config.yml"]
    assert json.loads((root / "store" / "control.json").read_text())["version"]["player"] == 1


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
    assert r.upstream_status["ok"] and r.upstream_status["down"]["got"] == ["player/v0003", "player/v0002",
                                                                            "config.yml", "control.json"]
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


def test_a_failed_version_download_keeps_the_previous_control_json(world, tmp_path):
    """The upstream publishes v4 and names it in control.json, but its download fails: the mirror and the hosts keep
    the control.json naming v3 (a version they have) until a later sync brings v4."""
    run, root = world

    class Flaky(FolderStore):
        broken = True

        def get(self, path, dest):
            if self.broken and path.startswith("versions/player/v0004/"):
                raise RuntimeError("xet download failed")
            return super().get(path, dest)
    up = Flaky(run)
    mirror = tmp_path / "m"
    r = Relay(mirror, [Host("pod1", str(root))], upstream=up, log=lambda m: None)
    r.sync_once()
    _version(run, 4)
    (run / "control.json").write_text(json.dumps({"format": "dz-control/1", "version": {"player": 4}}))
    res = r.sync_once()
    assert not r.upstream_status["down"]["ok"] and "xet" in r.upstream_status["down"]["error"]
    assert json.loads((mirror / "control.json").read_text())["version"]["player"] == 3
    assert json.loads((mirror / "versions/player/latest.json").read_text())["version"] == 3
    assert res["pod1"]["ok"] and json.loads((root / "store/control.json").read_text())["version"]["player"] == 3
    up.broken = False
    r.sync_once()
    assert json.loads((root / "store/control.json").read_text())["version"]["player"] == 4
    assert (root / "store/versions/player/v0004/meta.json").exists()


def test_mirror_down_fetches_the_version_control_json_names(world, tmp_path):
    run, root = world
    (run / "control.json").write_text(json.dumps({"format": "dz-control/1", "version": {"player": 1}}))
    r = Relay(tmp_path / "m", [Host("pod1", str(root))], upstream=FolderStore(run), log=lambda m: None)
    r.sync_once()
    assert r.upstream_status["ok"] and (tmp_path / "m/versions/player/v0001/meta.json").exists()
    assert (root / "store/versions/player/v0001/meta.json").exists()
    assert json.loads((root / "store/control.json").read_text())["version"]["player"] == 1


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


# ── the deck builder's files (draftzero.loop.builder): builder/shards/ pulled, versions/builder/ pushed ──


def _bshard(store: Path, actor: str, seq: int, complete: bool = True) -> Path:
    """A builder shard as BuilderActor.play_decks uploads it: decks.jsonl, games.jsonl, meta.json last."""
    d = store / bld.shard_dir(actor, seq)
    _w(d / "decks.jsonl", f'{{"name": "b-{actor}-{seq}"}}\n')
    _w(d / "games.jsonl", '{"winner": 0}\n')
    if complete:
        _w(d / "meta.json", json.dumps({"actor": actor, "seq": seq, "version": 0, "games": 1, "decided": 1}))
    return d


def _bversion(store: Path, n: int, latest: bool = True) -> None:
    """A builder version as BuilderLearner publishes it: ens.pt and meta.json (version 0 untrained: meta only),
    then latest.json."""
    meta = {"version": n, "net": n > 0, "sha256": {}}
    if n > 0:
        pt = _w(store / ctl.version_file(n, "ens.pt", bld.COMPONENT), f"ens {n}")
        meta["sha256"] = {"pt": ctl.sha256(pt)}
    _w(store / ctl.version_file(n, "meta.json", bld.COMPONENT), json.dumps(meta))
    if latest:
        _w(store / ctl.latest_path(bld.COMPONENT),
           json.dumps({"version": n, "path": ctl.version_dir(n, bld.COMPONENT)}))


def _a_minute_on(store: Path) -> None:
    """Age the host's pointers by a relay interval. rsync's quick check compares sizes and whole-second mtimes, and a
    pointer often keeps its size (latest.json v0002 -> v0003; control.json builder 2 -> 3): two syncs in the same
    second, as a test makes them, would see it unchanged (a relay's syncs are an interval apart)."""
    for p in [store / "control.json", store / "config.yml", *store.glob("versions/*/latest.json")]:
        if p.exists():
            t = p.stat().st_mtime - 60
            os.utime(p, (t, t))


def _control(store: Path, player: int, builder=None) -> None:
    v = {"player": player, **({"builder": builder} if builder is not None else {})}
    (store / "control.json").write_text(json.dumps({"format": "dz-control/1", "version": v, "stop": False}))


@pytest.fixture
def bworld(world):
    """The world with the deck builder on: builder versions 0-2 here, named by control.json; on the host two complete
    builder shards, one being written, and a stale builder version."""
    local, root = world
    for n in (0, 1, 2):
        _bversion(local, n)
    _control(local, 3, 2)
    remote = root / "store"
    _bshard(remote, "pod1-b", 1)
    _bshard(remote, "pod1-b", 2, complete=False)
    _bshard(remote, "pod1-c", 1)
    _w(remote / "heartbeats" / "pod1-b.json", json.dumps({"machine": "pod1-b", "role": "builder-actor"}))
    _w(remote / "metrics" / "builder-actor" / "pod1-b" / "000001.jsonl", '{"name": "data.builder_games"}\n')
    _bversion(remote, 0)                                                # stale: deleted by the push
    _w(remote / ctl.latest_path(bld.COMPONENT), json.dumps({"version": 0}))
    _w(remote / "builder" / "shards" / "pod1-b" / "000003" / ".tmp-x", "half-written")       # never a shard
    return local, root


def test_builder_shards_come_in_and_builder_versions_go_out(bworld, tmp_path):
    local, root = bworld
    remote = root / "store"
    st = Relay(local, [Host("pod1", str(root))], log=lambda m: None).sync_once()["pod1"]
    assert st["ok"] and st["pulled"] == 4 and st["builder_pulled"] == 2
    assert st["versions"] == ["builder/v0002", "builder/v0001", "player/v0003", "player/v0002"]
    assert st["pushed"] == ["config.yml", "control.json"]
    # pull: complete builder shards whole, beside the player's; the one being written not at all
    for actor in ("pod1-b", "pod1-c"):
        d = local / bld.shard_dir(actor, 1)
        assert sorted(p.name for p in d.iterdir()) == ["decks.jsonl", "games.jsonl", "meta.json"]
        assert (d / "decks.jsonl").read_text() == f'{{"name": "b-{actor}-1"}}\n'
    assert not (local / bld.shard_dir("pod1-b", 2)).exists() and not (local / "builder/shards/pod1-b/000003").exists()
    assert (local / "shards" / "pod1-a" / "000001" / "meta.json").exists()
    assert json.loads((local / "heartbeats" / "pod1-b.json").read_text())["role"] == "builder-actor"
    assert (local / "metrics" / "builder-actor" / "pod1-b" / "000001.jsonl").exists()
    # the builder learner here sees exactly the complete shards
    idx = bld.ShardIndex(LocalStore(local))
    assert sorted(m["path"] for m in idx.refresh()) == [bld.shard_dir("pod1-b", 1), bld.shard_dir("pod1-c", 1)]
    # push: latest + previous builder versions, the stale one gone, the pointers current
    vb = remote / "versions" / "builder"
    assert sorted(p.name for p in vb.iterdir()) == ["latest.json", "v0001", "v0002"]
    assert json.loads((vb / "latest.json").read_text())["version"] == 2
    assert json.loads((remote / "control.json").read_text())["version"] == {"player": 3, "builder": 2}
    # a builder actor on the host fetches the version control.json names from its own store
    meta, files = ctl.fetch_version(LocalStore(remote), 2, tmp_path / "bv", ("ens.pt",), component=bld.COMPONENT)
    assert meta["version"] == 2 and files["ens.pt"].read_text() == "ens 2"
    assert (remote / bld.shard_dir("pod1-b", 2) / "games.jsonl").exists()      # the host's shards stay
    assert not list((local / rl.TMP).rglob("*"))
    assert json.loads((local / rl.STATUS).read_text())["hosts"]["pod1"]["builder_pulled"] == 2
    # later: the shard being written completes, the builder learner publishes v3 and the controller names it
    _w(remote / bld.shard_dir("pod1-b", 2) / "meta.json", json.dumps({"actor": "pod1-b", "seq": 2}))
    _bversion(local, 3)
    _control(local, 3, 3)
    _a_minute_on(remote)
    st = Relay(local, [Host("pod1", str(root))], log=lambda m: None).sync_once()["pod1"]
    assert st["pulled"] == 1 and st["builder_pulled"] == 1
    assert (local / bld.shard_dir("pod1-b", 2) / "meta.json").exists()
    assert sorted(p.name for p in vb.iterdir()) == ["latest.json", "v0002", "v0003"]
    assert json.loads((remote / "control.json").read_text())["version"]["builder"] == 3
    assert Relay(local, [Host("pod1", str(root))], log=lambda m: None).sync_once()["pod1"]["pulled"] == 0


def test_builder_meta_json_is_copied_after_the_rest(bworld):
    local, root = bworld
    calls = []

    def runner(argv, **kw):
        calls.append(argv)
        return local_run(argv, **kw)
    Relay(local, [Host("pod1", str(root))], transport=lambda h: Transport(runner=runner),
          log=lambda m: None).sync_once()
    pulls = [a for a in calls if any(x.startswith("--files-from=") for x in a)]
    assert [a[-1] for a in pulls] == [f"{local.resolve()}/shards/"] * 2 + [f"{local.resolve()}/builder/shards/"] * 2
    assert all(a[-2].endswith("/store/builder/shards/") for a in pulls[2:])
    assert "--exclude=meta.json" in pulls[2] and "--exclude=meta.json" not in pulls[3]
    assert Path(next(x for x in pulls[3] if x.startswith("--files-from="))[13:]).name == "meta"
    # the pushes: both kinds' versions, both latest.jsons, then config + control
    pushes = [a for a in calls if a[-1].endswith("/store/") and "--list-only" not in a]
    assert [a[-2].rsplit("/./", 1)[-1] for a in pushes[:4]] == ["versions/builder/", "versions/player/",
                                                                "versions/builder/latest.json",
                                                                "versions/player/latest.json"]
    assert [Path(p).name for p in pushes[4][-3:-1]] == ["config.yml", "control.json"]


def test_builder_shards_wait_for_the_builder(world):
    """A run without the deck builder (no versions/builder/ here): the relay never looks for builder shards, so its
    rsync calls are the player's alone; once the builder learner publishes version 0, they come in."""
    local, root = world
    _bshard(root / "store", "pod1-b", 1)
    calls = []

    def runner(argv, **kw):
        calls.append(argv)
        return local_run(argv, **kw)
    r = Relay(local, [Host("pod1", str(root))], transport=lambda h: Transport(runner=runner), log=lambda m: None)
    st = r.sync_once()["pod1"]
    assert st["pulled"] == 2 and st["builder_pulled"] == 0 and not (local / "builder").exists()
    assert not any("builder/shards" in x or "versions/builder" in x for a in calls for x in a)
    _bversion(local, 0)
    st = r.sync_once()["pod1"]
    assert st["builder_pulled"] == 1 and (local / bld.shard_dir("pod1-b", 1) / "meta.json").exists()
    assert "builder/v0000" in st["versions"]


def test_push_sends_the_builder_pointers_as_they_were_before_the_versions(bworld):
    """The builder learner publishes v3 and the controller names it while the versions are in flight (right after
    the builder's versions rsync): the host keeps getting latest.json and control.json naming v2 until the next sync."""
    local, root = bworld
    remote = root / "store"
    bumped = []

    def runner(argv, **kw):
        r = local_run(argv, **kw)
        if "--delete-excluded" in argv and argv[-2].endswith("/./versions/builder/") and not bumped:
            _bversion(local, 3)
            _control(local, 3, 3)
            bumped.append(1)
        return r
    r = Relay(local, [Host("pod1", str(root))], transport=lambda h: Transport(runner=runner), log=lambda m: None)
    st = r.sync_once()["pod1"]
    assert bumped and st["ok"] and st["versions"][:2] == ["builder/v0002", "builder/v0001"]
    vb = remote / "versions" / "builder"
    named = json.loads((remote / "control.json").read_text())["version"]["builder"]
    assert named == 2 and json.loads((vb / "latest.json").read_text())["version"] == 2
    assert (vb / f"v{named:04d}" / "meta.json").exists() and not (vb / "v0003").exists()
    _a_minute_on(remote)
    r.sync_once()
    assert json.loads((remote / "control.json").read_text())["version"]["builder"] == 3
    assert (vb / "v0003" / "ens.pt").read_text() == "ens 3"


def test_push_keeps_the_builder_version_control_json_names(bworld):
    """The controller lags the builder learner (control.json names builder v0, latest is v2): the host keeps v0 too;
    a control.json naming a builder version this store lacks is not pushed."""
    local, root = bworld
    _control(local, 3, 0)
    st = Relay(local, [Host("pod1", str(root))], log=lambda m: None).sync_once()["pod1"]
    assert st["versions"][:3] == ["builder/v0002", "builder/v0001", "builder/v0000"]
    vb = root / "store" / "versions" / "builder"
    assert sorted(p.name for p in vb.iterdir()) == ["latest.json", "v0000", "v0001", "v0002"]
    _control(local, 3, 7)
    st = Relay(local, [Host("pod1", str(root))], log=lambda m: None).sync_once()["pod1"]
    assert st["ok"] and st["pushed"] == ["config.yml"]
    assert json.loads((root / "store" / "control.json").read_text())["version"]["builder"] == 0


def test_upstream_mirrors_builder_versions_and_uploads_builder_shards(bworld, tmp_path):
    run, root = bworld                      # the run folder plays the upstream store
    up = FolderStore(run)
    mirror = tmp_path / "relay-mirror"
    r = Relay(mirror, [Host("pod1", str(root))], upstream=up, log=lambda m: None)
    st = r.sync_once()["pod1"]
    assert r.upstream_status["ok"] and r.upstream_status["down"]["got"] == [
        "player/v0003", "player/v0002", "builder/v0002", "builder/v0001", "config.yml", "control.json"]
    assert st["versions"] == ["builder/v0002", "builder/v0001", "player/v0003", "player/v0002"]
    assert json.loads((root / "store/control.json").read_text())["version"] == {"player": 3, "builder": 2}
    assert (root / "store/versions/builder/v0002/ens.pt").read_text() == "ens 2"
    # up: the player's two shards and the builder's two, meta.json last in each batch
    bputs = [p for p in up.puts if p[0].startswith("builder/shards/")]
    assert len(bputs) == 2 and all(p[-1].endswith("/meta.json") for p in bputs)
    assert sorted(p[-1] for p in bputs) == [f"{bld.shard_dir(a, 1)}/meta.json" for a in ("pod1-b", "pod1-c")]
    assert (run / bld.shard_dir("pod1-c", 1) / "decks.jsonl").read_text() == '{"name": "b-pod1-c-1"}\n'
    assert r.upstream_status["up"]["shards"] == 4
    ledger = json.loads((mirror / "relay" / "uploaded.json").read_text())
    assert ledger["builder/shards"] == ["pod1-b/000001", "pod1-c/000001"] and len(ledger["shards"]) == 2
    n = len(up.puts)
    r.sync_once()
    assert len(up.puts) == n                                    # nothing new: nothing uploaded again


def test_upstream_mirrors_a_kind_control_json_names(bworld, tmp_path):
    """A relay told only of the player still mirrors the builder versions control.json names: otherwise it would
    hold back every control.json (a host is never pointed at a version it lacks)."""
    run, root = bworld
    _control(run, 3, 1)
    r = Relay(tmp_path / "m", [Host("pod1", str(root))], upstream=FolderStore(run), kinds=("player",),
              log=lambda m: None)
    r.sync_once()
    assert r.upstream_status["ok"] and (tmp_path / "m/versions/builder/v0001/meta.json").exists()
    assert json.loads((root / "store/control.json").read_text())["version"]["builder"] == 1


def test_a_player_only_upload_ledger_has_no_builder_entry(world, tmp_path):
    run, root = world
    r = Relay(tmp_path / "m", [Host("pod1", str(root))], upstream=FolderStore(run), log=lambda m: None)
    r.sync_once()
    assert set(json.loads((tmp_path / "m" / "relay" / "uploaded.json").read_text())) == {"shards", "files"}
