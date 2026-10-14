"""relay.py — carry a run's files between its store and actor machines that cannot reach it (contracts §4).

A RunPod pod sits behind NAT and gets no token; r1 must never hold secrets. An actor there plays against a LocalStore
on its own disk, `<root>/store`, started from the draftzero checkout on that machine (workers.base.actor_command):

    python -m draftzero.loop.cli actor --config <root>/store/config.yml --store <root>/store --machine <name>

and the relay, running beside the run's real store (a local folder on d1), keeps the two in step over rsync:

    relay = Relay("runs/smoke", [{"name": "pod1", "ssh": "root@1.2.3.4 -p 40066", "root": "/root/dz/run"}])
    relay.sync_once()          # first: the actors' config.yml and control.json must be there before they start
    relay.run(interval=60)     # until the run's control.json says stop (then one more sync for the last shards)

    python -m draftzero.loop.relay --store runs/smoke --host pod1 /root/dz/run "root@1.2.3.4 -p 40066" [--once]

Each sync, for every host (a host that fails is logged and retried next time; the others go on):

- **push** `versions/<kind>/v<N>/` for the latest version and the one before, and the one control.json names if it
  is older (older ones are deleted on the host; each version's files appear together), then `latest.json`, then
  `config.yml` and `control.json`. The pointers are read before the versions go (control.json first: the controller
  names a version only once it is latest) and pushed as read, so a host never points at a version it lacks; a
  version published meanwhile goes at the next sync. Nothing else is pushed.
- **pull** `<root>/store/shards/<actor>/<seq>/` for each shard that is complete there (its `meta.json` exists:
  actors write it last) and not yet here: every other file first, `meta.json` last, so a reader here also sees a
  complete shard or no meta.json. Then `heartbeats/` and `metrics/` (rsync's quick check skips unchanged files).
  Shards are never deleted on the host.

Transport: rsync. A host with "ssh" ("user@host -p port [-i key] [-o opt]") goes over SSH (connections shared through
an ssh ControlMaster); a host without one is a local path (tests, or a mounted disk). Pass `transport=` to plug in
another. The SSH proxy RunPod offers (`<podHostId>@ssh.runpod.io`) is a PTY only: it cannot carry rsync, so a
relay host needs a direct public IP and port.

**Upstream.** When the run's store is not a local folder (`hf://buckets/...`: the relay is then the only process with
the token), pass it as `upstream=` (CLI `--upstream SPEC`) and give the relay a folder of its own as the local
store: each sync first mirrors config.yml, control.json and the latest + previous versions down from the upstream
store into the folder, and after the pulls uploads every newly pulled complete shard (one put_many each, meta.json
last) and the heartbeat and metrics files that changed. Only the Store API is used (read_bytes, list, get, put_many).

Notes: give every actor machine a unique --machine name (shards/<actor>/ and heartbeats/<machine>.json are merged
from all hosts into one store). rsync stages pulled files in `<store>/.relay-tmp/`. In sync mode an actor behind the
relay sees only its own host's shards when it counts the version's games. `relay/status.json` in the store holds
the last sync's per-host result.
"""
from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import shlex
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

from draftzero.workers.base import Result, local_run
from draftzero.workers.ssh import SSHTarget

VERSION_RE = re.compile(r"^v(\d+)$")
PUSH_FILES = ("config.yml", "control.json")        # in this order: control.json last
PULL_DIRS = ("heartbeats", "metrics")
MULTIPLEX = ("ControlMaster=auto", "ControlPath=/tmp/dz-relay-%C", "ControlPersist=120")
STATUS = "relay/status.json"
TMP = ".relay-tmp"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] relay: {msg}", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class Host:
    name: str
    root: str                      # the host's run dir; its store is <root>/store
    ssh: Optional[str] = None      # "user@host -p port [-i key]"; None: root is a path on this machine

    @classmethod
    def of(cls, h: "Host | Mapping") -> "Host":
        if isinstance(h, Host):
            return h
        unknown = set(h) - {"name", "root", "ssh"}
        if unknown or not h.get("name") or not h.get("root"):
            raise ValueError(f"relay host {dict(h)}: needs name and root (and optionally ssh); unknown {sorted(unknown)}")
        return cls(str(h["name"]), str(h["root"]), h.get("ssh") or None)

    @property
    def store(self) -> str:
        return posixpath.join(self.root, "store")


class Transport:
    """rsync between this machine and one host: over SSH when `target` is set, else between local paths."""

    def __init__(self, target: Optional[SSHTarget] = None, *, runner: Callable[..., Result] = local_run,
                 rsync: str = "rsync", timeout: float = 3600):
        self.target, self.runner, self.rsync_bin, self.timeout = target, runner, rsync, timeout

    def remote(self, path: str) -> str:
        """rsync's name for a path on the host."""
        return self.target.remote(path) if self.target else path

    def rsync(self, args: list[str], *, mkdir: Optional[str] = None) -> Result:
        """Run rsync with the relay's flags. `mkdir`: create this directory on the host first (a push)."""
        argv = [self.rsync_bin, "-rlpt", "--no-o", "--no-g"]
        if self.target:
            argv += ["-z", "-e", self.target.rsync_shell()]
            if mkdir:
                argv += ["--rsync-path", f"mkdir -p {shlex.quote(mkdir)} && rsync"]
        elif mkdir:
            Path(mkdir).mkdir(parents=True, exist_ok=True)
        return self.runner(argv + list(args), timeout=self.timeout)

    def __repr__(self) -> str:
        return f"<Transport {self.target.dest + ':' + str(self.target.port) if self.target else 'local'}>"


def default_transport(host: Host, *, multiplex: bool = True) -> Transport:
    if not host.ssh:
        return Transport()
    t = SSHTarget.parse(host.ssh)
    if multiplex:
        t = SSHTarget(t.host, t.user, t.port, t.key, t.strict, tuple(t.options) + MULTIPLEX)
    return Transport(t)


def _missing_only(r: Result) -> bool:
    """rsync exit 23 because a source does not exist (yet): nothing to copy, not an error.

    Over SSH the receiver also reports "write error: Broken pipe" when the sender quits on the missing directory
    (seen on a RunPod pod before its actor wrote its first shard), so that line does not count against it."""
    errs = [l for l in r.err.splitlines() if l.startswith("rsync:")]
    missing = [l for l in errs if "No such file or directory" in l]
    other = [l for l in errs if l not in missing and "Broken pipe" not in l]
    return r.code == 23 and bool(missing) and not other


def _check(r: Result, what: str) -> Result:
    if not r.ok and not _missing_only(r):
        raise RuntimeError(f"{what}: rsync exit {r.code}: {(r.err or r.out).strip()[-400:]}")
    return r


def complete_shards(listing: str) -> set[str]:
    """`actor/seq` of every shard dir with a meta.json, from `rsync --list-only` of a shards/ dir."""
    out = set()
    for line in listing.splitlines():
        if not line.startswith("-"):
            continue
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        p = parts[4].split("/")
        if len(p) == 3 and p[2] == "meta.json":
            out.add(f"{p[0]}/{p[1]}")
    return out


class Relay:
    """rsync-over-ssh between one local store and the stores of actor hosts. See the module docstring."""

    def __init__(self, local_store_root, hosts: Iterable["Host | Mapping"], *,
                 transport: Optional[Callable[[Host], Transport]] = None, keep_versions: int = 2,
                 multiplex: bool = True, log: Callable[[str], None] = _log, status_path: Optional[str] = STATUS,
                 upstream=None, kinds: Iterable[str] = ("player",)):
        self.upstream, self.kinds, self.upstream_status = upstream, tuple(kinds), {}
        self.hosts = [Host.of(h) for h in hosts]
        names = [h.name for h in self.hosts]
        if len(set(names)) != len(names):
            raise ValueError(f"relay host names must be unique: {names}")
        if keep_versions < 1:
            raise ValueError("keep_versions must be at least 1")
        # a LocalStore, or its folder (careful: a Path has a .root too -- "/")
        root = local_store_root if isinstance(local_store_root, (str, os.PathLike)) else local_store_root.root
        self.root = Path(root).resolve()
        if self.root == Path(self.root.anchor):
            raise ValueError(f"relay store {self.root} is a filesystem root")
        self.root.mkdir(parents=True, exist_ok=True)
        make = transport or (lambda h: default_transport(h, multiplex=multiplex))
        self.transports = {h.name: make(h) for h in self.hosts}
        self.keep_versions, self.log, self.status_path = keep_versions, log, status_path

    # ── one round ──
    def sync_once(self) -> dict[str, dict]:
        """Push then pull, every host (with an upstream: mirror it down first, upload what came in last). Returns
        {host: {ok, pushed, versions, pulled, seconds[, error]}}; the upstream's result is in .upstream_status."""
        if self.upstream is not None:
            self.upstream_status = self._guard("upstream down", self.mirror_down)
        out = {}
        for h in self.hosts:
            t0 = time.monotonic()
            try:
                st = {"ok": True, **self.push(h), "pulled": self.pull(h)}
            except Exception as e:      # one unreachable pod must not stop the others
                st = {"ok": False, "error": f"{type(e).__name__}: {e}"[:600]}
                self.log(f"{h.name}: {st['error']}")
            st["seconds"] = round(time.monotonic() - t0, 2)
            if st["ok"] and (st["pulled"] or st["versions"]):
                self.log(f"{h.name}: pulled {st['pulled']} shards, versions {st['versions']}")
            out[h.name] = st
        if self.upstream is not None:
            self.upstream_status = {"down": self.upstream_status, "up": self._guard("upstream up", self.mirror_up)}
            self.upstream_status["ok"] = self.upstream_status["down"]["ok"] and self.upstream_status["up"]["ok"]
        self._write_status(out)
        return out

    def _guard(self, what: str, fn: Callable[[], dict]) -> dict:
        try:
            return {"ok": True, **fn()}
        except Exception as e:      # the store being down must not stop the hosts' round
            self.log(f"{what}: {type(e).__name__}: {e}")
            return {"ok": False, "error": f"{type(e).__name__}: {e}"[:600]}

    # ── upstream: the run's real store when this folder is only the relay's mirror (hf://buckets/...) ──
    def mirror_down(self) -> dict:
        """config.yml, control.json and the versions hosts get (latest + previous of each kind) from the upstream
        store into this folder: a version's files before its meta.json, latest.json after the version."""
        up, got = self.upstream, []
        for f in PUSH_FILES:
            b = up.read_bytes(f)
            if b is not None and _differs(self.root / f, b):
                _atomic_write(self.root / f, b)
                got.append(f)
        for kind in self.kinds:
            latest = up.read_bytes(f"versions/{kind}/latest.json")
            if latest is None:
                continue
            top = int(json.loads(latest)["version"])
            for v in range(top, max(top - self.keep_versions, -1), -1):
                name = f"v{v:04d}"
                if (self.root / "versions" / kind / name / "meta.json").exists():
                    continue
                files = sorted(up.list(f"versions/{kind}/{name}/"), key=lambda p: (p.endswith("/meta.json"), p))
                for p in files:
                    up.get(p, self.root / p)
                got += [f"{kind}/{name}"] if files else []
            if _differs(self.root / "versions" / kind / "latest.json", latest):
                _atomic_write(self.root / "versions" / kind / "latest.json", latest)
        return {"got": got}

    def mirror_up(self) -> dict:
        """Upload what the hosts sent: each complete shard not yet uploaded (one put_many, meta.json last), then the
        heartbeats and metrics files that changed. A ledger (relay/uploaded.json) remembers what went up."""
        up = self.upstream
        ledger_path = self.root / "relay" / "uploaded.json"
        try:
            ledger = json.loads(ledger_path.read_text())
        except (OSError, ValueError):
            ledger = {"shards": [], "files": {}}
        done = set(ledger["shards"])
        shards, changed = 0, {}
        try:
            for meta in sorted((self.root / "shards").glob("*/*/meta.json")):
                sid = str(meta.parent.relative_to(self.root / "shards"))
                if sid in done:
                    continue
                files = sorted((p for p in meta.parent.iterdir() if p.is_file()),
                               key=lambda p: (p.name == "meta.json", p.name))
                up.put_many({f"shards/{sid}/{p.name}": p for p in files}, message=f"relay: shard {sid}")
                done.add(sid)
                shards += 1
            for d in PULL_DIRS:
                for p in sorted((self.root / d).rglob("*")) if (self.root / d).is_dir() else ():
                    if not p.is_file() or p.name.startswith("."):
                        continue
                    rel, st = str(p.relative_to(self.root)), p.stat()
                    if ledger["files"].get(rel) != [st.st_mtime_ns, st.st_size]:
                        changed[rel] = (p, [st.st_mtime_ns, st.st_size])
            if changed:
                up.put_many({rel: p for rel, (p, _) in changed.items()}, message="relay: heartbeats and metrics")
                ledger["files"].update({rel: sig for rel, (_, sig) in changed.items()})
        finally:     # what went up stays recorded even when a later upload fails
            ledger["shards"] = sorted(done)
            _atomic_write(ledger_path, json.dumps(ledger, sort_keys=True).encode())
        return {"shards": shards, "files": len(changed)}

    def run(self, interval: float = 60, *, iterations: Optional[int] = None, stop_when_done: bool = True,
            sleep: Callable[[float], None] = time.sleep) -> dict:
        """sync_once every `interval` s; returns the last round's result after `iterations` rounds, or one round after
        the store's control.json says stop (so the actors' last shards arrive)."""
        n, stopping = 0, False
        while True:
            last = self.sync_once()
            n += 1
            if stopping or (iterations is not None and n >= iterations):
                return last
            if stop_when_done and self.run_stopped():
                self.log("the run is stopping: one more sync for the last shards")
                stopping = True
            sleep(interval)

    def run_stopped(self) -> bool:
        try:
            return bool(json.loads((self.root / "control.json").read_text()).get("stop"))
        except (OSError, ValueError):
            return False

    # ── push: versions, then the pointers ──
    def keep(self, kind_dir: Path, latest: Optional[bytes] = None, pin: Optional[int] = None) -> list[str]:
        """Version dirs of one kind to put on hosts: the latest (latest.json; `latest`: its bytes as already read)
        and the ones before it, newest first, `keep_versions` in all, plus `pin` (the version control.json names)
        when it is older. Without latest.json, the newest dirs that hold a meta.json."""
        found = {}
        for d in kind_dir.iterdir() if kind_dir.is_dir() else ():
            m = VERSION_RE.match(d.name)
            if m and d.is_dir():
                found[int(m.group(1))] = d.name
        top = None
        try:
            top = int(json.loads(latest if latest is not None else (kind_dir / "latest.json").read_bytes())["version"])
        except (OSError, ValueError, KeyError, TypeError):
            found = {v: n for v, n in found.items() if (kind_dir / n / "meta.json").exists()}
        nums = sorted((v for v in found if top is None or v <= top), reverse=True)[: self.keep_versions]
        if pin in found and pin not in nums:
            nums.append(pin)
        return [found[v] for v in nums]

    def push(self, h: Host) -> dict:
        """Versions, then latest.json, then config.yml and control.json. The pointers are read first (control.json
        before latest.json: the controller names a version only once it is latest) and pushed as read, so a version
        published or named while the versions are in flight waits for the next sync."""
        t = self.transports[h.name]
        dest = t.remote(h.store + "/")
        snap = {f: _read(self.root / f) for f in PUSH_FILES}
        vroot = self.root / "versions"
        kinds = sorted(p for p in vroot.iterdir() if p.is_dir()) if vroot.is_dir() else []
        latest = {k.name: _read(k / "latest.json") for k in kinds}
        named = _named_versions(snap["control.json"])
        versions, pointers = [], []
        for kind_dir in kinds:
            kind = kind_dir.name
            keep = self.keep(kind_dir, latest[kind], named.get(kind))
            if not keep:
                continue
            rules = ["-f", f"P /versions/{kind}/latest.json"]
            for v in keep:
                rules += ["-f", f"+ /versions/{kind}/{v}/***"]
            rules += ["-f", f"- /versions/{kind}/*"]
            _check(t.rsync(["-R", "--delete", "--delete-excluded", "--delay-updates", *rules,
                            f"{self.root}/./versions/{kind}/", dest], mkdir=h.store), f"push versions/{kind}")
            versions += [f"{kind}/{v}" for v in keep]
            if latest[kind] is not None:
                pointers.append((f"versions/{kind}/latest.json", latest[kind]))
        lacking = [f"{k}/v{v:04d}" for k, v in named.items() if f"{k}/v{v:04d}" not in versions]
        if lacking:         # never point a host at a version it lacks: it keeps the control.json it has
            self.log(f"{h.name}: control.json names {', '.join(lacking)}, which this store lacks: not pushing it")
            snap["control.json"] = None
        files = [f for f in PUSH_FILES if snap[f] is not None]
        tmp = self.root / TMP
        tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp, prefix="push-") as d:
            for rel, b in pointers + [(f, snap[f]) for f in files]:
                (Path(d) / rel).parent.mkdir(parents=True, exist_ok=True)
                (Path(d) / rel).write_bytes(b)
            for rel, _ in pointers:
                _check(t.rsync(["-R", f"{d}/./{rel}", dest], mkdir=h.store), f"push {rel}")
            if files:
                _check(t.rsync([*(f"{d}/{f}" for f in files), dest], mkdir=h.store), "push " + ", ".join(files))
        return {"versions": versions, "pushed": files}

    # ── pull: complete shards (meta.json last), heartbeats, metrics ──
    def pull(self, h: Host) -> int:
        t = self.transports[h.name]
        src = t.remote(posixpath.join(h.store, "shards") + "/")
        listing = _check(t.rsync(["--list-only", "--include=*/", "--include=meta.json", "--exclude=*", src]),
                         "list shards")
        local = self.root / "shards"
        new = sorted(s for s in complete_shards(listing.out) if not (local / s / "meta.json").exists())
        tmp = self.root / TMP
        tmp.mkdir(exist_ok=True)
        if new:
            local.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(dir=tmp, prefix="lists-") as d:
                data, meta = Path(d) / "data", Path(d) / "meta"
                data.write_text("".join(f"{s}/\n" for s in new))
                meta.write_text("".join(f"{s}/meta.json\n" for s in new))
                _check(t.rsync([f"--files-from={data}", "--exclude=meta.json", f"--temp-dir={tmp}", src, f"{local}/"]),
                       "pull shards")
                _check(t.rsync([f"--files-from={meta}", f"--temp-dir={tmp}", src, f"{local}/"]), "pull shard meta.json")
        _check(t.rsync([f"--temp-dir={tmp}", *(t.remote(posixpath.join(h.store, d)) for d in PULL_DIRS),
                        f"{self.root}/"]), "pull " + " and ".join(PULL_DIRS))
        return len(new)

    def _write_status(self, hosts: dict) -> None:
        if not self.status_path:
            return
        body = {"format": "dz-relay/1", "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "hosts": hosts, **({"upstream": self.upstream_status} if self.upstream is not None else {})}
        _atomic_write(self.root / self.status_path, json.dumps(body, indent=1, sort_keys=True).encode())


def _read(path: Path) -> Optional[bytes]:
    try:
        return path.read_bytes()
    except OSError:
        return None


def _named_versions(control: Optional[bytes]) -> dict[str, int]:
    """{kind: version} that a control.json names ({} when it names none, or doesn't parse)."""
    try:
        named = json.loads(control).get("version") or {}
        return {str(k): int(v) for k, v in named.items() if v is not None}
    except (TypeError, ValueError, AttributeError):
        return {}


def _differs(path: Path, data: bytes) -> bool:
    try:
        return path.read_bytes() != data
    except OSError:
        return True


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m draftzero.loop.relay", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True, help="the run's local store folder")
    ap.add_argument("--host", nargs="+", action="append", default=[], metavar=("NAME ROOT", "SSH"),
                    help='NAME ROOT ["user@host -p PORT"]: one actor host (repeat); no SSH: ROOT is a local path')
    ap.add_argument("--hosts", default=None, help="a JSON or YAML list of {name, root, ssh} instead of --host")
    ap.add_argument("--upstream", default=None, help="the run's store when --store is only the relay's mirror")
    ap.add_argument("--interval", type=float, default=60)
    ap.add_argument("--once", action="store_true", help="one sync, then exit")
    a = ap.parse_args(argv)
    hosts: list = []
    if a.hosts:
        text = Path(a.hosts).read_text()
        try:
            hosts = json.loads(text)
        except ValueError:
            import yaml
            hosts = yaml.safe_load(text)
    for h in a.host:
        if len(h) not in (2, 3):
            ap.error("--host NAME ROOT [SSH]")
        hosts.append({"name": h[0], "root": h[1], "ssh": h[2] if len(h) == 3 else None})
    if not hosts:
        ap.error("no hosts: --host or --hosts")
    upstream = None
    if a.upstream:
        from draftzero.loop.store import open_store
        upstream = open_store(a.upstream)
    relay = Relay(a.store, hosts, upstream=upstream)
    res = relay.sync_once() if a.once else relay.run(a.interval)
    print(json.dumps({**res, **({"@upstream": relay.upstream_status} if upstream else {})}, indent=1, sort_keys=True))
    return 0 if all(r["ok"] for r in res.values()) and relay.upstream_status.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
