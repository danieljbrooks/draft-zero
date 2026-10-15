"""The store's layout (contracts §4) and control.json, the file that says what every role should be doing.

    config.yml, manifest.json                     the run as launched (controller, at the start)
    control.json                                  {"format": "dz-control/1", "version": {"player": N}, "play": {...},
                                                   "evals": [jobs], "stop": bool, "paused": bool, "updated": iso}
    events.jsonl                                  plan changes {time, kind: "set", path, old, new, by, at_version}
                                                  (old/new as JSON: a whole section is a mapping)
    plan/requests/<uuid>.json                     a requested change {path, value, by, time} (`plan set`)
    plan/rejected/<uuid>.json                     ...one the controller refused, with the reason
    versions/player/v0001/                        model.pt, model.dzgw, meta.json
    versions/player/latest.json                   {version, path}
    shards/<actor>/000001/                        corpus.jsonl.gz, games.jsonl, meta.json (written last)
    evals/<job>/0000.games.jsonl, summary.json    evaluation games per shard, and the job's score
    heartbeats/<machine>.json                     liveness, one per role process (every 5 min, playing too)
    metrics/<component>/<machine>/000001.jsonl    draftzero-metrics/1 lines
    learner/lease.json, ingested.json, status.json
    status.md, status.json                        the controller's status page
    stop.request                                  present = stop the run

One writer per file: the controller writes control.json, events.jsonl, plan/rejected and the status page; the
learner versions/ and learner/; each actor shards/<actor>/; the evaluator evals/; `plan set` a request file.

control["play"] is the actors' live settings (the effective config's `actor` section, plus `games_per_version`
in sync mode); control["version"]["player"] is the version actors play (None before the learner publishes v0).
An evaluation job (opened by the controller, played by evaluators):

    {"job": "v0004-vs-prev", "version": 4, "opponent": "prev", "pairs": 100, "sims": 100, "seed": 3131,
     "created": iso, "shards": [{"id": 0, "first_pair": 0, "pairs": 10, "machine": "local-evaluator",
                                 "assigned": unix time, "done": false}, ...]}
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FORMAT = "dz-control/1"
COMPONENT = "player"

CONTROL = "control.json"
CONFIG = "config.yml"
MANIFEST = "manifest.json"
EVENTS = "events.jsonl"
REQUESTS = "plan/requests/"
REJECTED = "plan/rejected/"
STOP = "stop.request"
STATUS = "status.json"
STATUS_MD = "status.md"
LEASE = "learner/lease.json"
INGESTED = "learner/ingested.json"
LEARNER_STATUS = "learner/status.json"

# changes the controller applies as soon as it sees them; any other live field waits for the next version
IMMEDIATE = ("paused", "budget", "budget.")

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


# ------------------------------------------------------------------------------------------------ paths

def check_name(name: str, what: str = "machine") -> str:
    """Machine and actor names become store paths: letters, digits, '.', '_' and '-' only."""
    if not _NAME.match(name or ""):
        raise ValueError(f"{what} name {name!r}: use letters, digits, '.', '_' and '-' (at most 64)")
    return name


def version_dir(version: int, component: str = COMPONENT) -> str:
    return f"versions/{component}/v{int(version):04d}"


def version_file(version: int, name: str, component: str = COMPONENT) -> str:
    return f"{version_dir(version, component)}/{name}"


def latest_path(component: str = COMPONENT) -> str:
    return f"versions/{component}/latest.json"


def published_versions(store, component: str = COMPONENT) -> list[int]:
    """The versions of `component` with a meta.json (complete: it is written after the version's files)."""
    pat = re.compile(rf"^versions/{re.escape(component)}/v(\d+)/meta\.json$")
    return sorted(int(m.group(1)) for m in map(pat.match, store.list(f"versions/{component}/")) if m)


def refuse_version_zero_over(store, component: str = COMPONENT) -> None:
    """A learner publishes version 0 only on a fresh store. latest.json missing while a later version is published is
    a store that lost it, or a shared network folder that does not show it yet: publishing version 0 then would
    send every actor back to the start of the run."""
    later = [v for v in published_versions(store, component) if v > 0]
    if later:
        raise RuntimeError(f"{latest_path(component)} is missing but version {later[-1]} is published: refusing to "
                           "publish version 0 over a run in progress (restart the learner; if it persists, restore "
                           f"latest.json from {version_dir(later[-1], component)})")


def shard_dir(actor: str, seq: int) -> str:
    return f"shards/{actor}/{int(seq):06d}"


def parse_shard(path: str) -> tuple[str, int] | None:
    """(actor, seq) of a shard folder or of any file in one."""
    parts = path.strip("/").split("/")
    if len(parts) >= 3 and parts[0] == "shards" and parts[2].isdigit():
        return parts[1], int(parts[2])
    return None


def eval_games_path(job: str, shard: int) -> str:
    return f"evals/{job}/{int(shard):04d}.games.jsonl"


def eval_summary_path(job: str) -> str:
    return f"evals/{job}/summary.json"


def heartbeat_path(machine: str) -> str:
    return f"heartbeats/{machine}.json"


def metrics_path(component: str, machine: str, seq: int) -> str:
    return f"metrics/{component}/{machine}/{int(seq):06d}.jsonl"


# ------------------------------------------------------------------------------------------------ small helpers

def now_iso(t: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if t is None else t, timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def chunk_seed(run_seed: int, machine: str, seq: int) -> int:
    """An actor chunk's seed: int(sha256(f"{run_seed}|{machine}|{seq}")) % 2**31."""
    return int(hashlib.sha256(f"{run_seed}|{machine}|{seq}".encode()).hexdigest(), 16) % 2**31


def get(cfg: Any, dotted: str, default: Any = None) -> Any:
    """cfg's value at a dotted path, through dataclass attributes or dict keys."""
    cur = cfg
    for k in dotted.split("."):
        if isinstance(cur, dict):
            if k not in cur:
                return default
            cur = cur[k]
        elif hasattr(cur, k):
            cur = getattr(cur, k)
        else:
            return default
    return cur


def as_dict(x: Any) -> dict:
    """A config section (dataclass, namespace or dict) as a plain dict."""
    if x is None:
        return {}
    if isinstance(x, dict):
        return dict(x)
    import dataclasses
    if dataclasses.is_dataclass(x):
        return dataclasses.asdict(x)
    return dict(vars(x))


def plain(x: Any) -> Any:
    """A config value as JSON data: a section (dataclass) becomes a dict, anything else stays as it is."""
    import dataclasses
    return dataclasses.asdict(x) if dataclasses.is_dataclass(x) and not isinstance(x, type) else x


def parse_value(s: str) -> Any:
    """A command-line value: JSON if it parses (3e-4, true, null, [1,2], {"a": 1}), else YAML, else the string."""
    try:
        return json.loads(s)
    except ValueError:
        pass
    try:
        import yaml
        return yaml.safe_load(s)
    except Exception:                                    # noqa: BLE001
        return s


# ------------------------------------------------------------------------------------------------ control.json

def play_fields(cfg) -> dict:
    """What actors read from control.json: the effective config's actor section (+ games_per_version, sync)."""
    from draftzero import config
    d = config.to_dict(cfg)
    play = dict(d["actor"])
    if d.get("mode") == "sync":
        play["games_per_version"] = int(d["sync"]["games_per_version"])
    return play


def new_control(cfg) -> dict:
    return {"format": FORMAT, "version": {COMPONENT: None}, "play": play_fields(cfg), "evals": [], "stop": False,
            "paused": bool(get(cfg, "paused", False)), "updated": now_iso()}


def read(store) -> dict | None:
    c = store.read_json(CONTROL)
    if c is not None and c.get("format") != FORMAT:
        raise ValueError(f"control.json has format {c.get('format')!r}, not {FORMAT} (a run from another version?)")
    return c


def write(store, control: dict, message: str = "") -> None:
    control["updated"] = now_iso()
    store.write_json(CONTROL, control, message or f"control: v{player_version(control)}")


def player_version(control: dict | None, component: str = COMPONENT) -> int | None:
    if not control:
        return None
    return (control.get("version") or {}).get(component)


# ------------------------------------------------------------------------------------------------ plan changes

def read_events(store) -> list[dict]:
    b = store.read_bytes(EVENTS)
    return [] if not b else [json.loads(x) for x in b.decode().splitlines() if x.strip()]


def append_events(store, events: list[dict]) -> None:
    """events.jsonl plus `events` (the controller is its only writer, so read-modify-write is safe)."""
    if not events:
        return
    old = store.read_bytes(EVENTS) or b""
    if old and not old.endswith(b"\n"):
        old += b"\n"
    store.put(EVENTS, old + "".join(json.dumps(e, sort_keys=True, default=str) + "\n" for e in events).encode(),
              f"events: {len(events)} plan change(s)")


def request_change(store, path: str, value: Any, by: str) -> str:
    """Ask the controller to change a live config field: plan/requests/<UTC time to the ns>-<uuid>.json (names sort
    in the order the requests were made, which is the order they are applied)."""
    ns = time.time_ns()
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(ns // 10**9)) + f".{ns % 10**9:09d}"
    name = f"{REQUESTS}{stamp}-{uuid.uuid4().hex[:8]}.json"
    store.write_json(name, {"path": path, "value": value, "by": by, "time": now_iso()}, f"plan: {path}={value!r}")
    return name


def pending_requests(store) -> list[tuple[str, dict]]:
    """[(store path, request)] in the order they were made."""
    out = []
    for p in store.list(REQUESTS):
        if p.endswith(".json"):
            r = store.read_json(p)
            if r is not None:
                out.append((p, r))
    return out                                           # store.list is sorted: by name, i.e. by time


def immediate(path: str) -> bool:
    return any(path == p or (p.endswith(".") and path.startswith(p)) for p in IMMEDIATE)


def effective(base_cfg, store):
    """The launch config plus every plan change so far (config.effective over events.jsonl)."""
    from draftzero import config
    events = read_events(store)
    return config.effective(base_cfg, events) if events else base_cfg


def heartbeat(machine: str, role: str, run: str, **fields) -> tuple[str, bytes]:
    """(path, bytes) of a heartbeat, for a role to put alongside its other files."""
    import os
    import socket
    hb = {"machine": machine, "role": role, "run": run, "time": time.time(), "iso": now_iso(),
          "host": socket.gethostname(), "pid": os.getpid(), **fields}
    return heartbeat_path(machine), json.dumps(hb, indent=1, sort_keys=True, default=str).encode()


@contextlib.contextmanager
def heartbeating(store, beat, every_s: float, log=print):
    """Put `beat()` (a heartbeat's (path, bytes)) every `every_s` seconds from a thread while the block runs: a
    role is silent while Gorge.play runs, which a hung game stretches past timeout_s (its pair's replay adds a
    probe), longer than the controller's liveness window. A failed put is logged; the next one retries."""
    stop = threading.Event()

    def loop():
        while not stop.wait(every_s):
            try:
                path, hb = beat()
                store.put_many({path: hb}, "heartbeat")
            except Exception as e:                       # noqa: BLE001 - a heartbeat never stops a role
                log(f"heartbeat failed ({type(e).__name__}: {e})")
    t = threading.Thread(target=loop, name="heartbeat", daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=60)


def metrics_log(path: Path, cfg, component: str, machine: str):
    """A draftzero.metrics.MetricsLog writing to the local file `path` (uploaded by the role)."""
    from draftzero.metrics import MetricsLog
    return MetricsLog(Path(path), run=get(cfg, "run"), component=component, machine=machine)


def make_gorge(cfg):
    from draftzero.engine.gorge import Gorge
    return Gorge.from_config(cfg)


def next_seq(store, prefix: str) -> int:
    """1 + the highest 6-digit sequence number among the files or folders under `prefix`."""
    best = 0
    for p in store.list(prefix):
        head = p[len(prefix):].split("/")[0].split(".")[0]
        if head.isdigit():
            best = max(best, int(head))
    return best + 1


# ------------------------------------------------------------------------------------------------ versions

def fetch_version(store, version: int, dest_dir: Path, names: tuple[str, ...] = ("model.dzgw",),
                  component: str = COMPONENT) -> tuple[dict, dict[str, Path]]:
    """A published version's meta.json, and its files `names` in dest_dir (checked against the meta's SHA-256s;
    a file already there with the right hash is kept). A version without a network ("net": false, version 0
    when player.init is null) has no files: the dict is empty."""
    meta = store.read_json(version_file(version, "meta.json", component))
    if meta is None:
        raise FileNotFoundError(f"{version_dir(version, component)}/meta.json is not in the store")
    out: dict[str, Path] = {}
    if not meta.get("net"):
        return meta, out
    dest_dir = Path(dest_dir)
    for name in names:
        want = (meta.get("sha256") or {}).get(name.rsplit(".", 1)[-1])
        p = dest_dir / name
        if not (p.exists() and want and sha256(p) == want):
            if store.get(version_file(version, name, component), p) is None:
                raise FileNotFoundError(f"{version_file(version, name, component)} is not in the store")
            if want and sha256(p) != want:
                p.unlink()
                raise ValueError(f"{version_file(version, name, component)}: SHA-256 differs from its meta.json")
        out[name] = p
    return meta, out


# ------------------------------------------------------------------------------------------------ shards

class ShardIndex:
    """Every complete shard's meta.json (a shard folder is complete once its meta.json exists), read once each.

    `known` seeds it with metas a role already has (the learner's ingested.json), so a restart doesn't re-read
    every shard on a bucket.
    """

    def __init__(self, store, known: dict[str, dict] | None = None):
        self.store = store
        self.metas: dict[str, dict] = dict(known or {})

    def refresh(self) -> list[dict]:
        """Read the metas of shards that appeared since the last refresh; returns them (with "path": folder)."""
        new = []
        for p in self.store.list("shards/"):
            if not p.endswith("/meta.json"):
                continue
            d = p[: -len("/meta.json")]
            if d in self.metas or parse_shard(d) is None:
                continue
            m = self.store.read_json(p)
            if m is None:
                continue
            m["path"] = d
            self.metas[d] = m
            new.append(m)
        return sorted(new, key=lambda m: (m.get("version") or 0, m.get("seq", 0), m.get("actor", "")))

    def games(self, version: int | None = None) -> int:
        return sum(int(m.get("games", 0)) for m in self.metas.values()
                   if version is None or m.get("version") == version)
