"""`python -m draftzero.loop.cli <role>`: the loop's command line (contracts §4, §7; `dz` maps onto it).

    actor       --config C [--store S] [--machine M] [--work-dir DIR] [--max-chunks N]   play self-play chunks
    learner     --config C [--store S] [--machine M] [--work-dir DIR]                    train and publish versions
    evaluator   --config C [--store S] [--machine M] [--work-dir DIR]                    play evaluation shards
    controller  --config C [--store S] [--once] [--status-every SECS]                    control.json, plan, status
    local       --config C [--store S] [--actors N] [--work-dir DIR] [--max-hours H]     every role on this machine
                [--builder-actors M]
    builder-actor      --config C [--store S] [--machine M] [--work-dir DIR] [--max-chunks N]   (builder.learn)
    builder-learner    --config C [--store S] [--machine M] [--work-dir DIR]                    deck builder roles:
    builder-evaluator  --config C [--store S] [--machine M] [--work-dir DIR]                    draftzero.loop.builder
    status      [STORE] [--json]                                                         print status.md
    stop        [STORE]                                                                  ask the controller to stop
    plan        [STORE] set PATH=VALUE [PATH=VALUE ...] [--before-start]                 a live config change
    plan        [STORE] list                                                             pending, rejected, applied

S is a folder or hf://buckets/<owner>/<name>; it defaults to the config's `store`. --machine names this process in
the store (heartbeats/<machine>.json, shards/<machine>/): it must be unique per actor; the default is the host
name (with the role appended for the other roles). --work-dir defaults to runs/<run>/work/<role>-<machine>.
Roles poll the store every --poll seconds (default 30 on a folder, 60 on a bucket; local mode passes it on); the
controller writes its status page every --status-every seconds (default 60 on a folder, 300 on a bucket).
A controller started on a store whose run has started runs on the store's config.yml plus events.jsonl: C may differ
from it only in machine paths (draftzero.config.MACHINE), unless --allow-config-change (controller, and local, which
passes it on) lets C change other fields too, recorded in events.jsonl.
status, stop and plan refuse a STORE that is not a run's store (no config.yml or control.json, which the controller
writes when the run starts), so a mistyped path creates nothing and reports no success; `plan ... --before-start`
queues changes in a store whose run has not started yet.
"""
from __future__ import annotations

import argparse
import re
import signal
import socket
import sys
import time
from pathlib import Path

BUILDER_ROLES = ("builder-actor", "builder-learner", "builder-evaluator")
ROLES = ("actor", "learner", "evaluator", "controller", "local") + BUILDER_ROLES
COMMANDS = ROLES + ("status", "stop", "plan")
HELP = {
    "actor": "play self-play chunks with the current version and upload them as shards (many, on any machine)",
    "learner": "pack new shards, train on the newest window and publish versions (one per run, holding a lease)",
    "evaluator": "play the evaluation shards the controller assigns to this machine and score finished jobs",
    "controller": "write config.yml, manifest.json and control.json; apply plan changes; assign evaluation "
                  "shards; write status.md and status.json; stop the run at stop.request or the budget (one)",
    "local": "every role on this machine as separate processes sharing one store (`dz run`)",
    "builder-actor": "build sealed decks with the current builder version, play them and upload the results as "
                     "builder shards (builder.learn; many, on any machine)",
    "builder-learner": "fit the deck value ensemble on the newest builder shards and publish builder versions "
                       "(builder.learn; one per run, holding a lease)",
    "builder-evaluator": "play the builder evaluation shards the controller assigns to this machine (each builder "
                         "version's decks against best23's on held-out pools) and score finished jobs",
    "status": "print the run's status page (status.md, or status.json with --json)",
    "stop": "ask the controller to stop the run (writes stop.request)",
    "plan": "request a live config change (`set PATH=VALUE ...`) or list requested, rejected and applied ones",
}
ALLOW_HELP = ("on a store whose run has started, let this config change the run's config (the store's config.yml "
              "plus events.jsonl) beyond machine paths; the changes are recorded in events.jsonl. Without it such "
              "a config is refused (change live fields with `plan set`)")
PLAN_EPILOG = """examples:
  plan runs/smoke set learner.lr=3e-4             applied at the next version
  plan runs/smoke set paused=true                 paused and budget.* apply at once
  plan runs/smoke list
  plan runs/new set learner.lr=3e-4 --before-start   queued before `dz run` starts the run

Only live fields change (draftzero.config.LIVE: actor.sims, actor.pcr.*, actor.decided.*, actor.forced.*,
actor.record_every, actor.chunk_games, learner.lr, learner.window_positions, learner.min_new_positions,
learner.steps_per_position, learner.epochs, eval.every_versions, eval.pairs, eval.sims, budget.*, paused). A request
the controller rejects moves to plan/rejected/ with the reason; an applied one is a line of events.jsonl."""


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _host() -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", socket.gethostname().split(".")[0]) or "host"


def _parser(prog: str = "python -m draftzero.loop.cli") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="role", required=True)
    for role in ROLES:
        p = sub.add_parser(role, help=HELP[role], description=HELP[role])
        p.add_argument("--config", required=True, help="the run's config (YAML)")
        p.add_argument("--store", default=None, help="default: the config's store")
        p.add_argument("--poll", type=float, default=None,
                       help="seconds between looks at the store (default 30; 60 on a bucket)")
        if role != "controller":
            p.add_argument("--machine", default=None,
                           help="this process's name in the store; unique per actor (default: the host name)")
            p.add_argument("--work-dir", type=Path, default=None,
                           help="local cache and logs (default runs/<run>/work/...)")
        if role in ("actor", "builder-actor"):
            p.add_argument("--max-chunks", type=int, default=None, help="exit after this many chunks")
        if role == "controller":
            p.add_argument("--once", action="store_true", help="one tick, then exit")
            p.add_argument("--status-every", type=float, default=None, help="seconds between status pages")
        if role == "local":
            p.add_argument("--actors", type=int, default=1, help="actor processes (default 1)")
            p.add_argument("--max-hours", type=float, default=None,
                           help="stop the run after this long (exit code 2)")
            p.add_argument("--builder-actors", type=int, default=1,
                           help="builder actor processes when the config's builder.learn is on (default 1)")
        if role in ("controller", "local"):
            p.add_argument("--allow-config-change", action="store_true",
                           help=("passed on to the controller: " if role == "local" else "") + ALLOW_HELP)
    for role in ("status", "stop"):
        p = sub.add_parser(role, help=HELP[role], description=HELP[role])
        p.add_argument("store_pos", nargs="?", metavar="STORE", help="the run's store: a folder or hf://buckets/...")
        p.add_argument("--store", default=None, help="the same as STORE")
        if role == "status":
            p.add_argument("--json", action="store_true", help="status.json instead of status.md")
    p = sub.add_parser("plan", help=HELP["plan"], description=HELP["plan"], epilog=PLAN_EPILOG,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("words", nargs="+", metavar="[STORE] set PATH=VALUE... | [STORE] list")
    p.add_argument("--store", default=None, help="the run's store, instead of STORE")
    p.add_argument("--by", default=None, help="who asks (default user@host)")
    p.add_argument("--before-start", action="store_true",
                   help="the run has not started yet: accept a store without config.yml or control.json (a folder "
                        "is created); without it such a STORE is refused as a likely typo")
    return ap


def _sigterm_exits() -> None:
    """SIGTERM raises SystemExit, so a role's cleanup (and gorge's child kill) runs."""
    def handler(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, handler)


def main(argv=None, prog: str | None = None) -> int:
    """`prog` names the command in usage and help (`dz` passes "dz")."""
    a = (_parser(prog) if prog else _parser()).parse_args(argv)
    from draftzero.loop import control as ctl
    from draftzero.loop.store import open_store

    if a.role in ("status", "stop", "plan"):
        return _store_command(a, ctl, open_store)

    from draftzero import config
    cfg = config.load(a.config)
    store_spec = a.store or str(ctl.get(cfg, "store"))
    run = ctl.get(cfg, "run")
    if a.role == "local":
        from draftzero.loop import local
        wd = a.work_dir or Path("runs") / run / "work"
        return local.run(a.config, store_spec, wd, machine=a.machine or "local", actors=a.actors,
                         max_hours=a.max_hours, poll_s=a.poll, log=_log, allow_config_change=a.allow_config_change,
                         player=bool(ctl.get(cfg, "player.learn", True)),
                         builder=bool(ctl.get(cfg, "builder.learn", False)), builder_actors=a.builder_actors)

    _sigterm_exits()
    store = open_store(store_spec)
    from draftzero.loop.store import HFBucketStore
    remote = isinstance(store, HFBucketStore)
    poll = a.poll if a.poll is not None else (60.0 if remote else 30.0)
    if a.role == "controller":
        from draftzero.loop.controller import Controller
        every = a.status_every if a.status_every is not None else (300.0 if remote else 60.0)
        controller = Controller(store, cfg, log=_log, status_every_s=every, hb_cache_s=120.0 if remote else 0.0,
                                tracker=_tracker(cfg), allow_config_change=a.allow_config_change)
        return controller.run(poll_s=poll, once=a.once)
    machine = ctl.check_name(a.machine or (_host() if a.role == "actor" else f"{_host()}-{a.role}"))
    wd = a.work_dir or Path("runs") / run / "work" / f"{a.role}-{machine}"
    if a.role in BUILDER_ROLES:
        return _builder_role(a, cfg, store, machine, wd, poll)
    if a.role == "actor":
        from draftzero.loop.actor import Actor
        return Actor(store, cfg, machine, wd, log=_log, poll_s=poll).run(max_chunks=a.max_chunks)
    if a.role == "evaluator":
        from draftzero.loop.evaluator import Evaluator
        return Evaluator(store, cfg, machine, wd, log=_log, poll_s=poll).run()
    from draftzero.loop.learner import LeaseHeld, Learner
    learner = Learner(store, cfg, machine, wd, log=_log, poll_s=poll)
    while True:
        try:
            return learner.run()
        except LeaseHeld as e:
            _log(f"learner: {e}; waiting")
            c = ctl.read(store)
            if c and c.get("stop"):
                return 0
            time.sleep(poll)


def _builder_role(a, cfg, store, machine: str, wd: Path, poll: float) -> int:
    """builder-actor, builder-learner or builder-evaluator (draftzero.loop.builder); refused without builder.learn."""
    from draftzero.loop import builder as bld
    from draftzero.loop import control as ctl
    if not bld.enabled(cfg):
        print(f"{a.role}: the config's builder.learn is false: this run has no deck builder to train", file=sys.stderr)
        return 2
    if a.role == "builder-actor":
        return bld.BuilderActor(store, cfg, machine, wd, log=_log, poll_s=poll).run(max_chunks=a.max_chunks)
    if a.role == "builder-evaluator":
        return bld.BuilderEvaluator(store, cfg, machine, wd, log=_log, poll_s=poll).run()
    from draftzero.loop.learner import LeaseHeld
    learner = bld.BuilderLearner(store, cfg, machine, wd, log=_log, poll_s=poll)
    while True:
        try:
            return learner.run()
        except LeaseHeld as e:
            _log(f"builder learner: {e}; waiting")
            c = ctl.read(store)
            if c and c.get("stop"):
                return 0
            time.sleep(poll)


def _tracker(cfg):
    """draftzero.tracking.Tracker when tracking.trackio is on (None otherwise, or if it can't start)."""
    from draftzero.loop import control as ctl
    if not ctl.get(cfg, "tracking.trackio", False):
        return None
    try:
        from draftzero.tracking import Tracker
        return Tracker(cfg)
    except Exception as e:                               # noqa: BLE001 - tracking is a view; the run goes on
        _log(f"controller: no tracker ({type(e).__name__}: {e})")
        return None


def _store_command(a, ctl, open_store) -> int:
    words = list(getattr(a, "words", []) or [])
    pos = getattr(a, "store_pos", None)
    if a.role == "plan" and words and words[0] not in ("set", "list"):
        pos, words = words[0], words[1:]
    spec = a.store or pos
    if not spec:
        print(f"{a.role}: which store? (STORE or --store)", file=sys.stderr)
        return 2
    run_files, early, remote = (ctl.CONFIG, ctl.CONTROL), getattr(a, "before_start", False), "://" in str(spec)
    if not (early or remote or any((Path(spec) / f).is_file() for f in run_files)):
        return _not_a_store(a.role, spec)                # before open_store, which would create the folder
    store = open_store(spec)
    if not early and remote and not any(store.exists(f) for f in run_files):
        return _not_a_store(a.role, spec)
    if a.role == "status":
        b = store.read_bytes(ctl.STATUS if a.json else ctl.STATUS_MD)
        print(b.decode() if b else "no status yet (is the controller running?)")
        return 0
    if a.role == "stop":
        store.put(ctl.STOP, f"stop requested {ctl.now_iso()}\n".encode(), "stop requested")
        print("stop requested: the controller stops the run at its next tick")
        return 0
    if not words or words[0] not in ("set", "list"):
        print("plan: want `set PATH=VALUE ...` or `list`", file=sys.stderr)
        return 2
    if words[0] == "list":
        for _, r in ctl.pending_requests(store):
            print(f"pending   {r['path']} = {r['value']!r}  (by {r.get('by')}, {r.get('time')})")
        for p in store.list(ctl.REJECTED):
            r = store.read_json(p) or {}
            print(f"rejected  {r.get('path')} = {r.get('value')!r}: {r.get('error')}")
        for e in ctl.read_events(store):
            print(f"applied   {e['path']}: {e.get('old')!r} -> {e.get('new')!r} from version {e.get('at_version')} "
                  f"({e.get('time')}, by {e.get('by')})")
        return 0
    changes = []
    for w in words[1:]:
        if "=" not in w:
            print(f"plan set: {w!r} is not PATH=VALUE", file=sys.stderr)
            return 2
        k, _, v = w.partition("=")
        changes.append((k.strip(), ctl.parse_value(v)))
    if not changes:
        print("plan set: nothing to set", file=sys.stderr)
        return 2
    err = _check_changes(store, ctl, changes)
    if err:
        print(f"plan set: {err}", file=sys.stderr)
        return 2
    import getpass
    by = a.by or f"{getpass.getuser()}@{_host()}"
    for k, v in changes:
        name = ctl.request_change(store, k, v, by)
        print(f"requested {k} = {v!r} ({name}); the controller applies it "
              + ("now" if ctl.immediate(k) else "at the next version"))
    return 0


def _not_a_store(role: str, spec: str) -> int:
    print(f"{role}: {spec} is not a run's store (it has no config.yml or control.json, which the controller writes "
          "when the run starts): a typo?" + (" To queue a change before the run starts, add --before-start."
                                             if role == "plan" else ""), file=sys.stderr)
    return 2


def _check_changes(store, ctl, changes) -> str | None:
    """Validate the changes, in order, against the run's config in effect: the store's config.yml with events.jsonl
    replayed (ctl.effective, as the controller and the roles see it), when that loads here (the controller checks
    again either way)."""
    raw = store.read_bytes(ctl.CONFIG)
    if raw is None:
        return None
    try:
        import tempfile
        from draftzero import config
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.yml"
            p.write_bytes(raw)
            cfg = config.load(p)
        cfg = ctl.effective(cfg, store)
    except Exception:                                    # noqa: BLE001 - e.g. data paths of another machine
        return None
    for k, v in changes:
        try:
            cfg = config.apply_change(cfg, k, v)
        except Exception as e:                           # noqa: BLE001
            return f"{k}={v!r}: {e}"
    return None


if __name__ == "__main__":
    sys.exit(main())
