"""The controller (contracts §4): exactly one, the only writer of control.json, events.jsonl and the status page.
At the start it writes config.yml and manifest.json (once per store; a restart keeps them); then each tick it

  1. reads versions/player/latest.json; a newer version is a version boundary: control["version"]["player"]
     moves to it, and every version passed (none is skipped when several land between ticks) with
     v % eval.every_versions == 0 gets one evaluation job per opponent in eval.opponents (evaluator.make_job);
  2. applies plan requests (plan/requests/*.json): each is checked at once with config.apply_change (a bad one
     moves to plan/rejected/ with the reason); a valid one applies at the next version boundary, except `paused`
     and budget.* which apply at once; each applied change appends {time, kind: "set", path, old, new, by,
     at_version} to events.jsonl, so config.effective(launch config, events) is the config at any version;
  3. keeps control["play"] (the actors' live settings) and control["paused"] current;
  4. tracks evaluation jobs: marks shards whose games are in, assigns the others to live evaluators (a heartbeat
     newer than live_s; a shard whose evaluator went quiet, or that is very late, moves to another), and closes a
     job once its summary.json exists;
  5. stops the run (control["stop"]) on stop.request or budget.max_hours at once; on budget.max_versions or
     budget.max_games it first pauses the actors and stops once every evaluation job is closed;
  6. writes status.md and status.json at most every status_every_s (and whenever control.json changes).

    python -m draftzero.loop.cli controller --config C --store S [--once]
"""
from __future__ import annotations

import copy
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path

from draftzero.loop import control as ctl
from draftzero.loop import evaluator as ev


def default_manifest(store, cfg, log=print) -> None:
    """manifest.json via draftzero.manifest; the engine part is left out when gorge isn't built here."""
    from draftzero import manifest
    gorge = None
    try:
        gorge = ctl.make_gorge(cfg)
    except Exception as e:                               # noqa: BLE001 - a controller may run without the engine
        log(f"controller: no engine here for the manifest ({type(e).__name__}: {e})")
    manifest.write(store, manifest.collect(cfg, gorge=gorge))


def _ts(iso: str | None) -> float | None:
    try:
        return datetime.fromisoformat(iso).timestamp() if iso else None
    except ValueError:
        return None


class Controller:
    def __init__(self, store, cfg, *, log=print, status_every_s: float = 60.0, live_s: float | None = None,
                 hb_cache_s: float = 120.0, write_manifest=None, tracker=None):
        """status_every_s: status page interval; live_s: a heartbeat older than this is a dead machine (default
        actor.chunk_secs + 15 min, as an evaluator is silent while it plays a shard); hb_cache_s: heartbeats are
        re-read at most this often (each is a Hub call or two on a bucket)."""
        self.store, self.base, self.cfg, self.log = store, cfg, cfg, log
        self.status_every_s, self.hb_cache_s = status_every_s, hb_cache_s
        chunk = float(ctl.get(cfg, "actor.chunk_secs") or 3600)
        self.live_s = live_s if live_s is not None else chunk + 900
        self.late_s = 2 * self.live_s
        self.write_manifest = write_manifest or (lambda s, c: default_manifest(s, c, log))
        self.tracker = tracker
        self.index = ctl.ShardIndex(store)
        self.summaries: dict[str, dict] = {}
        self.mirrored: set[str] = set()
        self.started = False
        self.last_status = float("-inf")
        self._hbs: tuple[float, list[dict]] | None = None

    # ---------------------------------------------------------------------------- start
    def start(self) -> dict:
        from draftzero import config
        self.store.ensure()
        stored = self.store.read_bytes(ctl.CONFIG)
        if stored is None:
            self.store.put(ctl.CONFIG, config.dump(self.base).encode(), "config.yml")
        else:
            self._check_same_run(stored)
        if not self.store.exists(ctl.MANIFEST):
            self.write_manifest(self.store, self.base)
        self.cfg = ctl.effective(self.base, self.store)
        c = ctl.read(self.store)
        if c is None:
            c = ctl.new_control(self.cfg)
            c["started"] = ctl.now_iso()
            ctl.write(self.store, c, "control: the run starts")
            self.log(f"controller: run {ctl.get(self.cfg, 'run')} started on {self.store.describe()}")
        elif c.get("stop"):
            # starting a controller on a stopped run resumes it (a stop requested while it ran still stops it)
            c["stop"] = False
            c.pop("stop_reason", None)
            self.store.delete([ctl.STOP], "resume")
            ctl.write(self.store, c, "control: the run resumes")
            self.log(f"controller: run {ctl.get(self.cfg, 'run')} resumes at version {ctl.player_version(c)}")
        self.started = True
        return c

    def _check_same_run(self, stored: bytes) -> None:
        from draftzero import config
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.yml"
            p.write_bytes(stored)
            try:
                old = config.load(p)
            except Exception as e:                       # noqa: BLE001
                self.log(f"controller: the store's config.yml doesn't load here ({e}); keeping it")
                return
        if ctl.get(old, "run") != ctl.get(self.base, "run"):
            raise ValueError(f"the store holds run {ctl.get(old, 'run')!r}, not {ctl.get(self.base, 'run')!r}")
        if config.config_hash(old) != config.config_hash(self.base):
            self.log("controller: this config differs from the store's config.yml (machine paths?); the store's "
                     "copy stays the record")

    # ---------------------------------------------------------------------------- a tick
    def tick(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        c = self.start() if not self.started else ctl.read(self.store)
        if c is None:                                    # deleted under us: start over
            self.started = False
            c = self.start()
        before = copy.deepcopy(c)
        latest = self.store.read_json(ctl.latest_path())
        v_old = ctl.player_version(c)
        v_new = None if latest is None else int(latest["version"])
        boundary = v_new is not None and (v_old is None or v_new > v_old)
        self.apply_requests(boundary, at_version=v_new if boundary else (v_old or 0))
        if boundary:
            for v in range(0 if v_old is None else v_old + 1, v_new + 1):
                self.open_evals(c, v, now)
            c["version"][ctl.COMPONENT] = v_new
            self.log(f"controller: actors now play version {v_new}")
        c["play"] = ctl.play_fields(self.cfg)
        self.update_evals(c, now)
        draining, why = self.budget(c, latest, now)
        c["paused"] = bool(ctl.get(self.cfg, "paused", False)) or draining
        if why and not c.get("stop"):
            self.log(f"controller: stopping the run ({why})")
            c["stop"], c["stop_reason"] = True, why
        changed = {k: v for k, v in c.items() if k != "updated"} != {k: v for k, v in before.items() if k != "updated"}
        if changed:
            ctl.write(self.store, c)
        self.status(c, latest, now, force=changed)
        self.mirror()
        return c

    # ---------------------------------------------------------------------------- plan changes
    def apply_requests(self, boundary: bool, at_version: int) -> list[dict]:
        from draftzero import config
        events, applied = [], []
        for path, r in ctl.pending_requests(self.store):
            key, value = r.get("path"), r.get("value")
            try:
                if not isinstance(key, str):
                    raise ValueError("a request needs a dotted config path")
                new_cfg = config.apply_change(self.cfg, key, value)
            except Exception as e:                       # noqa: BLE001 - ConfigError, bad JSON shapes...
                reason = f"{type(e).__name__}: {e}"
                self.log(f"controller: plan request {key}={value!r} rejected: {reason}")
                self.store.put_many({ctl.REJECTED + path[len(ctl.REQUESTS):]:
                                     json.dumps({**r, "error": reason, "rejected": ctl.now_iso()}, indent=1).encode()},
                                    "plan: rejected")
                self.store.delete([path], "plan: rejected")
                continue
            if not (boundary or ctl.immediate(key)):
                continue                                 # waits for the next version
            events.append({"time": ctl.now_iso(), "kind": "set", "path": key, "old": ctl.get(self.cfg, key),
                           "new": ctl.get(new_cfg, key), "by": r.get("by"), "at_version": at_version,
                           "requested": r.get("time")})
            self.cfg = new_cfg
            applied.append(path)
            self.log(f"controller: plan change {key}: {events[-1]['old']!r} -> {events[-1]['new']!r} "
                     f"(by {r.get('by')}, from version {at_version})")
        if events:
            ctl.append_events(self.store, events)
            self.store.delete(applied, "plan: applied")
        return events

    # ---------------------------------------------------------------------------- evaluations
    def open_evals(self, c: dict, version: int, now: float) -> None:
        every = int(ctl.get(self.cfg, "eval.every_versions") or 0)
        if not every or version % every:
            return
        have = {j["job"] for j in c["evals"]}
        for opp in ctl.get(self.cfg, "eval.opponents") or []:
            if not ev.opponent_valid(version, opp):
                continue
            job = ev.make_job(version, opp, self.cfg, now)
            if job["job"] in have or self.store.exists(ctl.eval_summary_path(job["job"])):
                continue
            c["evals"].append(job)
            self.log(f"controller: evaluation {job['job']} opened ({job['pairs']} pairs, {len(job['shards'])} shards)")

    def evaluators(self, now: float) -> list[str]:
        return sorted(hb["machine"] for hb in self.heartbeats(now) if hb.get("role") == "evaluator"
                      and hb.get("state") != "stopped" and now - float(hb.get("time", 0)) < self.live_s)

    def update_evals(self, c: dict, now: float) -> None:
        still, live = [], None
        for job in c["evals"]:
            files = set(self.store.list(f"evals/{job['job']}/"))
            if ctl.eval_summary_path(job["job"]) in files:
                s = self.store.read_json(ctl.eval_summary_path(job["job"])) or {}
                self.summaries[job["job"]] = s
                self.log(f"controller: evaluation {job['job']} closed: score {s.get('score')} over "
                         f"{s.get('n_pairs')} pairs")
                continue
            for sh in job["shards"]:
                sh["done"] = ctl.eval_games_path(job["job"], sh["id"]) in files
            if any(not sh["done"] for sh in job["shards"]):
                live = self.evaluators(now) if live is None else live
                self.assign(job, live, now)
            still.append(job)
        c["evals"] = still

    def assign(self, job: dict, live: list[str], now: float) -> None:
        """Give each shard without a live, timely evaluator to the least-loaded live one (another, if possible)."""
        if not live:
            return
        load = {m: 0 for m in live}
        for sh in job["shards"]:
            if not sh["done"] and sh.get("machine") in load and sh.get("assigned") and now - sh["assigned"] < self.late_s:
                load[sh["machine"]] += 1
        for sh in job["shards"]:
            if sh["done"]:
                continue
            stale = sh.get("machine") not in load or not sh.get("assigned") or now - sh["assigned"] >= self.late_s
            if stale:
                pool = [m for m in live if m != sh.get("machine")] or live
                m = min(pool, key=lambda x: (load[x], x))
                sh.update(machine=m, assigned=now)
                load[m] += 1

    # ---------------------------------------------------------------------------- budget and stopping
    def budget(self, c: dict, latest: dict | None, now: float) -> tuple[bool, str | None]:
        """(drain: pause the actors, stop: the reason or None)."""
        if self.store.exists(ctl.STOP):
            return True, "stop.request"
        b = ctl.as_dict(ctl.get(self.cfg, "budget"))
        started = _ts(c.get("started"))
        if b.get("max_hours") is not None and started is not None and now - started >= float(b["max_hours"]) * 3600:
            return True, f"budget.max_hours {b['max_hours']}"
        drain = None
        if b.get("max_versions") is not None and latest is not None and int(latest["version"]) >= int(b["max_versions"]):
            drain = f"budget.max_versions {b['max_versions']}"
        if b.get("max_games") is not None:
            self.index.refresh()
            if self.index.games() >= int(b["max_games"]):
                drain = drain or f"budget.max_games {b['max_games']}"
        if drain and not c["evals"]:
            return True, drain
        return bool(drain), None

    # ---------------------------------------------------------------------------- status
    def heartbeats(self, now: float) -> list[dict]:
        """Every heartbeat, re-read at most every hb_cache_s."""
        if self._hbs is None or not 0 <= now - self._hbs[0] < self.hb_cache_s:
            out = [hb for hb in (self.store.read_json(p) for p in self.store.list("heartbeats/")) if hb]
            self._hbs = (now, sorted(out, key=lambda h: (h.get("role", ""), h.get("machine", ""))))
        return self._hbs[1]

    def closed_evals(self) -> list[dict]:
        for p in self.store.list("evals/"):
            if p.endswith("/summary.json"):
                job = p.split("/")[1]
                if job not in self.summaries:
                    self.summaries[job] = self.store.read_json(p) or {}
        return [self.summaries[j] for j in sorted(self.summaries)]

    def status(self, c: dict, latest: dict | None, now: float, force: bool = False) -> dict | None:
        if not force and now - self.last_status < self.status_every_s:
            return None
        self.last_status = now
        v = ctl.player_version(c)
        latest_meta = self.store.read_json(ctl.version_file(latest["version"], "meta.json")) if latest else None
        st = {"run": ctl.get(self.cfg, "run"), "mode": ctl.get(self.cfg, "mode", "async"), "time": now,
              "iso": ctl.now_iso(now), "store": self.store.describe(), "version": v,
              "latest": latest_meta and {k: latest_meta.get(k) for k in
                                         ("version", "parent", "net", "positions", "games", "new_positions", "metrics",
                                          "time", "train_s")},
              "stop": c.get("stop"), "stop_reason": c.get("stop_reason"), "paused": c.get("paused"),
              "started": c.get("started"), "budget": ctl.as_dict(ctl.get(self.cfg, "budget")),
              "learner": self.store.read_json(ctl.LEARNER_STATUS) or {},
              "machines": [{**{k: h.get(k) for k in ("machine", "role", "state", "version", "games", "chunks",
                                                      "games_per_hour", "shards")},
                            "seen_s_ago": round(now - float(h.get("time", now)))} for h in self.heartbeats(now)],
              "open_evals": [{**{k: j[k] for k in ("job", "version", "opponent", "pairs")},
                              "shards_done": sum(s["done"] for s in j["shards"]), "shards": len(j["shards"])}
                             for j in c["evals"]],
              "closed_evals": [{k: s.get(k) for k in ("job", "version", "opponent", "score", "se", "lo", "hi",
                                                      "n_pairs", "n_games", "sprt", "time")} for s in self.closed_evals()],
              "pending_requests": [r for _, r in ctl.pending_requests(self.store)],
              "events": ctl.read_events(self.store)[-10:]}
        self.store.put_many({ctl.STATUS: json.dumps(st, indent=1, sort_keys=True, default=str).encode(),
                             ctl.STATUS_MD: status_md(st).encode()}, "status")
        return st

    def mirror(self) -> None:
        """New metrics files to the tracker (when there is one)."""
        if self.tracker is None:
            return
        lines = []
        for p in self.store.list("metrics/"):
            if p.endswith(".jsonl") and p not in self.mirrored:
                b = self.store.read_bytes(p) or b""
                lines += [json.loads(x) for x in b.decode().splitlines() if x.strip()]
                self.mirrored.add(p)
        if lines:
            self.tracker.mirror(lines)

    def run(self, poll_s: float = 30.0, once: bool = False) -> int:
        while True:
            c = self.tick()
            if once:
                return 0
            if c.get("stop"):
                self.status(c, self.store.read_json(ctl.latest_path()), time.time(), force=True)
                return 0
            time.sleep(poll_s)


def _fmt(x, nd: int = 3) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def status_md(st: dict) -> str:
    """status.json as a short page (plain markdown)."""
    out = [f"# Run {st['run']}", "", f"*Updated {st['iso']}.* {st['mode']} mode, {st['store']}.", ""]
    state = " (stopping: " + str(st.get("stop_reason")) + ")" if st.get("stop") else (" (paused)" if st.get("paused") else "")
    out.append(f"- **Version played:** {_fmt(st['version'])}{state}")
    b = st.get("budget") or {}
    if any(v is not None for v in b.values()):
        out.append("- **Budget:** " + ", ".join(f"{k} {v}" for k, v in b.items() if v is not None)
                   + (f"; started {st['started']}" if st.get("started") else ""))
    lr = st.get("learner") or {}
    if lr:
        out.append(f"- **Learner:** {lr.get('state')}; {lr.get('shards', 0)} shards, {lr.get('games', 0)} games, "
                   f"{lr.get('positions', 0)} positions ingested; {lr.get('new_positions', 0)} new since the last version")
    lm = st.get("latest") or {}
    if lm:
        m = lm.get("metrics") or {}
        out.append(f"- **Newest version:** v{lm.get('version')} ({'network' if lm.get('net') else 'no network'}; "
                   f"{lm.get('positions', 0)} positions in its window)"
                   + ("; " + ", ".join(f"{k} {_fmt(v, 4)}" for k, v in m.items()) if m else ""))
    out += ["", "## Machines", "", "| Machine | Role | State | Version | Games | Games/h | Seen (s ago) |",
            "|---|---|---|---|---|---|---|"]
    for m in st["machines"]:
        out.append(f"| {m['machine']} | {m['role']} | {_fmt(m['state'])} | {_fmt(m['version'])} | {_fmt(m['games'])} | "
                   f"{_fmt(m['games_per_hour'], 0)} | {m['seen_s_ago']} |")
    out += ["", "## Evaluations", "", "| Job | Opponent | Pairs | Score | 95% interval | SPRT | State |",
            "|---|---|---|---|---|---|---|"]
    for j in st["open_evals"]:
        out.append(f"| {j['job']} | {j['opponent']} | {j['pairs']} | | | | {j['shards_done']}/{j['shards']} shards |")
    for s in reversed(st["closed_evals"]):
        ci = f"{_fmt(s.get('lo'))}–{_fmt(s.get('hi'))}" if s.get("lo") is not None else ""
        out.append(f"| {s.get('job')} | {s.get('opponent')} | {_fmt(s.get('n_pairs'))} | {_fmt(s.get('score'))} | {ci} | "
                   f"{s.get('sprt') or 'undecided'} | done |")
    if st.get("pending_requests") or st.get("events"):
        out += ["", "## Plan", ""]
        for r in st.get("pending_requests") or []:
            out.append(f"- pending: `{r.get('path')} = {r.get('value')!r}` (by {r.get('by')}, {r.get('time')})")
        for e in st.get("events") or []:
            out.append(f"- {e.get('time')}: `{e.get('path')}` {e.get('old')!r} -> {e.get('new')!r} "
                       f"from version {e.get('at_version')} (by {e.get('by')})")
    return "\n".join(out) + "\n"
