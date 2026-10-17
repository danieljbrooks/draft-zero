"""The controller (contracts §4): exactly one, the only writer of control.json, events.jsonl and the status page.
At the start it writes config.yml and manifest.json (once per store; a restart keeps them); then each tick it

  1. reads versions/player/latest.json; a newer version is a version boundary: control["version"]["player"]
     moves to it, and every version passed (none is skipped when several land between ticks) with
     v % eval.every_versions == 0 gets one evaluation job per opponent in eval.opponents (evaluator.make_job);
  2. applies plan requests (plan/requests/*.json, in the order made): each is checked at once with
     config.apply_change on the config with the earlier requests applied, waiting ones too (a bad one
     moves to plan/rejected/ with the reason); a valid one applies at the next version boundary, except `paused`
     and budget (budget.*) which apply at once; each applied change appends {time, kind: "set", path, old, new,
     by, at_version} to events.jsonl (values as JSON: a whole section is a mapping), so
     config.effective(launch config, events) is the config at any version;
  3. keeps control["play"] (the actors' live settings) and control["paused"] current;
  4. tracks evaluation jobs: marks shards whose games are in, assigns the others to live evaluators (a heartbeat
     newer than live_s; a shard whose evaluator went quiet, or that is very late, moves to another), and closes a
     job once its summary.json exists;
  5. stops the run (control["stop"]) on stop.request or budget.max_hours at once; on budget.max_versions or
     budget.max_games it first pauses the actors and stops once every evaluation job is closed;
  6. writes status.md and status.json at most every status_every_s (and whenever control.json changes).

With builder.learn (draftzero.loop.builder) it also moves control["version"]["builder"] to each new
versions/builder/latest.json, opens a builder evaluation job (control["builder_evals"]) for every builder version
v >= 1 with v % builder.eval.every_versions == 0 and assigns its shards to live builder evaluators, and counts
builder versions against budget.max_versions (the run drains once every learning component has reached it). In a
run where only the builder learns (player.learn false) its versions are the boundaries plan changes wait for.
With builder.learn off none of this happens and control.json has no builder fields.

    python -m draftzero.loop.cli controller --config C --store S [--once]
"""
from __future__ import annotations

import copy
import json
import time
from datetime import datetime

from draftzero.loop import builder as bld
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
                 hb_cache_s: float = 120.0, write_manifest=None, tracker=None, allow_config_change: bool = False):
        """status_every_s: status page interval; live_s: a heartbeat older than this is a dead machine (default
        actor.chunk_secs + 15 min; roles heartbeat every 5 minutes, while they play too); hb_cache_s: heartbeats
        are re-read at most this often (each is a Hub call or two on a bucket); allow_config_change: on a store
        that has a run, `cfg` may change the run's config (see _rebase), recorded in events.jsonl."""
        self.store, self.base, self.cfg, self.log = store, cfg, cfg, log
        self.allow_config_change = allow_config_change
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
            self._rebase(stored)
        if not self.store.exists(ctl.MANIFEST):
            self.write_manifest(self.store, self.base)
        self.cfg = ctl.effective(self.base, self.store)
        c = ctl.read(self.store)
        if c is None:
            c = ctl.new_control(self.cfg)
            c["started"] = ctl.now_iso()
            self._builder_fields(c)
            ctl.write(self.store, c, "control: the run starts")
            self.log(f"controller: run {ctl.get(self.cfg, 'run')} started on {self.store.describe()}")
        elif self._builder_fields(c):
            ctl.write(self.store, c, "control: the builder joins the run")
        elif c.get("stop"):
            # starting a controller on a stopped run resumes it (a stop requested while it ran still stops it)
            c["stop"] = False
            c.pop("stop_reason", None)
            self.store.delete([ctl.STOP], "resume")
            ctl.write(self.store, c, "control: the run resumes")
            self.log(f"controller: run {ctl.get(self.cfg, 'run')} resumes at version {ctl.player_version(c)}")
        self.started = True
        return c

    def _builder_fields(self, c: dict) -> bool:
        """With builder.learn: control["version"]["builder"] and control["builder_evals"], added if missing (True
        when it added them)."""
        if not bld.enabled(self.cfg) or (bld.COMPONENT in c.get("version", {}) and bld.EVALS in c):
            return False
        c.setdefault("version", {}).setdefault(bld.COMPONENT, None)
        c.setdefault(bld.EVALS, [])
        return True

    def _rebase(self, stored: bytes) -> None:
        """A restart: the run's config is the store's config.yml (plus events.jsonl); this controller's config
        supplies only the machine paths (config.MACHINE). A field it changes otherwise (a fixed one, or a live one
        other than by `plan set`) is refused (ValueError), unless allow_config_change, which records the changes
        in events.jsonl ("set" events at the current version), so the record stays the config in effect."""
        from draftzero import config
        try:
            old = config.loads(stored.decode())
        except Exception as e:                           # noqa: BLE001
            self.log(f"controller: the store's config.yml doesn't load here ({e}); running on this config")
            return
        given = self.base
        if ctl.get(old, "run") != ctl.get(given, "run"):
            raise ValueError(f"the store holds run {ctl.get(old, 'run')!r}, not {ctl.get(given, 'run')!r}")
        d = config.to_dict(old)
        for path in config.MACHINE:
            *head, last = path.split(".")
            node = d
            for k in head:
                node = node[k]
            node[last] = config.get(given, path)
        self.base = config.from_dict(d)
        launch = config.flatten(self.base)
        changes = {p: (a, b) for p, (a, b) in config.diff(ctl.effective(self.base, self.store), given).items()
                   if p not in config.MACHINE and b != launch[p]}       # the launch value: a plan change stays
        if not changes:
            return
        what = "; ".join(f"{p} {a!r} -> {b!r}" for p, (a, b) in sorted(changes.items()))
        if not self.allow_config_change:
            raise ValueError(f"this config changes the run's config (the store's config.yml and events.jsonl): "
                             f"{what}. Change live fields with `dz plan set`; to change these anyway, add "
                             "--allow-config-change (`dz run` or the controller; the changes are recorded in "
                             "events.jsonl)")
        c = ctl.read(self.store)
        builder_only = bld.enabled(self.base) and not bld.player_learns(self.base)   # its versions are the builder's
        at = (ctl.player_version(c, bld.COMPONENT) if builder_only else ctl.player_version(c)) or 0
        ctl.append_events(self.store, [{"time": ctl.now_iso(), "kind": "set", "path": p, "old": ctl.plain(a),
                                        "new": ctl.plain(b), "by": "controller restart", "at_version": at}
                                       for p, (a, b) in sorted(changes.items())])
        self.log(f"controller: config changed at restart, from version {at}: {what}")

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
        builder = bld.enabled(self.cfg)
        blatest = self.store.read_json(ctl.latest_path(bld.COMPONENT)) if builder else None
        if builder and not bld.player_learns(self.cfg):     # a builder-only run: plan changes wait for its versions
            b_old, b_new = ctl.player_version(c, bld.COMPONENT), None if blatest is None else int(blatest["version"])
            b_boundary = b_new is not None and (b_old is None or b_new > b_old)
            self.apply_requests(b_boundary, at_version=b_new if b_boundary else (b_old or 0))
        else:
            self.apply_requests(boundary, at_version=v_new if boundary else (v_old or 0))
        if boundary:
            for v in range(0 if v_old is None else v_old + 1, v_new + 1):
                self.open_evals(c, v, now)
            c["version"][ctl.COMPONENT] = v_new
            self.log(f"controller: actors now play version {v_new}")
        if builder:
            self.builder_versions(c, blatest, now)
        c["play"] = ctl.play_fields(self.cfg)
        self.update_evals(c, now)
        if builder:
            self.update_evals(c, now, key=bld.EVALS, role=bld.BuilderEvaluator.ROLE)
        draining, why = self.budget(c, latest, now, blatest)
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
        """Each request is checked against the config with the requests before it applied, waiting ones too (as
        `plan set` checks a multi-field change), and applied now if it is due."""
        from draftzero import config
        events, applied = [], []
        pending = self.cfg                               # self.cfg plus the requests accepted so far in this pass
        for path, r in ctl.pending_requests(self.store):
            key, value = r.get("path"), r.get("value")
            try:
                if not isinstance(key, str):
                    raise ValueError("a request needs a dotted config path")
                chained = config.apply_change(pending, key, value)
                due = boundary or ctl.immediate(key)
                new_cfg = config.apply_change(self.cfg, key, value) if due and pending is not self.cfg else chained
            except Exception as e:                       # noqa: BLE001 - ConfigError, bad JSON shapes...
                reason = f"{type(e).__name__}: {e}"
                self.log(f"controller: plan request {key}={value!r} rejected: {reason}")
                self.store.put_many({ctl.REJECTED + path[len(ctl.REQUESTS):]:
                                     json.dumps({**r, "error": reason, "rejected": ctl.now_iso()}, indent=1).encode()},
                                    "plan: rejected")
                self.store.delete([path], "plan: rejected")
                continue
            pending = chained
            if not due:
                continue                                 # waits for the next version
            events.append({"time": ctl.now_iso(), "kind": "set", "path": key, "old": ctl.plain(ctl.get(self.cfg, key)),
                           "new": ctl.plain(ctl.get(new_cfg, key)), "by": r.get("by"), "at_version": at_version,
                           "requested": r.get("time")})       # a section as a mapping, so config.effective replays it
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
            have.add(job["job"])                         # an opponent listed twice is one job
            c["evals"].append(job)
            self.log(f"controller: evaluation {job['job']} opened ({job['pairs']} pairs, {len(job['shards'])} shards)")

    def builder_versions(self, c: dict, blatest: dict | None, now: float) -> None:
        """A new builder version: an evaluation job for each version passed (see open_builder_evals), then
        control["version"]["builder"] moves to it."""
        self._builder_fields(c)
        if blatest is None:
            return
        old, new = ctl.player_version(c, bld.COMPONENT), int(blatest["version"])
        if old is not None and new <= old:
            return
        for v in range(0 if old is None else old + 1, new + 1):
            self.open_builder_evals(c, v, now)
        c["version"][bld.COMPONENT] = new
        self.log(f"controller: builder actors now build with builder version {new}")

    def open_builder_evals(self, c: dict, version: int, now: float) -> None:
        """Builder version `version` (>= 1, a multiple of builder.eval.every_versions) against builder.eval's
        opponent, piloted by the player version actors play now when the pilot follows the player."""
        every = int(ctl.get(self.cfg, "builder.eval.every_versions") or 0)
        if version < 1 or not every or version % every:
            return
        pv = None
        if bld.player_learns(self.cfg) and not ctl.get(self.cfg, "builder.pilot"):
            pv = ctl.player_version(c)
        job = bld.make_job(version, self.cfg, player_version=pv, now=now)
        if any(j["job"] == job["job"] for j in c[bld.EVALS]) or self.store.exists(ctl.eval_summary_path(job["job"])):
            return
        c[bld.EVALS].append(job)
        self.log(f"controller: builder evaluation {job['job']} opened ({job['pools']} pools, {len(job['shards'])} "
                 "shards)")

    def evaluators(self, now: float, role: str = "evaluator") -> list[str]:
        return sorted(hb["machine"] for hb in self.heartbeats(now) if hb.get("role") == role
                      and hb.get("state") != "stopped" and now - float(hb.get("time", 0)) < self.live_s)

    def update_evals(self, c: dict, now: float, key: str = "evals", role: str = "evaluator") -> None:
        """Close the jobs in c[key] whose summary.json exists; mark shards done; assign the others to live
        evaluators of `role` (the player's jobs: evals and evaluator; the builder's: builder_evals and
        builder-evaluator)."""
        still, live = [], None
        for job in c[key]:
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
                live = self.evaluators(now, role) if live is None else live
                self.assign(job, live, now)
            still.append(job)
        c[key] = still

    def assign(self, job: dict, live: list[str], now: float) -> None:
        """Give each shard without a live, timely evaluator to the least-loaded live one (another, if possible)."""
        if not live:
            return
        load = {m: 0 for m in live}
        for sh in job["shards"]:
            timely = sh.get("assigned") and now - sh["assigned"] < self.late_s
            if not sh["done"] and sh.get("machine") in load and timely:
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
    def budget(self, c: dict, latest: dict | None, now: float, blatest: dict | None = None
               ) -> tuple[bool, str | None]:
        """(drain: pause the actors, stop: the reason or None). budget.max_versions drains once every learning
        component (the player unless player.learn is false; the builder with builder.learn) has reached it."""
        if self.store.exists(ctl.STOP):
            return True, "stop.request"
        b = ctl.as_dict(ctl.get(self.cfg, "budget"))
        started = _ts(c.get("started"))
        if b.get("max_hours") is not None and started is not None and now - started >= float(b["max_hours"]) * 3600:
            return True, f"budget.max_hours {b['max_hours']}"
        drain = None
        mv = b.get("max_versions")
        if mv is not None:
            heads = [latest] if bld.player_learns(self.cfg) or not bld.enabled(self.cfg) else []
            heads += [blatest] if bld.enabled(self.cfg) else []
            if heads and all(h is not None and int(h["version"]) >= int(mv) for h in heads):
                drain = f"budget.max_versions {b['max_versions']}"
        if b.get("max_games") is not None:
            self.index.refresh()
            if self.index.games() >= int(b["max_games"]):
                drain = drain or f"budget.max_games {b['max_games']}"
        if drain and not c["evals"] and not c.get(bld.EVALS):
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
                                                      "n_pairs", "n_games", "sprt", "time")}
                             for s in self.closed_evals()],
              "pending_requests": [r for _, r in ctl.pending_requests(self.store)],
              "events": ctl.read_events(self.store)[-10:]}
        if bld.enabled(self.cfg):
            st["builder"] = self.builder_status(c)
            st["open_evals"] += [{**{k: j[k] for k in ("job", "version", "opponent")}, "pairs": j["pools"] * j["pairs"],
                                  "shards_done": sum(s["done"] for s in j["shards"]), "shards": len(j["shards"])}
                                 for j in c.get(bld.EVALS) or []]
        self.store.put_many({ctl.STATUS: json.dumps(st, indent=1, sort_keys=True, default=str).encode(),
                             ctl.STATUS_MD: status_md(st).encode()}, "status")
        return st

    def builder_status(self, c: dict) -> dict:
        bl = self.store.read_json(ctl.latest_path(bld.COMPONENT))
        meta = self.store.read_json(ctl.version_file(bl["version"], "meta.json", bld.COMPONENT)) if bl else None
        return {"version": ctl.player_version(c, bld.COMPONENT),
                "latest": meta and {k: meta.get(k) for k in ("version", "parent", "net", "games", "decided", "decks",
                                                             "lands", "metrics", "time", "train_s")},
                "learner": self.store.read_json(bld.LEARNER_STATUS) or {}}

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
                time.sleep(min(poll_s, 30.0))            # the roles see the stop and say so in their heartbeats
                self._hbs = None
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
    state = f" (stopping: {st.get('stop_reason')})" if st.get("stop") else (" (paused)" if st.get("paused") else "")
    out.append(f"- **Version played:** {_fmt(st['version'])}{state}")
    b = st.get("budget") or {}
    if any(v is not None for v in b.values()):
        out.append("- **Budget:** " + ", ".join(f"{k} {v}" for k, v in b.items() if v is not None)
                   + (f"; started {st['started']}" if st.get("started") else ""))
    lr = st.get("learner") or {}
    if lr:
        out.append(f"- **Learner:** {lr.get('state')}; {lr.get('shards', 0)} shards, {lr.get('games', 0)} games, "
                   f"{lr.get('positions', 0)} positions ingested; {lr.get('new_positions', 0)} new since the last "
                   "version")
    bs = st.get("builder")
    if bs:
        lb, bl = bs.get("latest") or {}, bs.get("learner") or {}
        m = lb.get("metrics") or {}
        out.append(f"- **Builder:** builds with version {_fmt(bs.get('version'))}"
                   + (f"; newest v{lb.get('version')} ({'trained' if lb.get('net') else 'untrained'}, "
                      f"{lb.get('decided', 0)} decided games in its window, lands {lb.get('lands')})" if lb else "")
                   + ("; held-out " + ", ".join(f"{k} {_fmt(v, 4)}" for k, v in m.items() if v is not None)
                      if any(v is not None for v in m.values()) else "")
                   + (f"; learner {bl.get('state')}, {bl.get('decided', 0)} decided games ingested" if bl else ""))
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
        out.append(f"| {s.get('job')} | {s.get('opponent')} | {_fmt(s.get('n_pairs'))} | {_fmt(s.get('score'))} | "
                   f"{ci} | "
                   f"{s.get('sprt') or 'undecided'} | done |")
    if st.get("pending_requests") or st.get("events"):
        out += ["", "## Plan", ""]
        for r in st.get("pending_requests") or []:
            out.append(f"- pending: `{r.get('path')} = {r.get('value')!r}` (by {r.get('by')}, {r.get('time')})")
        for e in st.get("events") or []:
            out.append(f"- {e.get('time')}: `{e.get('path')}` {e.get('old')!r} -> {e.get('new')!r} "
                       f"from version {e.get('at_version')} (by {e.get('by')})")
    return "\n".join(out) + "\n"
