"""The controller (docs/021 §3.1): the only writer of control.json. Each tick it

  1. moves control.json to the trainer's newest version (weights/latest.json);
  2. opens an evaluation job (evals.make_job) every eval.every_versions versions: the new version against the
     starting network (or the heuristic bot);
  3. assigns evaluation shards to live workers (a heartbeat within controller.live_s), reassigning a shard whose
     results are late, and keeps a worker's share of evaluation chunks under eval.max_share;
  4. scores each job's results with the sequential test and closes it (evals/<job>/summary.json) once the test
     decides or every shard is in;
  5. stops the run when stop.request appears in the store, or when the trainer is done and no evaluation is open;
  6. writes status.json and status.md: versions, machines, games an hour, the trainer's metrics, evaluations.

    dz selfplay controller --config configs/selfplay_gnn.yml --store <store>
"""
from __future__ import annotations

import copy
import json
import time

from draftzero.selfplay import control as ctl
from draftzero.selfplay import evals, records


class Controller:
    def __init__(self, store, cfg: dict, log=print):
        self.store, self.cfg, self.log = store, cfg, log
        self.last_status = 0.0

    def heartbeats(self) -> dict:
        out = {}
        for p in self.store.list("heartbeats/"):
            hb = self.store.read_json(p)
            if hb:
                out[hb["machine"]] = hb
        return out

    def tick(self, now: float | None = None) -> dict | None:
        now = time.time() if now is None else now
        ec = self.cfg["eval"]
        latest = self.store.read_json(ctl.LATEST)
        if latest is None:
            self._status(None, {}, now, note="waiting for the trainer to publish version 0", force=True)
            return None
        c = ctl.read(self.store)
        before = copy.deepcopy(c)
        if c is None:
            c = ctl.new_control(latest, self.cfg["play"])
        if latest["version"] > c["version"]:
            self.log(f"controller: version {latest['version']} published (was {c['version']})")
            c.update(version=latest["version"], weights=latest["path"], sha256=latest["sha256"])
            v = latest["version"]
            if ec["every_versions"] and v % int(ec["every_versions"]) == 0:
                b = 0 if ec["opponent"] == "start" else "heuristic"
                if not any(j["job"] == evals.job_id(v, b) for j in c["evals"]):
                    c["evals"].append(evals.make_job(v, b, ec, now))
                    self.log(f"controller: evaluation {evals.job_id(v, b)} opened")
        hbs = self.heartbeats()
        live = sorted(m for m, hb in hbs.items() if now - hb.get("time", 0) < self.cfg["controller"]["live_s"])
        still = []
        for job in c["evals"]:
            done = set(self.store.list(f"evals/{job['job']}/"))
            games = []
            for sh in job["shards"]:
                p = evals.result_path(job["job"], sh["id"])
                if p in done:
                    sh["done"] = True
                    games += self._read_results(p)
            sc = evals.score(games)
            test = evals.sprt(sc["wins"], sc["draws"], sc["losses"], ec["sprt_p0"], ec["sprt_p1"], ec["sprt_alpha"],
                              ec["sprt_beta"])
            job["result"] = {**sc, **test}
            if test["decision"] or all(sh["done"] for sh in job["shards"]):
                summary = {k: v for k, v in job.items() if k != "shards"}
                summary.update(closed=now, shards_done=sum(sh["done"] for sh in job["shards"]))
                self.store.write_json(f"evals/{job['job']}/summary.json", summary, f"evaluation {job['job']} closed")
                self.log(f"controller: evaluation {job['job']} closed: {json.dumps(job['result'])}")
                continue
            self._assign(job, live, now)
            still.append(job)
        c["evals"] = still
        trainer = self.store.read_json("learner/status.json") or {}
        if self.store.exists("stop.request") or (trainer.get("done") and not c["evals"]):
            if not c["stop"]:
                self.log("controller: stopping the run")
            c["stop"] = True
        if before is None or {k: v for k, v in c.items() if k != "updated"} != {k: v for k, v in before.items() if k != "updated"}:
            ctl.write(self.store, c)
        self._status(c, hbs, now, latest=latest, trainer=trainer)
        return c

    def _read_results(self, path: str) -> list[dict]:
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p = self.store.get(path, Path(d) / "r.jsonl.gz")
            return list(records.read(p)) if p else []

    def _assign(self, job: dict, live: list[str], now: float) -> None:
        ec = self.cfg["eval"]
        late = ec["late_factor"] * ec["shard_expected_s"]
        if not live:
            return
        load = {m: 0 for m in live}
        for sh in job["shards"]:
            if not sh["done"] and sh["machine"] in load and sh["assigned"] and now - sh["assigned"] < late:
                load[sh["machine"]] += 1
        for sh in job["shards"]:
            if sh["done"]:
                continue
            stale = sh["machine"] not in load or not sh["assigned"] or now - sh["assigned"] >= late
            if stale:
                # a late shard goes to another machine when there is one
                pool = [x for x in live if x != sh["machine"]] or live
                m = min(pool, key=lambda x: (load[x], x))
                sh.update(machine=m, assigned=now)
                load[m] += 1

    def _status(self, c, hbs: dict, now: float, note: str = "", latest: dict | None = None, trainer: dict | None = None,
                force: bool = False):
        if not force and now - self.last_status < self.cfg["controller"]["status_every_s"] and not (c or {}).get("stop"):
            return
        self.last_status = now
        machines = [{"machine": m, "version": hb.get("version"), "games": hb.get("games", 0),
                     "games_per_hour": hb.get("games_per_hour"), "job": hb.get("job"),
                     "seen_s_ago": round(now - hb.get("time", 0)), "workers": hb.get("workers"),
                     "device": hb.get("device"), "price_per_hour": hb.get("price_per_hour", 0.0)} for m, hb in sorted(hbs.items())]
        closed = []
        for p in self.store.list("evals/"):
            if p.endswith("summary.json"):
                s = self.store.read_json(p)
                if s:
                    closed.append(s)
        st = {"run": self.cfg["run"], "time": now, "note": note, "version": None if c is None else c["version"],
              "stop": None if c is None else c["stop"], "latest": latest, "trainer": trainer or {},
              "machines": machines, "open_evals": [] if c is None else [{k: v for k, v in j.items() if k != "shards"}
                                                                         for j in c["evals"]],
              "closed_evals": closed}
        self.store.put_many({ctl.STATUS: json.dumps(st, indent=1, sort_keys=True).encode(),
                             ctl.STATUS_MD: status_md(st).encode()}, "status")


def status_md(st: dict) -> str:
    """status.json as a short page (plain markdown)."""
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st["time"]))
    out = [f"# Self-play run {st['run']}", "", f"*Updated {t}.* {st.get('note') or ''}", ""]
    out.append(f"- **Version being played:** {st['version']}" + ("  (stopped)" if st.get("stop") else ""))
    tr = st.get("trainer") or {}
    if tr:
        out.append(f"- **Trainer:** {tr.get('state', '')}; games ingested {tr.get('games', 0)}, "
                   f"positions {tr.get('positions', 0)}")
        last = (st.get("latest") or {}).get("metrics") or {}
        if last:
            out.append(f"- **Last update:** " + ", ".join(f"{k} {v}" for k, v in last.items()))
    out += ["", "| Machine | Version | Games | Games/h | Job | Seen (s ago) |", "|---|---|---|---|---|---|"]
    for m in st["machines"]:
        out.append(f"| {m['machine']} | {m['version']} | {m['games']} | {m['games_per_hour']} | {m['job']} | {m['seen_s_ago']} |")
    out += ["", "| Evaluation | Games | Score | LLR | Decision |", "|---|---|---|---|---|"]
    for j in st["open_evals"] + st["closed_evals"]:
        r = j.get("result") or {}
        is_open = j in st["open_evals"]
        out.append(f"| {j['job']}{'' if is_open else ' (closed)'} | {r.get('games', 0)} | {r.get('score')} | "
                   f"{r.get('llr')} | {r.get('decision') or ('running' if is_open else 'undecided')} |")
    return "\n".join(out) + "\n"
