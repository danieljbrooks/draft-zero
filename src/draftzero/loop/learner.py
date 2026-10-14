"""The learner (contracts §4): exactly one per learned model, holding a lease. Each loop it

  1. holds the lease (learner/lease.json, renewed every LEASE_S / 3, also while training): a second learner waits
     until it expires;
  2. on a fresh store, publishes version 0: player.init (a .pt path or hf://...) when given, else a version with no
     network ("net": false, meta only), which actors play as search on its own priors;
  3. ingests new shards: downloads each shard's corpus and packs it locally (Gorge.pack, game ids prefixed
     "<actor>-<seq>-" so the crc32 holdout differs across actors and chunks); learner/ingested.json lists them;
  4. trains when due: async, once min_new_positions recorded positions arrived since the last version; sync, once
     the store holds sync.games_per_version games of the newest version (exactly once per version). Training is
     draftzero.nets.train on the newest window_positions positions, warm-started from the newest version's model.pt
     when learner.warm_start (else from random init with player.arch/config), for learner.epochs passes or, with
     steps_per_position, ceil(steps_per_position x new positions / batch) steps;
  5. exports model.dzgw (mlp), publishes versions/player/v{N+1} (model.pt, model.dzgw, meta.json) and latest.json
     last, with learn.* metrics from the run's final holdout evaluation (curves.jsonl);
  6. deletes local packs that fell out of the window. It stops on control "stop" or budget.max_versions.

    python -m draftzero.loop.cli learner --config C --store S [--machine M] [--work-dir DIR]
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

from draftzero.loop import control as ctl


class LeaseHeld(RuntimeError):
    pass


def default_train(argv: list[str], threads: int | None = None) -> dict:
    """draftzero.nets.train.main(argv), in this process; its SystemExit (bad data) becomes a RuntimeError."""
    import torch
    from draftzero.nets import train
    if threads:
        torch.set_num_threads(int(threads))
    try:
        return train.main(argv) or {}
    except SystemExit as e:
        raise RuntimeError(f"draftzero.nets.train {' '.join(argv)}: {e}") from None


def default_export(pt: Path, dzgw: Path) -> None:
    from draftzero.nets import export
    rc = export.main([str(pt), str(dzgw)])
    if rc != 0:
        raise RuntimeError(f"draftzero.nets.export {pt} {dzgw} exited {rc}")


def curve_metrics(curves: Path) -> dict:
    """The last evaluation in a nets.train curves.jsonl: its holdout set (else the first validation set)."""
    last = None
    if curves.exists():
        for line in curves.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("type") == "eval":
                    last = rec
    if not last:
        return {}
    ev = last.get("eval") or {}
    res = ev.get("holdout") or (next(iter(ev.values())) if ev else {})
    return {"step": last.get("step"), **{k: v for k, v in res.items() if isinstance(v, (int, float)) or v is None}}


LEARN = {   # learn.<name> from the holdout evaluation, and the sample size it is over
    "policy_ce": "n_policy", "policy_ce_uniform": "n_policy", "target_entropy": "n_policy",
    "top1_visits": "n_policy", "value_logloss": "n_value", "value_base_logloss": "n_value", "value_auc": "n_value",
    "value_brier": "n_value",
}


class Learner:
    LEASE_S = 1800.0
    STATUS_EVERY_S = 60.0
    CORPUS_TRIES = 5            # a shard whose meta is in but corpus isn't (a bucket batch is not atomic)

    def __init__(self, store, cfg, machine: str, work_dir: Path, *, gorge=None, train=None, export=None, log=print,
                 poll_s: float = 30.0):
        self.store, self.base, self.cfg, self.log, self.poll_s = store, cfg, cfg, log, poll_s
        self.machine = ctl.check_name(machine)
        self.wd = Path(work_dir).resolve()
        self.wd.mkdir(parents=True, exist_ok=True)
        self._gorge = gorge
        self._train = train
        self._export = export or default_export
        self.holder = f"{machine}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.lease_renewed = float("-inf")
        self.ingested: list[dict] = []          # {shard, actor, seq, version, games, positions, time, used}
        self.local: dict[str, list[Path]] = {}  # shard -> its local pack dirs
        self.tries: dict[str, int] = {}
        self.index: ctl.ShardIndex | None = None
        self.versions_made = 0
        self.last_status, self.state = 0.0, None

    @property
    def gorge(self):
        if self._gorge is None:
            self._gorge = ctl.make_gorge(self.cfg)
        return self._gorge

    def c(self, dotted: str, default=None):
        return ctl.get(self.cfg, dotted, default)

    # ---------------------------------------------------------------------------- lease
    def lease(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        if now - self.lease_renewed < self.LEASE_S / 3:
            return
        cur = self.store.read_json(ctl.LEASE)
        if cur and cur.get("holder") != self.holder and cur.get("expires", 0) > now:
            raise LeaseHeld(f"learner {cur['holder']} holds the lease until {time.ctime(cur['expires'])}")
        self.store.write_json(ctl.LEASE, {"holder": self.holder, "machine": self.machine, "expires": now + self.LEASE_S,
                                          "time": now}, f"lease: {self.holder}")
        self.lease_renewed = now

    def release(self) -> None:
        cur = self.store.read_json(ctl.LEASE)
        if cur and cur.get("holder") == self.holder:
            self.store.write_json(ctl.LEASE, {**cur, "expires": time.time(), "released": True}, "lease released")

    @contextlib.contextmanager
    def _renewing(self):
        """Renew the lease from a thread while a long call (training) runs."""
        stop = threading.Event()

        def loop():
            while not stop.wait(self.LEASE_S / 3):
                try:
                    self.lease()
                except Exception as e:                  # noqa: BLE001
                    self.log(f"learner: lease renewal failed: {e}")
        t = threading.Thread(target=loop, daemon=True)
        t.start()
        try:
            yield
        finally:
            stop.set()
            t.join(timeout=5)

    # ---------------------------------------------------------------------------- versions
    def latest(self) -> dict | None:
        return self.store.read_json(ctl.latest_path())

    def bootstrap(self) -> dict:
        """latest.json, publishing version 0 first on a fresh store."""
        latest = self.latest()
        if latest is not None:
            return latest
        init = self.c("player.init")
        meta = {"version": 0, "parent": None, "net": False, "arch": self.c("player.arch", "mlp"), "sha256": {},
                "shards": [], "positions": 0, "games": 0, "metrics": {}, "time": ctl.now_iso(), "source": init}
        files: dict = {}
        if init:
            src = self._resolve(str(init))
            d = self.wd / "versions" / "v0000"
            d.mkdir(parents=True, exist_ok=True)
            pt = d / "model.pt"
            shutil.copyfile(src, pt)
            files[ctl.version_file(0, "model.pt")] = pt
            meta.update(net=True, sha256={"pt": ctl.sha256(pt)})
            if meta["arch"] == "mlp":
                self._export(pt, d / "model.dzgw")
                files[ctl.version_file(0, "model.dzgw")] = d / "model.dzgw"
                meta["sha256"]["dzgw"] = ctl.sha256(d / "model.dzgw")
        latest = {"version": 0, "path": ctl.version_dir(0)}
        files[ctl.version_file(0, "meta.json")] = json.dumps(meta, indent=1, sort_keys=True).encode()
        files[ctl.latest_path()] = json.dumps(latest, indent=1, sort_keys=True).encode()
        self.store.put_many(files, "version 0")
        self.log(f"learner: published version 0 ({'from ' + str(init) if init else 'no network: search alone'})")
        return latest

    @staticmethod
    def _resolve(uri: str) -> Path:
        if uri.startswith("hf://"):
            from draftzero import hub
            return Path(hub.resolve(uri))
        p = Path(uri)
        if not p.exists():
            raise FileNotFoundError(f"player.init {uri}: no such file")
        return p

    # ---------------------------------------------------------------------------- ingest
    def load_state(self) -> None:
        """learner/ingested.json from an earlier learner: its shards count as old (their packs are rebuilt when a
        window needs them)."""
        self.ingested = list(self.store.read_json(ctl.INGESTED) or [])
        known = {e["shard"]: {**e, "path": e["shard"]} for e in self.ingested}
        self.index = ctl.ShardIndex(self.store, known)
        if self.ingested:
            self.log(f"learner: {len(self.ingested)} shards ingested earlier, "
                     f"{sum(e['positions'] for e in self.ingested)} positions")

    def ingest(self) -> list[dict]:
        if self.index is None:
            self.load_state()
        waiting = [m for m in self.index.metas.values() if m["path"] in self.tries]
        new = []
        for m in waiting + self.index.refresh():
            e = self._ingest_one(m)
            if e is not None:
                new.append(e)
        if new:
            self.log(f"learner: ingested {len(new)} shards ({sum(e['games'] for e in new)} games, "
                     f"{sum(e['positions'] for e in new)} positions); {self.new_positions()} new positions since "
                     "the last version")
        return new

    def _ingest_one(self, m: dict) -> dict | None:
        e = {"shard": m["path"], "actor": m.get("actor"), "seq": int(m.get("seq", 0)), "version": m.get("version"),
             "games": int(m.get("games", 0)), "positions": 0, "time": m.get("time"), "used": None}
        if int(m.get("records", 0)) > 0:
            try:
                dirs = self._pack(e)
            except FileNotFoundError:
                n = self.tries[m["path"]] = self.tries.get(m["path"], 0) + 1
                if n < self.CORPUS_TRIES:
                    return None
                self.log(f"learner: {m['path']} has no corpus after {n} tries; ingested without positions")
                e["missing"] = True
                dirs = []
            e["positions"] = sum(self._pack_n(d) for d in dirs)
        self.tries.pop(m["path"], None)
        self.ingested.append(e)
        return e

    def _pack(self, e: dict) -> list[Path]:
        """Fetch a shard's corpus and pack it into the local cache; the pack dirs (also kept in self.local)."""
        name = f"{e['actor']}-{e['seq']:06d}"
        corpus = self.wd / "corpus" / f"{name}.jsonl.gz"
        if self.store.get(f"{e['shard']}/corpus.jsonl.gz", corpus) is None:
            raise FileNotFoundError(f"{e['shard']}/corpus.jsonl.gz")
        out = self.wd / "packs" / name
        shutil.rmtree(out, ignore_errors=True)
        try:
            dirs = [Path(d) for d in self.gorge.pack([corpus], out, game_prefix=f"{e['actor']}-{e['seq']}-")]
        finally:
            corpus.unlink(missing_ok=True)
        self.local[e["shard"]] = dirs
        return dirs

    @staticmethod
    def _pack_n(d: Path) -> int:
        return int(json.loads((Path(d) / "meta.json").read_text()).get("n", 0))

    def new_positions(self) -> int:
        return sum(e["positions"] for e in self.ingested if e.get("used") is None)

    # ---------------------------------------------------------------------------- when to train
    def due(self, latest: dict) -> str | None:
        v = int(latest["version"])
        mx = self.c("budget.max_versions")
        if mx is not None and v >= int(mx):
            return None
        if self.c("mode", "async") == "sync":
            g = int(self.c("sync.games_per_version"))
            have = sum(e["games"] for e in self.ingested if e.get("version") == v)
            if have >= g and any(e["positions"] for e in self.ingested if e.get("used") is None):
                return f"sync: {have} games of version {v}"
            return None
        new = self.new_positions()
        if new >= int(self.c("learner.min_new_positions")) and new > 0:
            return f"{new} new positions"
        return None

    def window(self) -> list[dict]:
        """The newest shards holding learner.window_positions positions (oldest first)."""
        if self.c("mode", "async") == "sync":
            key = lambda e: (e.get("version") or 0, e["seq"], e.get("actor") or "")          # noqa: E731
        else:
            key = lambda e: (e.get("version") or 0, str(e.get("time") or ""), e["seq"],    # noqa: E731
                             e.get("actor") or "")
        out, n, cap = [], 0, int(self.c("learner.window_positions"))
        for e in sorted((e for e in self.ingested if e["positions"] > 0), key=key, reverse=True):
            out.append(e)
            n += e["positions"]
            if n >= cap:
                break
        return out[::-1]

    # ---------------------------------------------------------------------------- train and publish
    def train_args(self, out: Path, packs: list[Path], init: Path | None, version: int,
                   new_positions: int) -> list[str]:
        batch = int(self.c("learner.batch"))
        argv = ["--arch", str(self.c("player.arch", "mlp")),
                "--config", json.dumps(ctl.as_dict(self.c("player.config"))),
                "--train", *[str(p) for p in packs], "--out", str(out), "--batch", str(batch),
                "--lr", repr(float(self.c("learner.lr"))), "--warmup", str(int(self.c("learner.warmup"))),
                "--policy-target", str(self.c("learner.policy_target", "visits")),
                "--value-blend", repr(float(self.c("learner.value_blend"))),
                "--device", str(self.c("learner.device", "cpu")), "--in-ram",
                "--seed", str((int(self.c("seed", 1)) * 1_000_003 + version) % 2**31)]
        spp = self.c("learner.steps_per_position")
        if spp:
            argv += ["--epochs", "1000000", "--max-steps", str(max(1, math.ceil(float(spp) * new_positions / batch)))]
        else:
            argv += ["--epochs", repr(float(self.c("learner.epochs")))]
        if init is not None:
            argv += ["--init", str(init)]
        return argv

    def train_and_publish(self, latest: dict, reason: str = "") -> dict:
        parent = int(latest["version"])
        v = parent + 1
        win = self.window()
        new = [e for e in self.ingested if e.get("used") is None and e["positions"] > 0]
        new_pos = sum(e["positions"] for e in new)
        packs = []
        for e in win:
            dirs = self.local.get(e["shard"])
            if not dirs or not all(Path(d).exists() for d in dirs):
                dirs = self._pack(e)
            packs += dirs
        init = None
        if self.c("learner.warm_start", True):
            pmeta, files = ctl.fetch_version(self.store, parent, self.wd / "versions" / f"v{parent:04d}", ("model.pt",))
            init = files.get("model.pt")
        out = self.wd / "train" / f"v{v:04d}"
        shutil.rmtree(out, ignore_errors=True)
        argv = self.train_args(out, packs, init, v, new_pos)
        self.log(f"learner: training version {v} ({reason}): {len(win)} shards, "
                 f"{sum(e['positions'] for e in win)} positions ({new_pos} new), "
                 + (f"warm start from v{parent}" if init else "random init"))
        self.status("training", force=True, training=v)
        t0 = time.time()
        with self._renewing():
            summary = (self._train or (lambda a: default_train(a, self.c("learner.threads"))))(argv) or {}
        train_s = time.time() - t0
        pt = out / "best.pt" if (out / "best.pt").exists() else out / "last.pt"
        if not pt.exists():
            raise RuntimeError(f"training version {v} wrote no checkpoint in {out}")
        m = curve_metrics(out / "curves.jsonl")
        total = m.get("total")
        if total is not None and not math.isfinite(float(total)):
            raise RuntimeError(f"version {v}: the holdout loss is {total}; not publishing")
        files: dict = {ctl.version_file(v, "model.pt"): pt}
        sha = {"pt": ctl.sha256(pt)}
        arch = self.c("player.arch", "mlp")
        if arch == "mlp":
            dzgw = out / "model.dzgw"
            self._export(pt, dzgw)
            files[ctl.version_file(v, "model.dzgw")] = dzgw
            sha["dzgw"] = ctl.sha256(dzgw)
        for e in new:
            e["used"] = v
        metrics = {k: m.get(k) for k in LEARN if k in m}
        meta = {"version": v, "parent": parent, "net": True, "arch": arch, "sha256": sha,
                "shards": [e["shard"] for e in win], "positions": sum(e["positions"] for e in win),
                "games": sum(e["games"] for e in win), "new_positions": new_pos, "new_shards": len(new),
                "metrics": metrics, "time": ctl.now_iso(), "train_s": round(train_s, 1),
                "init": f"v{parent:04d}" if init else None,
                "steps": summary.get("steps"), "argv": [a if not a.startswith(str(self.wd)) else
                                                        os.path.relpath(a, self.wd) for a in argv]}
        mpath = out / "metrics.jsonl"
        mlog = ctl.metrics_log(mpath, self.cfg, "learner", self.machine)
        for k, nk in LEARN.items():
            if m.get(k) is not None:
                mlog.log(f"learn.{k}", float(m[k]), version=v, step=m.get("step"), n=m.get(nk))
        mlog.log("data.window_positions", meta["positions"], version=v)
        mlog.log("data.new_positions", new_pos, version=v)
        mlog.log("learn.train_s", train_s, version=v, step=summary.get("steps"))
        if mpath.exists():
            files[ctl.metrics_path("learner", self.machine, v)] = mpath
        files[ctl.version_file(v, "meta.json")] = json.dumps(meta, indent=1, sort_keys=True).encode()
        files[ctl.INGESTED] = json.dumps(self.ingested, indent=1, sort_keys=True).encode()
        files[ctl.latest_path()] = json.dumps({"version": v, "path": ctl.version_dir(v)}, indent=1).encode()
        self.store.put_many(files, f"version {v}")
        self.versions_made += 1
        self.log(f"learner: published version {v} in {train_s:.0f} s: "
                 + ", ".join(f"{k} {m[k]:.4f}" for k in ("policy_ce", "value_logloss", "top1_visits", "value_auc")
                             if isinstance(m.get(k), (int, float))))
        mine = self.wd / "versions" / f"v{v:04d}"          # the next warm start, without downloading it again
        mine.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pt, mine / "model.pt")
        self.prune(win, keep=out)
        return meta

    def prune(self, win: list[dict], keep: Path | None = None) -> None:
        """Delete local packs outside the window and old training folders (the store keeps everything)."""
        inside = {e["shard"] for e in win}
        for shard in [s for s in self.local if s not in inside]:
            for d in self.local.pop(shard):
                shutil.rmtree(Path(d).parent if Path(d).parent.parent == self.wd / "packs" else d, ignore_errors=True)
        for d in (self.wd / "train").glob("v*"):
            if keep is None or d != keep:
                shutil.rmtree(d, ignore_errors=True)
        for d in (self.wd / "versions").glob("v*"):
            if keep is None or d.name != keep.name:
                shutil.rmtree(d, ignore_errors=True)

    # ---------------------------------------------------------------------------- status and the loop
    def status(self, state: str, force: bool = False, done: bool = False, **extra) -> None:
        now = time.time()
        if not force and state == self.state and now - self.last_status < self.STATUS_EVERY_S:
            return
        self.state, self.last_status = state, now
        st = {"holder": self.holder, "machine": self.machine, "state": state, "done": done, "time": now,
              "iso": ctl.now_iso(now), "shards": len(self.ingested), "games": sum(e["games"] for e in self.ingested),
              "positions": sum(e["positions"] for e in self.ingested), "new_positions": self.new_positions(),
              "versions_made": self.versions_made, **extra}
        hb_path, hb = ctl.heartbeat(self.machine, "learner", self.c("run"), state=state)
        self.store.put_many({ctl.LEARNER_STATUS: json.dumps(st, indent=1, sort_keys=True).encode(), hb_path: hb},
                            f"learner: {state}")

    def step(self) -> str:
        """One pass: "stop", "published" or "wait"."""
        self.lease()
        c = ctl.read(self.store)
        if c and c.get("stop"):
            self.status("stopped", force=True, done=True)
            return "stop"
        self.cfg = ctl.effective(self.base, self.store)
        latest = self.bootstrap()
        new = self.ingest()
        reason = self.due(latest)
        if reason:
            self.train_and_publish(latest, reason)
            self.status("waiting", force=True)
            return "published"
        if new:
            self.store.write_json(ctl.INGESTED, self.ingested, f"ingested {len(self.ingested)} shards")
        mx = self.c("budget.max_versions")
        self.status("done" if mx is not None and int(latest["version"]) >= int(mx) else "waiting",
                    force=bool(new))
        return "wait"

    def run(self, max_versions: int | None = None) -> int:
        made = 0
        try:
            while True:
                state = self.step()
                if state == "stop":
                    return 0
                if state == "published":
                    made += 1
                    if max_versions is not None and made >= max_versions:
                        return 0
                    continue
                time.sleep(self.poll_s)
        finally:
            with contextlib.suppress(Exception):
                self.release()
