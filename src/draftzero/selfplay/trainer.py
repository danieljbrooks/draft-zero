"""The trainer (docs/021 §3.1): exactly one, running continuously. Each loop it

  1. holds the lease (learner/lease.json, renewed every lease_s / 3): a second trainer waits until it expires;
  2. on a fresh store, publishes the starting checkpoint as version 0;
  3. ingests new game batches (games/ minus learner/ingested.json) into training rows, kept per batch in its
     work dir (GNN: tables.graph_rows .npz; MLP: the batch files themselves);
  4. once min_new_games new games are in and publish_every_s has passed (or twice as many games arrived), trains
     on the newest window_games games, reuse x the new positions (gnn_train for the GNN; supervised.py with the
     soft table, the KL and init_checkpoint for the flat MLP), runs the offline checks on held-out games, and
     publishes version n+1: weights/v<n+1>.pt.gz and weights/latest.json;
  5. prunes old weights (keeping the newest keep_versions, every 10th, version 0 and any open evaluation's) and,
     every full_every_versions versions, saves its full state (optimizer too) as learner/full.pt.gz.

    dz selfplay trainer --config configs/selfplay_gnn.yml --store <store> [--work-dir DIR]
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

from draftzero.selfplay import config as sc
from draftzero.selfplay import control as ctl
from draftzero.selfplay import records, tables

LEASE, INGESTED, STATUS, FULL = "learner/lease.json", "learner/ingested.json", "learner/status.json", "learner/full.pt.gz"


class LeaseHeld(RuntimeError):
    pass


class Trainer:
    def __init__(self, store, cfg: dict, work_dir: Path | None = None, log=print):
        self.store, self.cfg, self.log = store, cfg, log
        self.tc = cfg["trainer"]
        self.wd = Path(work_dir or sc.path(f"runs/selfplay/{cfg['run']}/trainer"))
        self.wd.mkdir(parents=True, exist_ok=True)
        self.holder = f"{socket.gethostname().split('.')[0]}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.lease_renewed = float("-inf")
        self.ingested: list[dict] = []       # [{path, games, positions, rows}] in ingest order
        self.pending_games = self.pending_positions = 0
        self.last_publish = time.time()
        self.versions_made = 0
        self.gnn = None
        self.seen: set[str] = set()          # batches a previous trainer ingested: re-read, not counted as new

    # ---------------------------------------------------------------------------- lease
    def lease(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        if now - self.lease_renewed < self.tc["lease_s"] / 3:
            return
        cur = self.store.read_json(LEASE)
        if cur and cur.get("holder") != self.holder and cur.get("expires", 0) > now:
            raise LeaseHeld(f"trainer {cur['holder']} holds the lease until {time.ctime(cur['expires'])}")
        self.store.write_json(LEASE, {"holder": self.holder, "expires": now + self.tc["lease_s"], "time": now},
                              f"lease: {self.holder}")
        self.lease_renewed = now

    # ---------------------------------------------------------------------------- versions
    def bootstrap(self) -> dict:
        latest = self.store.read_json(ctl.LATEST)
        if latest is not None:
            return latest
        src = self.cfg["start"]
        local = self.wd / "weights" / "v0000.pt.gz"
        local.parent.mkdir(parents=True, exist_ok=True)
        if str(src).startswith("hf://"):
            from draftzero.gameplay.imitation_net import resolve
            shutil.copyfile(resolve(src), local)
        else:
            shutil.copyfile(sc.path(src), local)
        latest = {"version": 0, "path": ctl.weights_path(0), "sha256": ctl.sha256(local), "parent": None,
                  "source": str(src), "games": 0, "positions": 0, "metrics": {}, "time": time.time()}
        self.store.put_many({ctl.weights_path(0): local, ctl.LATEST: json.dumps(latest, indent=1).encode()},
                            "version 0: the starting network")
        self.log(f"trainer: published version 0 from {src}")
        return latest

    def _local_weights(self, version: int) -> Path:
        p = self.wd / "weights" / f"v{version:04d}.pt.gz"
        if not p.exists():
            if self.store.get(ctl.weights_path(version), p) is None:
                raise RuntimeError(f"version {version}'s weights are not in the store")
        return p

    # ---------------------------------------------------------------------------- ingest
    def ingest(self) -> int:
        done = {x["path"] for x in self.ingested}
        new = [p for p in self.store.list(ctl.games_prefix()) if p.endswith(".jsonl.gz") and p not in done]
        for path in new:
            local = self.wd / "batches" / path
            if self.store.get(path, local) is None:
                continue
            entry = {"path": path, "local": str(local)}
            # games, not seats: a self-play game has two training seats, a game against the start one
            train_games = len({(sg["game"]["pair"], sg["game"]["swap"]) for sg in records.read(local) if sg.get("train")})
            if self.cfg["network"] == "gnn":
                rows = tables.graph_rows(local, self.tc["td_lambda"], self.tc["heldout_share"])
                npz = self.wd / "rows" / (path.replace("/", "__") + ".npz")
                tables.save(rows, npz)
                entry.update(rows_file=str(npz), positions=int((~rows["heldout"]).sum()), rows=tables.n_rows(rows))
                local.unlink()                       # the rows are all the GNN trainer needs
            else:
                n = sum(len(sg["records"]) for sg in records.read(local) if sg.get("train"))
                entry.update(positions=n, rows=n)
            entry["games"] = train_games
            self.ingested.append(entry)
            if path not in self.seen:
                self.pending_games += train_games
                self.pending_positions += entry["positions"]
        if new:
            self.store.write_json(INGESTED, [{k: v for k, v in e.items() if k in ("path", "games", "positions")}
                                             for e in self.ingested], f"ingested {len(self.ingested)} batches")
            self.log(f"trainer: ingested {len(new)} batches; {self.pending_games} new games, "
                     f"{self.pending_positions} new positions since the last version")
        return len(new)

    def window(self) -> list[dict]:
        out, games = [], 0
        for e in reversed(self.ingested):
            out.append(e)
            games += e["games"]
            if games >= self.tc["window_games"]:
                break
        return list(reversed(out))

    def due(self, now: float) -> bool:
        """Enough new games, and new training positions among them (held-out games alone make no version)."""
        m = self.tc["min_new_games"]
        return self.pending_games >= m and self.pending_positions > 0 and (
            now - self.last_publish >= self.tc["publish_every_s"] or self.pending_games >= 2 * m)

    # ---------------------------------------------------------------------------- train + publish
    def update(self, latest: dict) -> dict:
        v = latest["version"]
        win = self.window()
        if self.cfg["network"] == "gnn":
            metrics, out = self._update_gnn(v, win)
        else:
            metrics, out = self._update_mlp(v, win)
        bad = [k for k, x in metrics.items() if isinstance(x, float) and not np.isfinite(x)]
        if bad:
            raise RuntimeError(f"version {v + 1} failed its sanity check: non-finite {bad}")
        nv = v + 1
        new = {"version": nv, "path": ctl.weights_path(nv), "sha256": ctl.sha256(out), "parent": v,
               "games": sum(e["games"] for e in win), "positions": sum(e["positions"] for e in win),
               "new_games": self.pending_games, "new_positions": self.pending_positions, "metrics": metrics,
               "time": time.time(), "trainer": self.holder}
        files = {ctl.weights_path(nv): out, ctl.LATEST: json.dumps(new, indent=1).encode()}
        if self.gnn is not None and nv % int(self.tc["full_every_versions"]) == 0:
            full = self.wd / "full.pt.gz"
            self.gnn.full_state(full, {"version": nv})
            files[FULL] = full
        self.store.put_many(files, f"version {nv}")
        self.log(f"trainer: published version {nv}: {json.dumps(metrics)}")
        self.pending_games = self.pending_positions = 0
        self.last_publish = time.time()
        self.versions_made += 1
        self.prune(nv)
        return new

    def _update_gnn(self, v: int, win: list[dict]) -> tuple[dict, Path]:
        from draftzero.selfplay import gnn_train
        cur = self._local_weights(v)
        if self.gnn is None:
            full = None
            if self.store.exists(FULL):
                full = self.wd / "full_in.pt.gz"
                self.store.get(FULL, full)
            self.gnn = gnn_train.GnnTrainer(cur, self._local_weights(0), self.tc, log=self.log, full_state=full)
        rows = tables.concat([tables.load(Path(e["rows_file"])) for e in win])
        m = self.gnn.train(rows, self.pending_positions)
        m.update({f"heldout_{k}": x for k, x in self.gnn.evaluate(rows).items()})
        out = self.wd / "weights" / f"v{v + 1:04d}.pt.gz"
        self.gnn.save_weights(out, {"version": v + 1, "parent": v, "selfplay": m})
        return m, out

    def _update_mlp(self, v: int, win: list[dict]) -> tuple[dict, Path]:
        """The flat MLP: supervised.py on the window's soft table, from version v, KL to version 0."""
        import yaml
        tdir = self.wd / "tables"
        shutil.rmtree(tdir, ignore_errors=True)
        info = tables.flat_tables([Path(e["local"]) for e in win], tdir, self.tc["td_lambda"], self.tc["heldout_share"])
        f = min(1.0, self.versions_made / max(1, self.tc["kl_anneal_versions"]))
        beta = self.tc["kl_weight"] + (self.tc["kl_weight_end"] - self.tc["kl_weight"]) * f
        steps = max(1, int(np.ceil(self.tc["reuse"] * max(self.pending_positions, 1) / self.tc["batch_rows"])))
        spec = {"tables_dir": str(tdir), "tables": [{"name": "selfplay", "kind": "soft", "value_column": "z_td"}],
                "init_checkpoint": str(self._local_weights(v)), "kl_ref": str(self._local_weights(0)),
                "kl_weight": float(beta), "kl_weight_end": None, "lr": self.tc["lr"],
                "warmup_steps": self.tc["warmup_steps"], "max_steps": steps, "max_epochs": None, "time_budget_s": None,
                "val_rows": max(1, info["val_rows"]), "eval_every_steps": steps, "patience_epochs": None}
        if self.tc["human_share"] > 0:
            hdir = sc.path(self.tc["human_tables_dir"])
            for h in hdir.glob("*.h5"):                  # supervised.py reads every table from one directory
                (tdir / h.name).symlink_to(h)
            spec["tables"] += [{"name": n, "kind": k, "group": "human"} for n, k in (
                ("turnstart", "priority_set"), ("replay_priority", "priority_set"), ("opp_priority", "priority_set"),
                ("replay_attack", "binary"), ("replay_target", "target"), ("opp_block", "target"))]
            spec["group_shares"] = {"human": self.tc["human_share"]}
        run = self.wd / f"mlp_v{v + 1:04d}"
        shutil.rmtree(run, ignore_errors=True)
        run.mkdir(parents=True)
        (run / "config.yml").write_text(yaml.safe_dump(spec))
        env = dict(os.environ, PYTHONPATH=str(sc.REPO / "src"))
        rc = subprocess.call([sys.executable, "-m", "draftzero.gameplay.supervised", "train", "--config",
                              str(run / "config.yml"), "--out", str(run)], env=env)
        out = run / "final.pt.gz" if (run / "final.pt.gz").exists() else run / "best.pt.gz"
        if rc != 0 or not out.exists():
            raise RuntimeError(f"supervised.py failed (rc {rc}) training version {v + 1}")
        m = {"steps": steps, "beta": round(float(beta), 4), **info}
        last = (run / "evals.jsonl").read_text().splitlines() if (run / "evals.jsonl").exists() else []
        if last:
            m["eval"] = json.loads(last[-1])
        dst = self.wd / "weights" / f"v{v + 1:04d}.pt.gz"
        shutil.copyfile(out, dst)
        return m, dst

    def prune(self, newest: int) -> None:
        c = ctl.read(self.store) or {}
        keep = {0, newest} | {int(j["a"]) for j in c.get("evals", [])} | {
            int(j["b"]) for j in c.get("evals", []) if j["b"] != "heuristic"}
        keep |= set(range(max(0, newest - int(self.tc["keep_versions"]) + 1), newest + 1))
        old = [p for p in self.store.list("weights/") if p.startswith("weights/v") and p.endswith(".pt.gz")]
        drop = [p for p in old if (n := int(p[len("weights/v"):-len(".pt.gz")])) not in keep and n % 10]
        if drop:
            self.store.delete(drop, f"prune {len(drop)} old versions")

    def status(self, state: str, done: bool = False) -> None:
        self.store.write_json(STATUS, {"holder": self.holder, "state": state, "done": done, "time": time.time(),
                                       "games": sum(e["games"] for e in self.ingested),
                                       "positions": sum(e["positions"] for e in self.ingested),
                                       "pending_games": self.pending_games, "versions_made": self.versions_made},
                              f"trainer: {state}")

    def run(self) -> None:
        while True:
            try:
                self.lease()
                break
            except LeaseHeld as e:
                self.log(f"trainer: {e}; waiting")
                time.sleep(self.tc["poll_s"])
        latest = self.bootstrap()
        self.seen = {e["path"] for e in (self.store.read_json(INGESTED) or [])}
        self.log(f"trainer {self.holder}: version {latest['version']}, store {self.store.describe()}"
                 + (f"; {len(self.seen)} batches already trained on" if self.seen else ""))
        self.status("running")
        while True:
            self.lease()
            c = ctl.read(self.store)
            if c and c.get("stop"):
                self.status("stopped", done=True)
                self.log("trainer: the run is stopped")
                return
            self.ingest()
            if self.due(time.time()):
                latest = self.update(latest)
                mv = self.tc["max_versions"]
                done = mv is not None and self.versions_made >= int(mv)
                self.status("done" if done else "running", done=done)
                if done:
                    self.log(f"trainer: made {self.versions_made} versions (max_versions); done")
                    return
            time.sleep(self.tc["poll_s"])
