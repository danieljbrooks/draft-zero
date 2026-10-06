"""The worker (docs/021 §3.1): plays games on one machine, for as long as control.json says to.

Each loop it reads control.json, makes sure a network server set is running for the version it needs (the
weights fetched from the store, checked against their sha256), then plays one job with
tools/imitation_scale/play.py and uploads the result with its heartbeat in one commit:

    an evaluation shard assigned to this machine (evals.py): version a against b, fixed deck pairs from the
        evaluation pool, no exploration, no records -> evals/<job>/<shard>.jsonl.gz
    otherwise a self-play chunk: the newest version against itself (or, a past_share of chunks, against the
        starting network), PIMC at play.simulations with exploration, records on, deck pairs from the training
        pool drawn with this machine's own seeds -> games/v<version>/<machine>/<chunk>.jsonl.gz

A batch that fails to upload stays in the spool and goes up with the next one. Servers: tools/imitation_scale/
graph_server.py for the GNN, tools/search_bench/value_server.py for the flat MLP, on 127.0.0.1; a new version's
servers start beside the old ones, so a version change costs no idle time.

    dz selfplay worker --config configs/selfplay_gnn.yml --store <store> [--machine NAME] [--work-dir DIR]
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from draftzero.selfplay import config as sc
from draftzero.selfplay import control as ctl
from draftzero.selfplay import evals, records

PLAY = sc.REPO / "tools" / "imitation_scale" / "play.py"


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(sc.REPO / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("MZ_ACTION_VOCAB", str(sc.REPO / "assets" / "vocab" / "FDN_SPG.tsv"))
    return env


def _healthy(port: int) -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2).read()
        return True
    except OSError:
        return False


def machine_size(wcfg: dict) -> dict:
    """Game workers and network servers for this machine, from its real cores and memory (cgroup, not nproc)."""
    from draftzero import resources
    cores = resources.cpu_quota() or os.cpu_count() or 2
    try:
        import psutil
        mem = resources.mem_limit_gb() or psutil.virtual_memory().total / 2 ** 30
    except ImportError:
        mem = resources.mem_limit_gb() or 16.0
    servers = wcfg["servers"] if wcfg["servers"] != "auto" else max(1, min(4, round(cores / 8)))
    h = str(wcfg["heap"]).lower()
    heap_gb = float(h[:-1]) / (1024 if h.endswith("m") else 1) if h[-1] in "mg" else float(h)
    by_mem = int((mem - 3 - 1.5 * servers) / (heap_gb + 0.45))
    workers = wcfg["workers"] if wcfg["workers"] != "auto" else max(1, min(int(cores) - servers, by_mem))
    return {"cores": round(float(cores), 1), "mem_gb": round(float(mem), 1), "workers": int(workers),
            "servers": int(servers)}


class ServerSet:
    """The network servers of one version on this machine."""

    def __init__(self, network: str, version: int, weights: Path, ports: list[int], device: str, log_dir: Path):
        self.version, self.ports, self.procs = version, ports, []
        script = sc.REPO / "tools" / ("imitation_scale/graph_server.py" if network == "gnn" else "search_bench/value_server.py")
        log_dir.mkdir(parents=True, exist_ok=True)
        for p in ports:
            if network == "gnn":
                args = [sys.executable, str(script), "--model", str(weights), "--port", str(p), "--device",
                        "cpu" if device == "cpu" else "auto", "--torch-threads", "2"]
            else:
                args = [sys.executable, str(script), "--model", str(weights), "--port", str(p), "--threads", "8", "--policy"]
            env = _env()
            if device == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = ""
            with open(log_dir / f"server_v{version:04d}_{p}.log", "ab") as lf:
                self.procs.append(subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT, env=env,
                                                   start_new_session=True))

    def wait(self, timeout: float = 600) -> None:
        t0 = time.time()
        for p, proc in zip(self.ports, self.procs):
            while not _healthy(p):
                if proc.poll() is not None:
                    raise RuntimeError(f"server for version {self.version} on port {p} exited ({proc.returncode})")
                if time.time() - t0 > timeout:
                    raise RuntimeError(f"server for version {self.version} on port {p} not up after {timeout:.0f} s")
                time.sleep(1)

    def stop(self) -> None:
        for proc in self.procs:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for proc in self.procs:
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)


class Worker:
    def __init__(self, store, cfg: dict, machine: str | None = None, work_dir: Path | None = None, log=print):
        self.store, self.cfg, self.log = store, cfg, log
        self.machine = machine or socket.gethostname().split(".")[0]
        self.wd = Path(work_dir or sc.path(f"runs/selfplay/{cfg['run']}/worker-{self.machine}"))
        self.wd.mkdir(parents=True, exist_ok=True)
        self.size = machine_size(cfg["worker"])
        self.servers: dict[int, ServerSet] = {}
        self.slots: dict[int, int] = {}
        self.belief = None
        self.stats = {"games": 0, "errors": 0, "chunks": 0, "evals": 0, "rate": None, "job": "starting"}
        self.done_shards: set[str] = set()
        self.rng = random.Random(f"{self.machine}-{time.time()}")
        self._stop = False

    # ---------------------------------------------------------------------------- servers
    def _weights(self, version: int, sha: str | None) -> Path:
        dst = self.wd / "weights" / f"v{version:04d}.pt.gz"
        if not dst.exists():
            got = self.store.get(ctl.weights_path(version), dst)
            if got is None:
                raise RuntimeError(f"no weights for version {version} in the store")
        if sha and ctl.sha256(dst) != sha:
            dst.unlink()
            raise RuntimeError(f"version {version}'s weights don't match their sha256")
        return dst

    def server(self, version: int, sha: str | None = None) -> ServerSet:
        s = self.servers.get(version)
        if s is not None and all(p.poll() is None for p in s.procs):
            return s
        used = set(self.slots.values())
        slot = next(i for i in range(64) if i not in used)
        base = int(self.cfg["worker"]["base_port"]) + 10 * slot
        ports = [base + i for i in range(self.size["servers"])]
        self.log(f"worker {self.machine}: starting version {version}'s servers on {ports}")
        s = ServerSet(self.cfg["network"], version, self._weights(version, sha), ports, self._device(), self.wd / "logs")
        s.wait()
        self.servers[version], self.slots[version] = s, slot
        return s

    def _device(self) -> str:
        d = self.cfg["worker"]["device"]
        if d != "auto":
            return d
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def retire(self, keep: set[int]) -> None:
        for v in [v for v in self.servers if v not in keep]:
            self.log(f"worker {self.machine}: stopping version {v}'s servers")
            self.servers.pop(v).stop()
            self.slots.pop(v, None)

    def belief_port(self) -> int | None:
        if not self.cfg["play"]["closed_decklists"]:
            return None
        port = int(self.cfg["worker"]["base_port"]) - 30
        if self.belief is None or self.belief.poll() is not None:
            if not _healthy(port):
                lf = open(self.wd / "logs" / "belief.log", "ab")
                self.belief = subprocess.Popen([sys.executable, str(sc.REPO / "tools/imitation_scale/belief_server.py"),
                                                "--port", str(port)], stdout=lf, stderr=subprocess.STDOUT, env=_env(),
                                               start_new_session=True)
            t0 = time.time()
            while not _healthy(port):
                if time.time() - t0 > 300:
                    raise RuntimeError("the belief service did not start")
                time.sleep(1)
        return port

    # ---------------------------------------------------------------------------- play
    def _play(self, out: Path, *, pool: str, pairs: int, seed: int, bot1: str, bot2: str, ports1: list[int],
              ports2: list[int] | None, record: bool, explore: bool, min_pair: int = 0, max_pair: int | None = None,
              budget: int) -> int:
        p = self.cfg["play"]
        args = [sys.executable, str(PLAY), "--deck-root", str(sc.path(self.cfg["deck_root"])), "--pool", str(sc.path(pool)),
                "--pairs", str(pairs), "--seed", str(seed), "--budget", str(budget), "--bot1", bot1, "--bot2", bot2,
                "--method", p["method"], "--workers", str(self.size["workers"]), "--heap", str(self.cfg["worker"]["heap"]),
                "--max-turns", str(p["max_turns"]), "--search-timeout", str(p["search_timeout"]),
                "--game-timeout", str(p["game_timeout"]), "--min-pair", str(min_pair), "--out", str(out)]
        if max_pair is not None:
            args += ["--max-pair", str(max_pair)]
        flag, flag2 = ("--graph-ports", "--bot2-graph-ports") if self.cfg["network"] == "gnn" else ("--ports", "--bot2-ports")
        args += [flag, ",".join(map(str, ports1))]
        if ports2:
            args += [flag2, ",".join(map(str, ports2))]
        if record:
            args.append("--record")
        if explore:
            args += ["--root-noise", str(p["root_noise"]), "--root-noise-alpha", str(p["root_noise_alpha"]),
                     "--sample-turns", str(p["sample_turns"])]
        bp = self.belief_port()
        args += ["--belief-port", str(bp)] if bp else ["--open-decklists"]
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "play.log", "ab") as lf:
            proc = subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT, env=_env(), start_new_session=True)
            try:
                return proc.wait()
            except BaseException:
                os.killpg(proc.pid, signal.SIGTERM)
                raise

    def _heartbeat(self, version: int | None) -> bytes:
        hb = {"machine": self.machine, "time": time.time(), "version": version, "games": self.stats["games"],
              "errors": self.stats["errors"], "chunks": self.stats["chunks"], "evals": self.stats["evals"],
              "games_per_hour": None if self.stats["rate"] is None else round(self.stats["rate"], 1),
              "job": self.stats["job"], "device": self._device(), "price_per_hour": self.cfg["worker"]["price_per_hour"],
              **self.size}
        return json.dumps(hb, indent=1).encode()

    def _upload(self, files: dict, version: int | None, message: str) -> bool:
        spool = self.wd / "spool"
        pending = {}
        if spool.exists():
            for f in sorted(spool.rglob("*")):
                if f.is_file():
                    pending[str(f.relative_to(spool))] = f
        pending.update(files)
        pending[f"heartbeats/{self.machine}.json"] = self._heartbeat(version)
        try:
            self.store.put_many(pending, message)
        except Exception as e:  # noqa: BLE001 - a failed upload waits in the spool for the next one
            self.log(f"worker {self.machine}: upload failed ({type(e).__name__}: {e}); spooled")
            for path, src in files.items():
                dst = spool / path
                dst.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(src, (bytes, bytearray)):
                    dst.write_bytes(src)
                else:
                    shutil.copyfile(src, dst)
            return False
        if spool.exists():
            shutil.rmtree(spool)
        return True

    def selfplay_chunk(self, c: dict) -> None:
        v = int(c["version"])
        cur = self.server(v, c.get("sha256"))
        past = v > 0 and self.rng.random() < float(self.cfg["play"]["past_share"])
        opp = self.server(0) if past else None
        chunk = f"{int(time.time())}-{self.stats['chunks']:05d}"
        seed = int.from_bytes(hashlib.sha1(f"{self.machine}|{chunk}".encode()).digest()[:4], "big") % 2_000_000
        n = self.cfg["worker"]["pairs_per_chunk"]
        pairs = max(1, math.ceil(self.size["workers"] / 2)) if n == "auto" else int(n)
        out = self.wd / "chunks" / chunk
        self.stats["job"] = f"self-play v{v}" + (" vs v0" if past else "")
        t0 = time.time()
        b = sc.bot(self.cfg)
        rc = self._play(out, pool=self.cfg["play"]["pool"], pairs=pairs, seed=seed, bot1=b, bot2=b, ports1=cur.ports,
                        ports2=opp.ports if opp else None, record=True, explore=True,
                        budget=int(self.cfg["play"]["simulations"]))
        name = records.batch_name(self.machine, chunk)
        batch = self.wd / "batches" / name
        summ = records.pack(out, batch, run=self.cfg["run"], machine=self.machine, chunk=chunk, version=v,
                            opponent_version=0 if past else None)
        hours = (time.time() - t0) / 3600
        rate = summ["games"] / hours if hours > 0 else None
        self.stats["rate"] = rate if self.stats["rate"] is None or rate is None else 0.7 * self.stats["rate"] + 0.3 * rate
        self.stats.update(games=self.stats["games"] + summ["games"], errors=self.stats["errors"] + summ["errors"],
                          chunks=self.stats["chunks"] + 1)
        self.log(f"worker {self.machine}: chunk {chunk} (v{v}{' vs v0' if past else ''}): {summ['games']} games, "
                 f"{summ['errors']} errors, {summ['decisions']} decisions, {summ['bytes'] / 1e6:.1f} MB, play.py rc {rc}")
        if summ["games"]:
            self._upload({f"{ctl.games_prefix(v)}{self.machine}/{name}": batch}, v,
                         f"games: {self.machine} v{v}, {summ['games']} games")
        else:
            self._upload({}, v, f"heartbeat {self.machine}")
        shutil.rmtree(out, ignore_errors=True)

    def eval_shard(self, c: dict, job: dict, sh: dict) -> None:
        a, b = job["a"], job["b"]
        sa = self.server(int(a))
        sb = None if b == "heuristic" else self.server(int(b))
        bot1 = sc.bot(self.cfg, job["simulations"])
        bot2 = f"heuristic@{job['heuristic_simulations']}" if b == "heuristic" else sc.bot(self.cfg, job["simulations"])
        key = f"{job['job']}/{sh['id']}"
        self.stats["job"] = f"eval {key}"
        out = self.wd / "evals" / job["job"] / f"{sh['id']:04d}"
        shutil.rmtree(out, ignore_errors=True)
        rc = self._play(out, pool=job["pool"], pairs=job["pairs"], seed=job["seed"], bot1=bot1, bot2=bot2,
                        ports1=sa.ports, ports2=sb.ports if sb else None, record=False, explore=False,
                        min_pair=sh["min_pair"], max_pair=sh["max_pair"], budget=job["simulations"])
        games = records.eval_results(out)
        for g in games:
            g.update(job=job["job"], shard=sh["id"], machine=self.machine)
        import gzip
        data = gzip.compress("".join(json.dumps(g) + "\n" for g in games).encode())
        self.stats["evals"] += 1
        self.log(f"worker {self.machine}: evaluation shard {key}: {len(games)} games, play.py rc {rc}")
        if games and self._upload({evals.result_path(job["job"], sh["id"]): data}, c["version"], f"evaluation {key}"):
            self.done_shards.add(key)

    def my_shard(self, c: dict):
        for job in c.get("evals", []):
            for sh in job["shards"]:
                if sh.get("machine") == self.machine and not sh.get("done") and f"{job['job']}/{sh['id']}" not in self.done_shards:
                    return job, sh
        return None

    def run(self, max_jobs: int | None = None) -> None:
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "_stop", True))
        self.log(f"worker {self.machine}: {self.size['workers']} game workers, {self.size['servers']} servers a "
                 f"version, {self.size['cores']} cores, {self.size['mem_gb']} GB; store {self.store.describe()}")
        jobs = 0
        try:
            while not self._stop and (max_jobs is None or jobs < max_jobs):
                c = ctl.read(self.store)
                if c is None:
                    self.log(f"worker {self.machine}: no control.json yet")
                    time.sleep(min(30, self.cfg["worker"]["poll_s"]))
                    continue
                if c.get("stop"):
                    self.log(f"worker {self.machine}: the run is stopped")
                    self._upload({}, c["version"], f"heartbeat {self.machine}")
                    break
                need = {int(c["version"]), 0}
                mine = self.my_shard(c)
                if mine:
                    job, sh = mine
                    need |= {int(job["a"])} | ({int(job["b"])} if job["b"] != "heuristic" else set())
                    self.retire(need)
                    self.eval_shard(c, job, sh)
                else:
                    self.retire(need)
                    self.selfplay_chunk(c)
                jobs += 1
        finally:
            self.retire(set())
            if self.belief is not None and self.belief.poll() is None:
                os.killpg(self.belief.pid, signal.SIGTERM)
