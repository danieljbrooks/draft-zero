"""Throughput benchmark: K JVMs x T game threads, for a fixed number of minutes.

Answers Will's review question (ROADMAP, "Will's review of experiment #1"): is one big JVM the
bottleneck, and does splitting the same threads across several JVMs play faster? Also used to
tune a single JVM (threads, heap, GC).

    python tools/throughput_bench.py --out bench/offline_1x24 --jvms 1 --threads 24 --minutes 10
    python tools/throughput_bench.py --out bench/net_6x4 --jvms 6 --threads 4 --mode network \\
        --checkpoint gen33 --minutes 10

Each JVM gets its own copy of the XMage working files (card DB, config, logs) with the jars
shared by symlink. On this v0.1 engine, JVMs sharing one XMage dir collide on the H2 card DB
(fixed upstream in v0.2), and they would also write the same magezero.log.

Network mode serves a checkpoint from models/<model>/ver<version>/. By default each JVM gets its
own inference server; --shared-server points every JVM at one. Comparing the two is the
question ROADMAP B3 asks Will.

Writes <out>/result.json: simulations/s summed over JVMs, searches that completed vs timed out,
games finished, and machine CPU use while it ran.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil
import yaml

from draftzero import paths as dzpaths
from magezero.resources import cpu_quota, mem_limit_gb
from magezero.runner import jvm_command, start_server, stop_server

REPO = Path(__file__).resolve().parent.parent


def cgroup_cores() -> float:
    """The container's real CPU quota (cgroup v1 or v2). Never os.cpu_count(): a pod capped at
    31 cores reported 256."""
    return cpu_quota() or float(os.cpu_count() or 1)


def cgroup_ram_gb() -> float:
    """The container's real memory limit. Never psutil's total: the same pod reported 1,007 GB
    of host RAM against a 116 GB limit, and a heap sized from that gets the JVM OOM-killed."""
    return mem_limit_gb() or psutil.virtual_memory().total / 2**30


def cgroup_cpu_seconds() -> float | None:
    """CPU time used by this container so far (cgroup v2 or v1). Never psutil.cpu_percent: in a
    container it reads the host's /proc/stat, so on a shared 256-core host it counts other
    tenants' load, and scaled to the quota it says nothing about ours."""
    try:
        for line in open("/sys/fs/cgroup/cpu.stat"):
            if line.startswith("usage_usec "):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    for f in ("/sys/fs/cgroup/cpuacct/cpuacct.usage", "/sys/fs/cgroup/cpu,cpuacct/cpuacct.usage"):
        try:
            return int(Path(f).read_text()) / 1e9
        except OSError:
            continue
    return None


def xmage_copy(src: Path, dst: Path) -> Path:
    """A private XMage working dir: jars shared by symlink, everything writable copied."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for item in src.iterdir():
        if item.name == "lib":
            (dst / "lib").symlink_to(item.resolve())
        elif item.name in ("magezero.log", "magezeroErrors.log", "seenFeatures.ser", "FeatureTable.txt", "WinRates.txt") \
                or item.name.startswith("magezero.log."):
            continue
        elif item.is_dir():
            shutil.copytree(item, dst / item.name)
        else:
            shutil.copy2(item, dst / item.name)
    return dst


def game_yml(path: Path, a: argparse.Namespace, port_a: int, port_b: int, out_dir: Path) -> None:
    cfg = yaml.safe_load((REPO / "configs/game.yml").read_text())
    offline = a.mode == "offline"
    for side, p in (("A", cfg["player_a"]), ("B", cfg["player_b"])):
        p["deckPath"] = ""
        p["deck_pool"] = str(a.resolved_pool)
        p["deck_pool_mode"] = "random"
        p["output_file"] = str((out_dir / f"{path.stem}_{side}.hdf5").resolve())
        p["type"] = "mcts"
        p["mcts"]["offline_mode"] = offline
        p["mcts"]["td_discount"] = a.td_discount
        p["mcts"]["search_budget"] = a.budget
        p["mcts"]["timeout_ms"] = a.timeout_ms
        for head in ("binary", "priority", "target", "opponent"):
            p["priors"][head] = False
    cfg["training"]["games"] = 100000          # never the limit: the run is stopped by time
    cfg["training"]["threads"] = a.threads
    cfg["training"]["max_minutes"] = a.max_minutes
    cfg["server"]["port"] = port_a
    cfg["server"]["opponent_port"] = port_b
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def gc_flags(gc: str) -> list[str]:
    return {"zgc": ["-XX:+UseZGC"],
            "zgc-gen": ["-XX:+UseZGC", "-XX:+ZGenerational"],
            "g1": ["-XX:+UseG1GC"]}[gc]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jvms", type=int, default=1)
    ap.add_argument("--threads", type=int, required=True, help="game threads per JVM")
    ap.add_argument("--heap", default=None, help="per JVM, e.g. 16g (default: 70%% of RAM / jvms)")
    ap.add_argument("--gc", choices=["zgc", "zgc-gen", "g1"], default="zgc")
    ap.add_argument("--mode", choices=["offline", "network"], default="offline")
    ap.add_argument("--shared-server", action="store_true", help="network mode: one server for all JVMs")
    ap.add_argument("--model", default="FDN_generalist")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--checkpoint", default=None, help="e.g. gen33; default: model.pt.gz")
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--timeout-ms", type=int, default=60000)
    ap.add_argument("--max-minutes", type=int, default=50)
    ap.add_argument("--td-discount", type=float, default=0.95)
    ap.add_argument("--pool", default="data/pools/train.txt")
    ap.add_argument("--xmage", default="xmage")
    ap.add_argument("--deck-root", default=os.environ.get("MZ_DECK_DIR", "data/decks"),
                    help="where the .dck files live (pool files hold deck names, not paths)")
    a = ap.parse_args()

    os.chdir(REPO)
    out = Path(a.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    # pool files hold deck names; the JVM needs one absolute .dck path per line, exactly as
    # draftzero.loop resolves them (without this every game fails with "deck size=0")
    a.resolved_pool = dzpaths.resolve_pool(dzpaths.read_pool(Path(a.pool)), Path(a.deck_root).resolve(),
                                           out / "pool_resolved.txt").resolve()
    ram_gb = cgroup_ram_gb()
    heap = a.heap or f"{max(2, int(ram_gb * 0.7 / a.jvms))}g"
    cores = cgroup_cores()
    print(f"[bench] {a.jvms} JVM x {a.threads} threads, {a.mode}, heap {heap}/JVM, gc {a.gc}, "
          f"budget {a.budget}, {a.minutes} min | cores {cores:.1f}, RAM {ram_gb:.0f} GB", flush=True)

    servers, ports = [], []
    procs, logs = [], []
    t0, cpu0 = time.time(), cgroup_cpu_seconds()
    try:
        if a.mode == "network":
            n_servers = 1 if a.shared_server else a.jvms
            for i in range(n_servers):
                port = 50100 + i
                servers.append(start_server(a.model, a.version, port, out, checkpoint=a.checkpoint))
                ports.append(port)
        for j in range(a.jvms):
            wd = xmage_copy(Path(a.xmage), out / f"xmage_{j}")
            port = ports[0 if a.shared_server else j] if ports else 50100
            yml = out / f"game_{j}.yml"
            game_yml(yml, a, port, port, out)
            cmd, _ = jvm_command(str(yml), heap)
            cmd = [c for c in cmd if c != "-XX:+UseZGC"]
            # A private temp dir per JVM: jhdf5 unpacks its native library into java.io.tmpdir,
            # and two JVMs starting at once race on that file. Locally the loser died at startup
            # with "No suitable HDF5 native library found", leaving a 2-JVM run doing 1 JVM's work.
            (wd / "tmp").mkdir(exist_ok=True)
            cmd[1:1] = gc_flags(a.gc) + [f"-Djava.io.tmpdir={(wd / 'tmp').resolve()}"]
            log = out / f"jvm_{j}.log"
            logs.append(log)
            procs.append(subprocess.Popen(cmd, cwd=wd, stdout=open(log, "w"), stderr=subprocess.STDOUT))
            if j < a.jvms - 1:
                time.sleep(5)   # stagger starts too, so card-DB bootstraps don't all hit the disk at once
        deadline = t0 + a.minutes * 60
        while time.time() < deadline and any(p.poll() is None for p in procs):
            time.sleep(5)
    finally:
        cpu1, t1 = cgroup_cpu_seconds(), time.time()      # before teardown, which is not the workload
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
        for s in servers:
            stop_server(s)
    wall = time.time() - t0
    cpu_used = cpu1 - cpu0 if cpu0 is not None and cpu1 is not None else None

    sims = searches = timeouts = completed = games = 0
    search_seconds = 0.0
    per_jvm = []
    for log in logs:
        # count straight from the per-search log lines, so unfinished games count too. Only the
        # "Player: X simulated N evaluations in T seconds" lines: the cumulative "Total: simulated"
        # lines would double-count every search.
        txt = log.read_text(errors="replace")
        pairs = [(int(n), float(t)) for n, t in
                 re.findall(r"Player: \S+ simulated (\d+) evaluations in ([0-9.]+) seconds", txt)]
        jv_sims = sum(n for n, _ in pairs)
        jv_secs = sum(t for _, t in pairs)
        jv_to = txt.count("timed out, ending search")
        jv_done = txt.count("required visits reached")
        jv_games = txt.count("GAME_SUMMARY")
        if len(pairs) == 0 or "Exception in thread \"main\"" in txt:
            print(f"[bench] WARNING {log.name}: {len(pairs)} searches"
                  f"{'; died with an exception in main' if 'Exception in thread' in txt else ''}", flush=True)
        per_jvm.append({"log": log.name, "sims": jv_sims, "died": "Exception in thread \"main\"" in txt, "searches": len(pairs), "search_seconds": round(jv_secs, 1),
                        "timeouts": jv_to, "completed": jv_done, "games": jv_games})
        sims += jv_sims; searches += len(pairs); search_seconds += jv_secs
        timeouts += jv_to; completed += jv_done; games += jv_games

    result = {
        "jvms": a.jvms, "threads_per_jvm": a.threads, "total_threads": a.jvms * a.threads,
        "mode": a.mode, "shared_server": a.shared_server, "heap_per_jvm": heap, "gc": a.gc,
        "budget": a.budget, "timeout_ms": a.timeout_ms, "minutes_requested": a.minutes,
        "wall_seconds": round(wall, 1), "cores": round(cores, 2), "ram_gb": round(ram_gb, 1),
        "sims": sims, "sims_per_sec": round(sims / wall, 2) if wall else None,
        "searches": searches, "search_timeouts": timeouts, "searches_completed": completed,
        "timeout_rate": round(timeouts / (timeouts + completed), 4) if (timeouts + completed) else None,
        "games_finished": games, "games_per_hour": round(games * 3600 / wall, 1) if wall else None,
        "cores_busy": round(cpu_used / (t1 - t0), 2) if cpu_used is not None else None,
        "quota_percent": round(100 * cpu_used / (t1 - t0) / cores, 1) if cpu_used is not None else None,
        "jvms_dead": sum(1 for j in per_jvm if j.get("died") or not j.get("searches")),
        "per_jvm": per_jvm,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    # the per-JVM XMage copies are ~0.3 GB each (the card DB); keep the logs, drop the copies
    for wd in out.glob("xmage_*"):
        shutil.rmtree(wd, ignore_errors=True)
    print(json.dumps({k: v for k, v in result.items() if k != "per_jvm"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
