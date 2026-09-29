"""Run docs/012's first experiment: one `bench` request per (run, decision), on a pool of bridge
workers kept full, run after run.

    python tools/search_bench/run.py --items data/search_bench/sb-v1 --out runs/search_bench/e2_offline \\
        --grid e2 --evaluator offline --workers 30
    python tools/search_bench/run.py ... --grid e2 --evaluator remote --ports 50052,50152,50252 --workers 36

A run is one configuration: method x budget x evaluator x discount (and its unit) x seed. The
methods (docs/012 §2.1), each a `bench` request over the item's worlds:

    clairvoyant  tree search on the item's real world (the filler hand and library orders)
    pimc1        tree search on the first belief world
    pimc4        one tree per belief world, the first four, a quarter of the budget each
    ismcts       one information-set tree over the eight belief worlds, re-dealt every iteration
    policy       the network's root policy, no search (E0 reference; remote only)

Runs go one after another, each over all its decisions at once, so its wall-clock at full load
gives pod-seconds per decision (docs/012 §2.6). Output, all resumable:

    <out>/decisions/<run_id>.jsonl   one row per decision (appended; a restart skips done items)
    <out>/runs.jsonl                 one row per finished batch: decisions, wall-clock, workers
    <out>/load.jsonl                 cgroup CPU, GPU and memory every 10 s
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from draftzero.gameplay.bridge import BridgePool  # noqa: E402

BUDGETS = (100, 300, 1000, 3000)
METHODS = ("clairvoyant", "pimc1", "pimc4", "ismcts")


def run_id(r: dict) -> str:
    rid = f"{r['method']}-b{r['budget']}-{r['evaluator']}"
    if r["method"] != "policy":
        rid += f"-d{r['discount']:g}{'' if r['unit'] == 'ply' else '-' + r['unit']}"
    if r.get("priors"):
        rid += "-pri"
    if r.get("seed", 0):
        rid += f"-s{r['seed']}"
    return rid


def grid(name: str, evaluator: str, a) -> list[dict]:
    def R(method, budget, discount=0.99, unit="ply", seed=0, priors=False):
        out = {"method": method, "budget": budget, "evaluator": evaluator, "discount": discount,
               "unit": unit, "seed": seed}
        if priors:
            out["priors"] = True
        return out
    budgets = [int(b) for b in a.budgets.split(",")] if a.budgets else list(BUDGETS)
    methods = a.methods.split(",") if a.methods else list(METHODS)
    if name == "e2":
        # cheap budgets first, so a complete low-budget picture exists early
        return [R(m, b) for b in budgets for m in methods]
    if name == "e2b":
        d_a, d_t = a.d_action, a.d_turn
        arms = [(1.0, "ply"), (0.95, "ply"), (0.9, "ply"), (d_a, "action"), (d_t, "turn")]
        return [R(m, 1000, d, u) for m in ("clairvoyant", "pimc4") for d, u in arms]
    if name == "e0":
        out = [R("pimc4", 1000, seed=1)]
        if evaluator == "remote":
            out.insert(0, R("policy", 0))
        return out
    if name == "policy":
        return [R("policy", 0)]
    if name == "priors":
        # follow-up: the network's policy heads as PUCT priors (MageZero's setPriors); needs a
        # policy-serving server (MageZero's own), not the value-only one
        return [R(m, b, priors=True) for b in budgets for m in methods]
    if name == "cal":
        return [R(m, b) for b in budgets for m in methods]
    raise SystemExit(f"unknown grid {name}")


def request_for(item: dict, r: dict) -> tuple[list, dict]:
    m = r["method"]
    opts = dict(item["request"])
    opts.update(budget=max(1, r["budget"]), discount=r["discount"], discountUnit=r["unit"],
                idSeed=item["build_seed"], timeoutSec=1500)
    if m in ("clairvoyant", "policy"):
        specs = [item["real"]]
        opts.update(method="policy" if m == "policy" else "tree", worldSeeds=[item["build_seed"]])
    elif m == "pimc1":
        specs = item["worlds"][:1]
        opts.update(method="tree")
    elif m == "pimc4":
        specs = item["worlds"][:4]
        opts.update(method="tree")
    elif m == "ismcts":
        specs = item["worlds"][:8]
        opts.update(method="ismcts")
    else:
        raise ValueError(m)
    opts["seed"] = 7 + 1000 * r.get("seed", 0)
    if r.get("priors"):
        opts["priors"] = True
    return specs, opts


def load_items(path: Path, split: str | None, limit: int | None, types: str | None) -> list[dict]:
    with gzip.open(path / "items.jsonl.gz", "rt") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if split:
        items = [it for it in items if it["split"] == split]
    if types:
        keep = set(types.split(","))
        items = [it for it in items if it["type"] in keep]
    if limit:
        rng = random.Random(1)
        items = sorted(rng.sample(items, min(limit, len(items))), key=lambda it: it["id"])
    return items


def cgroup_cpu_usec() -> int | None:
    try:
        for line in open("/sys/fs/cgroup/cpu.stat"):
            if line.startswith("usage_usec"):
                return int(line.split()[1])
    except OSError:
        pass
    try:
        return int(open("/sys/fs/cgroup/cpuacct/cpuacct.usage").read()) // 1000
    except OSError:
        return None


def sampler(out: Path, stop: threading.Event, tag: dict) -> None:
    prev = (time.time(), cgroup_cpu_usec())
    with open(out / "load.jsonl", "a") as f:
        while not stop.wait(10):
            now, cpu = time.time(), cgroup_cpu_usec()
            row = {"t": round(now), **tag}
            if cpu is not None and prev[1] is not None:
                row["cores"] = round((cpu - prev[1]) / 1e6 / (now - prev[0]), 2)
            prev = (now, cpu)
            try:
                g = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                u, mem = g.stdout.strip().split(",")[:2]
                row.update(gpu=int(u), vram_mb=int(mem))
            except Exception:  # noqa: BLE001 - no GPU here
                pass
            try:
                row["mem_gb"] = round(int(open("/sys/fs/cgroup/memory.current").read()) / 2**30, 1)
            except OSError:
                pass
            f.write(json.dumps(row) + "\n")
            f.flush()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", default=str(REPO / "data/search_bench/sb-v1"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", default="e2")
    ap.add_argument("--evaluator", choices=("offline", "remote"), default="offline")
    ap.add_argument("--ports", default="50052", help="inference servers, comma-separated (remote)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--heap", default="3g")
    ap.add_argument("--split", default="test")
    ap.add_argument("--types", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--budgets", default=None, help="override, e.g. 100,300")
    ap.add_argument("--methods", default=None, help="override, e.g. clairvoyant,pimc4")
    ap.add_argument("--d-action", type=float, default=0.9, help="E2b per-action discount (from E0)")
    ap.add_argument("--d-turn", type=float, default=0.7, help="E2b per-turn discount (from E0)")
    ap.add_argument("--only", default=None, help="run ids to run, comma-separated")
    a = ap.parse_args(argv)

    out = Path(a.out)
    (out / "decisions").mkdir(parents=True, exist_ok=True)
    items = load_items(Path(a.items), a.split or None, a.limit, a.types)
    runs = grid(a.grid, a.evaluator, a)
    if a.only:
        keep = set(a.only.split(","))
        runs = [r for r in runs if run_id(r) in keep]
    ports = [int(p) for p in a.ports.split(",")]
    print(f"{len(items)} decisions x {len(runs)} runs, {a.workers} workers, evaluator {a.evaluator}"
          + (f" on ports {ports}" if a.evaluator == "remote" else ""), flush=True)
    (out / "config.json").write_text(json.dumps({"argv": sys.argv, "runs": runs, "n_items": len(items)}, indent=1))

    stop = threading.Event()
    threading.Thread(target=sampler, args=(out, stop, {"evaluator": a.evaluator}), daemon=True).start()
    # worker runtime dirs are per output dir: two runners on one pod must not share a worker name
    pool = BridgePool(a.workers, prefix=f"sb_{out.name}_", heap=a.heap, timeout=3600)
    counter = iter(range(10**12))
    lock = threading.Lock()
    try:
        for r in runs:
            rid = run_id(r)
            path = out / "decisions" / f"{rid}.jsonl"
            done = set()
            if path.exists():
                for line in open(path):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not row.get("error"):
                        done.add(row["item_id"])
            todo = [it for it in items if it["id"] not in done]
            if not todo:
                print(f"[{rid}] done already", flush=True)
                continue
            # the slowest items first, so the batch does not end on a long tail
            todo.sort(key=lambda it: (it["type"] != "spell", it["n_options"]), reverse=False)
            t0 = time.time()
            n_err = 0
            with open(path, "a") as f, ThreadPoolExecutor(a.workers) as ex:
                def one(it):
                    specs, opts = request_for(it, r)
                    if a.evaluator == "remote":
                        with lock:
                            port = ports[next(counter) % len(ports)]
                        opts["evaluator"] = {"type": "remote", "host": "127.0.0.1", "port": port}
                    ts = time.time()
                    row = {"run_id": rid, "item_id": it["id"], "split": it["split"], "type": it["type"], **r}
                    try:
                        resp = pool.request("bench", None, specs=specs, **opts)
                        row.update(best=resp.get("best"), children=resp.get("children"),
                                   rootVisits=resp.get("rootVisits"), rootValue=resp.get("rootValue"),
                                   consistent=resp.get("consistent"), decision=(resp.get("decision") or {}).get("type"),
                                   stats=resp.get("stats"), timing_ms=resp.get("timing_ms"))
                    except Exception as e:  # noqa: BLE001 - record and continue
                        row["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
                    row["wall_s"] = round(time.time() - ts, 3)
                    return row
                futs = [ex.submit(one, it) for it in todo]
                for k, fu in enumerate(as_completed(futs), 1):
                    row = fu.result()
                    n_err += bool(row.get("error"))
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    if k % 100 == 0:
                        el = time.time() - t0
                        print(f"[{rid}] {k}/{len(todo)}  {el:.0f}s  {el / k:.3f} s/decision  errors {n_err}", flush=True)
            wall = time.time() - t0
            rec = {"run_id": rid, **r, "n": len(todo), "errors": n_err, "wall_s": round(wall, 1),
                   "workers": a.workers, "pod_s_per_decision": round(wall / len(todo), 4),
                   "complete": len(done) == 0, "t_end": round(time.time())}
            with open(out / "runs.jsonl", "a") as g:
                g.write(json.dumps(rec) + "\n")
            print(f"[{rid}] finished {len(todo)} in {wall:.0f}s ({wall / len(todo):.3f} pod-s/decision), errors {n_err}",
                  flush=True)
    finally:
        stop.set()
        pool.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
