"""What a rented vCPU is (docs/020): the host's CPU, how the container's share is enforced, and how fast one vCPU runs
alone and when every vCPU is busy.

    python tools/compute_bench/cpuprobe.py --out runs/compute_bench/<tag>/cpuprobe.json [--seconds 4]

The score is a fixed single-threaded workload (integer arithmetic, a dict and a list in pure Python: branchy,
cache-resident, like the game engine's bookkeeping more than like a matmul), run in 1 process, then in N at once
for N = 2, the cgroup quota and the vCPUs visible. Two vCPUs that are SMT siblings of one core score about 0.55-0.65
each when both run; a vCPU on a host whose other tenants keep the cores busy scores lower still. A crude memory
bandwidth figure (numpy copy, 1 thread) goes alongside.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

_here = Path(__file__).resolve()
if len(_here.parents) > 2 and (_here.parents[2] / "src").is_dir():    # in the repo: draftzero's cgroup reader
    sys.path.insert(0, str(_here.parents[2] / "src"))


def work(seconds: float) -> float:
    """Iterations a second of a fixed pure-Python kernel (after a short warm-up)."""
    def kernel(n: int) -> int:
        d, lst, acc = {}, [], 0
        for i in range(n):
            k = (i * 2654435761) & 1023
            d[k] = d.get(k, 0) + i
            if i & 7 == 0:
                lst.append(k)
            acc ^= (k << 3) + (acc >> 5)
        return acc + len(lst) + len(d)
    kernel(20000)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        kernel(20000)
        n += 1
    return n * 20000 / (time.perf_counter() - t0)


def _child(args):
    seconds, barrier_t = args
    while time.time() < barrier_t:       # start together
        time.sleep(0.001)
    return work(seconds)


def concurrent(n: int, seconds: float) -> list[float]:
    with mp.get_context("fork").Pool(n) as pool:
        start = time.time() + 1.0
        return pool.map(_child, [(seconds, start)] * n)


def mem_bandwidth_gbs() -> float | None:
    try:
        import numpy as np
    except ImportError:
        return None
    a = np.ones(64 * 2**20 // 8)            # 64 MB, past any L3
    b = np.empty_like(a)
    np.copyto(b, a)
    t0, k = time.perf_counter(), 0
    while time.perf_counter() - t0 < 1.0:
        np.copyto(b, a)
        k += 1
    return round(2 * a.nbytes * k / (time.perf_counter() - t0) / 1e9, 1)


def read(p: str) -> str | None:
    try:
        return open(p).read().strip()
    except OSError:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seconds", type=float, default=4.0)
    a = ap.parse_args(argv)
    try:
        from draftzero.resources import cpu_quota
        quota = cpu_quota() or os.cpu_count()
    except ImportError:                   # a bare python: read the cgroup itself
        q = (read("/sys/fs/cgroup/cpu.max") or "max").split()
        v1 = read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        quota = (int(q[0]) / int(q[1]) if q[0] != "max" else
                 int(v1[0]) / int(v1[1]) if v1[0] and v1[0] != "-1" else os.cpu_count())
    visible = os.cpu_count()
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else visible
    model = next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")), None) \
        if os.path.exists("/proc/cpuinfo") else None
    res = {"cpu_model": model, "visible_cpus": visible, "affinity_cpus": affinity, "cgroup_cores": quota,
           "cpuset": read("/sys/fs/cgroup/cpuset.cpus.effective") or read("/sys/fs/cgroup/cpuset/cpuset.cpus"),
           "cpu_max": read("/sys/fs/cgroup/cpu.max"),
           "throttled": read("/sys/fs/cgroup/cpu.stat"),
           "mem_bw_gbs_1thread": mem_bandwidth_gbs()}
    one = work(a.seconds)
    res["score_1"] = round(one)
    for n in sorted({2, int(round(quota)), min(visible, 2 * int(round(quota)))}):
        if n < 2 or n > 256:
            continue
        sc = concurrent(n, a.seconds)
        res[f"score_{n}_each_mean"] = round(sum(sc) / n)
        res[f"score_{n}_each_vs_1"] = round(sum(sc) / n / one, 3)
        res[f"score_{n}_total_vs_1"] = round(sum(sc) / one, 2)
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
