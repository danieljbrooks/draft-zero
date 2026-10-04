"""Samples a benchmark's load every --every seconds until killed (docs/020): the cgroup's busy cores and memory, the
GPU, the bridge JVMs (count, total and largest RSS), the inference servers' RSS and the cores' clock.

    python tools/compute_bench/monitor.py --out runs/compute_bench/<tag>/games_b100/load.jsonl &
"""
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path


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


def cgroup_mem_gb() -> float | None:
    for p in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            return round(int(open(p).read()) / 2**30, 2)
        except OSError:
            continue
    return None


def cgroup_anon_gb() -> float | None:
    """Memory that is not page cache (anon): what a job really holds."""
    try:
        for line in open("/sys/fs/cgroup/memory.stat"):
            if line.startswith("anon "):
                return round(int(line.split()[1]) / 2**30, 2)
    except OSError:
        pass
    try:
        for line in open("/sys/fs/cgroup/memory/memory.stat"):
            if line.startswith("total_rss "):
                return round(int(line.split()[1]) / 2**30, 2)
    except OSError:
        pass
    return None


def mean_mhz() -> float | None:
    try:
        v = [float(line.split(":")[1]) for line in open("/proc/cpuinfo") if line.startswith("cpu MHz")]
        return round(sum(v) / len(v)) if v else None
    except OSError:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--every", type=float, default=10.0)
    a = ap.parse_args(argv)
    import psutil
    a.out.parent.mkdir(parents=True, exist_ok=True)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    prev = (time.time(), cgroup_cpu_usec())
    with open(a.out, "a") as f:
        while not stop:
            time.sleep(a.every)
            now, cpu = time.time(), cgroup_cpu_usec()
            row = {"t": round(now, 1)}
            if cpu is not None and prev[1] is not None:
                row["cores"] = round((cpu - prev[1]) / 1e6 / (now - prev[0]), 2)
            prev = (now, cpu)
            row["mem_gb"], row["anon_gb"], row["mhz"] = cgroup_mem_gb(), cgroup_anon_gb(), mean_mhz()
            jv, srv = [], []
            for p in psutil.process_iter(["name", "cmdline", "memory_info"]):
                try:
                    cmd = " ".join(p.info["cmdline"] or [])
                    rss = p.info["memory_info"].rss / 2**30
                except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError, AttributeError):
                    continue
                if "org.draftzero.mzbridge.Worker" in cmd:
                    jv.append(rss)
                elif "value_server.py" in cmd or "belief_server.py" in cmd:
                    srv.append(rss)
            row.update(jvms=len(jv), jvm_rss_gb=round(sum(jv), 2), jvm_rss_max_gb=round(max(jv), 2) if jv else None,
                       servers_rss_gb=round(sum(srv), 2))
            try:
                g = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                u, m = g.stdout.strip().splitlines()[0].split(",")[:2]
                row.update(gpu=int(u), vram_mb=int(m))
            except Exception:  # noqa: BLE001 - no GPU here
                pass
            f.write(json.dumps(row) + "\n")
            f.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
