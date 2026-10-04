"""The facts of the machine a compute benchmark ran on (docs/020): what it advertises (lscpu, nproc, the host's
RAM, the pod's quote) and what the container may actually use (the cgroup's CPU quota and memory limit).

    python tools/compute_bench/sysinfo.py --out runs/compute_bench/<tag>/sysinfo.json [--price 0.24 --offer "..."]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001 - a missing tool is an empty answer
        return ""


def pod_env() -> dict:
    """RunPod's own facts about the pod, from PID 1's environment (an SSH session does not inherit it). Never the key."""
    keep = ("RUNPOD_POD_ID", "RUNPOD_GPU_COUNT", "RUNPOD_CPU_COUNT", "RUNPOD_DC_ID", "RUNPOD_POD_HOSTNAME",
            "RUNPOD_PUBLIC_IP", "RUNPOD_MEM_GB")
    try:
        env = dict(kv.split("=", 1) for kv in open("/proc/1/environ").read().split("\0") if "=" in kv)
    except OSError:
        return {}
    return {k: env[k] for k in keep if k in env}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--price", type=float, default=None, help="USD an hour, as quoted")
    ap.add_argument("--offer", default=None, help="what was rented, as quoted (cloud, GPU or CPU flavor, vCPU, RAM)")
    a = ap.parse_args(argv)
    from draftzero.resources import cpu_quota, mem_limit_gb
    lscpu = {}
    for line in sh("lscpu").splitlines():
        k, _, v = line.partition(":")
        lscpu[k.strip()] = v.strip()
    try:
        import psutil
        host_mem = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        host_mem = None
    try:
        import torch
        tinfo = {"torch": torch.__version__, "torch_threads_default": torch.get_num_threads(),
                 "cuda": torch.cuda.is_available(),
                 "mkldnn": bool(torch.backends.mkldnn.is_available())}
    except ImportError:
        tinfo = {}
    flags = lscpu.get("Flags", "")
    info = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "price_usd_hr": a.price, "offer": a.offer,
        "cpu_model": lscpu.get("Model name") or sh("sysctl -n machdep.cpu.brand_string"),
        "cpu_max_mhz": lscpu.get("CPU max MHz"), "cpu_base_mhz": lscpu.get("CPU MHz"),
        "sockets": lscpu.get("Socket(s)"), "cores_per_socket": lscpu.get("Core(s) per socket"),
        "threads_per_core": lscpu.get("Thread(s) per core"), "numa_nodes": lscpu.get("NUMA node(s)"),
        "l3": lscpu.get("L3 cache"), "hypervisor": lscpu.get("Hypervisor vendor"),
        "avx2": "avx2" in flags, "avx512f": "avx512f" in flags, "avx512_bf16": "avx512_bf16" in flags,
        "nproc": os.cpu_count(),
        "affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cgroup_cores": cpu_quota(), "cgroup_mem_gb": mem_limit_gb(), "host_mem_gb": host_mem,
        "cpuset": (open("/sys/fs/cgroup/cpuset.cpus.effective").read().strip()
                   if os.path.exists("/sys/fs/cgroup/cpuset.cpus.effective") else None),
        "gpu": sh("nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv,noheader"),
        "java": sh("java -version 2>&1 | head -1"),
        "kernel": platform.release(),
        **tinfo, "pod": pod_env(),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
