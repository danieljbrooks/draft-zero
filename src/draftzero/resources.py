"""resources.py — sample CPU / RAM / GPU use while a generation runs.

A ResourceMonitor runs a daemon thread that polls psutil (and nvidia-smi when a GPU is
present) every `interval` seconds. `summary()` folds the samples into one metrics row so
the dashboard can plot utilization next to games/hour: if games/hour is flat while the
CPU sits at 40%, the JVM thread count is the thing to raise, not the GPU.
"""
import shutil
import subprocess
import threading
import time
from typing import Optional

import psutil

SMI_QUERY = "utilization.gpu,utilization.memory,memory.used,memory.total"


def cpu_quota() -> Optional[float]:
    """Effective cores this container may use, or None when uncapped.

    Containers see every core on the host: a RunPod pod sold as 9 vCPU reports nproc=48
    and is then cgroup-capped to ~7.65. Sizing thread counts or reading cpu_percent off
    the visible count is therefore wrong by 6x, so read the cgroup instead.
    """
    try:  # cgroup v2
        raw = open("/sys/fs/cgroup/cpu.max").read().split()
        if raw[0] != "max":
            return int(raw[0]) / int(raw[1])
    except Exception:
        pass
    try:  # cgroup v1
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        per = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0 and per > 0:
            return q / per
    except Exception:
        pass
    return None


def mem_limit_gb() -> Optional[float]:
    """Container memory cap in GB, or None when uncapped."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(path).read().strip()
            if v != "max":
                n = int(v)
                if 0 < n < (1 << 62):
                    return n / 1024 ** 3
        except Exception:
            continue
    return None


def _gpu_sample() -> Optional[dict]:
    """One nvidia-smi reading, or None when there is no GPU (Mac / CPU-only pod)."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={SMI_QUERY}",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode != 0 or not out.stdout.strip():
            return None
        # first GPU only; these pods are single-GPU
        util, mem_util, used, total = (float(x) for x in out.stdout.strip().splitlines()[0].split(","))
        return {"gpu_util": util, "gpu_mem_util": mem_util, "gpu_mem_used_gb": used / 1024,
                "gpu_mem_total_gb": total / 1024}
    except Exception:
        return None


def cgroup_mem_used_gb() -> Optional[float]:
    """Memory this container is actually using, in GB.

    psutil reports the HOST: on a RunPod pod that is 251 GB shared with other tenants, so
    it happily reports more "used" than this container's own 46 GB cap. Only the cgroup
    counter answers "are we near our limit".
    """
    for path in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            return int(open(path).read().strip()) / 1024 ** 3
        except Exception:
            continue
    return None


def cgroup_mem_failcnt() -> Optional[int]:
    """How many times the container has hit its memory limit. Nonzero means RAM-bound."""
    try:
        return int(open("/sys/fs/cgroup/memory/memory.failcnt").read().strip())
    except Exception:
        return None


def _mean(xs: list[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


class ResourceMonitor:
    def __init__(self, interval: float = 10.0):
        self.interval = interval
        self._samples: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.has_gpu = _gpu_sample() is not None
        self.quota_cores = cpu_quota()
        self.visible_cores = psutil.cpu_count() or 1
        self.mem_cap_gb = mem_limit_gb()

    def _loop(self) -> None:
        psutil.cpu_percent(interval=None)  # prime the counter; first call always reads 0
        while not self._stop.wait(self.interval):
            vm = psutil.virtual_memory()
            raw = psutil.cpu_percent(interval=None)
            used_gb = (vm.total - vm.available) / 1024 ** 3
            s = {"cpu_percent": raw,
                 "ram_used_gb": used_gb,
                 "ram_percent": vm.percent}
            # what matters on a capped pod is use against the quota, not against the host
            if self.quota_cores:
                s["cpu_quota_percent"] = min(100.0, raw * self.visible_cores / self.quota_cores)
            cg = cgroup_mem_used_gb()
            if cg is not None:
                s["container_mem_gb"] = cg
                if self.mem_cap_gb:
                    s["container_mem_percent"] = 100.0 * cg / self.mem_cap_gb
            if self.has_gpu:
                g = _gpu_sample()
                if g:
                    s.update(g)
            with self._lock:
                self._samples.append(s)

    def start(self) -> "ResourceMonitor":
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="resmon")
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 5)
            self._thread = None

    def summary(self, reset: bool = True) -> dict:
        """Mean and peak of everything sampled since the last reset."""
        with self._lock:
            samples = list(self._samples)
            if reset:
                self._samples.clear()
        if not samples:
            return {"samples": 0}
        keys = {k for s in samples for k in s}
        row: dict = {"samples": len(samples), "cpu_count": self.visible_cores,
                     "cpu_quota_cores": self.quota_cores,
                     "ram_total_gb": psutil.virtual_memory().total / 1024 ** 3,
                     "ram_cap_gb": self.mem_cap_gb,
                     "mem_failcnt": cgroup_mem_failcnt()}
        for k in sorted(keys):
            vals = [s[k] for s in samples if k in s]
            row[f"{k}_mean"] = _mean(vals)
            row[f"{k}_max"] = max(vals) if vals else None
        return row
