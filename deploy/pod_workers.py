"""How many game workers a pod can run: one per core of its CPU quota (less one for the servers), and as many JVM
heaps of the given size as fit in its memory limit beside ~8 GB for the servers (cgroup v1 or v2).
    python deploy/pod_workers.py <heap GB>"""
import os
import sys


def cores() -> float:
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        return int(q) / int(p) if q != "max" else os.cpu_count()
    except Exception:
        try:
            q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            return q / int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read()) if q > 0 else os.cpu_count()
        except Exception:
            return os.cpu_count()


def mem_gb() -> float:
    for f in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(f).read().strip()
            if v != "max" and int(v) < 1 << 50:
                return int(v) / 2 ** 30
        except Exception:
            pass
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30


heap = float(sys.argv[1])
print(max(2, min(int(cores()) - 1, int((mem_gb() - 8) / heap))))
