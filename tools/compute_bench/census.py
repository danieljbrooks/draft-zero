"""A census of what RunPod's vCPUs are (docs/020): rent each offer for a few minutes, read the host's CPU and how the
container's share is enforced, run tools/compute_bench/cpuprobe.py, remove the pod. A few cents an offer.

    python tools/compute_bench/census.py --out runs/compute_bench/census \\
        "3090-secure|NVIDIA GeForce RTX 3090||32" "cpu3c-2|cpu3c|EU-SE-1|2" ...

An offer is "tag|gpu type id or cpu flavor|data center (optional)|vCPUs (the floor for a GPU pod, the count for a CPU
pod)", optionally "|community". Pods start from RunPod's small base image (the probe needs only python3).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pods  # noqa: E402

SSHO = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=15",
        "-o", "LogLevel=ERROR", "-i", str(Path("~/.ssh/id_ed25519").expanduser())]
FACTS = r"""
lscpu | grep -E '^(Model name|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|CPU max MHz|L3 cache|NUMA node\(s\))'
echo "nproc: $(nproc)"
echo "cpuset: $(cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || cat /sys/fs/cgroup/cpuset/cpuset.cpus 2>/dev/null)"
echo "cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null) $(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null))"
echo "mem.max: $(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null)"
echo "host_mem_kb: $(grep MemTotal /proc/meminfo | awk '{print $2}')"
echo "loadavg: $(cat /proc/loadavg)"
"""


class Args:
    pass


def one(spec: str, out: Path, wait_s: int = 600) -> dict:
    parts = spec.split("|")
    tag, kind, dc, v = parts[:4]
    community = len(parts) > 4 and parts[4] == "community"
    a = Args()
    a.name, a.image, a.disk, a.community, a.dc = f"cb-census-{tag}", "runpod/base:1.4.0-ubuntu2404", 10, community, dc or None
    if kind in pods.CPU_FLAVORS:
        a.cpu, a.vcpu, a.gpu = kind, int(v), None
    else:
        a.cpu, a.gpu, a.min_vcpu, a.min_ram = None, kind, int(v or 8), 16
    rec = {"tag": tag, "offer": spec, "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        p = pods.create(a)
    except SystemExit as e:
        rec["error"] = str(e)[:300]
        return rec
    pid = p["id"]
    rec.update(pod=pid, price=p.get("costPerHr"), vcpu=p.get("vcpuCount"), ram_gb=p.get("memoryInGb"))
    try:
        t0 = time.time()
        while time.time() - t0 < wait_s:
            b = pods.brief(pods.get(pid))
            if b.get("publicIp") and b.get("ssh_port"):
                break
            time.sleep(10)
        else:
            rec["error"] = f"no public ip/port after {wait_s} s"
            return rec
        host = ["-p", str(b["ssh_port"]), f"root@{b['publicIp']}"]
        for _ in range(30):
            if subprocess.run(["ssh", *SSHO, *host, "true"], capture_output=True).returncode == 0:
                break
            time.sleep(5)
        rec["up_s"] = round(time.time() - t0)
        rec["datacenter"] = (b.get("machine") or {}).get("dataCenterId")
        subprocess.run(["scp", *SSHO, "-P", str(b["ssh_port"]), str(HERE / "cpuprobe.py"),
                        f"root@{b['publicIp']}:/root/cpuprobe.py"], capture_output=True, timeout=120)
        f = subprocess.run(["ssh", *SSHO, *host, FACTS], capture_output=True, text=True, timeout=120).stdout
        rec["facts"] = dict(ln.split(":", 1) for ln in f.splitlines() if ":" in ln)
        rec["facts"] = {k.strip(): v.strip() for k, v in rec["facts"].items()}
        pr = subprocess.run(["ssh", *SSHO, *host, "python3 /root/cpuprobe.py --seconds 3"], capture_output=True,
                            text=True, timeout=900).stdout
        rec["probe"] = json.loads(pr[pr.index("{"):]) if "{" in pr else {"raw": pr[-500:]}
    except Exception as e:  # noqa: BLE001 - record and remove the pod
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
    finally:
        pods._req("DELETE", f"{pods.REST}/pods/{pid}")
        rec["removed"] = True
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("offers", nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--wait", type=int, default=600, help="seconds to wait for a public IP")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    for spec in a.offers:
        rec = one(spec, a.out, a.wait)
        (a.out / f"{rec['tag']}.json").write_text(json.dumps(rec, indent=1))
        f, pr = rec.get("facts") or {}, rec.get("probe") or {}
        print(f"{rec['tag']:16} {rec.get('error') or ''} {f.get('Model name', '')} | vCPU {rec.get('vcpu')} "
              f"nproc {f.get('nproc')} cpuset {f.get('cpuset')} cpu.max {f.get('cpu.max')} | score1 {pr.get('score_1')} "
              + " ".join(f"{k}={v}" for k, v in pr.items() if k.endswith("each_vs_1")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
