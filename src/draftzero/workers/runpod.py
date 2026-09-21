"""runpod.py — a rented GPU pod. An SSH host that has to be created and, crucially, destroyed.

Notes that cost real time to learn:

* Self-play is CPU-bound XMage/MCTS, so pick a pod by **vCPU**, not VRAM. The GPU only
  serves small batched inference and sits near-idle (2-8% in measurements).
* `runpodctl gpu list` shows only cards with stock. The GraphQL catalogue lists every card
  including unavailable ones, and `lowestPrice.minVcpu` is the only place vCPU is exposed.
* Network volumes exist **only in Secure Cloud**, so persistence and the cheapest community
  pricing are mutually exclusive.
* The container is cgroup-capped well below what it advertises: a pod sold as 9 vCPU /
  50 GB reports nproc=48 / 251 GB and is capped to ~7.65 cores / 46 GB. Never size threads
  from nproc.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from draftzero.workers.base import Result, WorkerConfig, local_run
from draftzero.workers.ssh import SSHTarget, SSHWorker

CTL = os.path.expanduser(os.environ.get("RUNPODCTL", "~/.local/bin/runpodctl"))
GQL = "https://api.runpod.io/graphql"


def _ctl(*args: str) -> Result:
    return local_run([CTL, *args])


def _ctl_json(*args: str):
    r = _ctl(*args)
    if not r.ok:
        raise RuntimeError(f"runpodctl {' '.join(args)} failed: {r.err or r.out}")
    return json.loads(r.out)


def catalogue_vcpu() -> dict[str, dict]:
    """displayName -> {minVcpu, minMemory}. Only the GraphQL API exposes vCPU."""
    key = os.environ.get("RUNPOD_API_KEY", "")
    q = ('{"query":"query { gpuTypes { displayName lowestPrice(input:{gpuCount:1}) '
         '{ minVcpu minMemory } } }"}')
    out = subprocess.run(["curl", "-s", "-X", "POST", f"{GQL}?api_key={key}",
                          "-H", "Content-Type: application/json", "-d", q],
                         capture_output=True, text=True).stdout
    try:
        types = json.loads(out)["data"]["gpuTypes"]
    except Exception:
        return {}
    return {g["displayName"]: (g.get("lowestPrice") or {}) for g in types}


def rank_offers(min_vcpu: int = 8, secure_only: bool = True) -> list[dict]:
    """In-stock GPUs ranked by vCPU per dollar -- the metric that matters for self-play."""
    vcpu = catalogue_vcpu()
    rows = []
    for g in _ctl_json("gpu", "list"):
        if not g.get("available"):
            continue
        lp = vcpu.get(g["displayName"]) or {}
        v, m = lp.get("minVcpu"), lp.get("minMemory")
        price = g.get("securePricePerHr") if secure_only else (
            g.get("communityPricePerHr") or g.get("securePricePerHr"))
        dcs = [d["dataCenterId"] for d in (g.get("dataCenterAvailability") or [])
               if d.get("stockStatus") not in (None, "", "none")]
        if not (v and price and v >= min_vcpu) or (secure_only and not dcs):
            continue
        rows.append({"name": g["displayName"], "gpu_id": g["gpuId"], "price": price,
                     "vcpu": v, "ram": m, "datacenters": dcs,
                     "vcpu_per_dollar": round(v / price, 1)})
    return sorted(rows, key=lambda r: -r["vcpu_per_dollar"])


class RunPodWorker(SSHWorker):
    """Rented, billed per second, and destroyed when the run finishes."""

    ephemeral = True

    def __init__(self, gpu_id: str, datacenter: str, volume_id: Optional[str] = None,
                 image: str = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404",
                 container_disk_gb: int = 60, ssh_key: str = "~/.ssh/id_ed25519",
                 name: str = "draftzero", cfg: Optional[WorkerConfig] = None):
        self.gpu_id, self.datacenter, self.volume_id = gpu_id, datacenter, volume_id
        self.image, self.container_disk_gb, self.pod_name = image, container_disk_gb, name
        self.ssh_key, self.pod_id, self.pod = ssh_key, None, None
        super().__init__(SSHTarget(host="", key=ssh_key),
                         cfg or WorkerConfig(name=name, workdir="/workspace/draftzero",
                                             deck_root="/workspace/decks",
                                             persist="/workspace/persist"))

    # ── lifecycle ──
    def provision(self, wait: str = "10m") -> None:
        args = ["pod", "create", "--name", self.pod_name, "--gpu-id", self.gpu_id,
                "--cloud-type", "SECURE", "--data-center-ids", self.datacenter,
                "--image", self.image, "--container-disk-in-gb", str(self.container_disk_gb),
                "--ports", "22/tcp", "--wait", "--wait-timeout", wait]
        if self.volume_id:
            args += ["--network-volume-id", self.volume_id]
        self.pod = _ctl_json(*args)
        self.pod_id = self.pod.get("id") or (self.pod.get("ssh") or {}).get("id")
        ssh = self.pod.get("ssh") or {}
        self.target = SSHTarget(host=ssh["ip"], user="root", port=int(ssh["port"]), key=self.ssh_key)
        if not self.wait_ready():
            raise RuntimeError(f"pod {self.pod_id} never answered ssh")

    def teardown(self) -> Result:
        """Destroy the pod. Billing only stops on remove; a stopped pod still bills disk."""
        if not self.pod_id:
            return Result(0, "no pod to tear down")
        _ctl("pod", "stop", self.pod_id)
        r = _ctl("pod", "remove", self.pod_id)
        if r.ok:
            self.pod_id = None
        return r

    # ── facts about the machine, as opposed to what it advertises ──
    def quota(self) -> dict:
        r = self.run(
            "echo cores=$(awk '{print $1/$2}' /sys/fs/cgroup/cpu.max 2>/dev/null || "
            "echo $(( $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us) / $(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us) )));"
            " echo ram_gb=$(( $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null ||"
            " cat /sys/fs/cgroup/memory.max) / 1024/1024/1024 ));"
            " echo nproc=$(nproc)")
        out = {}
        for line in r.out.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
        return out

    def balance(self) -> Optional[float]:
        try:
            return _ctl_json("user").get("clientBalance")
        except Exception:
            return None


def ensure_volume(name: str, size_gb: int, datacenter: str) -> str:
    """Return the id of a network volume with this name, creating it if needed."""
    for v in _ctl_json("network-volume", "list") or []:
        if v.get("name") == name:
            return v["id"]
    return _ctl_json("network-volume", "create", "--name", name,
                     "--size", str(size_gb), "--data-center-id", datacenter)["id"]
