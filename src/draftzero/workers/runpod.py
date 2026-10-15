"""runpod.py — rent RunPod pods for actors, fill them, and give them back.

A pod is an SSH host that has to be created and, crucially, destroyed. Actors on it play against a LocalStore on the
pod's disk and the relay (draftzero.loop.relay, running beside the run's store on d1) pulls their shards over SSH,
so a pod holds no DraftZero or Hugging Face token. The RunPod key stays on d1: `api_key()` reads it with
`secrets.load("/root/.runpod/env")`; it travels only in an Authorization header and is scrubbed from every error.

The expected flow (the integration agent's first real pod; spend a dollar or two, never add credits):

    from draftzero.workers import runpod as rp
    from draftzero.loop.relay import Relay
    api = rp.RunPodAPI()
    api.balance()                                  # {clientBalance, currentSpendPerHr, spendLimit}
    offers = api.offers(min_vcpu=8)                # Community GPU pods, most vCPU per dollar first, in stock only
    spec = rp.PodSpec(name="dz-a1", gpu=offers[0]["gpu_id"], max_price=0.25, min_vcpu=8)
    b = rp.Bundle(repo=Path("/work/dz-main"), engine_root=Path("/work/gorge-fork"),
                  decks=Path("data/gorge/decks"), pool=Path("data/gorge/pool.tsv"))   # or assets/sample/...
    pod = rp.up(api, spec, max_hours=2, files=b)   # create -> wait for SSH -> arm -> bundle -> bootstrap; any failure
                                                   # after create terminates the pod
    relay = Relay("runs/x", [{"name": "dz-a1", "ssh": pod.target.spec, "root": b.remote_run}])
    relay.sync_once()                              # config.yml + control.json (+ versions) onto the pod
    pod.worker(b).start_actor(store=f"{b.remote_run}/store", machine="dz-a1")
    relay.run(interval=60)                         # until the run stops; then
    api.terminate(pod.id); api.list()              # billing stops only on terminate; check nothing is left

Steps, each usable alone:

- `offers()`: GPU types on Community Cloud ranked by vCPU per dollar (self-play is CPU-bound; the GPU idles), from the
  GraphQL catalogue (`lowestPrice` is the only place vCPU is exposed). `cpu_quote()`: a CPU flavor's price and the
  data centers that have it (the v2 catalog; a CPU create must be pinned to one).
- `create(spec)`: REST `POST /v1/pods`. **Community Cloud only**, refused unless `spec.allow_secure` -- and CPU pods
  exist only on Secure Cloud, so they need it (docs/020 §7). Refused when the current quote is above
  `spec.max_price`, when the balance is below an hour of it, or when a pod of that name exists (a create can fail
  after the pod exists: a retry must not make a second one, and every pod of that name is then terminated). A pod
  that bills above max_price is terminated at once. This is the only way pods are rented here: `RunPodCtl` (runpodctl,
  which has no price cap) only reads the account.
- `wait_ssh(id)`: until the pod has a public IP and port 22 mapping that answers with an SSH banner. Through RunPod's
  SSH proxy (`<podHostId>@ssh.runpod.io`) there is only a PTY: no rsync, so no relay. A network that blocks high
  ports (docs/005) cannot run relay actors.
- `arm(target, max_hours)`: deploy/runpod_arm.sh piped over SSH: the pod terminates itself after max_hours with its
  own key, even if d1 dies (retrying the terminate for up to 30 min, each try logged in /root/selfdestruct.log; it
  arms nothing and fails without the pod's key). Do it first; `up` arms it for what is left of its
  max_hours since the create.
- `bundle(target, b)`: rsync the dzgorge binary + bin/dzgorge.build.json and .cards (at the same absolute path as
  on d1, so a config's engine.root works unchanged), the deck pool to <repo>/data/gorge/, and the repo (git archive
  of HEAD, or rsync of the working tree) to <remote_root>/draftzero.
- `bootstrap(target, b)`: deploy/pod_bootstrap.sh on the pod: a venv with CPU torch, `pip install -e .`, `dz version`
  and a one-game dzgorge smoke.

Network volumes (docs/035): `create_volume(name, size_gb, dc)` rents a MooseFS volume in one data center ($0.07 a
GB-month; it bills until deleted, pods or not), and `PodSpec(volume=ID)` mounts it at /workspace on a Secure Cloud pod
pinned to that data center. Several pods can mount one volume at once, so a run's store can live on it: every role
opens the same folder (`--store /workspace/runs/<run>/store`), with no relay and no token on the pods. `delete_volume`
is permanent and is for people to run (`dz pod volume rm`).

Notes that cost real time to learn (docs/005, docs/020): pods are cgroup-capped below what they advertise (size
workers from draftzero.resources.cpu_quota, never nproc); Community hosts often stall pulling big images (the default
image is RunPod's 0.7 GB base; CPU torch comes from pip); stock moves by the minute; a stopped pod still bills.

    python -m draftzero.workers.runpod offers [--min-vcpu 8] [--max-price 0.3]
    python -m draftzero.workers.runpod balance | ls | get ID | rm ID
"""
from __future__ import annotations

import argparse
import json
import os
import posixpath
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional

from draftzero import secrets
from draftzero.data.paths import GORGE_DECKS, GORGE_POOL, REPO
from draftzero.workers.base import DEFAULT_EXCLUDES, RSYNC_FLAGS, Result, WorkerConfig, local_run, q
from draftzero.workers.ssh import SSHTarget, SSHWorker

REST = "https://rest.runpod.io/v1"
GQL = "https://api.runpod.io/graphql"
CATALOG_CPU = "https://api.runpod.io/v2/catalog/cpus"
VOLUMES = f"{REST}/networkvolumes"
VOLUME_MOUNT = "/workspace"
VOLUME_USD_PER_GB_MONTH = 0.07                   # STANDARD volumes, the first TB (docs.runpod.io/pods/pricing, Oct 2026)
MIN_VOLUME_GB, MAX_VOLUME_GB = 10, 50            # RunPod's floor; this module's cap (a volume grows but never shrinks)
IMAGE = "runpod/base:1.4.0-ubuntu2404"          # 0.7 GB; deploy/pod_bootstrap.sh installs CPU torch
IMAGE_PYTORCH = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"   # 10.7 GB, CUDA torch: for training on a pod
CPU_FLAVORS = {"cpu3c": 2, "cpu3g": 4, "cpu3m": 8, "cpu5c": 2, "cpu5g": 4, "cpu5m": 8}   # flavor -> GB per vCPU
ENV_FILES = ("/root/.runpod/env", "~/.runpod/env")
CTL = os.environ.get("RUNPODCTL", "~/.local/bin/runpodctl")
USER_AGENT = "curl/8.7.1"      # RunPod's API filter refuses urllib's default User-Agent
ARM_SCRIPT = REPO / "deploy" / "runpod_arm.sh"
BOOTSTRAP = "deploy/pod_bootstrap.sh"
MAX_ARM_HOURS = 72.0
MIN_ARM_S = 60                                   # a pod whose SSH comes up with less of max_hours left is removed
SSH_KEY = "~/.ssh/id_ed25519"


class RunPodError(RuntimeError):
    pass


def api_key(env_file: str | None = None) -> str:
    """The RunPod API key: from the env file (parsed, never sourced), else $RUNPOD_API_KEY. Never print it."""
    for f in ([env_file] if env_file else ENV_FILES):
        v = secrets.load(f, default_key="RUNPOD_API_KEY", export=False).get("RUNPOD_API_KEY")
        if v:
            return v
    if os.environ.get("RUNPOD_API_KEY"):
        return os.environ["RUNPOD_API_KEY"]
    raise RunPodError(f"no RunPod API key: put `export RUNPOD_API_KEY=...` in {ENV_FILES[0]} (chmod 600)")


Http = Callable[[str, str, Optional[bytes], dict], tuple[int, str]]


def urllib_http(method: str, url: str, data: Optional[bytes], headers: dict) -> tuple[int, str]:
    r = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=60) as f:
            return f.status, f.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


@dataclass
class PodSpec:
    """What to rent. `max_price` ($/h) is required: offers above it are refused, a pod billing above it is removed."""
    name: str
    max_price: float
    gpu: Optional[str] = None             # a GPU type id ("NVIDIA GeForce RTX 3090"): a GPU pod, bought for its vCPUs
    cpu: Optional[str] = None             # a CPU flavor (cpu3c, cpu5c, ...): Secure Cloud only
    vcpu: int = 8                         # CPU pods: vCPUs
    min_vcpu: int = 8                     # GPU pods: vCPU floor
    min_ram: int = 16                     # GPU pods: GB floor
    image: str = IMAGE
    disk_gb: int = 30
    datacenters: tuple[str, ...] = ()
    cloud: str = "COMMUNITY"
    allow_secure: bool = False            # Secure Cloud costs more; set only when Dan said so (and for CPU pods)
    cuda: tuple[str, ...] = ()            # allowedCudaVersions, for a CUDA image
    env: dict = field(default_factory=dict)
    volume: Optional[str] = None          # a network volume id: Secure Cloud, pinned to the volume's data center
    volume_mount: str = VOLUME_MOUNT

    def validate(self) -> "PodSpec":
        if bool(self.gpu) == bool(self.cpu):
            raise ValueError("a pod is either gpu= or cpu=")
        if not (self.max_price and self.max_price > 0):
            raise ValueError("max_price ($/h) must be positive")
        if self.cloud not in ("COMMUNITY", "SECURE"):
            raise ValueError(f"cloud {self.cloud!r}: COMMUNITY or SECURE")
        if self.cpu:
            if self.cpu not in CPU_FLAVORS:
                raise ValueError(f"cpu flavor {self.cpu!r}: one of {', '.join(CPU_FLAVORS)}")
            if self.cloud != "SECURE":
                raise ValueError("CPU pods exist only on Secure Cloud: cloud='SECURE' (and allow_secure=True)")
            if not self.datacenters:
                raise ValueError("a CPU pod must be pinned to a data center that has it (cpu_quote)")
        if self.cloud == "SECURE" and not self.allow_secure:
            raise ValueError("Secure Cloud is refused unless allow_secure=True: actors rent Community Cloud")
        if self.volume:
            if self.cloud != "SECURE":
                raise ValueError("network volumes exist only on Secure Cloud: cloud='SECURE' (and allow_secure=True)")
            if len(self.datacenters) != 1:
                raise ValueError("a pod with a network volume is pinned to the volume's data center: exactly one")
            if not posixpath.isabs(self.volume_mount) or self.volume_mount == "/":
                raise ValueError(f"volume_mount {self.volume_mount!r}: an absolute path other than /")
        return self


def pod_request(spec: PodSpec) -> dict:
    """The REST `POST /v1/pods` body for a spec."""
    spec.validate()
    body = {"name": spec.name, "imageName": spec.image, "containerDiskInGb": spec.disk_gb, "ports": ["22/tcp"],
            "cloudType": spec.cloud}
    if spec.cpu:
        body.update(computeType="CPU", cpuFlavorIds=[spec.cpu], vcpuCount=spec.vcpu, cpuFlavorPriority="custom")
    else:
        body.update(computeType="GPU", gpuTypeIds=[spec.gpu], gpuCount=1, minVCPUPerGPU=spec.min_vcpu,
                    minRAMPerGPU=spec.min_ram)
        if spec.cloud == "COMMUNITY":
            body["supportPublicIp"] = True       # without one there is no direct SSH, so no relay
        if spec.cuda:
            body["allowedCudaVersions"] = list(spec.cuda)
    if spec.datacenters:
        body.update(dataCenterIds=list(spec.datacenters), dataCenterPriority="custom")
    if spec.volume:
        body.update(networkVolumeId=spec.volume, volumeMountPath=spec.volume_mount)
    if spec.env:
        body["env"] = {str(k): str(v) for k, v in spec.env.items()}
    return body


def ssh_endpoint(pod: Mapping) -> Optional[tuple[str, int]]:
    """(public IP, port mapped to 22) of a REST pod, or None while it has none."""
    ip = pod.get("publicIp")
    port = (pod.get("portMappings") or {}).get("22")
    return (ip, int(port)) if ip and port else None


def ssh_banner(host: str, port: int, timeout: float = 8.0) -> bool:
    """True when host:port answers with an SSH banner (what runpodctl --wait checks; no key needed)."""
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.settimeout(timeout)
            return s.recv(64).startswith(b"SSH-")
    except OSError:
        return False


def brief(p: Mapping) -> dict:
    """The fields worth printing about a pod."""
    ep = ssh_endpoint(p)
    m = p.get("machine") or {}
    return {k: p.get(k) for k in ("id", "name", "desiredStatus", "costPerHr", "vcpuCount", "memoryInGb",
                                  "lastStatusChange")} | {
        "ssh": f"root@{ep[0]} -p {ep[1]}" if ep else None, "gpu": (p.get("gpu") or {}).get("displayName"),
        "volume": p.get("networkVolumeId") or (p.get("networkVolume") or {}).get("id"),
        "machine": {k: m.get(k) for k in ("dataCenterId", "cpuTypeId", "gpuTypeId", "location")}}


class RunPodAPI:
    """RunPod's REST and GraphQL APIs with the key in a header. `http` is pluggable (tests pass a fake)."""

    def __init__(self, key: str | None = None, *, http: Http | None = None, env_file: str | None = None,
                 log: Callable[[str], None] = lambda m: print(m, file=sys.stderr),
                 sleep: Callable[[float], None] = time.sleep):
        self._key, self._env_file = key, env_file
        self.http = http or urllib_http
        self.log, self.sleep = log, sleep

    @property
    def key(self) -> str:
        if self._key is None:
            self._key = api_key(self._env_file)
        return self._key

    def __repr__(self) -> str:
        return f"<RunPodAPI key={secrets.redact(self._key) if self._key else 'unloaded'}>"

    def _scrub(self, text: str) -> str:
        return text.replace(self._key, secrets.redact(self._key)) if self._key else text

    def request(self, method: str, url: str, body=None):
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}", "User-Agent": USER_AGENT}
        status, text = self.http(method, url, data, headers)
        if status >= 400:
            raise RunPodError(f"{method} {url}: HTTP {status}: {self._scrub(text)[:800]}")
        return json.loads(text) if text and text.strip() else {}

    def gql(self, query: str, variables: dict | None = None) -> dict:
        r = self.request("POST", GQL, {"query": query, **({"variables": variables} if variables else {})})
        if r.get("errors"):
            raise RunPodError(f"GraphQL: {self._scrub(json.dumps(r['errors']))[:800]}")
        return r.get("data") or {}

    # ── money ──
    def balance(self) -> dict:
        """{clientBalance, currentSpendPerHr, spendLimit}. Read-only: nothing here adds credits."""
        return self.gql("query { myself { clientBalance currentSpendPerHr spendLimit } }")["myself"]

    def require_balance(self, usd: float) -> float:
        b = float(self.balance().get("clientBalance") or 0.0)
        if b < usd:
            raise RunPodError(f"balance ${b:.2f} is below the ${usd:.2f} this needs; RunPod credits are never added here")
        return b

    # ── offers ──
    def offers(self, min_vcpu: int = 8, *, max_price: float | None = None, community_only: bool = True) -> list[dict]:
        """In-stock GPU offers, most vCPU per dollar first: {gpu_id, offer, cloud, price, vcpu, ram, stock,
        usd_per_vcpu_hr}. A quote is the cheapest host meeting the vCPU floor, so the pod can come smaller: check
        vcpuCount after create."""
        rows = []
        for secure in ((False,) if community_only else (False, True)):
            d = self.gql("query { gpuTypes { id displayName lowestPrice(input:{gpuCount:1, secureCloud:%s, "
                         "minVcpuCount:%d}) { uninterruptablePrice minVcpu minMemory stockStatus } } }"
                         % ("true" if secure else "false", int(min_vcpu)))
            for g in d.get("gpuTypes") or []:
                lp = g.get("lowestPrice") or {}
                price, v = lp.get("uninterruptablePrice"), lp.get("minVcpu")
                if not (price and v and lp.get("stockStatus")):
                    continue
                if max_price is not None and price > max_price:
                    continue
                rows.append({"gpu_id": g["id"], "offer": g.get("displayName") or g["id"],
                             "cloud": "SECURE" if secure else "COMMUNITY", "price": price, "vcpu": v,
                             "ram": lp.get("minMemory"), "stock": lp["stockStatus"],
                             "usd_per_vcpu_hr": round(price / v, 4)})
        return sorted(rows, key=lambda r: (r["usd_per_vcpu_hr"], r["price"]))

    def quote(self, spec: PodSpec) -> Optional[float]:
        """The current $/h for a spec (None: not in stock)."""
        if spec.cpu:
            price, dcs = self.cpu_quote(spec.cpu, spec.vcpu)
            return price if set(spec.datacenters) & set(dcs) else None
        for o in self.offers(spec.min_vcpu, community_only=spec.cloud == "COMMUNITY"):
            if o["gpu_id"] == spec.gpu and o["cloud"] == spec.cloud:
                return o["price"]
        return None

    def cpu_quote(self, flavor: str, vcpu: int) -> tuple[float, list[str]]:
        """A CPU flavor at a size: ($/h, data centers that have it), from the v2 catalog (Secure Cloud)."""
        d = self.request("GET", f"{CATALOG_CPU}/{flavor}?include=AVAILABILITY&product=POD&vcpuCount={int(vcpu)}")
        per = float((d.get("price") or {}).get("securePerVcpu") or 0)
        dcs = [x["id"] for x in d.get("dataCenters") or [] if x.get("availability") not in (None, "NONE")]
        return round(per * vcpu, 4), dcs

    # ── pods ──
    def list(self) -> list[dict]:
        r = self.request("GET", f"{REST}/pods")
        return r if isinstance(r, list) else r.get("pods", [])

    def get(self, pod_id: str) -> dict:
        return self.request("GET", f"{REST}/pods/{pod_id}")

    def find(self, name: str) -> list[dict]:
        return [p for p in self.list() if p.get("name") == name]

    def terminate(self, pod_id: str) -> None:
        """Remove the pod: billing stops only here (a stopped pod still bills its disk)."""
        self.request("DELETE", f"{REST}/pods/{pod_id}")
        self.log(f"runpod: terminated {pod_id}")

    def create(self, spec: PodSpec, *, balance_hours: float = 1.0) -> dict:
        """Rent a pod (REST). Refuses: Secure Cloud without allow_secure, a quote above max_price or none in stock,
        a balance below `balance_hours` at max_price, a pod of the same name, a network volume that does not exist
        or is in another data center than the pod's. Terminates a pod billing above max_price.
        A POST that fails (a timeout, a lost or garbled reply, an HTTP error) may still have made the pod: every pod
        of that name is then terminated (`remove_named`) and the error says what happened to it."""
        body = pod_request(spec)
        if spec.volume:
            dc = self.volume(spec.volume).get("dataCenterId")
            if (dc,) != tuple(spec.datacenters):
                raise RunPodError(f"network volume {spec.volume} is in {dc}, the pod is pinned to "
                                  f"{', '.join(spec.datacenters)}: pin it to {dc}")
        if self.find(spec.name):
            raise RunPodError(f"a pod named {spec.name!r} exists: check `runpodctl pod list` before any retry")
        self.require_balance(spec.max_price * balance_hours)
        price = self.quote(spec)
        if price is None:
            raise RunPodError(f"no {spec.gpu or spec.cpu} offer in stock on {spec.cloud} Cloud")
        if price > spec.max_price:
            raise RunPodError(f"{spec.gpu or spec.cpu} costs ${price}/h, above max_price ${spec.max_price}/h")
        try:
            pod = self.request("POST", f"{REST}/pods", body)
            if not (isinstance(pod, dict) and pod.get("id")):
                raise RunPodError(f"create returned no pod id: {self._scrub(json.dumps(pod))[:400]}")
        except BaseException as e:      # Ctrl-C too: the POST may have gone through
            outcome = self.remove_named(spec.name)
            if not isinstance(e, Exception):
                raise
            raise RunPodError(f"creating pod {spec.name!r} failed ({type(e).__name__}: {self._scrub(str(e))[:400]}); "
                              f"{outcome}") from e
        pid = pod["id"]
        cost, vcpu = pod.get("costPerHr"), pod.get("vcpuCount")
        if cost is not None and float(cost) > spec.max_price:
            self.terminate(pid)
            raise RunPodError(f"pod {pid} bills ${cost}/h, above max_price ${spec.max_price}/h: terminated")
        if spec.gpu and vcpu is not None and float(vcpu) < spec.min_vcpu:   # a quote is a floor, not a promise
            self.terminate(pid)
            raise RunPodError(f"pod {pid} came with {vcpu} vCPU, below min_vcpu {spec.min_vcpu}: terminated")
        self.log(f"runpod: created {pid} {spec.name} ({spec.gpu or spec.cpu}, ${cost}/h, {pod.get('vcpuCount')} vCPU)")
        return pod

    def remove_named(self, name: str, *, looks: int = 3, wait: float = 10.0) -> str:
        """After a failed create: terminate every pod named `name` (create refuses a name in use, so any is the one it
        made), looking `looks` times `wait` s apart since a pod can show up late. Never raises; logs loudly when a pod
        may be left billing, and returns what happened."""
        problem = None
        for i in range(looks):
            try:
                ids = [p.get("id") for p in self.find(name)]
            except Exception as e:                       # noqa: BLE001
                problem = f"cannot list the pods ({self._scrub(str(e))[:200]})"
                break
            if ids:
                failed = []
                for pid in ids:
                    try:
                        self.terminate(pid)
                    except Exception as e:               # noqa: BLE001
                        failed.append(f"{pid} ({self._scrub(str(e))[:200]})")
                if not failed:
                    return f"removed pod {', '.join(ids)} that it made anyway"
                problem = f"terminating {', '.join(failed)} failed"
                break
            if i + 1 < looks:
                self.sleep(wait)
        if problem is None:
            msg = f"no pod named {name!r} showed up, but one may still appear: check `dz pod list` in a few minutes"
            self.log(f"runpod: WARNING: {msg}")
            return msg
        msg = (f"TERMINATE FAILED: {problem}: a pod named {name!r} may be billing; remove it with "
               f"`dz pod down --name {name}`")
        self.log(f"runpod: {msg}")
        return msg

    # ── network volumes ──
    def volumes(self) -> list[dict]:
        """[{id, name, size, dataCenterId}]"""
        r = self.request("GET", VOLUMES)
        return r if isinstance(r, list) else r.get("networkVolumes", [])

    def volume(self, vid: str) -> dict:
        try:
            return self.request("GET", f"{VOLUMES}/{vid}")
        except RunPodError as e:
            if "HTTP 404" in str(e):
                raise RunPodError(f"no network volume {vid} on this account (`dz pod volume list`)") from None
            raise

    def create_volume(self, name: str, size_gb: int, datacenter: str, *, max_gb: int = MAX_VOLUME_GB) -> dict:
        """Rent a STANDARD network volume. It bills about size_gb x $0.07 a month until deleted, attached or not.
        Refuses a size outside [10, max_gb] GB and a name already in use (a retry must not make a second one)."""
        size_gb = int(size_gb)
        if not (MIN_VOLUME_GB <= size_gb <= max_gb):
            raise RunPodError(f"size {size_gb} GB: between {MIN_VOLUME_GB} and {max_gb} (volumes grow, never shrink)")
        if not datacenter:
            raise RunPodError("a network volume lives in one data center: give one")
        same = [v for v in self.volumes() if v.get("name") == name]
        if same:
            raise RunPodError(f"a network volume named {name!r} exists ({same[0].get('id')}): use it, or another name")
        try:
            v = self.request("POST", VOLUMES, {"name": name, "size": size_gb, "dataCenterId": datacenter})
        except RunPodError as e:
            raise RunPodError(f"{e}; a lost reply may still have made it: check `dz pod volume list`") from None
        if not (isinstance(v, dict) and v.get("id")):
            raise RunPodError(f"create returned no volume id: {self._scrub(json.dumps(v))[:400]}; check "
                              "`dz pod volume list`")
        self.log(f"runpod: created network volume {v['id']} {name} ({size_gb} GB in {datacenter}): it bills about "
                 f"${size_gb * VOLUME_USD_PER_GB_MONTH:.2f} a month until deleted")
        return v

    def delete_volume(self, vid: str) -> None:
        """Delete a network volume and everything on it: permanent. RunPod refuses while a pod has it mounted."""
        self.request("DELETE", f"{VOLUMES}/{vid}")
        self.log(f"runpod: deleted network volume {vid}")

    def wait_ssh(self, pod_id: str, *, timeout: float = 900, poll: float = 10, key: str = SSH_KEY,
                 probe: Callable[[str, int], bool] = ssh_banner, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> SSHTarget:
        """Wait until the pod's public SSH port answers; return its SSHTarget (no host key checking: pods are new)."""
        end = clock() + timeout
        last = None
        while True:
            pod = self.get(pod_id)
            ep = ssh_endpoint(pod)
            if ep and probe(*ep):
                self.log(f"runpod: {pod_id} ssh root@{ep[0]} -p {ep[1]}")
                return SSHTarget(host=ep[0], user="root", port=ep[1], key=key, strict=False)
            last = ep
            if clock() >= end:
                raise RunPodError(
                    f"pod {pod_id}: no SSH banner after {timeout:.0f}s (endpoint {last}). Check `runpodctl pod logs "
                    f"{pod_id}` and whether this network lets out high ports (docs/005); the SSH proxy has no rsync")
            sleep(poll)


class RunPodCtl:
    """runpodctl (2.x) for read-only account queries: user, gpu list. The key goes in the child's environment.
    It rents nothing: runpodctl has no price cap, so pods are created only by RunPodAPI.create, which holds the money
    guards (max_price, balance, a unique name, removal of a pod that bills above the cap or failed to come up)."""

    def __init__(self, ctl: str = CTL, runner: Callable[..., Result] = local_run, key: str | None = None):
        self.ctl, self.runner, self._key = os.path.expanduser(ctl), runner, key

    def _run(self, argv: list[str], timeout: float = 120) -> object:
        env = dict(os.environ)
        if self._key:
            env["RUNPOD_API_KEY"] = self._key
        r = self.runner(argv, timeout=timeout, env=env)
        if not r.ok:
            raise RunPodError(f"runpodctl {' '.join(argv[1:3])} failed (exit {r.code}): {(r.err or r.out)[-500:]}")
        return json.loads(r.out) if r.out.strip() else {}

    def user(self) -> dict:
        """Account and balance: clientBalance, currentSpendPerHr, spendLimit (and email: do not print it)."""
        return self._run([self.ctl, "user", "-o", "json"])

    def gpus(self) -> list:
        return self._run([self.ctl, "gpu", "list", "-o", "json"])


# ── on the pod ──
def arm(target: SSHTarget, max_hours: float, *, runner: Callable[..., Result] = local_run,
        script: Path = ARM_SCRIPT) -> Result:
    """Arm (or move) the pod's self-destruct: deploy/runpod_arm.sh, piped over SSH, terminates the pod after
    max_hours with the pod's own key. Arm first, before anything that can hang."""
    if not (0 < max_hours <= MAX_ARM_HOURS):
        raise ValueError(f"max_hours must be in (0, {MAX_ARM_HOURS}]")
    secs = int(round(max_hours * 3600))
    r = runner(target.ssh_argv(f"bash -s -- {secs}"), timeout=120, input=Path(script).read_text())
    if not r.ok or "armed:" not in r.out:
        raise RunPodError(f"arming the self-destruct failed (exit {r.code}): {(r.err or r.out)[-400:]}")
    return r


@dataclass
class Step:
    what: str
    argv: list[str]
    input: Optional[str] = None
    timeout: float = 1800


@dataclass
class Bundle:
    """What a pod needs to run actors, and where it goes there.

    remote_root/draftzero      the repo (cwd of actors: a config's relative data paths resolve here)
    remote_root/draftzero/data/gorge/{decks/, pool.tsv}
    remote_engine/bin/{dzgorge, dzgorge.build.json}, remote_engine/.cards/   (default: the same path as engine_root)
    remote_root/run/store      the actors' store (the relay host's root is remote_root/run)
    remote_root/venv           made by deploy/pod_bootstrap.sh
    """
    repo: Path = REPO
    engine_root: Path = Path(os.environ.get("GORGE_DIR", "/work/gorge-fork"))
    decks: Path = GORGE_DECKS
    pool: Path = GORGE_POOL
    remote_root: str = "/root/dz"
    remote_engine_root: Optional[str] = None
    repo_mode: str = "git"          # git: archive of HEAD (committed code only) | rsync: the working tree

    @property
    def remote_repo(self) -> str:
        return posixpath.join(self.remote_root, "draftzero")

    @property
    def remote_engine(self) -> str:
        return self.remote_engine_root or str(self.engine_root)

    @property
    def remote_run(self) -> str:
        return posixpath.join(self.remote_root, "run")

    @property
    def remote_python(self) -> str:
        return posixpath.join(self.remote_root, "venv/bin/python")

    def missing(self) -> list[str]:
        need = [self.engine_root / "bin/dzgorge", self.engine_root / "bin/dzgorge.build.json",
                self.engine_root / ".cards", self.decks, self.pool, self.repo / "pyproject.toml"]
        return [str(p) for p in need if not Path(p).exists()]


def bundle_steps(target: SSHTarget, b: Bundle, tmp: Path) -> list[Step]:
    """The commands that fill a pod (run in order by `bundle`)."""
    if b.repo_mode not in ("git", "rsync"):
        raise ValueError(f"repo_mode {b.repo_mode!r}: git or rsync")
    rs = ["rsync", *RSYNC_FLAGS, "-e", target.rsync_shell()]
    eng, repo = Path(b.engine_root), Path(b.repo)
    steps = [Step("prepare the pod", target.ssh_argv(
        "command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync >/dev/null); "
        f"mkdir -p {q(b.remote_repo)}/data/gorge {q(b.remote_engine)}/bin {q(b.remote_run)}/store"), timeout=600)]
    if b.repo_mode == "git":
        tar = tmp / "draftzero.tar.gz"
        remote_tar = posixpath.join(b.remote_root, "draftzero.tar.gz")
        steps += [Step("archive the repo (HEAD)",
                       ["git", "-C", str(repo), "archive", "--format=tar.gz", "-o", str(tar), "HEAD"]),
                  Step("send the repo", [*rs, str(tar), target.remote(remote_tar)]),
                  Step("unpack the repo",
                       target.ssh_argv(f"tar -xzf {q(remote_tar)} -C {q(b.remote_repo)} && rm -f {q(remote_tar)}"))]
    else:
        ex = [x for p in DEFAULT_EXCLUDES for x in ("--exclude", p)]
        steps.append(Step("send the working tree", [*rs, *ex, f"{repo}/", target.remote(f"{b.remote_repo}/")]))
    steps += [
        Step("send dzgorge", [*rs, str(eng / "bin/dzgorge"), str(eng / "bin/dzgorge.build.json"),
                              target.remote(f"{b.remote_engine}/bin/")]),
        Step("send .cards", [*rs, "--delete", f"{Path(eng / '.cards').resolve()}/",
                             target.remote(f"{b.remote_engine}/.cards/")]),
        Step("send the decks", [*rs, "--delete", f"{Path(b.decks).resolve()}/",
                                target.remote(f"{b.remote_repo}/data/gorge/decks/")]),
        Step("send the pool", [*rs, str(Path(b.pool).resolve()), target.remote(f"{b.remote_repo}/data/gorge/pool.tsv")]),
    ]
    return steps


def _run_steps(steps: list[Step], runner: Callable[..., Result], log: Callable[[str], None]) -> None:
    for s in steps:
        t0 = time.monotonic()
        r = runner(s.argv, timeout=s.timeout, input=s.input) if s.input is not None else runner(s.argv, timeout=s.timeout)
        if not r.ok:
            raise RunPodError(f"bundle: {s.what} failed (exit {r.code}): {(r.err or r.out)[-500:]}")
        log(f"bundle: {s.what} ({time.monotonic() - t0:.0f}s)")


def bundle(target: SSHTarget, b: Bundle, *, runner: Callable[..., Result] = local_run,
           log: Callable[[str], None] = lambda m: print(m, file=sys.stderr)) -> None:
    """Copy the engine, its card corpus, the deck pool and the repo to the pod."""
    missing = b.missing()
    if missing:
        raise RunPodError(f"bundle: missing here: {', '.join(missing)}")
    with tempfile.TemporaryDirectory(prefix="dz-bundle-") as tmp:
        _run_steps(bundle_steps(target, b, Path(tmp)), runner, log)


def bootstrap_argv(target: SSHTarget, b: Bundle) -> list[str]:
    return target.ssh_argv(f"bash {q(posixpath.join(b.remote_repo, BOOTSTRAP))} {q(b.remote_root)} {q(b.remote_engine)}")


def bootstrap(target: SSHTarget, b: Bundle, *, runner: Callable[..., Result] = local_run, timeout: float = 2400) -> Result:
    """deploy/pod_bootstrap.sh on the pod: python deps, `dz version`, a one-game dzgorge smoke."""
    r = runner(bootstrap_argv(target, b), timeout=timeout)
    if not r.ok or "pod_bootstrap: ready" not in r.out:
        raise RunPodError(f"pod bootstrap failed (exit {r.code}): {(r.err or r.out)[-800:]}")
    return r


@dataclass
class Pod:
    id: str
    target: SSHTarget
    info: dict

    def worker(self, b: Bundle, name: str | None = None) -> SSHWorker:
        """An SSHWorker for this pod's bundle: actors run from the repo with the venv's python."""
        return SSHWorker(self.target, WorkerConfig(name=name or self.info.get("name") or self.id,
                                                   workdir=b.remote_repo, python=b.remote_python))


def up(api: RunPodAPI, spec: PodSpec, *, max_hours: float, files: Bundle | None = None, setup: bool = True,
       key: str = SSH_KEY, wait_timeout: float = 900, runner: Callable[..., Result] = local_run,
       probe: Callable[[str, int], bool] = ssh_banner, sleep: Callable[[float], None] = time.sleep,
       keep_on_failure: bool = False, clock: Callable[[], float] = time.monotonic) -> Pod:
    """Create -> wait for SSH -> arm the self-destruct -> (bundle `files` -> bootstrap if `setup`).
    max_hours is the pod's whole life, counted from the create (it bills from there): SSH must come up within it,
    and the self-destruct is armed for what is left, so the pod costs at most max_hours x max_price even if this
    machine dies. Any failure after the create terminates the pod. keep_on_failure keeps it (its id in the error)
    only once it is armed, so it still removes itself; a pod that is not armed yet is always terminated. The balance
    must cover max_hours at max_price (1-4 h of it)."""
    if not (0 < max_hours <= MAX_ARM_HOURS):
        raise ValueError(f"max_hours must be in (0, {MAX_ARM_HOURS}]")
    if files is not None and files.missing():
        raise RunPodError(f"bundle: missing here: {', '.join(files.missing())}")
    t0 = clock()
    pod = api.create(spec, balance_hours=max(1.0, min(max_hours, 4.0)))
    pid, armed = pod["id"], False
    try:
        target = api.wait_ssh(pid, timeout=min(wait_timeout, max_hours * 3600 - (clock() - t0)), key=key,
                              probe=probe, sleep=sleep)
        left = max_hours * 3600 - (clock() - t0)
        if left < MIN_ARM_S:
            raise RunPodError(f"pod {pid}: SSH came up with {max(left, 0):.0f} s of --max-hours {max_hours:g} left")
        arm(target, left / 3600, runner=runner)
        armed = True
        if files is not None:
            bundle(target, files, runner=runner, log=api.log)
            if setup:
                bootstrap(target, files, runner=runner)
    except BaseException as e:
        if keep_on_failure and armed:
            raise RunPodError(f"pod {pid} kept after: {e} (it terminates itself {max_hours:g} h after its create; "
                              f"`dz pod down {pid}` removes it sooner)") from e
        try:
            api.terminate(pid)
        except Exception as te:   # the original error matters more; say loudly that a pod may be billing
            api.log(f"runpod: TERMINATE FAILED for {pid} ({te}): remove it by hand (runpodctl pod delete {pid})")
        raise
    return Pod(pid, target, pod)


class RunPodWorker(SSHWorker):
    """A rented pod as a Worker: provision() = up() without a bundle; teardown() terminates it."""

    ephemeral = True

    def __init__(self, spec: PodSpec, *, max_hours: float, key: str = SSH_KEY, api: RunPodAPI | None = None,
                 cfg: WorkerConfig | None = None, runner: Callable[..., Result] = local_run):
        spec.validate()
        self.spec, self.max_hours, self.key = spec, max_hours, key
        self.api = api or RunPodAPI()
        self.pod_id: Optional[str] = None
        b = Bundle()
        super().__init__(SSHTarget(host="", key=key, strict=False),
                         cfg or WorkerConfig(name=spec.name, workdir=b.remote_repo, python=b.remote_python),
                         runner=runner)

    def provision(self, probe: Callable[[str, int], bool] = ssh_banner,
                  sleep: Callable[[float], None] = time.sleep) -> None:
        pod = up(self.api, self.spec, max_hours=self.max_hours, key=self.key, runner=self.runner, probe=probe, sleep=sleep)
        self.pod_id, self.target = pod.id, pod.target

    def teardown(self) -> Result:
        """Terminate the pod. Billing only stops on terminate; a stopped pod still bills disk."""
        if not self.pod_id:
            return Result(0, "no pod to tear down")
        self.api.terminate(self.pod_id)
        self.pod_id = None
        return Result(0, "terminated")

    # ── facts about the machine, as opposed to what it advertises ──
    def quota(self) -> dict:
        r = self.run(
            "echo cores=$(awk '{print $1/$2}' /sys/fs/cgroup/cpu.max 2>/dev/null || "
            "echo $(( $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us) / $(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us) )));"
            " echo ram_gb=$(( $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null ||"
            " cat /sys/fs/cgroup/memory.max) / 1024/1024/1024 ));"
            " echo nproc=$(nproc)")
        return dict(line.split("=", 1) for line in r.out.splitlines() if "=" in line)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m draftzero.workers.runpod", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("offers")
    o.add_argument("--min-vcpu", type=int, default=8)
    o.add_argument("--max-price", type=float, default=None)
    o.add_argument("--secure", action="store_true", help="also list Secure Cloud")
    sub.add_parser("balance")
    sub.add_parser("ls")
    sub.add_parser("get").add_argument("id")
    sub.add_parser("rm").add_argument("id")
    a = ap.parse_args(argv)
    api = RunPodAPI()
    if a.cmd == "offers":
        for x in api.offers(a.min_vcpu, max_price=a.max_price, community_only=not a.secure):
            print(f"{x['cloud']:9} {x['offer'][:30]:30} ${x['price']:<6} {x['vcpu']:>3} vCPU {x['ram'] or 0:>4} GB "
                  f"${x['usd_per_vcpu_hr']:.4f}/vCPU-h {x['stock']}")
    elif a.cmd == "balance":
        print(json.dumps(api.balance()))
    elif a.cmd == "ls":
        for p in api.list():
            print(json.dumps(brief(p)))
    elif a.cmd == "get":
        print(json.dumps(brief(api.get(a.id)), indent=1))
    elif a.cmd == "rm":
        api.terminate(a.id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
