"""RunPod pods for the compute benchmark (docs/020), from the laptop: list offers, create, inspect, remove.

    python tools/compute_bench/pods.py offers                    # CPU flavors and GPU types: $/hr, vCPU, RAM, per vCPU
    python tools/compute_bench/pods.py create --name cb-cpu5c --cpu cpu5c --vcpu 32
    python tools/compute_bench/pods.py create --name cb-2000ada --gpu "NVIDIA RTX 2000 Ada Generation" --min-vcpu 32 --min-ram 30
    python tools/compute_bench/pods.py get <pod id>              # status, ssh host:port, vCPU, RAM, $/hr
    python tools/compute_bench/pods.py ls
    python tools/compute_bench/pods.py rm <pod id>

The key comes from RUNPOD_API_KEY (~/.runpod/env) and goes in an Authorization header, never a URL. RunPod's API
sits behind a filter that refuses urllib's default User-Agent, so requests carry curl's.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

REST = "https://rest.runpod.io/v1"
GQL = "https://api.runpod.io/graphql"
IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
CPU_FLAVORS = {"cpu3c": 2, "cpu3g": 4, "cpu3m": 8, "cpu5c": 2, "cpu5g": 4, "cpu5m": 8}   # flavor -> GB per vCPU


def _req(method: str, url: str, body: dict | None = None) -> dict | list:
    key = os.environ["RUNPOD_API_KEY"]
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}", "User-Agent": "curl/8.7.1"})
    try:
        with urllib.request.urlopen(r, timeout=60) as f:
            txt = f.read().decode()
            return json.loads(txt) if txt.strip() else {}
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {url}: HTTP {e.code}: {e.read().decode()[:1000]}")


def gql(query: str) -> dict:
    return _req("POST", GQL, {"query": query})


def offers(min_vcpu: int = 16) -> list[dict]:
    rows = []
    for fl, gb in CPU_FLAVORS.items():
        for v in (8, 16, 32):
            iid = f"{fl}-{v}-{v * gb}"
            d = gql('query { cpuFlavors { id specifics(input:{instanceId:"%s"}) { securePrice } } }' % iid)
            p = d["data"]["cpuFlavors"][0]["specifics"]["securePrice"]
            rows.append({"kind": "cpu", "offer": iid, "cloud": "secure", "price": p, "vcpu": v, "ram": v * gb})
    for secure in ("true", "false"):
        d = gql("query { gpuTypes { id displayName lowestPrice(input:{gpuCount:1, secureCloud:%s, minVcpuCount:%d}) "
                "{ uninterruptablePrice minVcpu minMemory stockStatus } } }" % (secure, min_vcpu))
        for g in d["data"]["gpuTypes"]:
            lp = g.get("lowestPrice") or {}
            if lp.get("uninterruptablePrice") and lp.get("stockStatus"):
                rows.append({"kind": "gpu", "offer": g["displayName"], "gpu_id": g["id"],
                             "cloud": "secure" if secure == "true" else "community", "price": lp["uninterruptablePrice"],
                             "vcpu": lp["minVcpu"], "ram": lp["minMemory"], "stock": lp["stockStatus"]})
    for r in rows:
        r["usd_per_vcpu_hr"] = round(r["price"] / r["vcpu"], 4) if r.get("vcpu") else None
        r["usd_per_gb_hr"] = round(r["price"] / r["ram"], 4) if r.get("ram") else None
    return sorted(rows, key=lambda r: r["usd_per_vcpu_hr"] or 9)


def create(a) -> dict:
    body = {"name": a.name, "imageName": a.image, "containerDiskInGb": a.disk, "ports": ["22/tcp"],
            "cloudType": "COMMUNITY" if a.community else "SECURE"}
    if a.cpu:
        body.update(computeType="CPU", cpuFlavorIds=[a.cpu], vcpuCount=a.vcpu, cpuFlavorPriority="custom")
    else:
        body.update(computeType="GPU", gpuTypeIds=[a.gpu], gpuCount=1, minVCPUPerGPU=a.min_vcpu,
                    minRAMPerGPU=a.min_ram, allowedCudaVersions=["12.8", "12.9", "13.0"])
        if a.community:
            body["supportPublicIp"] = True
    if a.dc:
        body.update(dataCenterIds=a.dc.split(","), dataCenterPriority="custom")
    return _req("POST", f"{REST}/pods", body)


def get(pod_id: str) -> dict:
    return _req("GET", f"{REST}/pods/{pod_id}")


def brief(p: dict) -> dict:
    pm = p.get("portMappings") or {}
    return {k: p.get(k) for k in ("id", "name", "desiredStatus", "costPerHr", "vcpuCount", "memoryInGb", "publicIp",
                                  "lastStatusChange")} | {"ssh_port": pm.get("22"), "gpu": (p.get("gpu") or {}).get("displayName"),
                                                          "machine": {k: (p.get("machine") or {}).get(k) for k in
                                                                      ("dataCenterId", "cpuTypeId", "gpuTypeId", "location")}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("offers")
    o.add_argument("--min-vcpu", type=int, default=16)
    o.add_argument("--json", default=None)
    c = sub.add_parser("create")
    c.add_argument("--name", required=True)
    c.add_argument("--cpu", default=None, help="a CPU flavor: " + ", ".join(CPU_FLAVORS))
    c.add_argument("--vcpu", type=int, default=32)
    c.add_argument("--gpu", default=None, help="a GPU type id, e.g. 'NVIDIA GeForce RTX 4090'")
    c.add_argument("--min-vcpu", type=int, default=16)
    c.add_argument("--min-ram", type=int, default=30)
    c.add_argument("--community", action="store_true")
    c.add_argument("--dc", default=None)
    c.add_argument("--image", default=IMAGE)
    c.add_argument("--disk", type=int, default=40)
    g = sub.add_parser("get")
    g.add_argument("id")
    sub.add_parser("ls")
    r = sub.add_parser("rm")
    r.add_argument("id")
    a = ap.parse_args(argv)
    if a.cmd == "offers":
        rows = offers(a.min_vcpu)
        if a.json:
            open(a.json, "w").write(json.dumps(rows, indent=1))
        for x in rows:
            print(f"{x['kind']:3} {x['cloud']:9} {x['offer'][:34]:34} ${x['price']:<6} {x['vcpu']:>3} vCPU {x['ram']:>4} GB "
                  f"${x['usd_per_vcpu_hr']:.4f}/vCPU-h ${x['usd_per_gb_hr']:.4f}/GB-h {x.get('stock') or ''}")
    elif a.cmd == "create":
        if bool(a.cpu) == bool(a.gpu):
            ap.error("one of --cpu or --gpu")
        print(json.dumps(brief(create(a)), indent=1))
    elif a.cmd == "get":
        print(json.dumps(brief(get(a.id)), indent=1))
    elif a.cmd == "ls":
        for p in _req("GET", f"{REST}/pods"):
            print(json.dumps(brief(p)))
    elif a.cmd == "rm":
        print(_req("DELETE", f"{REST}/pods/{a.id}") or "removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
