"""Workers: places actors can run.

    from draftzero.workers import make_worker
    w = make_worker("local")
    w = make_worker("ssh", ssh="dan@gpubox.lan -p 22", workdir="~/draftzero")
    w = make_worker("runpod", gpu="NVIDIA GeForce RTX 3070", max_price=0.20, max_hours=2)   # rents on provision()

An actor on a worker plays against a store on the worker's own disk; draftzero.loop.relay carries its shards to the
run's store. workers.runpod rents and returns RunPod pods (its docstring has the whole flow).
"""
from draftzero.workers.base import Result, Worker, WorkerConfig, actor_command
from draftzero.workers.local import LocalWorker
from draftzero.workers.ssh import SSHTarget, SSHWorker

__all__ = ["Worker", "WorkerConfig", "Result", "LocalWorker", "SSHWorker", "SSHTarget", "actor_command",
           "make_worker"]


def make_worker(kind: str, **kw) -> Worker:
    kind = kind.lower()
    if kind == "local":
        return LocalWorker(WorkerConfig(name="local", workdir=kw.get("workdir", "."), python=kw.get("python", "python")))
    if kind == "ssh":
        target = (SSHTarget.parse(kw["ssh"]) if "ssh" in kw else
                  SSHTarget(host=kw["host"], user=kw.get("user", "root"), port=int(kw.get("port", 22)), key=kw.get("key")))
        return SSHWorker(target, WorkerConfig(name=kw.get("name", target.host), workdir=kw.get("workdir", "~/draftzero"),
                                              python=kw.get("python", "python")))
    if kind == "runpod":
        from draftzero.workers.runpod import PodSpec, RunPodWorker   # lazily: the API key is read on first use
        spec = PodSpec(name=kw.get("name", "draftzero"), max_price=float(kw["max_price"]), gpu=kw.get("gpu"),
                       cpu=kw.get("cpu"), vcpu=int(kw.get("vcpu", 8)), min_vcpu=int(kw.get("min_vcpu", 8)),
                       datacenters=tuple(kw.get("datacenters", ())), allow_secure=bool(kw.get("allow_secure", False)))
        return RunPodWorker(spec, max_hours=float(kw["max_hours"]), key=kw.get("key", "~/.ssh/id_ed25519"))
    raise ValueError(f"unknown worker kind: {kind!r} (local, ssh, runpod)")
