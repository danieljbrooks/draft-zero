"""Workers: places self-play can run.

    from draftzero.workers import make_worker
    w = make_worker("local")
    w = make_worker("ssh", host="gpubox.lan", user="dan")
    w = make_worker("runpod", gpu_id="NVIDIA RTX 4000 Ada Generation", datacenter="EU-RO-1")
"""
from draftzero.workers.base import Result, Worker, WorkerConfig
from draftzero.workers.local import LocalWorker
from draftzero.workers.ssh import SSHTarget, SSHWorker

__all__ = ["Worker", "WorkerConfig", "Result", "LocalWorker", "SSHWorker", "SSHTarget",
           "make_worker"]


def make_worker(kind: str, **kw) -> Worker:
    kind = kind.lower()
    if kind == "local":
        return LocalWorker(WorkerConfig(name="local", workdir=kw.get("workdir", ".")))
    if kind == "ssh":
        target = SSHTarget(host=kw["host"], user=kw.get("user", "root"),
                           port=int(kw.get("port", 22)), key=kw.get("key"))
        return SSHWorker(target, WorkerConfig(
            name=kw.get("name", kw["host"]), workdir=kw.get("workdir", "~/draftzero"),
            deck_root=kw.get("deck_root"), persist=kw.get("persist")))
    if kind == "runpod":
        from draftzero.workers.runpod import RunPodWorker  # imported lazily: needs runpodctl
        return RunPodWorker(gpu_id=kw["gpu_id"], datacenter=kw["datacenter"],
                            volume_id=kw.get("volume_id"), name=kw.get("name", "draftzero"),
                            ssh_key=kw.get("key", "~/.ssh/id_ed25519"))
    raise ValueError(f"unknown worker kind: {kind!r} (local, ssh, runpod)")
