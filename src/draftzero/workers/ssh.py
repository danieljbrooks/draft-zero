"""ssh.py — any machine reachable over SSH: a box on the LAN, a colo server, a rented pod.

This is the workhorse. RunPodWorker subclasses it, because once a pod exists it is simply
an SSH host.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from draftzero.workers.base import (DEFAULT_EXCLUDES, RSYNC_FLAGS, Result, Worker,
                                    WorkerConfig, local_run)


@dataclass
class SSHTarget:
    host: str
    user: str = "root"
    port: int = 22
    key: Optional[str] = None

    @property
    def dest(self) -> str:
        return f"{self.user}@{self.host}"

    def opts(self) -> list[str]:
        o = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20",
             "-p", str(self.port)]
        if self.key:
            o += ["-i", str(Path(self.key).expanduser())]
        return o

    def rsync_shell(self) -> str:
        key = f" -i {Path(self.key).expanduser()}" if self.key else ""
        return f"ssh -o StrictHostKeyChecking=accept-new -p {self.port}{key}"


class SSHWorker(Worker):
    ephemeral = False

    def __init__(self, target: SSHTarget, cfg: Optional[WorkerConfig] = None):
        super().__init__(cfg or WorkerConfig(name=target.host))
        self.target = target

    def run(self, cmd: str, timeout: Optional[int] = None) -> Result:
        return local_run(["ssh", *self.target.opts(), self.target.dest, cmd], timeout=timeout)

    def push(self, local: Path, remote: str, excludes: Optional[list[str]] = None) -> Result:
        ex = []
        for pattern in (excludes if excludes is not None else DEFAULT_EXCLUDES):
            ex += ["--exclude", pattern]
        return local_run(["rsync", *RSYNC_FLAGS, *ex, "-e", self.target.rsync_shell(),
                          f"{Path(local)}/", f"{self.target.dest}:{remote}/"])

    def fetch(self, remote: str, local: Path) -> Result:
        Path(local).mkdir(parents=True, exist_ok=True)
        return local_run(["rsync", *RSYNC_FLAGS, "-e", self.target.rsync_shell(),
                          f"{self.target.dest}:{remote}/", f"{Path(local)}/"])

    def wait_ready(self, tries: int = 30, delay: int = 10) -> bool:
        import time
        for _ in range(tries):
            if self.run("echo ok", timeout=30).ok:
                return True
            time.sleep(delay)
        return False
