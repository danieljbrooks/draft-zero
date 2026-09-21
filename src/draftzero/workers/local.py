"""local.py — run the loop on this machine. No provisioning, no network, no billing."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from draftzero.workers.base import Result, Worker, WorkerConfig, local_run


class LocalWorker(Worker):
    ephemeral = False

    def __init__(self, cfg: Optional[WorkerConfig] = None):
        super().__init__(cfg or WorkerConfig(name="local"))

    def run(self, cmd: str, timeout: Optional[int] = None) -> Result:
        return local_run(cmd, timeout=timeout, cwd=self.cfg.workdir)

    def push(self, local: Path, remote: str, excludes: Optional[list[str]] = None) -> Result:
        """A local worker already has the code. Copy only when src and dst differ."""
        src, dst = Path(local).resolve(), Path(remote).resolve()
        if src == dst:
            return Result(0, "local worker: source and destination are the same, nothing to push")
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return Result(0, f"copied {src} -> {dst}")

    def fetch(self, remote: str, local: Path) -> Result:
        return self.push(Path(remote), str(local))
