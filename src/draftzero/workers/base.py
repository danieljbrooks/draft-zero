"""base.py — what DraftZero needs from a machine that can run games.

A worker is somewhere self-play can happen: this laptop, a box on the LAN, a rented GPU
pod. They differ in only two places -- whether the machine has to be created and destroyed,
and how a command gets to it. Everything above this interface (configs, pools, the loop,
the watchdog) is identical, so adding a worker means implementing `run`/`push`/`fetch`
and, if the machine is rented, `provision`/`teardown`.
"""
from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Network volumes (RunPod uses MooseFS) reject chown, which plain `rsync -a` attempts and
# then fails the whole transfer on. Never use -a for a worker push.
RSYNC_FLAGS = ["-rlptz", "--no-o", "--no-g"]

DEFAULT_EXCLUDES = [".git", ".venv", "__pycache__", "*.pyc", ".pytest_cache",
                    "runs", "models", "data/decks", "logs"]


@dataclass
class Result:
    code: int
    out: str
    err: str = ""

    @property
    def ok(self) -> bool:
        return self.code == 0

    def check(self, what: str) -> "Result":
        if not self.ok:
            raise RuntimeError(f"{what} failed (exit {self.code}): {(self.err or self.out)[:500]}")
        return self


@dataclass
class WorkerConfig:
    name: str = "worker"
    workdir: str = "."
    deck_root: Optional[str] = None
    persist: Optional[str] = None
    env: dict = field(default_factory=dict)


class Worker(ABC):
    """One machine that can run the DraftZero loop."""

    #: rented workers cost money per hour and must be torn down; local ones do not
    ephemeral: bool = False

    def __init__(self, cfg: WorkerConfig):
        self.cfg = cfg

    # ── lifecycle (no-ops for machines you already own) ──
    def provision(self) -> None:
        return None

    def teardown(self) -> None:
        return None

    # ── the two things that actually differ ──
    @abstractmethod
    def run(self, cmd: str, timeout: Optional[int] = None) -> Result:
        """Run a shell command in the worker's workdir."""

    @abstractmethod
    def push(self, local: Path, remote: str, excludes: Optional[list[str]] = None) -> Result:
        """Mirror a local directory onto the worker."""

    @abstractmethod
    def fetch(self, remote: str, local: Path) -> Result:
        """Copy a path off the worker."""

    # ── shared behaviour ──
    def setup(self) -> Result:
        """Install system and python dependencies. Idempotent."""
        return self.run(f"cd {self.cfg.workdir} && bash deploy/bootstrap.sh")

    def launch(self, config: str, mode: str = "--fresh", extra_env: Optional[dict] = None) -> Result:
        env = {**self.cfg.env, **(extra_env or {})}
        if self.cfg.deck_root:
            env.setdefault("MZ_DECK_DIR", self.cfg.deck_root)
        if self.cfg.persist:
            env.setdefault("DZ_PERSIST", self.cfg.persist)
        exports = " ".join(f"{k}={shell_quote(str(v))}" for k, v in env.items())
        return self.run(f"cd {self.cfg.workdir} && {exports} bash deploy/launch.sh {config} {mode}")

    def status(self) -> dict:
        """Games played and whether the loop is still alive."""
        r = self.run(
            f"cd {self.cfg.workdir} && "
            "D=$(ls -1dt runs/*/ 2>/dev/null | head -1); "
            'echo "run=${D%/}"; '
            'echo "games=$(cat ${D}games.jsonl 2>/dev/null | wc -l)"; '
            'echo "alive=$(pgrep -fc \'draftzero.loop\' || echo 0)"')
        out = {}
        for line in r.out.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
        return out

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.cfg.name}>"


def shell_quote(s: str) -> str:
    import shlex
    return shlex.quote(s)


def local_run(cmd: list[str] | str, timeout: Optional[int] = None, cwd: Optional[str] = None) -> Result:
    shell = isinstance(cmd, str)
    p = subprocess.run(cmd, shell=shell, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    return Result(p.returncode, p.stdout, p.stderr)
