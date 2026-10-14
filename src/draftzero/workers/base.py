"""base.py — what DraftZero needs from a machine that can run actors.

A worker is somewhere self-play can happen: this machine, a box on the LAN (r1), a rented pod. They differ in only
two places -- whether the machine has to be created and destroyed, and how a command and files get to it. Everything
above this interface (configs, the store, the loop) is identical, so adding a worker means implementing
`run`/`push`/`fetch` and, if the machine is rented, `provision`/`teardown`.

In v0.2.0 a worker runs actors (`python -m draftzero.loop.cli actor`) against a LocalStore on its own disk, and the
relay (`draftzero.loop.relay`) carries their shards back and new versions out, so a worker needs no tokens and no
inbound access beyond SSH. `setup()` runs deploy/pod_bootstrap.sh, `start_actor()` starts one detached actor
process, and `status()` reads the worker's heartbeats and counts its actor processes.
"""
from __future__ import annotations

import json
import posixpath
import shlex
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

# Network volumes (RunPod uses MooseFS) reject chown, which plain `rsync -a` attempts and
# then fails the whole transfer on. Never use -a for a worker push.
RSYNC_FLAGS = ["-rlptz", "--no-o", "--no-g"]

# For pushing a checkout: anchored (leading /) where a bare name would also hit a package directory --
# "data" alone would drop src/draftzero/data.
DEFAULT_EXCLUDES = [".git", ".venv", "__pycache__", "*.pyc", ".pytest_cache", "*.egg-info",
                    "/runs/", "/models/", "/logs/", "/data/", "/ext/", ".cards/"]

ACTOR_MODULE = "draftzero.loop.cli"


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
            raise RuntimeError(f"{what} failed (exit {self.code}): {(self.err or self.out)[-500:]}")
        return self


@dataclass
class WorkerConfig:
    name: str = "worker"
    workdir: str = "."            # the draftzero checkout on the worker; actors run from it (relative data paths)
    python: str = "python"        # an interpreter with draftzero installed (a pod: <root>/venv/bin/python)
    env: dict = field(default_factory=dict)


def q(s) -> str:
    return shlex.quote(str(s))


shell_quote = q


def actor_command(*, workdir: str, python: str, store: str, machine: str, config: str | None = None,
                  log: str | None = None, env: Mapping[str, str] | None = None) -> str:
    """The shell line that starts one actor, detached (nohup, stdin closed, output to a log), and prints its pid.

    `store` is the worker's own store folder; its config.yml (pushed there by the relay) is the default --config,
    so start actors after the relay's first sync. The log goes beside the store: <store>/../actor-<machine>.log."""
    config = config or posixpath.join(store, "config.yml")
    log = log or posixpath.join(posixpath.dirname(store.rstrip("/")) or ".", f"actor-{machine}.log")
    exports = "".join(f"{k}={q(v)} " for k, v in (env or {}).items())
    return (f"cd {q(workdir)} && mkdir -p {q(posixpath.dirname(log) or '.')} && "
            f"{exports}nohup {q(python)} -m {ACTOR_MODULE} actor --config {q(config)} --store {q(store)} "
            f"--machine {q(machine)} > {q(log)} 2>&1 < /dev/null & echo $!")


class Worker(ABC):
    """One machine that can run DraftZero actors."""

    #: rented workers cost money per hour and must be torn down; local ones do not
    ephemeral: bool = False

    def __init__(self, cfg: WorkerConfig):
        self.cfg = cfg

    # ── lifecycle (no-ops for machines you already own) ──
    def provision(self) -> None:
        return None

    def teardown(self) -> None:
        return None

    # ── the things that actually differ ──
    @abstractmethod
    def run(self, cmd: str, timeout: Optional[int] = None, input: Optional[str] = None) -> Result:
        """Run a shell command on the worker (stdin from `input`)."""

    @abstractmethod
    def push(self, local: Path, remote: str, excludes: Optional[list[str]] = None) -> Result:
        """Mirror a local directory onto the worker."""

    @abstractmethod
    def fetch(self, remote: str, local: Path) -> Result:
        """Copy a path off the worker."""

    # ── shared behaviour ──
    def setup(self, root: str, engine_root: str, timeout: int = 3600) -> Result:
        """Install the python side and check it plays (deploy/pod_bootstrap.sh). Idempotent."""
        return self.run(f"bash {q(posixpath.join(self.cfg.workdir, 'deploy/pod_bootstrap.sh'))} {q(root)} {q(engine_root)}",
                        timeout=timeout)

    def start_actor(self, store: str, machine: str, config: str | None = None, log: str | None = None) -> Result:
        """Start one actor process against the worker's store; Result.out is its pid."""
        return self.run(actor_command(workdir=self.cfg.workdir, python=self.cfg.python, store=store, machine=machine,
                                      config=config, log=log, env=self.cfg.env), timeout=60)

    def status(self, store: str) -> dict:
        """{"actors": running actor processes, "heartbeats": {machine: heartbeat}} from the worker's store."""
        # [d]raftzero: a pattern that does not match its own command line (pgrep -f would count this shell)
        r = self.run(f"echo actors=$(pgrep -fc '[d]raftzero.loop.cli actor' || true); "
                     f"for f in {q(store)}/heartbeats/*.json; do [ -f \"$f\" ] && echo \"hb=$(basename \"$f\" .json) $(tr -d '\\n' < \"$f\")\"; done",
                     timeout=60)
        out: dict = {"actors": 0, "heartbeats": {}}
        for line in r.out.splitlines():
            if line.startswith("actors="):
                out["actors"] = int(line[7:].strip() or 0)
            elif line.startswith("hb="):
                name, _, body = line[3:].partition(" ")
                try:
                    out["heartbeats"][name] = json.loads(body)
                except ValueError:
                    out["heartbeats"][name] = {"unparsed": body[:200]}
        return out

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.cfg.name}>"


def local_run(cmd: list[str] | str, timeout: Optional[float] = None, cwd: Optional[str] = None,
              input: Optional[str] = None, env: Optional[Mapping[str, str]] = None) -> Result:
    """Run a command here and capture it. A timeout is a failed Result (code 124), not an exception."""
    shell = isinstance(cmd, str)
    try:
        p = subprocess.run(cmd, shell=shell, capture_output=True, text=True, timeout=timeout, cwd=cwd,
                           input=input, env=dict(env) if env is not None else None)
    except subprocess.TimeoutExpired as e:
        def s(x):
            return x.decode(errors="replace") if isinstance(x, bytes) else (x or "")
        return Result(124, s(e.stdout), s(e.stderr) + f"\ntimed out after {timeout}s")
    return Result(p.returncode, p.stdout, p.stderr)
