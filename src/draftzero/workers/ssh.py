"""ssh.py — any machine reachable over SSH: a box on the LAN, r1, a rented pod.

This is the workhorse. RunPodWorker subclasses it, because once a pod exists it is simply an SSH host, and the relay
(draftzero.loop.relay) reaches its hosts through an SSHTarget too:

    t = SSHTarget.parse("root@213.192.2.92 -p 40066")      # the "ssh" field of a relay host
    t.ssh_argv("echo ok")      ["ssh", "-o", "BatchMode=yes", ..., "-p", "40066", "root@213.192.2.92", "echo ok"]
    t.rsync_shell()            the `rsync -e` value with the same options

Every connection is non-interactive (BatchMode: a missing key fails instead of prompting). A rented pod gets a fresh
host key on an address another pod may have used, so its targets are `strict=False` (no known_hosts entry kept);
machines we own keep accept-new host key checking.
"""
from __future__ import annotations

import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from draftzero.workers.base import (DEFAULT_EXCLUDES, RSYNC_FLAGS, Result, Worker, WorkerConfig, local_run)


@dataclass
class SSHTarget:
    host: str
    user: str = "root"
    port: int = 22
    key: Optional[str] = None
    strict: bool = True                    # False: no host key checking (rented pods)
    options: tuple[str, ...] = field(default_factory=tuple)   # extra `-o` values, e.g. "ControlMaster=auto"

    @classmethod
    def parse(cls, spec: str, **kw) -> "SSHTarget":
        """From an ssh-style string: "user@host", "user@host -p 2222", "-p 2222 -i ~/.ssh/k user@host -o X=Y"."""
        args = shlex.split(spec)
        port, key, opts, dest = 22, None, [], None
        i = 0
        while i < len(args):
            a = args[i]
            if a in ("-p", "-i", "-o"):
                if i + 1 >= len(args):
                    raise ValueError(f"ssh spec {spec!r}: {a} needs a value")
                v = args[i + 1]
                if a == "-p":
                    port = int(v)
                elif a == "-i":
                    key = v
                else:
                    opts.append(v)
                i += 2
                continue
            if a.startswith("-"):
                raise ValueError(f"ssh spec {spec!r}: option {a} is not supported (use -p, -i, -o)")
            if dest is not None:
                raise ValueError(f"ssh spec {spec!r}: two destinations ({dest}, {a})")
            dest = a
            i += 1
        if not dest:
            raise ValueError(f"ssh spec {spec!r} has no user@host")
        user, _, host = dest.rpartition("@")
        return cls(host=host, user=user or "root", port=port, key=kw.pop("key", key),
                   options=tuple(opts) + tuple(kw.pop("options", ())), **kw)

    @property
    def dest(self) -> str:
        return f"{self.user}@{self.host}"

    @property
    def spec(self) -> str:
        """Back to the string `parse` reads."""
        s = f"{self.dest} -p {self.port}"
        if self.key:
            s += f" -i {shlex.quote(str(self.key))}"
        return s + "".join(f" -o {shlex.quote(o)}" for o in self.options)

    def opts(self) -> list[str]:
        o = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30"]
        if self.strict:
            o += ["-o", "StrictHostKeyChecking=accept-new"]
        else:
            o += ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR"]
        for extra in self.options:
            o += ["-o", extra]
        o += ["-p", str(self.port)]
        if self.key:
            o += ["-i", str(Path(self.key).expanduser())]
        return o

    def ssh_argv(self, cmd: Optional[str] = None) -> list[str]:
        return ["ssh", *self.opts(), self.dest, *([cmd] if cmd is not None else [])]

    def rsync_shell(self) -> str:
        """`rsync -e` value: ssh with the same options."""
        return shlex.join(["ssh", *self.opts()])

    def remote(self, path: str) -> str:
        """rsync's name for a path on this host."""
        return f"{self.dest}:{path}"


class SSHWorker(Worker):
    ephemeral = False

    def __init__(self, target: SSHTarget, cfg: Optional[WorkerConfig] = None,
                 runner: Callable[..., Result] = local_run):
        super().__init__(cfg or WorkerConfig(name=target.host))
        self.target = target
        self.runner = runner

    def run(self, cmd: str, timeout: Optional[int] = None, input: Optional[str] = None) -> Result:
        return self.runner(self.target.ssh_argv(cmd), timeout=timeout, input=input)

    def push(self, local: Path, remote: str, excludes: Optional[list[str]] = None) -> Result:
        ex = []
        for pattern in (excludes if excludes is not None else DEFAULT_EXCLUDES):
            ex += ["--exclude", pattern]
        return self.runner(["rsync", *RSYNC_FLAGS, *ex, "-e", self.target.rsync_shell(),
                            f"{Path(local)}/", self.target.remote(f"{remote}/")])

    def fetch(self, remote: str, local: Path) -> Result:
        Path(local).mkdir(parents=True, exist_ok=True)
        return self.runner(["rsync", *RSYNC_FLAGS, "-e", self.target.rsync_shell(),
                            self.target.remote(f"{remote}/"), f"{Path(local)}/"])

    def wait_ready(self, tries: int = 30, delay: float = 10, sleep: Callable[[float], None] = time.sleep) -> bool:
        for i in range(tries):
            if self.run("echo ok", timeout=30).ok:
                return True
            if i + 1 < tries:
                sleep(delay)
        return False
