"""Local mode (`dz run`): every role on this machine, as separate processes sharing one store (normally a folder).
The same programs as on many machines; only the store differs.

    python -m draftzero.loop.cli local --config C [--store S] [--actors N] [--work-dir DIR] [--max-hours H]

Starts the controller, the learner, one evaluator and N actors (`python -m draftzero.loop.cli <role>`), each in its
own session with its log in <work-dir>/logs/<role>.log and its cache in <work-dir>/<role>-<machine>. It returns
once the controller has stopped the run (budget, `stop`) and the others have followed. If a role dies, or at
--max-hours, it asks the run to stop (stop.request) and waits up to `grace_s` for the roles to exit; on Ctrl-C it
stops them at once. Whatever is left gets SIGTERM, then SIGKILL, including gorge processes they started.

Return code: 0 the run stopped normally, 1 a role failed, 2 --max-hours, 130 Ctrl-C.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from draftzero.loop import control as ctl

REPO_SRC = Path(__file__).resolve().parents[2]


def role_commands(config: str, store: str, work_dir: Path, machine: str, actors: int,
                  poll_s: float | None = None) -> dict[str, list[str]]:
    """{log name: argv} for every role."""
    base = [sys.executable, "-m", "draftzero.loop.cli"]
    common = ["--config", str(config), "--store", str(store)] + (["--poll", str(poll_s)] if poll_s else [])

    def role(name: str, m: str) -> list[str]:
        return base + [name, *common, "--machine", m, "--work-dir", str(work_dir / f"{name}-{m}")]
    cmds = {"controller": base + ["controller", *common],
            "learner": role("learner", f"{machine}-learner"),
            "evaluator": role("evaluator", f"{machine}-evaluator")}
    for i in range(actors):
        cmds[f"actor-{i}"] = role("actor", f"{machine}-{i}")
    return cmds


def descendants(pid: int) -> list[int]:
    """Every process below pid (Linux /proc; empty elsewhere): gorge runs in its own session, so killing a role's
    process group alone would leave it running."""
    children: dict[int, list[int]] = {}
    try:
        for d in os.listdir("/proc"):
            if d.isdigit():
                try:
                    with open(f"/proc/{d}/stat") as f:
                        ppid = int(f.read().rsplit(")", 1)[1].split()[1])
                    children.setdefault(ppid, []).append(int(d))
                except (OSError, IndexError, ValueError):
                    pass
    except OSError:
        return []
    out, todo = [], [pid]
    while todo:
        for c in children.get(todo.pop(), []):
            out.append(c)
            todo.append(c)
    return out


def _signal_all(procs: dict, sig: int) -> None:
    for p in procs.values():
        if p.poll() is None:
            for pid in descendants(p.pid):
                try:
                    os.kill(pid, sig)
                except (ProcessLookupError, PermissionError):
                    pass
            try:
                os.killpg(p.pid, sig)
            except (ProcessLookupError, PermissionError):
                pass


def run(config: str, store: str, work_dir: Path, *, machine: str = "local", actors: int = 1,
        max_hours: float | None = None, grace_s: float = 180.0, poll_s: float | None = None, log=print,
        commands: dict[str, list[str]] | None = None, tick_s: float = 2.0) -> int:
    work_dir = Path(work_dir).resolve()
    (work_dir / "logs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_SRC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONUNBUFFERED", "1")
    cmds = commands or role_commands(config, store, work_dir, machine, actors, poll_s)
    st = None
    if commands is None:
        from draftzero.loop.store import open_store
        st = open_store(store)
    procs: dict[str, subprocess.Popen] = {}
    logs = []
    for name, argv in cmds.items():
        lf = open(work_dir / "logs" / f"{name}.log", "ab")
        logs.append(lf)
        procs[name] = subprocess.Popen(argv, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        log(f"local: {name} started (pid {procs[name].pid}), log {work_dir / 'logs' / (name + '.log')}")
    deadline = None if max_hours is None else time.time() + max_hours * 3600
    rc, stopping_since = 0, None

    def ask_stop(why: str, request: bool = True) -> None:
        nonlocal stopping_since
        if stopping_since is None:
            log(f"local: {why}; " + ("asking the run to stop" if request else "waiting for the other roles"))
            stopping_since = time.time()
            if st is None:
                _signal_all(procs, signal.SIGTERM)
            elif request:
                st.put(ctl.STOP, f"stop requested by local mode: {why}\n".encode(), "stop requested")

    try:
        while any(p.poll() is None for p in procs.values()):
            for name, p in procs.items():
                if p.poll() not in (None, 0) and stopping_since is None:
                    rc = 1
                    ask_stop(f"{name} exited with {p.returncode} (see its log)")
            if "controller" in procs and procs["controller"].poll() is not None:
                ask_stop("the controller has stopped the run", request=False)   # control.json says stop
            if deadline is not None and time.time() > deadline and stopping_since is None:
                rc = 2
                ask_stop(f"--max-hours {max_hours} reached")
            if stopping_since is not None and time.time() - stopping_since > grace_s:
                log(f"local: roles still running {grace_s:.0f} s after the stop; terminating them")
                break
            time.sleep(tick_s)
    except KeyboardInterrupt:
        log("local: interrupted; stopping everything")
        rc = 130
    finally:
        below = {pid for p in procs.values() if p.poll() is None for pid in descendants(p.pid)}
        _signal_all(procs, signal.SIGTERM)
        t0 = time.time()
        while any(p.poll() is None for p in procs.values()) and time.time() - t0 < 30:
            time.sleep(0.5)
        _signal_all(procs, signal.SIGKILL)
        for pid in below:                                # orphans whose role already exited
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        for p in procs.values():
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        for lf in logs:
            lf.close()
    for name, p in procs.items():
        log(f"local: {name} exited with {p.returncode}")
    return rc
