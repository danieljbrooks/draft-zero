"""Local mode (`dz run`): every role on this machine, as separate processes sharing one store (normally a folder).
The same programs as on many machines; only the store differs.

    python -m draftzero.loop.cli local --config C [--store S] [--actors N] [--work-dir DIR] [--max-hours H]
                                       [--allow-config-change]

Starts the controller, the learner, one evaluator and N actors (`python -m draftzero.loop.cli <role>`), each in its
own session with its log in <work-dir>/logs/<role>.log and its cache in <work-dir>/<role>-<machine>
(--allow-config-change goes on to the controller). With builder.learn it also starts the builder learner, one builder
evaluator and M builder actors (--builder-actors, default 1); with player.learn false, no player learner, evaluator
or actors. It returns once the controller has stopped the run (budget,
`stop`) and the others have followed. If a role dies, or at --max-hours, it asks the run to stop (stop.request) and
waits up to `grace_s` for the roles to exit. If the controller itself fails, nothing would act on a stop.request, so
it stops the others at once and leaves none in the store (a stop.request it wrote that no controller acted on is
withdrawn at the end): the next `dz run` resumes the run. On Ctrl-C, SIGTERM or SIGHUP (kill, a harness timeout, a
closed terminal) it stops them at once. Whatever is left gets SIGTERM, then SIGKILL, including gorge processes they
started; a second signal does not cut that short.

Return code: 0 the run stopped normally, 1 a role failed, 2 --max-hours, 130 Ctrl-C, 143 SIGTERM, 129 SIGHUP.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from draftzero.loop import control as ctl

REPO_SRC = Path(__file__).resolve().parents[2]


def role_commands(config: str, store: str, work_dir: Path, machine: str, actors: int,
                  poll_s: float | None = None, allow_config_change: bool = False, *, player: bool = True,
                  builder: bool = False, builder_actors: int = 1) -> dict[str, list[str]]:
    """{log name: argv} for every role; allow_config_change goes to the controller (--allow-config-change).
    player: the player's learner, evaluator and actors (player.learn); builder: the builder's (builder.learn)."""
    base = [sys.executable, "-m", "draftzero.loop.cli"]
    common = ["--config", str(config), "--store", str(store)] + (["--poll", str(poll_s)] if poll_s else [])

    def role(name: str, m: str) -> list[str]:
        return base + [name, *common, "--machine", m, "--work-dir", str(work_dir / f"{name}-{m}")]
    cmds = {"controller": base + ["controller", *common] + (["--allow-config-change"] if allow_config_change else [])}
    if player:
        cmds["learner"] = role("learner", f"{machine}-learner")
        cmds["evaluator"] = role("evaluator", f"{machine}-evaluator")
        for i in range(actors):
            cmds[f"actor-{i}"] = role("actor", f"{machine}-{i}")
    if builder:
        cmds["builder-learner"] = role("builder-learner", f"{machine}-builder-learner")
        cmds["builder-evaluator"] = role("builder-evaluator", f"{machine}-builder-evaluator")
        for i in range(builder_actors):
            cmds[f"builder-actor-{i}"] = role("builder-actor", f"{machine}-b{i}")
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


class Terminated(Exception):
    """SIGTERM or SIGHUP reached local mode: the roles are stopped as on Ctrl-C."""

    def __init__(self, signum: int):
        super().__init__(signal.Signals(signum).name)
        self.signum = signum


def _raise_terminated(signum, frame):
    raise Terminated(signum)


def _set_handlers(sigs, handler) -> dict:
    """Set `handler` for each signal and return the previous handlers (only the main thread can: {} elsewhere)."""
    if threading.current_thread() is not threading.main_thread():
        return {}
    return {s: signal.signal(s, handler) for s in sigs}


def run(config: str, store: str, work_dir: Path, *, machine: str = "local", actors: int = 1,
        max_hours: float | None = None, grace_s: float = 180.0, poll_s: float | None = None, log=print,
        commands: dict[str, list[str]] | None = None, tick_s: float = 2.0, allow_config_change: bool = False,
        player: bool = True, builder: bool = False, builder_actors: int = 1) -> int:
    work_dir = Path(work_dir).resolve()
    (work_dir / "logs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_SRC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONUNBUFFERED", "1")
    cmds = commands or role_commands(config, store, work_dir, machine, actors, poll_s,
                                     allow_config_change=allow_config_change, player=player, builder=builder,
                                     builder_actors=builder_actors)
    st = None
    if commands is None:
        from draftzero.loop.store import open_store
        st = open_store(store)
    procs: dict[str, subprocess.Popen] = {}
    logs = []
    deadline = None if max_hours is None else time.time() + max_hours * 3600
    rc, stopping_since, requested, orphaned = 0, None, None, False

    def ask_stop(why: str, request: bool = True) -> None:
        nonlocal stopping_since, requested
        if stopping_since is None:
            log(f"local: {why}; " + ("asking the run to stop" if request else "waiting for the other roles"))
            stopping_since = time.time()
            if st is None:
                _signal_all(procs, signal.SIGTERM)
            elif request:
                requested = f"stop requested by local mode: {why}\n".encode()
                st.put(ctl.STOP, requested, "stop requested")

    # The roles run in their own sessions, so a signal to this process reaches none of them: SIGTERM and SIGHUP
    # (kill, a harness timeout, a closed terminal) must end in the finally below, as Ctrl-C does.
    previous = _set_handlers((signal.SIGTERM, signal.SIGHUP), _raise_terminated)
    try:
        for name, argv in cmds.items():
            lf = open(work_dir / "logs" / f"{name}.log", "ab")
            logs.append(lf)
            procs[name] = subprocess.Popen(argv, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            log(f"local: {name} started (pid {procs[name].pid}), log {work_dir / 'logs' / (name + '.log')}")
        while any(p.poll() is None for p in procs.values()):
            ctrl = procs.get("controller")
            if ctrl is not None and ctrl.poll() not in (None, 0) and not orphaned:
                # no controller is left to turn a stop.request into control.json stop: stop the others here, and
                # write no request (it would stop the next `dz run` on this store at its first tick)
                rc, orphaned = 1, True
                log(f"local: the controller exited with {ctrl.returncode} (see its log); stopping the other roles")
                stopping_since = stopping_since or time.time()
                _signal_all(procs, signal.SIGTERM)
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
    except Terminated as e:
        log(f"local: {e}; stopping everything")
        rc = 128 + e.signum
    finally:
        # a second Ctrl-C, kill or hang-up must not cut the cleanup short (it would skip the SIGKILL pass)
        previous = {**_set_handlers((signal.SIGINT, signal.SIGTERM, signal.SIGHUP), signal.SIG_IGN), **previous}
        try:
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
        finally:
            for s, h in previous.items():
                signal.signal(s, h if h is not None else signal.SIG_DFL)
    if requested is not None:
        _withdraw(st, requested, log)
    for name, p in procs.items():
        log(f"local: {name} exited with {p.returncode}")
    return rc


def _withdraw(st, requested: bytes, log) -> None:
    """Remove the stop.request local mode wrote when no controller acted on it (control.json doesn't say stop):
    left there, it would stop the next `dz run` on this store at its first tick. A controller that did act on it
    deletes it itself when the run resumes."""
    try:
        if not (st.read_json(ctl.CONTROL) or {}).get("stop") and st.read_bytes(ctl.STOP) == requested:
            st.delete([ctl.STOP], "local mode: withdraw a stop request no controller acted on")
            log("local: no controller acted on the stop request; withdrew it (the next `dz run` resumes the run)")
    except Exception as e:                               # noqa: BLE001 - say so; the roles are already stopped
        log(f"local: could not check {ctl.STOP} ({type(e).__name__}: {e}); delete it before the next `dz run`")
