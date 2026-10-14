"""Local mode's process handling (draftzero.loop.local) with stand-in role commands: logs, a clean stop once the
controller finishes, a failed role, the time limit, and no stray child processes."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from draftzero.loop import local

PY = sys.executable


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:                                                 # a zombie is not alive
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


def test_role_commands(tmp_path):
    cmds = local.role_commands("c.yml", "runs/x", tmp_path, "box", 2, poll_s=5)
    assert list(cmds) == ["controller", "learner", "evaluator", "actor-0", "actor-1"]
    assert cmds["controller"][1:] == ["-m", "draftzero.loop.cli", "controller", "--config", "c.yml",
                                      "--store", "runs/x", "--poll", "5"]
    a1 = cmds["actor-1"]
    assert a1[a1.index("--machine") + 1] == "box-1" and a1[a1.index("--work-dir") + 1] == str(tmp_path / "actor-box-1")
    assert cmds["learner"][cmds["learner"].index("--machine") + 1] == "box-learner"


def test_stops_the_others_when_the_controller_finishes(tmp_path):
    pidfile = tmp_path / "grandchild.pid"
    # an "actor" that starts a long-lived child in its own session (as gorge does) and then waits
    actor = (f"import subprocess, time; p = subprocess.Popen(['sleep', '120'], start_new_session=True); "
             f"open({str(pidfile)!r}, 'w').write(str(p.pid)); print('actor up', flush=True); time.sleep(120)")
    cmds = {"controller": [PY, "-c", "import time; time.sleep(1.5); print('controller done')"],
            "actor-0": [PY, "-c", actor]}
    t0 = time.time()
    rc = local.run("c.yml", "unused", tmp_path / "w", commands=cmds, log=lambda *_: None, grace_s=0.5, tick_s=0.2)
    assert rc == 0 and time.time() - t0 < 60
    assert "controller done" in (tmp_path / "w" / "logs" / "controller.log").read_text()
    assert "actor up" in (tmp_path / "w" / "logs" / "actor-0.log").read_text()
    gc = int(pidfile.read_text())
    for _ in range(50):
        if not _alive(gc):
            break
        time.sleep(0.1)
    assert not _alive(gc), "a role's child process outlived local mode"


def test_a_failed_role_fails_the_run_and_the_time_limit(tmp_path):
    cmds = {"controller": [PY, "-c", "import time; time.sleep(120)"],
            "learner": [PY, "-c", "raise SystemExit(3)"]}
    assert local.run("c", "s", tmp_path / "a", commands=cmds, log=lambda *_: None, grace_s=0.5, tick_s=0.2) == 1
    cmds = {"controller": [PY, "-c", "import time; time.sleep(120)"]}
    assert local.run("c", "s", tmp_path / "b", commands=cmds, log=lambda *_: None, grace_s=0.5, tick_s=0.2,
                     max_hours=1 / 3600) == 2


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP], ids=["SIGTERM", "SIGHUP"])
def test_sigterm_or_sighup_stops_the_roles_as_ctrl_c_does(tmp_path, sig):
    """`kill`, a closed terminal or a harness timeout must not orphan the roles (they run in their own sessions)."""
    role = "import os, sys, time; open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(300)"
    cmds = {name: [PY, "-c", role, str(tmp_path / f"{name}.pid")] for name in ("controller", "actor-0")}
    parent = (f"import sys; from draftzero.loop import local; sys.exit(local.run('c', 's', {str(tmp_path / 'w')!r}, "
              f"commands={cmds!r}, log=lambda *a: None, grace_s=60, tick_s=0.2))")
    env = dict(os.environ, PYTHONPATH=str(Path(local.__file__).resolve().parents[2]))
    p = subprocess.Popen([PY, "-c", parent], env=env)
    pids = []
    try:
        for _ in range(200):
            if all((tmp_path / f"{n}.pid").exists() and (tmp_path / f"{n}.pid").read_text() for n in cmds):
                break
            time.sleep(0.05)
        pids = [int((tmp_path / f"{n}.pid").read_text()) for n in cmds]
        p.send_signal(sig)
        assert p.wait(timeout=60) == 128 + sig
        for _ in range(50):
            if not any(_alive(pid) for pid in pids):
                break
            time.sleep(0.1)
        assert not [pid for pid in pids if _alive(pid)], "roles outlived local mode"
    finally:
        p.kill()
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(not Path("/proc").exists(), reason="descendants() reads /proc")
def test_descendants_finds_grandchildren():
    import subprocess
    p = subprocess.Popen([PY, "-c", "import subprocess, time; subprocess.Popen(['sleep', '30']); time.sleep(30)"])
    try:
        for _ in range(50):
            if len(local.descendants(p.pid)) >= 1:
                break
            time.sleep(0.1)
        kids = local.descendants(p.pid)
        assert kids and local.descendants(os.getpid()) and p.pid in local.descendants(os.getpid())
    finally:
        for pid in local.descendants(p.pid):
            os.kill(pid, 9)
        p.kill()
        p.wait()
