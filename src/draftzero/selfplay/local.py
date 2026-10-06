"""Single-machine mode (docs/021 §3): the controller, the trainer and one worker on this box, as three processes
sharing a store (normally a local folder). The same programs as on many machines; only the store differs.

    dz selfplay local --config configs/selfplay_smoke.yml --store runs/selfplay/smoke --max-hours 1

Logs go to <work-dir>/logs/{controller,trainer,worker}.log. It returns when all three have exited (the trainer
after trainer.max_versions, the controller and worker once control.json says stop), and kills them all at
--max-hours, or on Ctrl-C.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from draftzero.selfplay import config as sc


def run(config_path: str, store: str, work_dir: Path, machine: str = "local", max_hours: float | None = None,
        log=print) -> int:
    work_dir = Path(work_dir)
    (work_dir / "logs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(sc.REPO / "src") + (os.pathsep + os.environ["PYTHONPATH"]
                                                             if os.environ.get("PYTHONPATH") else ""))
    base = [sys.executable, "-m", "draftzero.selfplay.cli"]
    roles = {
        "controller": base + ["controller", "--config", config_path, "--store", store],
        "trainer": base + ["trainer", "--config", config_path, "--store", store, "--work-dir", str(work_dir / "trainer")],
        "worker": base + ["worker", "--config", config_path, "--store", store, "--machine", machine,
                          "--work-dir", str(work_dir / f"worker-{machine}")],
    }
    procs = {}
    for role, args in roles.items():
        lf = open(work_dir / "logs" / f"{role}.log", "ab")
        procs[role] = subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        log(f"local: {role} started (pid {procs[role].pid}), log {work_dir / 'logs' / (role + '.log')}")
    deadline = None if max_hours is None else time.time() + max_hours * 3600
    rc = 0

    def kill_all(sig=signal.SIGTERM):
        for p in procs.values():
            if p.poll() is None:
                try:
                    os.killpg(p.pid, sig)
                except ProcessLookupError:
                    pass
    try:
        while any(p.poll() is None for p in procs.values()):
            if deadline is not None and time.time() > deadline:
                log(f"local: --max-hours {max_hours} reached; stopping everything")
                rc = 2
                break
            time.sleep(5)
    except KeyboardInterrupt:
        log("local: interrupted; stopping everything")
        rc = 130
    finally:
        kill_all()
        t0 = time.time()
        while any(p.poll() is None for p in procs.values()) and time.time() - t0 < 30:
            time.sleep(1)
        kill_all(signal.SIGKILL)
    for role, p in procs.items():
        log(f"local: {role} exited with {p.returncode}")
        if p.returncode not in (0, None, -signal.SIGTERM) and rc == 0:
            rc = 1
    return rc
