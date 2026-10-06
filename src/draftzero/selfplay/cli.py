"""`dz selfplay <role>`: the self-play loop's command line (docs/021 §3).

    dz selfplay worker      --config C --store S [--machine NAME] [--work-dir DIR] [--max-jobs N]
    dz selfplay trainer     --config C --store S [--work-dir DIR]
    dz selfplay controller  --config C --store S [--once]
    dz selfplay local       --config C --store S [--work-dir DIR] [--max-hours H]     all three on this machine
    dz selfplay status      --store S                                                 print status.md
    dz selfplay stop        --store S                                                 ask the controller to stop the run

S is a local folder (one machine) or hf://<dataset>[?weights=<private repo>] (many machines; docs/021 §3.2).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dz selfplay", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="role", required=True)
    for role in ("worker", "trainer", "controller", "local"):
        p = sub.add_parser(role)
        p.add_argument("--config", required=True)
        p.add_argument("--store", required=True)
        if role in ("worker", "trainer", "local"):
            p.add_argument("--work-dir", type=Path, default=None)
        if role in ("worker", "local"):
            p.add_argument("--machine", default=None)
        if role == "worker":
            p.add_argument("--max-jobs", type=int, default=None)
        if role == "controller":
            p.add_argument("--once", action="store_true", help="one tick, then exit")
        if role == "local":
            p.add_argument("--max-hours", type=float, default=None, help="a hard stop for everything")
    for role in ("status", "stop"):
        sub.add_parser(role).add_argument("--store", required=True)
    a = ap.parse_args(argv)

    from draftzero.selfplay.store import open_store
    store = open_store(a.store)
    if a.role == "status":
        b = store.read_bytes("status.md")
        print(b.decode() if b else "no status yet")
        return 0
    if a.role == "stop":
        store.put("stop.request", f"stop requested {time.ctime()}\n".encode(), "stop requested")
        print("stop requested: the controller stops the run at its next tick")
        return 0

    from draftzero.selfplay import config as sc
    cfg = sc.load(a.config)
    if a.role == "worker":
        from draftzero.selfplay.worker import Worker
        Worker(store, cfg, machine=a.machine, work_dir=a.work_dir, log=_log).run(max_jobs=a.max_jobs)
    elif a.role == "trainer":
        from draftzero.selfplay.trainer import Trainer
        Trainer(store, cfg, work_dir=a.work_dir, log=_log).run()
    elif a.role == "controller":
        from draftzero.selfplay.controller import Controller
        c = Controller(store, cfg, log=_log)
        while True:
            ctl = c.tick()
            if a.once or (ctl and ctl.get("stop")):
                break
            time.sleep(cfg["controller"]["poll_s"])
    elif a.role == "local":
        from draftzero.selfplay import local
        wd = a.work_dir or sc.path(f"runs/selfplay/{cfg['run']}")
        return local.run(a.config, a.store, wd, machine=a.machine or "local", max_hours=a.max_hours, log=_log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
