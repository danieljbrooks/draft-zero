"""Serve an experiment #2 network for the search benchmark: R replicas of MageZero v0.2's
inference server on ports base, base+100, ... (one process each, so one GIL each).

    MZ_ACTION_VOCAB=$PWD/assets/vocab/FDN_SPG.tsv python tools/search_bench/serve.py \\
        --deck FDN_exp2 --checkpoint gen18 --replicas 6 --base-port 50052 --clients 8

Runs until killed; kills its servers on exit.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from draftzero import engine  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="FDN_exp2")
    ap.add_argument("--checkpoint", default="gen18")
    ap.add_argument("--replicas", type=int, default=4)
    ap.add_argument("--base-port", type=int, default=50052)
    ap.add_argument("--clients", type=int, default=8, help="bridge workers per replica")
    ap.add_argument("--log-dir", default="runs/search_bench/servers")
    a = ap.parse_args()
    log = Path(a.log_dir)
    log.mkdir(parents=True, exist_ok=True)
    procs = []

    def stop(*_):
        for p in procs:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for i in range(a.replicas):
        procs.append(engine.start_server(a.deck, 1, a.base_port + 100 * i, log, checkpoint=a.checkpoint,
                                         clients=a.clients))
    print("ports", ",".join(str(a.base_port + 100 * i) for i in range(a.replicas)), flush=True)
    while True:
        time.sleep(30)
        for p in procs:
            if p.poll() is not None:
                print(f"server pid {p.pid} exited {p.returncode}", flush=True)
                stop()


if __name__ == "__main__":
    raise SystemExit(main())
