"""watchdog.py — keep the weights safe, and stop paying for a pod that is not working.

Runs as its own process beside the trainer, so a hung JVM or a wedged trainer still gets
noticed. Every `--interval` seconds it:

  1. counts games in the run's games.jsonl,
  2. mirrors checkpoints and run artifacts to `--persist` (the network volume, which
     outlives the pod),
  3. decides whether the run is done (target reached), stalled (no new game for
     `--stall-minutes`), or still healthy.

On done or stalled it syncs once more, writes STATUS.json, and — only when that final
sync verified — runs --on-complete. That command is whatever this worker needs: destroy
a rented pod, post a notification, or nothing at all on a machine you own.

The ordering is the whole point: a failed sync never runs --on-complete. Losing a worker
is cheap, losing the weights is not.

  python -m draftzero.watchdog --run-dir runs/<id> --target 20000 --persist data/persist \
      --on-complete "runpodctl pod remove $RUNPOD_POD_ID"
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

def log(msg: str) -> None:
    print(f"[watchdog {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def count_games(run_dir: Path) -> int:
    f = run_dir / "games.jsonl"
    if not f.exists():
        return 0
    try:
        with f.open() as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return -1  # unreadable this tick; treat as "no reading", not as "no progress"


def sync(src_dirs: list[Path], dest: Path) -> tuple[bool, str]:
    """Mirror each src into dest/<name>. Returns (ok, detail)."""
    dest.mkdir(parents=True, exist_ok=True)
    if not shutil.which("rsync"):
        return False, "rsync not installed"
    for src in src_dirs:
        if not src.exists():
            continue
        # --no-o/--no-g: a RunPod network volume is MooseFS and refuses chown, which
        # plain -a attempts and then fails the whole transfer on.
        r = subprocess.run(["rsync", "-rlptD", "--no-o", "--no-g", "--delete",
                            f"{src}/", str(dest / src.name) + "/"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return False, f"rsync {src.name} failed: {r.stderr.strip()[:200]}"
    return True, "ok"


def verify(dest: Path, run_name: str) -> tuple[bool, str]:
    """A sync only counts if the checkpoints actually landed."""
    models = dest / "models"
    ckpts = list(models.rglob("*.pt.gz")) if models.exists() else []
    if not ckpts:
        return False, "no checkpoints present in persist dir"
    if not (dest / "runs" / run_name).exists() and not (dest / run_name).exists():
        return False, "run artifacts not present in persist dir"
    total_mb = sum(c.stat().st_size for c in ckpts) / 1024 ** 2
    return True, f"{len(ckpts)} checkpoints, {total_mb:.1f} MB"


def run_on_complete(cmd: str) -> tuple[bool, str]:
    """Whatever this worker does when the run ends. Kept as a shell command so the
    watchdog stays worker-agnostic: a pod destroys itself, a LAN box does nothing."""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        return r.returncode == 0, (r.stdout or r.stderr).strip()[:300] or f"exit {r.returncode}"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--target", type=int, default=20000)
    ap.add_argument("--persist", required=True, type=Path)
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--interval", type=int, default=300, help="seconds between checks")
    ap.add_argument("--stall-minutes", type=int, default=45)
    ap.add_argument("--max-hours", type=float, default=0,
                    help="stop after this many hours of wall clock (0 = no limit). "
                         "A rented worker bills by the second, so this is the budget cap: "
                         "hours x rate. Deterministic, and needs no API credentials on the worker.")
    ap.add_argument("--on-complete", default=os.environ.get("DZ_ON_COMPLETE", ""),
                    help="shell command to run after a VERIFIED final sync (e.g. destroy the pod)")
    a = ap.parse_args()

    if a.on_complete:
        log(f"on-complete: {a.on_complete}")

    started = time.time()
    last_count, last_progress = -1, started
    log(f"watching {a.run_dir} -> target {a.target} games, persist {a.persist}")

    while True:
        n = count_games(a.run_dir)
        now = time.time()
        if n > last_count:
            last_count, last_progress = n, now
        idle_min = (now - last_progress) / 60

        ok, detail = sync([a.models, a.run_dir], a.persist)
        if not ok:
            log(f"sync FAILED: {detail}")

        elapsed_h = (now - started) / 3600
        done = n >= a.target
        stalled = idle_min >= a.stall_minutes
        over_budget = bool(a.max_hours) and elapsed_h >= a.max_hours
        budget = f" | {elapsed_h:.1f}/{a.max_hours:g}h" if a.max_hours else ""
        log(f"{n}/{a.target} games | idle {idle_min:.0f}m{budget} | sync {'ok' if ok else 'FAILED'}")

        if done or stalled or over_budget:
            reason = ("target reached" if done else
                      f"budget cap: {elapsed_h:.1f}h of {a.max_hours:g}h" if over_budget else
                      f"stalled {idle_min:.0f}m with no new game")
            log(f"stopping: {reason}")
            ok, detail = sync([a.models, a.run_dir], a.persist)
            vok, vdetail = verify(a.persist, a.run_dir.name)
            status = {"reason": reason, "games": n, "target": a.target,
                      "elapsed_hours": round(elapsed_h, 2), "max_hours": a.max_hours or None,
                      "final_sync_ok": ok, "sync_detail": detail,
                      "verified": vok, "verify_detail": vdetail,
                      "at": datetime.now(timezone.utc).isoformat(),
                      "on_complete": a.on_complete or None}
            (a.persist).mkdir(parents=True, exist_ok=True)
            (a.persist / "STATUS.json").write_text(json.dumps(status, indent=2))
            log(f"final sync {'ok' if ok else 'FAILED'} ({detail}); verify {'ok' if vok else 'FAILED'} ({vdetail})")

            if not (ok and vok):
                log("NOT running --on-complete: weights are not safely persisted. "
                    "Worker left alive for inspection.")
                return 1
            if a.on_complete:
                tok, tdetail = run_on_complete(a.on_complete)
                log(f"on-complete {'ok' if tok else 'FAILED'}: {tdetail}")
                return 0 if tok else 1
            log("weights persisted; no --on-complete, leaving the worker alive")
            return 0

        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
