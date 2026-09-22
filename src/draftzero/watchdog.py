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
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from draftzero import alerts
from draftzero import hfsync
from draftzero import secrets

def log(msg: str) -> None:
    print(f"[watchdog {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def seconds_since_activity(run_dir: Path) -> float:
    """Age of the most recent write anywhere in the run directory.

    games.jsonl only grows when a whole CHUNK finishes -- about every 27 minutes here --
    so using it alone as the liveness signal declares a perfectly healthy run dead the
    moment one chunk runs long. The JVM writes its log continuously, so file activity is
    the honest signal. This exact mistake destroyed a working pod: idle_min was climbing
    past the threshold while the JVM log had been written 0 seconds earlier.
    """
    newest = 0.0
    try:
        for f in Path(run_dir).rglob("*"):
            if f.is_file():
                try:
                    newest = max(newest, f.stat().st_mtime)
                except OSError:
                    continue
    except OSError:
        return 0.0
    return (time.time() - newest) if newest else 0.0


def _sync_set(a) -> list:
    """What must survive the worker. Weights alone are not enough: without the replay
    shards a resumed run trains its next generation on almost nothing."""
    out = [a.models, a.run_dir]
    if a.data and Path(a.data).exists():
        out.append(a.data)
    return out


def _tail(run_dir: Path, lines: int = 25) -> str:
    """Last lines of the trainer log, so an alert carries the traceback not just a verdict."""
    for cand in (Path(run_dir).parent.parent / "logs" / "loop.log", Path("logs/loop.log")):
        try:
            return "\n".join(cand.read_text().splitlines()[-lines:])
        except OSError:
            continue
    return ""


def read_state(run_dir: Path) -> dict:
    f = Path(run_dir) / "run.json"
    try:
        return json.loads(f.read_text()) if f.exists() else {}
    except (OSError, ValueError):
        return {}


def gens_done(run_dir: Path) -> int:
    """Completed generations. A checkpoint exists for each, so this is the safe stop point."""
    return len(read_state(run_dir).get("gens", {}))


def loop_pids() -> list[int]:
    """PIDs of actual trainer processes.

    A bare `pgrep -f draftzero.loop` also matches any shell whose command line merely
    mentions the module -- an ssh wrapper, a grep, this watchdog's own launcher. Killing
    one of those would be at best confusing and at worst fatal to the run, so require the
    executable to be python AND the argv to contain `-m draftzero.loop`, and never match
    our own process.
    """
    me = os.getpid()
    out = []
    try:
        listing = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True)
    except Exception:
        return []
    for line in listing.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_s, _, args = line.partition(" ")
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if pid == me:
            continue
        argv = args.split()
        if not argv or "python" not in os.path.basename(argv[0]):
            continue
        if "-m" in argv and "draftzero.loop" in argv:
            out.append(pid)
    return out


def stop_loop(timeout: int = 300) -> str:
    """Ask the trainer to exit, then make sure it has. Only ever called at a generation
    boundary (or after the grace period), so nothing half-written is lost."""
    pids = loop_pids()
    if not pids:
        return "trainer already exited"
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not loop_pids():
            return f"trainer stopped ({len(pids)} pid(s))"
        time.sleep(5)
    for pid in loop_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return "trainer killed after timeout"


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


def _watchdog_count() -> int:
    """How many watchdog processes are running, this one included."""
    try:
        out = subprocess.run(["pgrep", "-fc", "draftzero.watchdog"], capture_output=True, text=True)
        return int(out.stdout.strip() or "0")
    except Exception:
        return 1


def main() -> int:
    # Singleton guard. Never let two watchdogs run at once: a second one doubles restarts
    # and, since a restart used to relaunch launch.sh (which starts a watchdog), that once
    # cascaded into hundreds. Even with that fixed, two watchdogs racing on the sync and HF
    # push is a bug, so a duplicate exits immediately.
    if _watchdog_count() > 1:
        print("[watchdog] another watchdog is already running; exiting to stay singleton", flush=True)
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--target", type=int, default=20000)
    ap.add_argument("--persist", required=True, type=Path)
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--data", type=Path, default=Path("data"),
                    help="replay-buffer HDF5 shards. Synced too: the first run backed up only "
                         "weights and run artifacts, so when the pod died the training states "
                         "went with it and a resume would have had an empty replay buffer.")
    ap.add_argument("--interval", type=int, default=300, help="seconds between checks")
    ap.add_argument("--stall-minutes", type=int, default=45)
    ap.add_argument("--hf-env", default=os.environ.get("DZ_HF_ENV", "/root/.dz_env"),
                    help="file with HF_TOKEN and HF_REPO for the off-pod checkpoint push")
    ap.add_argument("--restart-cmd", default=os.environ.get("DZ_RESTART_CMD", ""),
                    help="shell command that relaunches the trainer (typically launch.sh "
                         "--resume). Without it a crashed trainer just idles the worker until "
                         "the stall timer fires, which is hours of paid-for nothing.")
    ap.add_argument("--max-restarts", type=int, default=3)
    ap.add_argument("--grace-hours", type=float, default=3.0,
                    help="once the cap or target is hit, wait up to this long for the current "
                         "generation to finish before stopping. A checkpoint is only written at "
                         "a generation boundary, so cutting mid-generation throws that work away. "
                         "If the grace expires the run is stopped anyway, to protect the budget.")
    ap.add_argument("--max-hours", type=float, default=0,
                    help="stop after this many hours of wall clock (0 = no limit). "
                         "A rented worker bills by the second, so this is the budget cap: "
                         "hours x rate. Deterministic, and needs no API credentials on the worker.")
    ap.add_argument("--on-complete", default=os.environ.get("DZ_ON_COMPLETE", ""),
                    help="shell command to run after a VERIFIED final sync (e.g. destroy the pod)")
    a = ap.parse_args()

    # Off-pod durable store. On a pod with no network volume this is the ONLY copy that
    # survives termination, so it is loaded here and verified in the repo before the pod may
    # be destroyed. Parsed, never sourced (a bare-value file once leaked a token by sourcing).
    if a.hf_env and Path(a.hf_env).expanduser().exists():
        secrets.load(a.hf_env)
    hf_on = hfsync.configured()
    log(f"HF push: {'on -> ' + os.environ['HF_REPO'] if hf_on else 'off (no HF_TOKEN/HF_REPO)'}")

    if a.on_complete:
        log(f"on-complete: {a.on_complete}")
    if a.restart_cmd:
        log(f"restart-cmd: {a.restart_cmd} (up to {a.max_restarts}x)")
    if os.environ.get("DZ_ALERT_TOPIC"):
        log(f"alerts -> ntfy topic set" + (", email on" if os.environ.get("DZ_ALERT_EMAIL") else ""))
    restarts = 0
    sync_warned = False
    hf_pushed_gens = 0

    started = time.time()
    last_count, last_progress = -1, started
    pending_since: Optional[float] = None
    pending_reason, pending_gens = "", 0
    log(f"watching {a.run_dir} -> target {a.target} games, persist {a.persist}")

    while True:
        n = count_games(a.run_dir)
        now = time.time()
        if n > last_count:
            last_count, last_progress = n, now
        idle_min = (now - last_progress) / 60
        # Two independent liveness signals. A run is only stalled when BOTH say so: no new
        # games AND nothing written to the run dir AND no trainer process. Any one of these
        # alone produces false positives that cost a pod.
        quiet_min = seconds_since_activity(a.run_dir) / 60
        trainer_alive = bool(loop_pids())

        ok, detail = sync(_sync_set(a), a.persist)
        if not ok:
            log(f"sync FAILED: {detail}")

        # Push off the pod once per completed generation (checkpoints only change then).
        cur_gens = gens_done(a.run_dir)
        if hf_on and cur_gens > hf_pushed_gens:
            hok, hdetail = hfsync.push(a.run_dir, a.models, gen=cur_gens - 1)
            log(f"HF push: {hdetail}")
            if hok:
                hf_pushed_gens = cur_gens
            else:
                alerts.send(alerts.SYNC_FAILED,
                            "Off-pod checkpoint push to Hugging Face failed. The pod will "
                            "NOT self-terminate until weights are safely off it.",
                            hdetail, a.run_dir)
            if not sync_warned:
                sync_warned = True
                alerts.send(alerts.SYNC_FAILED,
                            "Checkpoint sync to the volume is failing. Weights may not be "
                            "safe, and the worker will NOT self-terminate while this is true.",
                            detail, a.run_dir)

        elapsed_h = (now - started) / 3600
        done = n >= a.target

        # The trainer died and the run is not finished. Restart it: the loop resumes at the
        # interrupted stage, so the cost is one chunk, versus idling a rented machine until
        # the stall timer eventually declares the whole run dead.
        if not trainer_alive and not done and a.restart_cmd and pending_since is None:
            if restarts < a.max_restarts:
                restarts += 1
                backoff = 30 * restarts
                log(f"TRAINER GONE ({n} games). restart {restarts}/{a.max_restarts} in {backoff}s")
                alerts.send(alerts.CRASH,
                            f"Trainer stopped at {n} games (gen {gens_done(a.run_dir)}). "
                            f"Restarting ({restarts}/{a.max_restarts}).",
                            _tail(a.run_dir), a.run_dir)
                time.sleep(backoff)
                r = subprocess.run(a.restart_cmd, shell=True, capture_output=True,
                                   text=True, timeout=600)
                ok_restart = r.returncode == 0 and bool(loop_pids())
                log(f"restart {'ok' if ok_restart else 'FAILED'}: "
                    f"{(r.stdout or r.stderr).strip()[:200]}")
                alerts.send(alerts.RESTART if ok_restart else alerts.RESTART_FAILED,
                            f"Restart {restarts}/{a.max_restarts} "
                            f"{'succeeded' if ok_restart else 'FAILED'} at {n} games.",
                            (r.stdout or r.stderr)[-1000:], a.run_dir)
                if ok_restart:
                    last_progress = time.time()   # give the new trainer a fresh stall window
                    time.sleep(a.interval)
                    continue
            else:
                log(f"restart budget exhausted ({a.max_restarts}); letting the stall timer run")
                alerts.send(alerts.RESTART_FAILED,
                            f"Trainer down and {a.max_restarts} restarts exhausted at {n} games. "
                            f"Worker will shut down.", _tail(a.run_dir), a.run_dir)
        stalled = (idle_min >= a.stall_minutes
                   and quiet_min >= a.stall_minutes
                   and not trainer_alive)
        over_budget = bool(a.max_hours) and elapsed_h >= a.max_hours
        budget = f" | {elapsed_h:.1f}/{a.max_hours:g}h" if a.max_hours else ""
        live = "alive" if trainer_alive else "NO TRAINER"
        log(f"{n}/{a.target} games | idle {idle_min:.0f}m | quiet {quiet_min:.0f}m | {live}"
            f"{budget} | sync {'ok' if ok else 'FAILED'}")

        if (done or stalled or over_budget) and pending_since is None:
            pending_reason = ("target reached" if done else
                              f"budget cap: {elapsed_h:.1f}h of {a.max_hours:g}h" if over_budget else
                              f"stalled {idle_min:.0f}m with no new game")
            pending_since, pending_gens = now, gens_done(a.run_dir)
            log(f"stop requested ({pending_reason}); finishing generation {pending_gens} "
                f"before shutting down (grace {a.grace_hours:g}h)")

        if pending_since is not None:
            waited_h = (now - pending_since) / 3600
            at_boundary = gens_done(a.run_dir) > pending_gens or read_state(a.run_dir).get("completed_at")
            expired = waited_h >= a.grace_hours
            if stalled and not at_boundary:
                at_boundary = True   # nothing is progressing; waiting for a boundary is pointless
            if not (at_boundary or expired):
                log(f"waiting for generation boundary: {waited_h:.1f}/{a.grace_hours:g}h")
                time.sleep(a.interval)
                continue
            reason = pending_reason + (" (generation completed)" if at_boundary else
                                       f" (grace {a.grace_hours:g}h expired mid-generation)")
            log(f"stopping: {reason}")
            alerts.send(alerts.DONE if done else alerts.BUDGET if over_budget else alerts.STALLED,
                        f"Run stopping: {reason}. {n} games, "
                        f"{gens_done(a.run_dir)} generations.", "", a.run_dir)
            log(stop_loop())
            ok, detail = sync(_sync_set(a), a.persist)
            vok, vdetail = verify(a.persist, a.run_dir.name)

            # Off-pod durability gate. On a pod with no network volume, `persist` IS the
            # container disk that dies with the pod, so a passing local verify() means
            # nothing for survival. When HF is configured, the weights are only safe once a
            # final push lands AND the repo confirms it, and that gate -- not the local one
            # -- decides whether the pod may be destroyed.
            hf_ok, hf_detail = (True, "HF off")
            if hf_on:
                hfsync.push(a.run_dir, a.models, gen=gens_done(a.run_dir) - 1)
                hf_ok, hf_detail = hfsync.has_checkpoint(a.run_dir.name)
                log(f"final HF push -> {hf_detail}")

            status = {"reason": reason, "games": n, "target": a.target,
                      "elapsed_hours": round(elapsed_h, 2), "max_hours": a.max_hours or None,
                      "generations_completed": gens_done(a.run_dir),
                      "final_sync_ok": ok, "sync_detail": detail,
                      "verified": vok, "verify_detail": vdetail,
                      "hf_verified": hf_ok, "hf_detail": hf_detail,
                      "at": datetime.now(timezone.utc).isoformat(),
                      "on_complete": a.on_complete or None}
            (a.persist).mkdir(parents=True, exist_ok=True)
            (a.persist / "STATUS.json").write_text(json.dumps(status, indent=2))
            log(f"final sync {'ok' if ok else 'FAILED'} ({detail}); "
                f"verify {'ok' if vok else 'FAILED'} ({vdetail})")

            # Safe if EITHER durable store confirms the weights: a network volume (local
            # verify) or the HF repo. On a volume-less pod only the HF gate can pass, so it
            # is required there; local verify alone must never authorize destroying the pod.
            safe = (ok and vok) or (hf_on and hf_ok)
            if not safe:
                why = f"local verify {vdetail}" + (f"; HF {hf_detail}" if hf_on else "")
                log(f"NOT running --on-complete: weights are not safely off the pod ({why}). "
                    "Worker left alive for inspection.")
                alerts.send(alerts.SYNC_FAILED,
                            f"Run stopped but weights are NOT safely off the pod. Leaving it "
                            f"alive so nothing is lost. {why}", "", a.run_dir)
                return 1
            if a.on_complete:
                where = hf_detail if hf_on else vdetail
                alerts.send(alerts.TERMINATING,
                            f"Weights verified ({where}). Destroying the worker now. "
                            f"{n} games, {gens_done(a.run_dir)} generations.",
                            "", a.run_dir)
                tok, tdetail = run_on_complete(a.on_complete)
                log(f"on-complete {'ok' if tok else 'FAILED'}: {tdetail}")
                return 0 if tok else 1
            log("weights persisted; no --on-complete, leaving the worker alive")
            return 0

        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
