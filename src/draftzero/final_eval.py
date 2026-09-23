"""Final strength eval: one large, paired eval of the final checkpoint, run after the loop stops.

The in-run milestone evals are 40 games each (+/-15pp), too noisy to settle a trend. At the end
of a run the final checkpoint plays PAIRS paired games on the held-out eval decks against each
opponent: "offline" (the same MCTS search with no network) or a past checkpoint such as "gen10".
Decks are drawn from the same fixed seed the milestone evals use, so the first 20 pairs are
exactly the milestone matchups, and every opponent faces the same decks.

Results go to games.jsonl (kind final_eval), one metrics.jsonl row per opponent, and
final_eval.json; then they are pushed to HF and sent as an alert. Meant to run detached from the
watchdog's --on-complete via deploy/final_eval.sh, which tears the worker down afterwards:

  python -m draftzero.final_eval --config configs/fdn_l40s.yml --run-dir runs/<id> \
      --vs offline gen10 --pairs 100
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from draftzero import alerts, hfsync, secrets
from draftzero.loop import (OPPONENT_PORT, PRIMARY_PORT, TMP_DIR, Run, game_yml, load_config,
                            play_job, read_games)
from magezero import metrics
from magezero.runner import start_server, stop_server
from magezero.util.config import load_curriculum, resolve_gen

KIND = "final_eval"
# 40 games per JVM job: the size the milestone evals already run at, well inside jvm.max_minutes.
CHUNK_PAIRS = 20
BASELINES = ("offline", "minimax")


def eval_pairs(pool: list[str], pairs: int, seed: str = "eval-offline") -> list[tuple[str, str]]:
    """The draw eval_generation makes, extended: pair i here is pair i of every milestone eval."""
    rng = random.Random(seed)
    return [tuple(rng.sample(pool, 2)) for _ in range(pairs)]


def final_checkpoint(run: Run) -> int:
    """Newest generation that has its own checkpoint."""
    for g in sorted((int(k) for k in run.state.get("gens", {})), reverse=True):
        if run.has_checkpoint(f"gen{g}"):
            return g
    raise RuntimeError("no per-generation checkpoint found")


def summarize(games: list[dict], gen: int, opponent: str) -> dict:
    mine = [g for g in games
            if g.get("kind") == KIND and g.get("gen") == gen and g.get("opponent") == opponent]
    wins = sum(g.get("winner") == "A" for g in mine)
    p, lo, hi = metrics.wilson(wins, len(mine)) if mine else (None, None, None)
    return {"kind": KIND, "gen": gen, "agent_checkpoint": f"gen{gen}", "opponent": opponent,
            "games": len(mine), "wins": wins, "a_winrate": p, "a_winrate_lo": lo, "a_winrate_hi": hi}


def describe(row: dict) -> str:
    if not row["games"]:
        return f"gen {row['gen']} vs {row['opponent']}: no games completed"
    return (f"gen {row['gen']} vs {row['opponent']}: {row['wins']}/{row['games']} "
            f"({100 * row['a_winrate']:.0f}%, 95% CI {100 * row['a_winrate_lo']:.0f}"
            f"-{100 * row['a_winrate_hi']:.0f}%)")


def play_vs(run: Run, gen: int, settings, opponent: str, pairs: list[tuple[str, str]],
            scratch: Path) -> None:
    """Paired games, agent (player A, checkpoint gen<gen>) vs one opponent, in 40-game jobs."""
    offline_b = opponent in BASELINES
    frozen = None
    try:
        if not offline_b:
            frozen = start_server_retry(run.model, run.version, OPPONENT_PORT, run.dir, checkpoint=opponent)
        for start in range(0, len(pairs), CHUNK_PAIRS):
            chunk = pairs[start:start + CHUNK_PAIRS]
            pool_a = [d for x, y in chunk for d in (x, y)]
            pool_b = [d for x, y in chunk for d in (y, x)]
            tag = f"{opponent}_{start // CHUNK_PAIRS:02d}"
            fa, fb = TMP_DIR / f"final_{tag}_A.txt", TMP_DIR / f"final_{tag}_B.txt"
            fa.parent.mkdir(parents=True, exist_ok=True)
            fa.write_text("\n".join(pool_a) + "\n")
            fb.write_text("\n".join(pool_b) + "\n")
            name = f"final_{tag}"
            yml = game_yml(run, f"gen{gen}_{name}", settings, len(pool_a),
                           pool_a=str(fa), pool_b=str(fb), mode="sequential",
                           out_a=scratch / f"{name}_A.hdf5", out_b=scratch / f"{name}_B.hdf5",
                           offline_a=False, offline_b=offline_b,
                           type_b="minimax" if opponent == "minimax" else "mcts", b_port=OPPONENT_PORT)
            play_job(run, gen, name, yml, KIND, opponent, "a")
            print(f"[final_eval] {describe(summarize(read_games(run), gen, opponent))} "
                  f"after {start + len(chunk)}/{len(pairs)} pairs", flush=True)
    finally:
        if frozen:
            stop_server(frozen)


def start_server_retry(*args, attempts: int = 3, wait_s: int = 20, **kw):
    """start_server, retried. The loop's last server has only just released the port when we
    start, and a server that exits before it is ready has happened once mid-run (gen 16) and
    cleared on the watchdog's retry. The final eval has no watchdog behind it, so retry here."""
    for i in range(attempts):
        try:
            return start_server(*args, **kw)
        except RuntimeError as e:
            if i == attempts - 1:
                raise
            print(f"[final_eval] server start failed ({e}); retrying in {wait_s}s", flush=True)
            time.sleep(wait_s)


def push_results(run_dir: Path, gen: int, attempts: int = 3) -> str:
    # Weights were pushed and verified before the watchdog handed off; only the result files
    # are new, so skip the models (models_dir=None) instead of re-uploading every checkpoint.
    detail = ""
    for i in range(attempts):
        ok, detail = hfsync.push(run_dir, None, gen=gen)
        if ok:
            return detail
        time.sleep(30 * (i + 1))
    return f"FAILED: {detail}"


def _raise_on_term(signum, frame):  # noqa: ARG001
    # deploy/final_eval.sh runs us under `timeout`; turn its SIGTERM into an exception so the
    # results that exist still get written, pushed and alerted before the worker goes away.
    raise TimeoutError("final eval timed out (SIGTERM)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--vs", nargs="+", default=["offline"],
                    help="opponents: offline, minimax, or a checkpoint name like gen10")
    ap.add_argument("--pairs", type=int, default=100, help="paired games per opponent (2 games each)")
    ap.add_argument("--hf-env", default="/root/.dz_env")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit without playing")
    a = ap.parse_args()

    if a.hf_env and Path(a.hf_env).expanduser().exists():
        secrets.load(a.hf_env)
    signal.signal(signal.SIGTERM, _raise_on_term)

    t0 = time.time()
    results: list[dict] = []
    error = None
    gen = None
    try:
        cfg = load_config(a.config)
        run = Run(cfg, a.run_dir)
        gen = final_checkpoint(run)
        settings = resolve_gen(load_curriculum(cfg["curriculum"]), gen)
        pairs = eval_pairs(run.eval_pool, a.pairs)
        missing = [o for o in a.vs if o not in BASELINES and not run.has_checkpoint(o)]
        if missing:
            raise RuntimeError(f"no checkpoint for opponent(s): {', '.join(missing)}")
        print(f"[final_eval] gen{gen} vs {', '.join(a.vs)}: {a.pairs} pairs "
              f"({2 * a.pairs} games) each, {len(run.eval_pool)} eval decks", flush=True)
        if a.dry_run:
            for x, y in pairs[:3]:
                print(f"  pair: {Path(x).name} vs {Path(y).name}")
            return 0

        scratch = run.dir / "scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        server = start_server_retry(run.model, run.version, PRIMARY_PORT, run.dir, checkpoint=f"gen{gen}")
        try:
            for opp in a.vs:
                t1 = time.time()
                play_vs(run, gen, settings, opp, pairs, scratch)
                row = {**summarize(read_games(run), gen, opp), "seconds": round(time.time() - t1)}
                metrics.append_jsonl(run.dir / "metrics.jsonl", row)
                results.append(row)
                print(f"[final_eval] DONE {describe(row)}", flush=True)
        finally:
            stop_server(server)
            shutil.rmtree(scratch, ignore_errors=True)
    except BaseException as e:  # noqa: BLE001 - report whatever we have, then let the wrapper tear down
        error = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    if a.dry_run:   # a plan check: never write results, push, or alert
        return 0 if error is None else 1
    # Opponents that were cut short (timeout, crash) still have their partial games counted.
    if gen is not None:
        done = {r["opponent"] for r in results}
        try:
            games = read_games(Run(load_config(a.config), a.run_dir))
            results += [summarize(games, gen, o) for o in a.vs if o not in done]
        except Exception:  # noqa: BLE001
            pass
    summary = {"status": "ok" if error is None else "failed", "error": error, "gen": gen,
               "pairs": a.pairs, "results": results, "hours": round((time.time() - t0) / 3600, 2),
               "at": datetime.now(timezone.utc).isoformat()}
    (a.run_dir / "final_eval.json").write_text(json.dumps(summary, indent=2))
    hf = push_results(a.run_dir, gen) if gen is not None else "skipped (no checkpoint)"
    lines = [describe(r) for r in results] or ["no results"]
    print(f"[final_eval] {summary['status']}: " + " | ".join(lines) + f" | HF {hf}", flush=True)
    alerts.send(alerts.DONE if error is None else alerts.CRASH,
                "Final eval " + ("done" if error is None else f"FAILED ({error})") + ": " + "; ".join(lines),
                f"HF: {hf}. Tearing down the worker next.", a.run_dir)
    return 0 if error is None else 1


if __name__ == "__main__":
    sys.exit(main())
