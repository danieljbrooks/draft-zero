"""Evaluation jobs (docs/021 §2-3): one version against the starting network (or the heuristic bot), on fixed
deck pairs from the evaluation pool, in small shards that any worker can play.

A job:
    {"job": "v0004-vs-v0000", "a": 4, "b": 0 | "heuristic", "simulations": 100, "pairs": 500, "seed": ...,
     "pool": "data/pools/eval.txt", "created": <time>,
     "shards": [{"id": 0, "min_pair": 0, "max_pair": 9, "machine": null, "assigned": null, "done": false}, ...]}

A shard is play.py's deck pairs min_pair..max_pair (two games each, the bots swapping seats; both games share
their seed, so the deal's luck cancels). Version a is play.py's bot1, b its bot2, each with its own servers.
Results go to evals/<job>/<shard>.jsonl.gz (play.py's games.jsonl lines); the controller scores them with a
sequential test (SPRT) and closes the job once the test decides or every shard is in.
"""
from __future__ import annotations

import math
import time


def job_id(a: int, b) -> str:
    return f"v{a:04d}-vs-" + (b if isinstance(b, str) else f"v{b:04d}")


def make_job(a: int, b, ecfg: dict, now: float | None = None) -> dict:
    pairs, size = int(ecfg["pairs"]), max(1, int(ecfg["shard_pairs"]))
    shards = [{"id": i, "min_pair": lo, "max_pair": min(pairs, lo + size) - 1, "machine": None, "assigned": None,
               "done": False} for i, lo in enumerate(range(0, pairs, size))]
    return {"job": job_id(a, b), "a": a, "b": b, "simulations": int(ecfg["simulations"]), "pairs": pairs,
            "seed": int(ecfg["seed"]), "pool": ecfg["pool"], "heuristic_simulations": int(ecfg["heuristic_simulations"]),
            "created": time.time() if now is None else now, "shards": shards}


def result_path(job: str, shard: int) -> str:
    return f"evals/{job}/{shard:04d}.jsonl.gz"


def score(games: list[dict]) -> dict:
    """bot1's (version a's) record over the valid games: a win 1, a draw (no winner) 0.5."""
    w = d = l_ = errors = 0
    for g in games:
        if g.get("error"):
            errors += 1
            continue
        role = g.get("winner_role")
        if role == "bot1":
            w += 1
        elif role == "bot2":
            l_ += 1
        else:
            d += 1
    n = w + d + l_
    return {"games": n, "wins": w, "draws": d, "losses": l_, "errors": errors,
            "score": None if n == 0 else round((w + 0.5 * d) / n, 4)}


def sprt(wins: int, draws: int, losses: int, p0: float = 0.5, p1: float = 0.55, alpha: float = 0.05,
         beta: float = 0.05) -> dict:
    """A sequential probability ratio test of H1: score = p1 against H0: score = p0, on game scores (the normal
    approximation fishtest uses). decision: "better" (H1 accepted), "not better" (H0 accepted) or None (go on)."""
    n = wins + draws + losses
    lower, upper = math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)
    if n == 0:
        return {"llr": 0.0, "lower": round(lower, 3), "upper": round(upper, 3), "decision": None}
    m = (wins + 0.5 * draws) / n
    var = (wins * (1 - m) ** 2 + draws * (0.5 - m) ** 2 + losses * m ** 2) / n
    var = max(var, 1e-6)
    llr = n * (p1 - p0) * (2 * m - p0 - p1) / (2 * var)
    decision = "better" if llr >= upper else "not better" if llr <= lower else None
    return {"llr": round(llr, 3), "lower": round(lower, 3), "upper": round(upper, 3), "decision": decision}
