"""Paired-game scoring of `dzgorge play` records, and a sequential test on the result.

A paired run deals each deck pair twice with the policies swapped (leg 0: policy A in seat 0; leg 1: swapped).
Both legs share the pair's seed, so the deal's luck mostly cancels and the pair, not the game, is the sampling
unit. One games.jsonl row per game (gorge fork cmd/dzgorge/play.go `gameRec`):

    {game, pair, leg, seed, decks[2], side[2] ("A"|"B" per seat), pols[2], winner (seat, -1 if none),
     result "A"|"B"|"draw"|"stall"|"error", turns, ms, in_hand[2], ...}

`result` names the winning POLICY (the -a or -b spec), not the seat; `side` says which policy sat where.

    score_pairs(games)                  side A's score over complete pairs, with its standard error
    score_pairs(games, hero_side="B")   the same from B's side
    sprt(w, d, l)                       "better" | "not better" | None (keep playing)

Scoring follows gorge/recipe/score.py (v0.1.0): a game is worth 1 for a hero win, 0 for a loss and 0.5 for a
draw or a turn- or decision-capped stall; the score is the mean over complete pairs of the pair means, and the
standard error is the SD of the pair means over sqrt(pairs). Unlike score.py, an engine error is not half a
point: it is dropped, and its pair with it (a pair needs both legs to be complete).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

Z95 = 1.959964


def game_points(g: dict, hero_side: str = "A") -> float | None:
    """The hero's points in one game: 1 win, 0 loss, 0.5 draw or stall; None for an engine error (or no result)."""
    r = g.get("result")
    if r == hero_side:
        return 1.0
    if r in ("A", "B"):
        return 0.0
    if r in ("draw", "stall"):
        return 0.5
    return None


def score_pairs(games: Iterable[dict], *, hero_side: str = "A", paired: bool = True) -> dict:
    """Side `hero_side`'s score over complete deck pairs.

    Returns {score, se, lo, hi, n_pairs, n_games, wins, losses, draws, stalls, errors, dropped, duplicates}:
    score and se are None without a complete pair (se is 0.0 for a single pair, as in score.py); lo/hi is the
    normal 95% interval clipped to [0, 1]; wins/losses/draws/stalls count the games of complete pairs; errors
    counts every error row; dropped counts the scored games left out because their pair was incomplete;
    duplicates counts repeated (pair, seed, leg) rows (a replayed pair), of which only the first is kept.

    A pair is keyed by (pair, seed), so files from runs with other seeds never merge into one pair. With
    paired=False every game is its own unit (single-leg runs: `-single`, or -a == -b).
    """
    if hero_side not in ("A", "B"):
        raise ValueError(f"hero_side must be 'A' or 'B', not {hero_side!r}")
    legs: dict[tuple, dict[int, float]] = defaultdict(dict)
    results: dict[tuple, dict[int, str]] = defaultdict(dict)
    errors = duplicates = 0
    for i, g in enumerate(games):
        pts = game_points(g, hero_side)
        if pts is None:
            errors += 1
            continue
        key = (g.get("pair"), g.get("seed")) if paired else (i,)
        leg = int(g.get("leg", 0)) if paired else 0
        if leg in legs[key]:
            duplicates += 1
            continue
        legs[key][leg] = pts
        results[key][leg] = g.get("result")
    need = 2 if paired else 1
    full = [k for k, v in legs.items() if len(v) >= need]
    means = [sum(legs[k].values()) / len(legs[k]) for k in full]
    counted = [r for k in full for r in results[k].values()]
    n = len(means)
    other = "B" if hero_side == "A" else "A"
    out = {"score": None, "se": None, "lo": None, "hi": None, "n_pairs": n, "n_games": len(counted),
           "wins": counted.count(hero_side), "losses": counted.count(other), "draws": counted.count("draw"),
           "stalls": counted.count("stall"), "errors": errors,
           "dropped": sum(len(v) for k, v in legs.items() if len(v) < need), "duplicates": duplicates}
    if not n:
        return out
    m = sum(means) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in means) / max(n - 1, 1))
    se = sd / math.sqrt(n)
    out.update(score=m, se=se, lo=max(0.0, m - Z95 * se), hi=min(1.0, m + Z95 * se))
    return out


def policies(games: Iterable[dict]) -> dict[str, set[str]]:
    """{"A": {specs}, "B": {specs}}: the policy specs seen on each side (one each in a well-formed run)."""
    out: dict[str, set[str]] = {"A": set(), "B": set()}
    for g in games:
        for side, pol in zip(g.get("side") or (), g.get("pols") or ()):
            if side in out:
                out[side].add(pol)
    return out


def sprt_llr(wins: int, draws: int, losses: int, p0: float = 0.5, p1: float = 0.55, alpha: float = 0.05,
             beta: float = 0.05) -> dict:
    """The SPRT's state: {llr, lower, upper, decision}. See `sprt`."""
    n = wins + draws + losses
    lower, upper = math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)
    if n == 0:
        return {"llr": 0.0, "lower": lower, "upper": upper, "decision": None}
    m = (wins + 0.5 * draws) / n
    var = max((wins * (1 - m) ** 2 + draws * (0.5 - m) ** 2 + losses * m ** 2) / n, 1e-6)
    llr = n * (p1 - p0) * (2 * m - p0 - p1) / (2 * var)
    decision = "better" if llr >= upper else "not better" if llr <= lower else None
    return {"llr": llr, "lower": lower, "upper": upper, "decision": decision}


def sprt(w: int, d: int, l: int, p0: float = .5, p1: float = .55, alpha: float = .05,  # noqa: E741
         beta: float = .05) -> str | None:
    """A sequential probability ratio test of H1: score = p1 against H0: score = p0 on game scores, with the
    normal approximation fishtest uses (from v0.1.0's selfplay/evals.py). "better" (H1 accepted), "not better"
    (H0 accepted) or None (play on). Stalls count as draws. It treats games as independent, though the two games
    of a pair share a deal and are correlated."""
    return sprt_llr(w, d, l, p0, p1, alpha, beta)["decision"]
