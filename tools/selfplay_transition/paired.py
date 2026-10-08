"""docs/028: score a match from play.py's games.jsonl, by deck pair.

Each deck pair is played twice with the seats swapped (the same deal when the bots have their own servers), so the
pair is the unit: its score is bot1's points over its two games (a win 1, a 50-turn stop 0.5). Only pairs with both
games free of engine errors count, so an error can't drop just one side's loss. The standard error is across pairs.

    python tools/selfplay_transition/paired.py runs/sp/h2h/<run> [<run> ...] [--json out.json]
    python tools/selfplay_transition/paired.py --diff <run a> <run b>     # the same deals: a's score minus b's

Several runs on one line are merged (designated shards of one match).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(runs: list[Path]) -> dict:
    """(pair, swap) -> the game's line, over the runs (the first valid copy of a game wins)."""
    games = {}
    for r in runs:
        f = Path(r) / "games.jsonl" if Path(r).is_dir() else Path(r)
        for ln in f.read_text().splitlines():
            if not ln.strip():
                continue
            g = json.loads(ln)
            k = (g["pair"], g["swap"])
            if k not in games or (games[k].get("error") and not g.get("error")):
                games[k] = g
    return games


def points(g: dict) -> float:
    if g.get("winner_role") is None:
        return 0.5
    return 1.0 if g.get("winner_role") == "bot1" else 0.0


def score(games: dict) -> dict:
    pairs = sorted({p for p, _ in games})
    full, err_pairs, n_err, n_cap, turns = [], 0, 0, 0, []
    for p in pairs:
        gs = [games.get((p, s)) for s in (False, True)]
        n_err += sum(1 for g in gs if g is not None and g.get("error"))
        if any(g is None or g.get("error") for g in gs):
            err_pairs += any(g is not None and g.get("error") for g in gs)
            continue
        n_cap += sum(g.get("winner_role") is None for g in gs)
        turns += [g.get("turns") or 0 for g in gs]
        full.append(sum(points(g) for g in gs) / 2)
    n = len(full)
    m = sum(full) / n if n else float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in full) / (n - 1)) if n > 1 else float("nan")
    return {"pairs": n, "games": 2 * n, "score": round(m, 4), "se": round(sd / math.sqrt(n), 4) if n > 1 else None,
            "split_pairs": sum(1 for x in full if x == 0.5), "swept_pairs": sum(1 for x in full if x in (0, 1)),
            "error_games": n_err, "error_pairs": err_pairs, "capped_games": n_cap,
            "cap_rate": round(n_cap / (2 * n), 4) if n else None,
            "mean_turns": round(sum(turns) / len(turns), 2) if turns else None, "games_listed": len(games)}


def diff(a: dict, b: dict) -> dict:
    """a's bot1 score minus b's on the deals both played cleanly (by pair)."""
    d = []
    for p in sorted({p for p, _ in a} & {p for p, _ in b}):
        ga = [a.get((p, s)) for s in (False, True)]
        gb = [b.get((p, s)) for s in (False, True)]
        if any(g is None or g.get("error") for g in ga + gb):
            continue
        d.append(sum(points(g) for g in ga) / 2 - sum(points(g) for g in gb) / 2)
    n = len(d)
    m = sum(d) / n if n else float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else float("nan")
    return {"pairs": n, "diff": round(m, 4), "se": round(sd / math.sqrt(n), 4) if n > 1 else None,
            "pairs_differing": sum(1 for x in d if x != 0)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="*")
    ap.add_argument("--diff", type=Path, nargs=2)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    if a.diff:
        res = diff(load([a.diff[0]]), load([a.diff[1]]))
    else:
        res = score(load(a.runs))
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
