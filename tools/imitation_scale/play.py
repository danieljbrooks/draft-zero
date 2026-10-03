"""Experiment #4's games (docs/017 §6.5-6.6): two bots, IS-MCTS at every decision for both, over
deck pairs from a pool, each pair played twice with the seats swapped.

    python tools/imitation_scale/play.py --pool data/pools/eval.txt --deck-root $MZ_DECK_DIR \\
        --pairs 100 --budget 1000 --bot1 il_bc --bot2 heuristic --ports 50052,50152 --workers 28 \\
        --out runs/imitation_scale/play_stage5

Bots (--bot1, --bot2), the four mixes of docs/017 §6.5, each with uniform priors at the opponent's
nodes and IS-MCTS policies per actor and decision type:

    heuristic    the default (uniform) prior, the heuristic at the leaves: experiment #3's offline search
    il_bc        the network's policy as the prior, its value at the leaves
    il_heur      the network's policy as the prior, the heuristic at the leaves
    bc           the default prior, the network's value at the leaves

and two with no search (docs/018, policy-only evaluation): at every decision with a policy head
(priority, target, binary) the network's own policy over the options, read once on the live game,
with no simulations and no belief worlds. Decisions without a head are searched as il_bc, at
--policy-fallback-budget simulations.

    policy        sampled from the policy (temperature 1): plays like the humans it imitates
    policy_greedy the policy's most likely option

A bot can carry its own budget, `name@simulations` (heuristic@100, il_bc@1000); without one it searches at
--budget. Experiment #4's games (docs/018): policy against heuristic@100, il_bc@100/300/1000 against
heuristic@100, and il_bc@1000 against heuristic@1000.

Every game's seats carry inHand (the cards seen in each player's hand), for the games-in-hand win
rate against 17lands (tools/imitation_scale/gih.py).

A network bot needs a MageZero inference server (tools/search_bench/serve.py, which returns the
policy heads); --ports spreads the games over several replicas.

Closed decklists (docs/017 §3.4) are the default: each bot samples the opponent's hidden cards from
the belief service (tools/imitation_scale/belief_server.py, on --belief-port), which leaves the
opponent deck's draft out of its pool. --open-decklists instead re-deals from the real decklist. --record also writes each game's
training records (both seats: features, visit counts by action index, root value, decision type,
and the game's result from that seat) to records/p<pair><s if swapped>.jsonl.gz, one file per game,
for stage 6 (tools/imitation_scale/selfplay_tables.py).

Output in --out: games.jsonl (one line per game, appended, so a stopped run resumes), summary.json.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from draftzero import paths as dzpaths  # noqa: E402
from draftzero.gameplay.bridge import BridgeError, BridgePool  # noqa: E402

BOTS = {
    "heuristic": {"evaluator": "offline", "priors": False, "leaf": "heuristic"},
    "il_bc": {"evaluator": "remote", "priors": True, "leaf": "net"},
    "il_heur": {"evaluator": "remote", "priors": True, "leaf": "heuristic"},
    "bc": {"evaluator": "remote", "priors": False, "leaf": "net"},
    "policy": {"evaluator": "remote", "priors": True, "leaf": "net", "policy_temp": 1.0},
    "policy_greedy": {"evaluator": "remote", "priors": True, "leaf": "net", "policy_temp": 0.0},
}


def parse_bot(spec: str) -> tuple[str, int | None]:
    """'il_bc@1000' -> ('il_bc', 1000); 'heuristic' -> ('heuristic', None)."""
    name, _, budget = spec.partition("@")
    if name not in BOTS:
        raise ValueError(f"unknown bot {name!r}: one of {sorted(BOTS)}, optionally @<simulations>")
    if budget and (not budget.isdigit() or int(budget) < 1):
        raise ValueError(f"bad budget in {spec!r}")
    return name, int(budget) if budget else None


def seat_options(bot: str, budget: int, port: int | None, timeout_s: float, belief_port: int | None = None,
                 opponent_deck: str | None = None, policy_fallback_budget: int = 100) -> dict:
    bot, own = parse_bot(bot)
    budget = own or budget
    b = BOTS[bot]
    s = {"budget": budget, "leaf": b["leaf"], "timeoutSec": timeout_s}
    if "policy_temp" in b:
        s.update(budget=policy_fallback_budget, policyOnly=True, policyTemp=b["policy_temp"])
    if belief_port is not None:
        s["belief"] = {"port": belief_port, "exclude": opponent_deck, "worlds": 8}
    if b["evaluator"] == "remote":
        s["evaluator"] = {"type": "remote", "host": "127.0.0.1", "port": port}
        if b["priors"]:
            s.update(priors=True, opponentPriors="uniform", isPolicyPerWorld=True)
    else:
        s["evaluator"] = {"type": "offline"}
    return s


def deck_pairs(stems: list[str], n: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a, b = rng.sample(stems, 2)
        out.append((a, b))
    return out


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def summarize(games: list[dict], bot1: str, bot2: str) -> dict:
    """bot1's score: a win 1, a game stopped without a winner 0.5, a loss 0; and a 95% interval."""
    done = [g for g in games if not g.get("error")]
    pts = lambda g: 1.0 if g.get("winner_role") == "bot1" else 0.5 if g.get("winner_role") is None else 0.0  # noqa: E731
    score = sum(pts(g) for g in done)
    lo, hi = wilson(score, len(done))
    by_pair = {}
    for g in done:
        by_pair.setdefault(g["pair"], []).append(pts(g))
    return {"bot1": bot1, "bot2": bot2, "games": len(done), "errors": len(games) - len(done),
            "bot1_score": score / len(done) if done else None, "ci95": [lo, hi],
            "no_winner": sum(g.get("winner_role") is None for g in done),
            "pairs_both_won_by_bot1": sum(1 for v in by_pair.values() if len(v) == 2 and sum(v) == 2),
            "pairs_split": sum(1 for v in by_pair.values() if len(v) == 2 and sum(v) == 1),
            "mean_turns": sum(g["turns"] for g in done) / len(done) if done else None,
            "game_seconds": sum(g["seconds"] for g in done)}


def game_tasks(pairs: list, seed: int, mirror: bool) -> list[dict]:
    """Each deck pair twice: bot1 in seat A (deck1, first to play), then in seat B. A pair's two
    games share the game seed (the same shuffles, so deck luck cancels between the bots), except in
    a mirror match (self-play), where the swapped game would replay the first exactly."""
    tasks = []
    for k, (da, db) in enumerate(pairs):
        for swap in (False, True):
            tasks.append({"pair": k, "swap": swap, "deck1": da, "deck2": db,
                          "game_seed": seed * 1000 + k + (500_000 if mirror and swap else 0)})
    return tasks


def shard_tasks(tasks: list[dict], shard: str | None) -> list[dict]:
    """--shard i/n: this pod's share of the games, by deck pair (pair % n == i), so a pair's two games
    stay together and n pods' games.jsonl files concatenate into the whole run."""
    if not shard:
        return tasks
    i, n = (int(x) for x in shard.split("/"))
    if not 0 <= i < n:
        raise SystemExit(f"--shard {shard}: need 0 <= i < n")
    return [t for t in tasks if t["pair"] % n == i]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=str(REPO / "data/pools/eval.txt"))
    ap.add_argument("--deck-root", required=True)
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--policy-fallback-budget", type=int, default=100,
                    help="policy bots: simulations for the decisions without a policy head")
    ap.add_argument("--bot1", required=True, help=f"one of {sorted(BOTS)}, optionally @<simulations>")
    ap.add_argument("--bot2", required=True, help="as --bot1")
    ap.add_argument("--ports", default="50052", help="inference servers, comma-separated (network bots)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--heap", default="3g")
    ap.add_argument("--max-turns", type=int, default=50)
    ap.add_argument("--search-timeout", type=float, default=300.0)
    ap.add_argument("--game-timeout", type=float, default=7200.0)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--belief-port", type=int, default=50070, help="the belief service (closed decklists)")
    ap.add_argument("--open-decklists", action="store_true", help="re-deal from the real decklist instead")
    ap.add_argument("--shard", default=None, metavar="I/N", help="play only deck pairs with pair %% N == I (one pod of N)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    for spec in (a.bot1, a.bot2):
        try:
            parse_bot(spec)
        except ValueError as e:
            ap.error(str(e))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stems = dzpaths.read_pool(Path(a.pool))
    pairs = deck_pairs(stems, a.pairs, a.seed)
    ports = [int(p) for p in a.ports.split(",")]
    tasks = shard_tasks(game_tasks(pairs, a.seed, mirror=a.bot1 == a.bot2), a.shard)
    done_keys = set()
    games_file = out / "games.jsonl"
    if games_file.exists():
        for line in games_file.read_text().splitlines():
            g = json.loads(line)
            if not g.get("error"):
                done_keys.add((g["pair"], g["swap"]))
    todo = [t for t in tasks if (t["pair"], t["swap"]) not in done_keys]
    (out / "config.json").write_text(json.dumps({**vars(a), "n_tasks": len(tasks)}, indent=1))
    print(f"{len(tasks)} games, {len(todo)} to play", flush=True)

    belief = None if a.open_decklists else a.belief_port
    if belief is not None:
        import urllib.request
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{belief}/healthz", timeout=5).read()
        except OSError:
            raise SystemExit(f"no belief service on port {belief}: start tools/imitation_scale/belief_server.py "
                             f"--port {belief}, or pass --open-decklists")
    lock = threading.Lock()
    pool = BridgePool(a.workers, prefix="play", heap=a.heap)
    rec_dir = out / "records"
    t0 = time.time()

    def play(i_t):
        i, t = i_t
        port = ports[i % len(ports)]
        # the decks keep their seats (deck1 in A, which plays first), the bots swap seats: over a
        # pair each bot plays each deck once and goes first once, so deck strength and the play
        # order cancel in the pair
        bot_a, bot_b = (a.bot2, a.bot1) if t["swap"] else (a.bot1, a.bot2)
        deck_a, deck_b = t["deck1"], t["deck2"]
        opts = dict(deckA=str(dzpaths.deck_path(deck_a, Path(a.deck_root)).resolve()),
                    deckB=str(dzpaths.deck_path(deck_b, Path(a.deck_root)).resolve()),
                    seatA=seat_options(bot_a, a.budget, port, a.search_timeout, belief, deck_b, a.policy_fallback_budget),
                    seatB=seat_options(bot_b, a.budget, port, a.search_timeout, belief, deck_a, a.policy_fallback_budget),
                    seed=t["game_seed"], starting="A", maxTurns=a.max_turns, record=a.record)
        g = {**t, "botA": bot_a, "botB": bot_b, "budget": a.budget, "closed_decklists": belief is not None}
        ts = time.time()
        try:
            r = pool.request("play", None, timeout=a.game_timeout, **opts)
            recs = r.pop("records", None)
            # bot1 sat in seat A unless swapped: credit the role, so a mirror match (bot1 = bot2) still splits
            role = None if r["winner"] is None else ("bot1" if (r["winner"] == "A") != t["swap"] else "bot2")
            g.update(winner=r["winner"], winner_role=role, winner_bot=None if role is None else (a.bot1 if role == "bot1" else a.bot2),
                     turns=r["turns"], seats=r["seats"], seconds=round(time.time() - ts, 1))
            if recs is not None and a.record:
                # one file per game, written whole before the game counts as done (a pod stopped
                # mid-run loses only unfinished games)
                lines = []
                for seat, bot in (("A", bot_a), ("B", bot_b)):
                    z = None if r["winner"] is None else (1 if r["winner"] == seat else -1)
                    lines.append(json.dumps({"pair": t["pair"], "swap": t["swap"], "seat": seat, "bot": bot,
                                             "result": z, "records": recs[seat]}) + "\n")
                rec_dir.mkdir(parents=True, exist_ok=True)
                dst = rec_dir / f"p{t['pair']:05d}{'s' if t['swap'] else ''}.jsonl.gz"
                tmp = dst.with_name(dst.name + ".tmp")
                with gzip.open(tmp, "wt") as f:
                    f.write("".join(lines))
                os.replace(tmp, dst)
        except (BridgeError, Exception) as e:  # noqa: BLE001 - one bad game must not stop the run
            g.update(error=f"{type(e).__name__}: {e}"[:500], seconds=round(time.time() - ts, 1))
        with lock:
            with open(games_file, "a") as f:
                f.write(json.dumps(g) + "\n")
            n = sum(1 for _ in open(games_file))
            print(f"  {n}/{len(tasks)} games  last: pair {t['pair']} {'swapped' if t['swap'] else ''} "
                  f"{g.get('winner_role') or g.get('error', 'no winner')} ({g['seconds']} s)  "
                  f"elapsed {time.time() - t0:.0f} s", flush=True)
        return g

    from concurrent.futures import ThreadPoolExecutor
    try:
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(play, enumerate(todo)))
    finally:
        pool.close()
    games = [json.loads(line) for line in games_file.read_text().splitlines()]
    s = summarize(games, a.bot1, a.bot2)
    s["wall_seconds"] = round(time.time() - t0)
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
