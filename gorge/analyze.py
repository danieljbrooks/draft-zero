"""
Read dzgorge game records (gorge/cmd/dzgorge): win rates, 17lands card statistics, throughput.

  python gorge/analyze.py wins data/gorge/runs/bench/*.jsonl
  python gorge/analyze.py gih data/gorge/runs/bench/bot_self.jsonl [--min-games 30] [--json out.json]
  python gorge/analyze.py speed data/gorge/runs/bench/*.summary.json

wins   A's score (a draw or a turn-capped game counts half; engine errors are dropped), with a 95%
       interval. In paired runs both games of a deck pair share a seed and swap the policies, so the
       pair is the sampling unit.
gih    17lands' games-in-hand win rate from the simulated games: per card, the win rate of the seats
       that had it in hand (kept opening hand or drawn), over games with a winner. Spearman rank
       correlation with 17lands' FDN Premier Draft GIH WR (assets/reference/FDN_gih.json), over
       commons and over every non-basic card with at least --min-games games in hand, as in
       docs/019 §4.4. Also the spread of the commons' win rates and the colour-pair records.
speed  games per hour from each run's summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / "assets" / "reference" / "FDN_gih.json"
RARITY = REPO / "assets" / "gameplay" / "FDN_card_ids.tsv"
POOL = REPO / "data" / "gorge" / "pool.tsv"
BASICS = {"Plains", "Island", "Swamp", "Mountain", "Forest"}


def load(paths) -> list[dict]:
    out = []
    for p in paths:
        with open(p) as f:
            out += [json.loads(l) for l in f if l.strip()]
    return out


def points(g: dict) -> float | None:
    """A's points in one game: 1 win, 0.5 draw or stall, 0 loss; None for an engine error."""
    return {"A": 1.0, "B": 0.0, "draw": 0.5, "stall": 0.5}.get(g["result"])


def score(games: list[dict]) -> dict:
    by_pair = defaultdict(list)
    for g in games:
        p = points(g)
        if p is not None:
            by_pair[g["pair"]].append(p)
    pair_means = np.array([np.mean(v) for v in by_pair.values()])
    n_games = sum(len(v) for v in by_pair.values())
    if len(pair_means) == 0:
        return {"games": 0}
    s = float(np.mean(pair_means))
    se = float(np.std(pair_means, ddof=1) / math.sqrt(len(pair_means))) if len(pair_means) > 1 else float("nan")
    return {"games": n_games, "pairs": len(pair_means), "score": s, "lo": s - 1.96 * se, "hi": s + 1.96 * se,
            "errors": sum(g["result"] == "error" for g in games),
            "stalls": sum(g["result"] == "stall" for g in games),
            "turns": float(np.mean([g["turns"] for g in games]))}


def cmd_wins(a):
    for p in a.games:
        games = load([p])
        r = score(games)
        if not r["games"]:
            print(f"{p}: no games")
            continue
        pols = games[0]["pols"] if games[0]["side"][0] == "A" else games[0]["pols"][::-1]
        print(f"{Path(p).stem:24s} {pols[0]:>28s} vs {pols[1]:<28s} {r['score']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"
              f"  {r['games']} games / {r['pairs']} pairs, {r['stalls']} stalls, {r['errors']} errors, {r['turns']:.1f} turns")


def ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b) -> float:
    ra, rb = ranks(np.asarray(a, float)), ranks(np.asarray(b, float))
    if len(ra) < 3 or ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def rarity() -> dict[str, str]:
    out: dict[str, str] = {}
    with open(RARITY) as f:
        for r in csv.DictReader((l for l in f if not l.startswith("#")), delimiter="\t"):
            if r.get("expansion") == "FDN" or r["name"] not in out:
                out[r["name"]] = r.get("rarity", "")
    return out


def gih(games: list[dict], side: str | None = None):
    """card -> [games in hand, wins]; deck colours -> [games, wins]; player-games counted."""
    cards: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    decks: list[tuple[str, int]] = []
    seats = 0
    for g in games:
        if g["result"] not in ("A", "B"):
            continue
        for s in (0, 1):
            if side and g["side"][s] != side:
                continue
            seats += 1
            won = int(g["winner"] == s)
            for c in set(g["in_hand"][s] or []):
                cards[c][0] += 1
                cards[c][1] += won
            decks.append((g["decks"][s], won))
    return cards, decks, seats


def deck_colors() -> dict[str, str]:
    with open(POOL) as f:
        return {r["deck"]: r["main_colors"] for r in csv.DictReader(f, delimiter="\t")}


def gih_report(games: list[dict], min_games: int, side: str | None = None) -> dict:
    ref_all = json.loads(REFERENCE.read_text())
    ref = ref_all["cards"]
    rar = rarity()
    cards, decks, seats = gih(games, side)
    out = {"games": len(games), "player_games": seats, "min_games": min_games}
    for label, keep in (("commons", lambda c: rar.get(c) == "common"), ("all", lambda c: True)):
        rows = [(c, n, w / n, ref[c]["gih_wr"]) for c, (n, w) in cards.items()
                if c not in BASICS and c in ref and n >= min_games and keep(c)]
        sim = np.array([r[2] for r in rows])
        hum = np.array([r[3] for r in rows])
        out[label] = {"cards": len(rows), "spearman": spearman(sim, hum),
                      "sd_sim": float(sim.std()) if len(rows) else float("nan"),
                      "sd_17lands": float(hum.std()) if len(rows) else float("nan"),
                      "rows": sorted(rows, key=lambda r: -r[2])}
    # Colour pairs: the decks' main colours, two-colour decks only.
    col = deck_colors()
    by = defaultdict(lambda: [0, 0])
    for d, w in decks:
        c = col.get(d, "")
        if len(c) == 2:
            by[c][0] += 1
            by[c][1] += w
    refc = ref_all["colors"]
    pairs = sorted(c for c in by if c in refc)
    out["colors"] = {"rows": {c: {"games": by[c][0], "wr": by[c][1] / by[c][0], "ref_wr": refc[c]["wr"]} for c in pairs},
                     "spearman": spearman([by[c][1] / by[c][0] for c in pairs], [refc[c]["wr"] for c in pairs])}
    return out


def cmd_gih(a):
    games = load(a.games)
    r = gih_report(games, a.min_games, a.side)
    print(f"{r['games']:,} games, {r['player_games']:,} player-games with a winner")
    for label in ("commons", "all"):
        x = r[label]
        print(f"{label:8s} {x['cards']:3d} cards with {a.min_games}+ games in hand: Spearman {x['spearman']:.3f} vs 17lands;"
              f" SD of win rates {100 * x['sd_sim']:.1f} pts (17lands {100 * x['sd_17lands']:.1f})")
    c = r["colors"]
    print(f"colour pairs: Spearman {c['spearman']:.3f} vs 17lands; " +
          ", ".join(f"{k} {v['wr']:.3f}" for k, v in sorted(c["rows"].items(), key=lambda kv: -kv[1]["wr"])))
    rows = r["commons"]["rows"]
    if rows:
        avg_s, avg_h = np.mean([x[2] for x in rows]), np.mean([x[3] for x in rows])
        print("\nbest commons in the simulation (points above each source's commons average):")
        for card, n, s, h in rows[:8]:
            print(f"  {card:28s} sim {100 * (s - avg_s):+5.1f} ({n:6d})  17lands {100 * (h - avg_h):+5.1f}")
        print("worst:")
        for card, n, s, h in rows[-8:]:
            print(f"  {card:28s} sim {100 * (s - avg_s):+5.1f} ({n:6d})  17lands {100 * (h - avg_h):+5.1f}")
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1))


def cmd_speed(a):
    print(f"{'run':24s} {'workers':>7s} {'games':>7s} {'games/h':>10s} {'games/h/worker':>15s} {'turns':>6s} {'ms/searched':>12s}")
    for p in a.summaries:
        s = json.loads(Path(p).read_text())
        ms = s.get("az_ms_per_searched")
        print(f"{Path(p).name.split('.')[0]:24s} {s['workers']:7d} {s['games']:7d} {s['games_per_hour']:10,.0f}"
              f" {s['games_per_hour_per_worker']:15,.0f} {s['mean_turns']:6.1f} {'' if ms is None else f'{ms:12.1f}'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("wins")
    w.add_argument("games", nargs="+")
    w.set_defaults(fn=cmd_wins)
    g = sub.add_parser("gih")
    g.add_argument("games", nargs="+")
    g.add_argument("--min-games", type=int, default=30)
    g.add_argument("--side", choices=["A", "B"], default=None, help="count only one side's seats")
    g.add_argument("--json", default=None)
    g.set_defaults(fn=cmd_gih)
    s = sub.add_parser("speed")
    s.add_argument("summaries", nargs="+")
    s.set_defaults(fn=cmd_speed)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
