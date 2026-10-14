"""
Read dzgorge game records (`dzgorge play -out`, gorge fork cmd/dzgorge): scores, ratings, 17lands card
statistics, throughput.

  python -m draftzero.eval.analyze wins RUN.jsonl [...]
  python -m draftzero.eval.analyze ratings DIR_OR_FILE [...] [--min-games 200] [--anchor sbtactical] [--json OUT]
  python -m draftzero.eval.analyze gih RUN.jsonl [...] [--min-games 30] [--side A] [--json OUT]
  python -m draftzero.eval.analyze speed RUN.jsonl.summary.json [...]

wins     A's score in each file over complete deck pairs (eval.paired: a draw or stall counts half, engine
         errors are dropped with their pair), with a 95% interval over pairs.
ratings  one Bradley-Terry fit (eval.ratings) over every non-self-play games file; a player is its policy spec
         with the shared search keys dropped and a network named by its directory.
gih      17lands' games-in-hand win rate from the simulated games and its Spearman correlation with 17lands'
         (eval.gih), over commons and over every non-basic card with --min-games games in hand (use 100 to
         match docs/019), plus the colour pairs' records.
speed    games per hour from each run's summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from draftzero.eval import gih as gihmod
from draftzero.eval.paired import policies, score_pairs
from draftzero.eval.ratings import game_triples, table


def load(paths) -> list[dict]:
    """games.jsonl rows; a killed run's partial last line is skipped."""
    out = []
    for p in paths:
        with open(p) as f:
            for line in f:
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return out


def games_files(paths) -> list[Path]:
    """Files as given, and every games file under directories (no chunk, audit or visit files)."""
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(f for f in p.rglob("*.jsonl")
                          if not any(x in f.name for x in (".chunk", "audit", "visits", "corpus")))
        else:
            out.append(p)
    return out


def _pct(x) -> str:
    return "  n/a" if x is None else f"{100 * x:5.1f}"


def cmd_wins(a):
    for p in a.games:
        games = load([p])
        r = score_pairs(games)
        if not r["n_pairs"]:
            print(f"{p}: no complete pairs ({len(games)} games, {r['errors']} errors)")
            continue
        pols = policies(games)
        turns = sum(g.get("turns", 0) for g in games) / max(len(games), 1)
        print(f"{Path(p).stem:24s} {'|'.join(sorted(pols['A'])):>28s} vs {'|'.join(sorted(pols['B'])):<28s}"
              f" {_pct(r['score'])}% [{_pct(r['lo'])}, {_pct(r['hi'])}]  {r['n_games']} games / {r['n_pairs']} pairs,"
              f" {r['stalls']} stalls, {r['errors']} errors, {turns:.1f} turns")


def cmd_ratings(a):
    triples = []
    for f in games_files(a.paths):
        triples += game_triples(load([f]))
    rows = table(triples, anchor=a.anchor, min_games=a.min_games)
    print(f"{len(triples):,} games; Elo points, {a.anchor} = 0")
    for r in rows:
        print(f"{r['elo']:7.0f} ± {r['se']:3.0f}  {r['games']:6.0f} games  {r['player']}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))


def cmd_gih(a):
    games = load(a.games)
    r = gihmod.gih_report(games, a.min_games, a.side)
    print(f"{r['games']:,} games, {r['player_games']:,} player-games with a winner")
    for label in ("commons", "all"):
        x = r[label]
        rho = "n/a" if x["spearman"] is None else f"{x['spearman']:.3f}"
        sd = "n/a" if x["sd_sim"] is None else f"{100 * x['sd_sim']:.1f} pts (17lands {100 * x['sd_17lands']:.1f})"
        print(f"{label:8s} {x['cards']:3d} cards with {a.min_games}+ games in hand: Spearman {rho} vs 17lands;"
              f" SD of win rates {sd}")
    c = r["colors"]
    rho = "n/a" if c["spearman"] is None else f"{c['spearman']:.3f}"
    print(f"colour pairs: Spearman {rho} vs 17lands; " +
          ", ".join(f"{k} {v['wr']:.3f}" for k, v in sorted(c["rows"].items(), key=lambda kv: -kv[1]["wr"])))
    rows = r["commons"]["rows"]
    if rows:
        avg_s = sum(x[2] for x in rows) / len(rows)
        avg_h = sum(x[3] for x in rows) / len(rows)
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
    r = sub.add_parser("ratings")
    r.add_argument("paths", nargs="+")
    r.add_argument("--min-games", type=float, default=200)
    r.add_argument("--anchor", default="sbtactical")
    r.add_argument("--json", default=None)
    r.set_defaults(fn=cmd_ratings)
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
