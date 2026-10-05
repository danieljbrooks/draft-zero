"""PIMC against IS-MCTS in the MLP's games (docs/019 §4.6, docs/018): the score against heuristic@100 by search budget
for each search method, the same games compared pair by pair, and what the games cost.

    python tools/imitation_scale/pimc_games.py [--figure]

Methods, all with run 2's MLP against heuristic@100 on phase C's deck pairs and seeds, both bots on the same method:
    IS-MCTS              docs/019's games (mlp-*). Their belief service failed every call, so every decision searched
                         worlds re-dealt from the opponent's real decklist.
    PIMC, real decklist  pimc-open-mlp-*: one world re-dealt from the opponent's real decklist (the same information)
    PIMC, belief         pimc-mlp-*: one world sampled from the belief service (real 17lands decks that fit the cards
                         seen so far), the fair setting

A budget pools its runs (shards, top-ups), one result per (deck pair, seat swap), without engine errors; a game
capped at 50 turns counts 0.5. "Same games" is the mean difference over the (pair, swap) games both methods have,
± 1.96 standard errors of the per-pair differences (the two games of a pair are not independent).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st
from pathlib import Path

G = "runs/exp4/games"
BUDGETS = ["0", "100", "300", "1000"]
METHODS = {
    "IS-MCTS": {"0": [f"{G}/mlp-pvh-t0"], "100": [f"{G}/mlp-ilbc100"], "300": [f"{G}/mlp-ilbc300"],
                "1000": [f"{G}/mlp-ilbc1000-s*", f"{G}/mlp-ilbc1000-top"]},
    "PIMC, real decklist": {"0": [f"{G}/pimc-open-mlp-pvh-t0"], "100": [f"{G}/pimc-open-mlp-ilbc100"],
                            "300": [f"{G}/pimc-open-mlp-ilbc300"], "1000": [f"{G}/pimc-open-mlp-ilbc1000*"]},
    "PIMC, belief": {"0": [f"{G}/pimc-mlp-pvh-t0"], "100": [f"{G}/pimc-mlp-ilbc100"],
                     "300": [f"{G}/pimc-mlp-ilbc300"], "1000": [f"{G}/pimc-mlp-ilbc1000*"]},
}


def load(patterns):
    """(pair, swap) -> game record, engine errors left out; and the number of errors."""
    out, errors = {}, 0
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            f = Path(d) / "games.jsonl"
            if not f.exists():
                continue
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                g = json.loads(line)
                if g.get("error"):
                    errors += 1
                    continue
                out[(g["pair"], g["swap"])] = g
    return out, errors


def points(g) -> float:
    return 1.0 if g.get("winner_role") == "bot1" else 0.5 if g.get("winner_role") is None else 0.0


def seat_totals(games, bot1: bool) -> dict:
    """Search totals of bot1's (or bot2's) seat over the games."""
    t = {}
    for g in games.values():
        bot1_in_a = not g["swap"]   # bot1 sits in A unless the pair's game is swapped
        seat = "A" if bot1_in_a == bot1 else "B"
        v = (g.get("seats") or {}).get(seat) or {}
        for k in ("decisions", "sims", "engineSteps", "searchSeconds", "worldsBuilt", "openFallbacks", "fallbacks"):
            t[k] = t.get(k, 0) + (v.get(k) or 0)
    return t


def same_games(a, b):
    """Mean of b minus a over the (pair, swap) games both have, and its 95% half-width over pairs."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    by_pair = {}
    for k in keys:
        by_pair.setdefault(k[0], []).append(points(b[k]) - points(a[k]))
    d = [sum(v) / len(v) for v in by_pair.values()]
    w = [len(v) for v in by_pair.values()]
    mean = sum(x * n for x, n in zip(d, w)) / sum(w)
    half = 1.96 * st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    return mean, half, len(keys)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", action="store_true", help="also draw docs/img/019-pimc-light.png")
    a = ap.parse_args(argv)
    data = {(m, b): load(p) for m, spec in METHODS.items() for b, p in spec.items()}
    print("| Simulations | Method | Valid games (errors) | Score | Same games as IS-MCTS | Median game | "
          "Network bot s/decision | Heuristic s/decision | Engine steps/simulation (network, heuristic) | "
          "Belief worlds built / decisions |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    table = {}
    for b in BUDGETS:
        for m in METHODS:
            games, err = data[(m, b)]
            if not games:
                continue
            sc = sum(points(g) for g in games.values()) / len(games)
            table[(m, b)] = (sc, len(games))
            same = "" if m == "IS-MCTS" else same_games(data[("IS-MCTS", b)][0], games)
            same = "" if not same else f"{100 * same[0]:+.1f} ± {100 * same[1]:.1f} ({same[2]})"
            med = st.median(g["seconds"] for g in games.values()) / 60
            n1, h = seat_totals(games, True), seat_totals(games, False)
            spd = lambda t: f"{t['searchSeconds'] / t['decisions']:.2f}" if t.get("decisions") else "–"   # noqa: E731
            sps = lambda t: f"{t['engineSteps'] / t['sims']:.1f}" if t.get("sims") else "–"   # noqa: E731
            built = n1.get("worldsBuilt", 0) + h.get("worldsBuilt", 0)
            dec = n1.get("decisions", 0) + h.get("decisions", 0)
            print(f"| {b} | {m} | {len(games)} ({err}) | {100 * sc:.1f}% | {same} | {med:.1f} min | {spd(n1)} | "
                  f"{spd(h)} | {sps(n1)}, {sps(h)} | {built:,} / {dec:,} |")
    if a.figure:
        figure(table)
    return 0


def figure(table):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    T = dict(surface="#fcfcfb", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7")
    colour = {"IS-MCTS": "#1baf7a", "PIMC, real decklist": "#2a78d6", "PIMC, belief": "#eb6834"}
    style = {"IS-MCTS": "-", "PIMC, real decklist": "-", "PIMC, belief": (0, (4, 2))}
    labels = {"0": "policy alone\n(no search)", "100": "100", "300": "300", "1000": "1,000"}
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 9})
    fig, ax = plt.subplots(figsize=(8.6, 4.6), facecolor=T["surface"])
    xs = {b: i for i, b in enumerate(BUDGETS)}
    off = {"IS-MCTS": -0.06, "PIMC, real decklist": 0.0, "PIMC, belief": 0.06}
    for m in METHODS:
        pts = [(b, table[(m, b)]) for b in BUDGETS if (m, b) in table]
        if not pts:
            continue
        x = [xs[b] + off[m] for b, _ in pts]
        y = [100 * v[0] for _, v in pts]
        ax.plot(x, y, color=colour[m], lw=2.2, ls=style[m], zorder=2)
        ax.plot(x, y, "o", ms=7.5, color=colour[m], zorder=3)
    ax.axhline(50, color=T["muted"], lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(len(BUDGETS) - 0.6, 49, "even with the baseline", fontsize=8, color=T["muted"], ha="right", va="top")
    ax.set_ylim(25, 80)
    ax.set_yticks(range(30, 81, 10), [f"{v}%" for v in range(30, 81, 10)])
    ax.set_ylabel("Games won against the baseline", fontsize=9, color=T["ink2"])
    ax.set_xticks(range(len(BUDGETS)), [labels[b] for b in BUDGETS], fontsize=8.5, color=T["ink2"])
    ax.set_xlim(-0.4, len(BUDGETS) - 0.6)
    ax.set_xlabel("Search: simulations per decision (both bots use the same search method)", fontsize=9,
                  color=T["ink2"])
    ax.set_facecolor(T["surface"])
    ax.grid(color=T["grid"], lw=0.8, axis="y")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(T["axis"])
    ax.tick_params(colors=T["ink2"], labelsize=8.5, length=0)
    names = {"IS-MCTS": "IS-MCTS (knows the opponent's decklist)",
             "PIMC, real decklist": "PIMC, one world (knows the opponent's decklist)",
             "PIMC, belief": "PIMC, one world (guesses the opponent's deck)"}
    ax.legend(handles=[Line2D([], [], color=colour[m], lw=2.2, ls=style[m], marker="o", label=names[m])
                       for m in METHODS if any((m, b) in table for b in BUDGETS)],
              loc="upper left", frameon=False, fontsize=8.5, labelcolor=T["ink2"])
    fig.tight_layout()
    p = Path("docs/img/019-pimc-light.png")
    fig.savefig(p, dpi=150, facecolor=T["surface"])
    print(p)


if __name__ == "__main__":
    raise SystemExit(main())
