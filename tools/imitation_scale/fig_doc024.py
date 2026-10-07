"""The games figures of docs/024 (the full GNN), in light and dark: the PIMC ladder against heuristic@100, and deck win
rate by colour pair in greedy self-play. The card win-rate scatter (024-gih) comes from tools/imitation_scale/fig_gih.py.

    docs/img/024-ladder-{light,dark}.png   win rate against heuristic@100 by the network's simulations (PIMC, guessed
                                           decks): the full GNN, docs/023's GNN and the MLP (docs/019 §4.6), and the full
                                           GNN's policy with its epoch-7 value head at 1,000 (a hollow diamond)
    docs/img/024-colours-{light,dark}.png  deck win rate by colour pair: the GNN's, the MLP's and the transformer's
                                           greedy self-play, and 17lands' top players

    python tools/imitation_scale/fig_doc024.py

The ladder pools docs/019's way: a budget's runs (shards, top-ups), one result per (deck pair, seat swap), engine
errors left out, a game capped at 50 turns counting 0.5. Colour pairs count both decks of every game with a winner,
by the deck's two main colours (a splash ignored), which reproduces docs/019's numbers for the MLP and the transformer.
"""
from __future__ import annotations

import glob
import json
import math
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

G, E = "runs/gnn/games", "runs/exp4/games"
OUT = Path("docs/img")
BUDGETS = ["0", "100", "300", "1000"]
LABELS = {"0": "policy alone\n(no search)", "100": "100", "300": "300", "1000": "1,000"}
LADDER = {
    "GNN, full run (this doc)": {"0": [f"{G}/pimc-gnn-full-pvh-t0"], "100": [f"{G}/pimc-gnn-full-gnn100"],
                                 "300": [f"{G}/pimc-gnn-full-gnn300"], "1000": [f"{G}/pimc-gnn-full-gnn1000-s*"]},
    "GNN, docs/023 (3 epochs)": {"0": [f"{G}/pimc-gnn-stage3-pvh-t0"], "100": [f"{G}/pimc-gnn-stage3-gnn100"],
                                 "300": [f"{G}/pimc-gnn-stage3-gnn300"]},
    "MLP (docs/019 §4.6)": {"0": [f"{E}/pimc-mlp-pvh-t0"], "100": [f"{E}/pimc-mlp-ilbc100"],
                            "300": [f"{E}/pimc-mlp-ilbc300"],
                            "1000": [f"{E}/pimc-mlp-ilbc1000-s*", f"{E}/pimc-mlp-ilbc1000-top"]},
}
# the full GNN's policy with the value from its epoch-7 checkpoint (graph_server.py --value-model; docs/024 §5)
HYBRID = ("Full run's policy, epoch-7 value head", "1000", [f"{G}/pimc-gnn-full-v7-gnn1000-s*"])
SELFPLAY = {"GNN": [f"{G}/gnn-full-selfplay-t0-s*"], "MLP": [f"{E}/mlp-selfplay-t0"],
            "Transformer": [f"{E}/c1-selfplay-t0-s*"]}
# 17lands' top players' deck win rate by main colours (docs/018, C1; docs/019's figure 6)
TOP_PLAYERS = {"WU": 0.656, "WG": 0.646, "WB": 0.634, "RG": 0.653, "WR": 0.650, "UB": 0.640, "UG": 0.622,
               "BG": 0.612, "UR": 0.638, "BR": 0.639}
# categorical slots validated for colour-vision separation (fig_mlp_curves.py's): the full run keeps slot 0, docs/023's
# GNN slot 1, the MLP slot 2 and the transformer slot 3, as in figure 3
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"]},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "series": ["#3987e5", "#d95926", "#199e70", "#9b72e0"]},
}


def records(patterns):
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            f = Path(d) / "games.jsonl"
            for line in f.read_text().splitlines():
                if line.strip():
                    yield json.loads(line)


def ladder_scores(patterns):
    out = {}
    for g in records(patterns):
        if g.get("error"):
            continue
        out[(g["pair"], g["swap"])] = 1.0 if g.get("winner_role") == "bot1" else 0.5 if g.get("winner_role") is None else 0.0
    return out


def main_colours(deck):
    s = deck.split("_")[-1].split("p")[0]
    return "".join(c for c in "WUBRG" if c in s) if len(s) == 2 else None


def colour_rates(patterns):
    n, w = {}, {}
    for g in records(patterns):
        if g.get("error") or g.get("winner") is None:
            continue
        for deck, seat in ((g["deck1"], "A"), (g["deck2"], "B")):
            m = main_colours(deck)
            if m:
                n[m] = n.get(m, 0) + 1
                w[m] = w.get(m, 0) + (g["winner"] == seat)
    return {k: w[k] / n[k] for k in n}


def style(ax, t, grid="y"):
    ax.set_facecolor(t["surface"])
    if grid:
        ax.grid(color=t["grid"], lw=0.8, axis=grid)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(t["axis"])
    ax.tick_params(colors=t["ink2"], labelsize=8.5, length=0)


def ladder(table, hybrid, theme):
    t = THEMES[theme]
    xs = {b: i for i, b in enumerate(BUDGETS)}
    fig, ax = plt.subplots(figsize=(8.6, 4.8), facecolor=t["surface"])
    for k, net in enumerate(LADDER):
        col = t["series"][k]
        pts = [(b, table[(net, b)]) for b in BUDGETS if (net, b) in table]
        ax.plot([xs[b] for b, _ in pts], [100 * v[0] for _, v in pts], color=col, lw=2, zorder=2 + (k == 0),
                ls="-" if k != 1 else (0, (4, 2)))
        for b, (sc, n, half) in pts:
            ax.plot(xs[b], 100 * sc, "o", ms=8, color=col, markeredgecolor=t["surface"], markeredgewidth=2,
                    zorder=4 + (k == 0))
    for b in BUDGETS:   # the full run's value at each budget: above its point when highest there, else below
        sc = table[(next(iter(LADDER)), b)][0]
        above = all(sc >= table[(n, b)][0] for n in list(LADDER)[1:] if (n, b) in table)
        ax.annotate(f"{100 * sc:.1f}%", (xs[b], 100 * sc), textcoords="offset points", xytext=(0, 10 if above else -11),
                    ha="center", va="bottom" if above else "top", fontsize=9, color=t["ink"], weight="bold")
    hx = xs[HYBRID[1]] + 0.13       # beside the budget's other points, clear of the MLP's
    ax.plot(hx, 100 * hybrid, "D", ms=8, color=t["surface"], markeredgecolor=t["series"][0], markeredgewidth=2, zorder=6)
    ax.annotate(f"{100 * hybrid:.1f}%", (hx, 100 * hybrid), textcoords="offset points", xytext=(9, 0), ha="left",
                va="center", fontsize=9, color=t["ink"], weight="bold")
    ax.axhline(50, color=t["muted"], lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(len(BUDGETS) - 0.55, 49, "even with the baseline", fontsize=8, color=t["muted"], ha="right", va="top")
    ax.set_ylim(30, 82)
    ax.set_yticks(range(30, 81, 10), [f"{v}%" for v in range(30, 81, 10)])
    ax.set_ylabel("Games won against the baseline (about 100 each)", fontsize=9, color=t["ink2"])
    ax.set_xticks(range(len(BUDGETS)), [LABELS[b] for b in BUDGETS], fontsize=8.5, color=t["ink2"])
    ax.set_xlim(-0.4, len(BUDGETS) - 0.35)
    ax.set_xlabel("Search: the network's simulations per decision (PIMC, guessed decks)", fontsize=9, color=t["ink2"])
    ax.set_title("Against MageZero's heuristic bot at 100 simulations", loc="left", fontsize=10, color=t["ink"])
    style(ax, t)
    ax.legend(handles=[Line2D([], [], color=t["series"][k], lw=2, marker="o", ls="-" if k != 1 else (0, (4, 2)),
                              label=n) for k, n in enumerate(LADDER)]
             + [Line2D([], [], color=t["surface"], marker="D", markeredgecolor=t["series"][0], markeredgewidth=2, ms=7,
                       lw=0, label=HYBRID[0])],
              loc="lower right", frameon=False, fontsize=8.5, labelcolor=t["ink2"])
    fig.tight_layout()
    p = OUT / f"024-ladder-{theme}.png"
    fig.savefig(p, dpi=150, facecolor=t["surface"])
    plt.close(fig)
    print(p)


def colours(rates, theme):
    t = THEMES[theme]
    series = [("GNN's self-play (this doc)", rates["GNN"], t["series"][0]),
              ("MLP's self-play", rates["MLP"], t["series"][2]),
              ("Transformer's self-play", rates["Transformer"], t["series"][3]),
              ("17lands' top players", TOP_PLAYERS, t["ink2"])]
    order = sorted(TOP_PLAYERS, key=lambda k: -rates["GNN"][k])
    fig, ax = plt.subplots(figsize=(8.6, 4.2), facecolor=t["surface"])
    for i, c in enumerate(order):
        for (label, r, col), off in zip(series, (-0.21, -0.07, 0.07, 0.21)):
            mean = sum(r.values()) / len(r)
            ax.plot(i + off, 100 * (r[c] - mean), "D" if label.startswith("17") else "o", ms=7 if off != -0.21 else 8.5,
                    color=col, zorder=3 + (off == -0.21), markeredgecolor=t["surface"], markeredgewidth=1.2)
    ax.axhline(0, color=t["muted"], lw=1, ls=(0, (4, 3)), zorder=0)
    ax.set_xticks(range(len(order)), order, fontsize=9, color=t["ink"])
    ax.set_yticks(range(-10, 11, 5), [f"{v:+d}" if v else "0" for v in range(-10, 11, 5)])
    ax.set_ylabel("Deck win rate, points above\nor below the source's average", fontsize=9, color=t["ink2"])
    ax.set_xlabel("Deck colours (sorted by the GNN's self-play)", fontsize=9, color=t["ink2"])
    ax.set_title("Deck win rate by colour pair", loc="left", fontsize=10, color=t["ink"])
    style(ax, t)
    ax.legend(handles=[Line2D([], [], color=col, marker="D" if label.startswith("17") else "o", lw=0, ms=7,
                              label=label) for label, _, col in series],
              loc="lower left", frameon=False, fontsize=8.5, labelcolor=t["ink2"], ncol=2)
    fig.tight_layout()
    p = OUT / f"024-colours-{theme}.png"
    fig.savefig(p, dpi=150, facecolor=t["surface"])
    plt.close(fig)
    print(p)


def main() -> int:
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 9})
    table = {}
    for net, spec in LADDER.items():
        for b, pats in spec.items():
            g = ladder_scores(pats)
            v = list(g.values())
            sc = sum(v) / len(v)
            half = 1.96 * st.pstdev(v) / math.sqrt(len(v))
            table[(net, b)] = (sc, len(v), half)
            print(f"{net:26s} {b:>5s}: {100 * sc:.1f}% of {len(v)} (+/- {100 * half:.1f})")
    rates = {n: colour_rates(p) for n, p in SELFPLAY.items()}
    for n, r in rates.items():
        print(n, " ".join(f"{k} {r[k]:.3f}" for k in sorted(r, key=lambda k: -r[k])))
    h = list(ladder_scores(HYBRID[2]).values())
    hybrid = sum(h) / len(h)
    print(f"{HYBRID[0]:26s} {HYBRID[1]:>5s}: {100 * hybrid:.1f}% of {len(h)}")
    for theme in THEMES:
        ladder(table, hybrid, theme)
        colours(rates, theme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
