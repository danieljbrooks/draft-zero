"""docs/025's figures, light PNGs into docs/img/: the search ladder (win rate against heuristic search at 100 simulations,
by simulations a decision) and games per dollar against XMage.

    python3 mtgkernel/py/fig_doc025.py --runs mtgkernel/runs --spec mtgkernel/py/fig_doc025_spec.json

Needs matplotlib. The spec lists, for each line of the ladder, the games.jsonl files (relative to --runs) of each
search budget, and the bars of the games-per-dollar chart.

The ladder pools every run of the same matchup (games.jsonl files), counting wins 1, draws and truncated games 0.5,
errors excluded, as dzk's summary does.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
IMG = HERE.parents[1] / "docs" / "img"
# dataviz reference palette, light mode
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]


def score(paths):
    """bot1's score and valid games over the games.jsonl files."""
    s = n = 0
    for p in paths:
        for line in open(p):
            if not line.strip():
                continue
            g = json.loads(line)
            if g.get("error"):
                continue
            n += 1
            w = g.get("winner_role")
            s += 1 if w == "bot1" else 0 if w == "bot2" else 0.5
    return s / n, n


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def ladder(runs, series, out):
    """series: [(label, {budget: [games.jsonl, ...]})]"""
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    drawn = []
    for i, (label, pts) in enumerate(series):
        xs, ys, ns = [], [], []
        for b in sorted(pts):
            paths = [runs / p for p in pts[b]]
            if not all(p.exists() for p in paths):
                continue
            y, n = score(paths)
            xs.append(b)
            ys.append(100 * y)
            ns.append(n)
        if xs:
            drawn.append((i, label, xs, ys))
    top = max(range(len(drawn)), key=lambda j: drawn[j][3][-1]) if drawn else 0
    for j, (i, label, xs, ys) in enumerate(drawn):
        ax.plot(xs, ys, color=SERIES[i], linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=2, label=label, zorder=3)
        # value labels above the top series' points and below the others', so close lines don't collide
        dy = 9 if j == top or i == 0 else -14
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.0f}%", (x, y), xytext=(0, dy), textcoords="offset points", ha="center", fontsize=8,
                        color=INK2)
    # end labels, nudged apart vertically
    ends = sorted(((ys[-1], xs[-1], label) for _, label, xs, ys in drawn))
    placed = []
    for y, x, label in ends:
        yl = max([y] + [p + 3.2 for p in placed])
        placed.append(yl)
        ax.annotate(label, (x, y), xytext=(10, (yl - y) * 4.2), textcoords="offset points", va="center",
                    fontsize=8.5, color=INK)
    ax.axhline(50, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.set_xscale("log")
    ax.set_xticks([100, 300, 1000, 3000, 10000])
    ax.set_xticklabels(["100", "300", "1,000", "3,000", "10,000"])
    ax.set_xlim(70, 30000)
    ax.set_ylim(40, 90)
    ax.set_xlabel("simulations per decision", color=INK2, fontsize=9)
    ax.set_ylabel("won against heuristic search at 100 (%)", color=INK2, fontsize=9)
    ax.set_title("Paired games on the 62 evaluation deck pairs", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    ax.set_xlim(70, 60000)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)


def per_dollar(rows, out):
    """rows: [(label, games per dollar, highlight)] -> horizontal bars, log scale."""
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(rows) + 1.0), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ys = list(range(len(rows)))[::-1]
    for y, (label, v, hi) in zip(ys, rows):
        ax.barh(y, v, height=0.6, color=SERIES[0] if hi else "#9fbfe8", zorder=3)
        ax.annotate(f"{v:,.0f}", (v, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
                    color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(5, max(r[1] for r in rows) * 40)
    ax.set_xlabel("self-play games per dollar (log scale)", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, default=HERE.parent / "runs")
    ap.add_argument("--spec", type=Path, required=True, help="JSON: {ladder: [[label, {budget: [paths]}]], dollars: [[label, value, highlight]]}")
    a = ap.parse_args()
    spec = json.loads(a.spec.read_text())
    IMG.mkdir(parents=True, exist_ok=True)
    ladder(a.runs, [(l, {int(k): v for k, v in pts.items()}) for l, pts in spec["ladder"]], IMG / "025-ladder-light.png")
    per_dollar([tuple(r) for r in spec["dollars"]], IMG / "025-games-per-dollar-light.png")


if __name__ == "__main__":
    main()
