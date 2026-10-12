"""
docs/032's figures (light PNGs in docs/img/).

  python -m sealed.figs builders   # win rate against the human decks at 100 simulations, per builder
  python -m sealed.figs phaseA     # phase A per generation: heuristic bot vs search at 100 simulations

Colours: the dataviz reference palette's first two categorical slots (validated: CVD dE 24.7, normal 33.6),
blue = learned only from self-play, orange = uses human data.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import cards as cardsmod  # noqa: E402

REPO = cardsmod.REPO
EVAL = os.path.join(REPO, "data/sealed/eval")
IMG = os.path.join(REPO, "docs/img")
SELF, HUMAN = "#2a78d6", "#eb6834"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def result(name):
    p = os.path.join(EVAL, name, "result.json")
    return json.load(open(p)) if os.path.exists(p) else None


def builders(rows, out):
    """rows: (label, eval dir, self_play?) in display order, top first."""
    pts = [(lab, result(d), sp) for lab, d, sp in rows]
    pts = [(lab, r["vs_human"], sp) for lab, r, sp in pts if r and r.get("vs_human")]
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(pts) + 1.1), facecolor=SURF)
    style(ax)
    for i, (lab, v, sp) in enumerate(pts):
        y = len(pts) - 1 - i
        c = SELF if sp else HUMAN
        w, se = 100 * v["winrate"], 100 * (v.get("se") or 0)
        ax.plot([w - se, w + se], [y, y], color=c, lw=2, solid_capstyle="round")
        ax.plot(w, y, "o", ms=8, color=c, mec=SURF, mew=2)
        ax.text(w + se + 0.4, y, f"{w:.1f}%", va="center", fontsize=9, color=INK)
    ax.axvline(50, color=INK2, lw=1, ls="--")
    ax.text(50.3, len(pts) - 1, "50%: as good as\nthe human decks", fontsize=8, color=INK2, ha="left", va="center")
    ax.set_yticks(range(len(pts)))
    ax.set_yticklabels([p[0] for p in reversed(pts)], fontsize=9, color=INK)
    ax.set_xlabel("Won against the human decks (%), 100 simulations", fontsize=9,
                  color=INK2)
    ax.set_xlim(min(45, min(100 * p[1]["winrate"] for p in pts) - 3), max(100 * p[1]["winrate"] for p in pts) + 5)
    ax.set_ylim(-0.6, len(pts) - 0.2)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=SELF, label="learned only from its own games"),
                       Line2D([], [], marker="o", ls="", color=HUMAN, label="uses human data")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURF)
    print("wrote", out, file=sys.stderr)


def phase_a(out):
    gens, bot, search = [], [], {}
    d = os.path.join(REPO, "data/sealed/A")
    for g in range(0, 40):
        m = os.path.join(d, f"gen{g}", "metrics.json")
        if not os.path.exists(m):
            continue
        j = json.load(open(m))
        gens.append(g)
        bot.append(100 * j["vs_human"]["winrate"])
        r = result(f"A-gen{g}-s100-fixed") or result(f"A-gen{g}-s100")
        if r and r.get("vs_human"):
            search[g] = 100 * r["vs_human"]["winrate"]
    fig, ax = plt.subplots(figsize=(7.5, 3.4), facecolor=SURF)
    style(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.plot(gens, bot, color=INK2, lw=2, marker="o", ms=6, mec=SURF, mew=1.5)
    ax.plot(list(search), list(search.values()), color=SELF, lw=2, marker="o", ms=8, mec=SURF, mew=1.5)
    ax.text(gens[-1] + 0.3, bot[-1], "played by gorge's\nheuristic bot", fontsize=8, color=INK2, va="center")
    ks = list(search)
    ax.text(ks[-1] + 0.3, search[ks[-1]], "played by the search\nat 100 simulations", fontsize=8, color=SELF,
            va="center")
    ax.axhline(50, color=INK2, lw=1, ls="--")
    ax.set_xlabel("Phase A generation (self-play games played by gorge's heuristic bot)", fontsize=9, color=INK2)
    ax.set_ylabel("Won against the human decks (%)", fontsize=9, color=INK2)
    ax.set_xlim(-0.5, gens[-1] + 3.5)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURF)
    print("wrote", out, file=sys.stderr)


if __name__ == "__main__":
    what = sys.argv[1]
    os.makedirs(IMG, exist_ok=True)
    if what == "builders":
        rows = json.loads(sys.argv[2]) if len(sys.argv) > 2 else [
            ["Self-play: phase A, generation 2", "A-gen2-s100-fixed", True],
            ["Self-play: phase A, generation 5", "A-gen5-s100", True],
            ["Self-play: phase A, generation 1", "A-gen1-s100", True],
            ["17lands heuristic (GIH win rate)", "gih-s100", False],
            ["Imitation of the human decks", "imit-s100", False]]
        builders(rows, os.path.join(IMG, "032-builders-light.png"))
    elif what == "phaseA":
        phase_a(os.path.join(IMG, "032-phaseA-light.png"))
