"""
docs/032's figures (light PNGs in docs/img/).

  python -m draftzero.build.figs builders   # win rate against the human decks at 100 simulations, per builder
  python -m draftzero.build.figs phaseA     # phase A per generation: heuristic bot vs search at 100 simulations

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


def colour_reference():
    """Each main-colour pair's game win rate in 17lands' sealed data (all builds, Bo1 and Bo3); was
    realism.colour_reference, which stays in v0.1.0."""
    import collections
    w, n = collections.Counter(), collections.Counter()
    for line in open(os.path.join(REPO, "data/sealed/sealed.jsonl")):
        p = json.loads(line)
        for b in p["builds"]:
            k = "".join(sorted(b["main_colors"] or "", key="WUBRG".index))
            w[k] += b["wins"]
            n[k] += b["games"]
    return {k: w[k] / n[k] for k in n if n[k] >= 500}


def colours(out):
    """Main colours by training depth, one row per colour pair (ordered by the pair's 17lands sealed win rate), with
    the humans' share and win rate. Shares on the 500 val pools each builder was scored on."""
    import collections
    from . import decks as D
    C = D.C
    pools = D.load_pools("sealed", "val")[:500]

    def mc(d):
        k = "".join(sorted(D.deck_colors(d).split("+")[0], key="WUBRG".index))
        return k if len(k) == 2 else ("3+" if len(k) >= 3 else "1")
    runs = [("heuristic bot", "A-gen2-s100-fixed", "#86b6ef"), ("25 simulations", "B-gen4-s100", "#3987e5"),
            ("100 simulations (gen 0)", "C-gen0-agree", "#1c5cab"), ("100 simulations (gen 1)", "C-gen1-agree", "#0d366b")]
    share = {}
    for lab, d, _ in runs:
        built = [C.vec(json.loads(l)) for l in open(os.path.join(EVAL, d, "built.jsonl"))][:500]
        share[lab] = collections.Counter(mc(x) for x in built)
    share["humans"] = collections.Counter(mc(p["ref_v"]) for p in pools)
    wr = colour_reference()
    names = {"W": "white", "U": "blue", "B": "black", "R": "red", "G": "green"}
    pairs = sorted([k for k in wr if len(k) == 2], key=lambda k: -wr[k])
    rows = pairs + ["3+"]
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(rows) + 1.3), facecolor=SURF)
    style(ax)
    n = 500
    for i, k in enumerate(rows):
        y = len(rows) - 1 - i
        xs = [100 * share[lab][k] / n for lab, _, _ in runs]
        ax.plot([min(xs), max(xs)], [y, y], color=GRID, lw=3, solid_capstyle="round", zorder=1)
        for (lab, _, col), x in zip(runs, xs):
            ax.plot(x, y, "o", ms=8, color=col, mec=SURF, mew=1.5, zorder=3)
        ax.plot(100 * share["humans"][k] / n, y, "D", ms=8, color=HUMAN, mec=SURF, mew=1.5, zorder=4)
        lbl = "-".join(names[c] for c in k) if k != "3+" else "three or more colours"
        ax.text(-1.5, y, lbl, ha="right", va="center", fontsize=9, color=INK)
        if k in wr:
            ax.text(61, y, f"{100 * wr[k]:.1f}%", ha="left", va="center", fontsize=9, color=INK2)
    ax.text(61, len(rows) - 0.35, "17lands\nwin rate", ha="left", va="bottom", fontsize=8, color=INK2)
    ax.set_yticks([])
    ax.set_xlim(-0.5, 60)
    ax.set_ylim(-0.7, len(rows) - 0.2)
    ax.set_xlabel("Share of decks with these main colours (%), 500 val pools", fontsize=9, color=INK2)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker="o", ls="", color=c, label=f"builder: {lab}") for lab, _, c in runs]
    hs.append(Line2D([], [], marker="D", ls="", color=HUMAN, label="humans"))
    ax.legend(handles=hs, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8,
              labelcolor=INK2)
    fig.subplots_adjust(left=0.22, right=0.86, top=0.82, bottom=0.12)
    fig.savefig(out, dpi=160, facecolor=SURF)
    print("wrote", out, file=sys.stderr)
    return share, wr, rows


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
    elif what == "colours":
        colours(os.path.join(IMG, "032-colours-light.png"))
    elif what == "phaseA":
        phase_a(os.path.join(IMG, "032-phaseA-light.png"))
