"""The figures of docs/019 (experiment #4's report). Light theme only: the doc uses plain markdown images, which
render everywhere (GitHub, Claude) where <picture> and mermaid don't.

    docs/img/019-pipeline-light.png  from a 17lands replay to training examples (docs/008 §4)
    docs/img/019-curves-light.png    validation curves of the two final networks by training decisions seen
    docs/img/019-ladder-light.png    win rate against heuristic@100 by search budget, the transformer and the MLP
    docs/img/019-cards-light.png     agreement with 17lands' card win rates, and 17lands' top commons by rank
    docs/img/019-colours-light.png   deck win rate by colour pair, self-play against 17lands' top players
    docs/img/019-games-light.png     the games behind each agent, this work against other game-playing systems

    python tools/imitation_scale/fig_doc019.py [--running 1000,3000]

The ladder pools a budget's runs (shards, tails, top-ups), one result per (deck pair, seat swap), and leaves out
games that ended in an engine error; a game capped at 50 turns counts 0.5. The card and colour-pair numbers are
docs/018's (C3 and the MLP's self-play), from tools/imitation_scale/gih.py and gih_ceiling.py.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

G = "runs/exp4/games"
OUT = Path("docs/img")
BUDGETS = ["0", "100", "300", "1000", "3000"]
LABELS = {"0": "policy alone\n(no search)", "100": "100", "300": "300", "1000": "1,000", "3000": "3,000"}
NETS = {
    "Transformer": {"0": [f"{G}/c1-pvh-t0-s*"], "100": [f"{G}/c2-ilbc100"], "300": [f"{G}/c2-ilbc300"],
                    "1000": [f"{G}/c2-ilbc1000"]},
    "MLP": {"0": [f"{G}/mlp-pvh-t0"], "100": [f"{G}/mlp-ilbc100"], "300": [f"{G}/mlp-ilbc300"],
            "1000": [f"{G}/mlp-ilbc1000-s*", f"{G}/mlp-ilbc1000-top"],
            "3000": [f"{G}/mlp-ilbc3000-s*", f"{G}/mlp-ilbc3000-top"]},
}
RUNS = {"Transformer": "runs/exp4/main/evals.jsonl", "MLP": "runs/exp4/mlp_1ep/evals.jsonl"}

# The reference palette's first slots (blue, aqua; orange for human data), validated as a set; muted grey for context.
T = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
         box="#f0efec", Transformer="#2a78d6", MLP="#1baf7a", selfplay="#2a78d6", human="#eb6834")

# (label, games, kind, value label), top to bottom. Sources: docs/014 (#2a), docs/018 (this work), 17lands,
# Silver et al. 2016 (AlphaGo's KGS games), 2017 (AlphaGo Zero, 3-day run), 2018 (AlphaZero), Perolat et al. 2022.
GAMES = [
    ("DraftZero #2a: training run, from scratch", 2_673, "selfplay", "2.7k"),
    ("This work: top players' games (training split)", 145_903, "human", "146k"),
    ("All 17lands FDN Premier Draft games", 791_159, "human", "791k"),
    ("AlphaGo (2016): human games (KGS)", 160_000, "human", "160k"),
    ("AlphaGo Zero (2017), 3-day run: self-play", 4_900_000, "selfplay", "4.9M"),
    ("AlphaZero (2018): chess self-play", 44_000_000, "selfplay", "44M"),
    ("DeepNash (2022): Stratego self-play", 5_500_000_000, "selfplay", "5.5B"),
]

# Spearman with 17lands' GIH WR (docs/018, C3 and the MLP's self-play), and the noise ceiling at ~17,700 player-games
# (17lands' own games subsampled; median and 5-95%).
CORR = {"Commons (90 cards)": {"Transformer, greedy": 0.41, "MLP, greedy": 0.34, "Transformer, sampled": 0.27,
                               "ceiling": (0.82, 0.74, 0.86)},
        "Every non-basic card (267)": {"Transformer, greedy": 0.43, "MLP, greedy": 0.40, "Transformer, sampled": 0.45,
                                       "ceiling": (0.79, 0.74, 0.82)}}
# 17lands' top 15 commons by GIH WR, and their rank among 90 commons in the transformer's greedy self-play (docs/018).
TOP15 = [("Bake into a Pie", 44), ("Burst Lightning", 83), ("Stab", 43), ("Dazzling Angel", 1),
         ("Luminous Rebuke", 10), ("Refute", 21), ("Healer's Hawk", 5), ("Helpful Hunter", 15),
         ("Banishing Light", 3), ("Felidar Savior", 6), ("Bigfin Bouncer", 11), ("Llanowar Elves", 37),
         ("Infestation Sage", 55), ("Gorehorn Raider", 74), ("Think Twice", 36)]
# Deck win rate by main colours: the transformer's greedy self-play, and 17lands' top players (docs/018, C1 and C3).
COLOURS = [("WU", 0.59, 0.656), ("WG", 0.54, 0.646), ("WB", 0.53, 0.634), ("RG", 0.51, 0.653), ("WR", 0.50, 0.650),
           ("UB", 0.49, 0.640), ("UG", 0.48, 0.622), ("BG", 0.48, 0.612), ("UR", 0.45, 0.638), ("BR", 0.40, 0.639)]


def style(ax, grid="y"):
    ax.set_facecolor(T["surface"])
    if grid:
        ax.grid(color=T["grid"], lw=0.8, axis=grid)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(T["axis"])
    ax.tick_params(colors=T["ink2"], labelsize=8.5, length=0)


def save(fig, name):
    p = OUT / f"{name}-light.png"
    fig.savefig(p, dpi=150, facecolor=T["surface"])
    plt.close(fig)
    print(p)


def pipeline():
    fig, ax = plt.subplots(figsize=(12.5, 3.3), facecolor=T["surface"])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 32)
    ax.axis("off")
    w, h, y0 = 15.5, 15, 12
    xs = [0.5, 21, 41.5, 62, 82.5]
    texts = ["17lands replay\n\none game from one player's\nview: each turn's plays\nand end-of-turn snapshots",
             "Rebuild the turn's start\n\nthe last snapshot + the draw;\ninfer tapped lands,\ncounters, attachments",
             "Load it into XMage\n\nplace every card in its\nzone, then resume the game",
             "Replay the turn\n\nboth players' recorded plays;\ntry up to 12 orderings\nof what isn't recorded",
             "Training examples\n\nthe position, the legal\nmoves, and the move\nthe player made"]
    for i, (x, t) in enumerate(zip(xs, texts)):
        last = i == len(xs) - 1
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=T["human"] if last else T["box"], ec=T["axis"] if not last else T["human"],
                                    lw=1, alpha=0.18 if last else 1.0))
        head, _, body = t.partition("\n\n")
        ax.text(x + w / 2, y0 + h - 2.2, head, ha="center", va="top", fontsize=9.5, color=T["ink"], weight="bold")
        ax.text(x + w / 2, y0 + h - 5.4, body, ha="center", va="top", fontsize=8.3, color=T["ink2"], linespacing=1.35)
        if not last:
            ax.add_patch(FancyArrowPatch((x + w + 0.6, y0 + h / 2), (xs[i + 1] - 0.6, y0 + h / 2),
                                         arrowstyle="-|>", mutation_scale=12, color=T["muted"], lw=1.3))
    ax.text((xs[3] + w + xs[4]) / 2, y0 + h / 2 + 1.2, "matches\n(~85%)", ha="center", va="bottom",
            fontsize=7.6, color=T["ink2"])
    bx = xs[3] + w / 2
    ax.add_patch(FancyArrowPatch((bx, y0 - 0.5), (bx, 6.2), arrowstyle="-|>", mutation_scale=12, color=T["muted"],
                                 lw=1.3))
    ax.text(bx + 1, 8.8, "doesn't match", ha="left", va="center", fontsize=7.6, color=T["ink2"])
    ax.add_patch(FancyBboxPatch((bx - 6, 1.2), 12, 4.6, boxstyle="round,pad=0.3,rounding_size=1", fc=T["surface"],
                                ec=T["axis"], lw=1, ls=(0, (3, 2))))
    ax.text(bx, 3.5, "turn dropped", ha="center", va="center", fontsize=8.3, color=T["ink2"])
    fig.tight_layout(pad=0.4)
    save(fig, "019-pipeline")


def games(patterns):
    out = {}
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
                    continue
                pts = 1.0 if g.get("winner_role") == "bot1" else 0.5 if g.get("winner_role") is None else 0.0
                out[(g["pair"], g["swap"])] = pts
    return out


def ladder(running):
    table = {}
    for net, spec in NETS.items():
        for b, pats in spec.items():
            g = games(pats)
            if g:
                table[(net, b)] = (sum(g.values()) / len(g), len(g))
                print(f"{net:12s} {b:>5s}: won {100 * table[(net, b)][0]:.1f}% of {len(g)} paired games")
    xs = {b: i for i, b in enumerate(BUDGETS)}
    fig, ax = plt.subplots(figsize=(8.6, 4.6), facecolor=T["surface"])
    for net in NETS:
        pts = [(b, table[(net, b)]) for b in BUDGETS if (net, b) in table]
        ax.plot([xs[b] for b, _ in pts], [100 * v[0] for _, v in pts], color=T[net], lw=2.2, zorder=2)
        for b, (sc, n) in pts:
            hollow = net == "MLP" and b in running
            ax.plot(xs[b], 100 * sc, "o", ms=8.5, color=T["surface"] if hollow else T[net], markeredgecolor=T[net],
                    markeredgewidth=2, zorder=4)
    for b in BUDGETS:   # each budget's two labels: the higher score's above its point, the lower's below
        here = sorted([(table[(n, b)], n) for n in NETS if (n, b) in table], key=lambda v: v[0][0])
        for k, ((sc, n), net) in enumerate(here):
            above = k == len(here) - 1
            lab = f"{100 * sc:.0f}%\n{n:,} games" + (" so far" if net == "MLP" and b in running else "")
            # the lower label sits below and to the right, clear of both lines and of the 50% line
            first = b == BUDGETS[0]   # the line rises to the right of the first point: label it to the left
            ax.annotate(lab, (xs[b], 100 * sc), textcoords="offset points",
                        xytext=((-6, 6) if first else (0, 9)) if above else (9, -5),
                        ha=("right" if first else "center") if above else "left", va="bottom" if above else "top", fontsize=7.8,
                        color=T["ink2"], linespacing=1.15)
    ax.axhline(50, color=T["muted"], lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(len(BUDGETS) - 0.45, 49, "even with the baseline", fontsize=8, color=T["muted"], ha="right", va="top")
    ax.set_ylim(25, 80)
    ax.set_yticks(range(30, 81, 10), [f"{v}%" for v in range(30, 81, 10)])
    ax.set_ylabel("Games won against the baseline", fontsize=9, color=T["ink2"])
    ax.set_xticks(range(len(BUDGETS)), [LABELS[b] for b in BUDGETS], fontsize=8.5, color=T["ink2"])
    ax.set_xlim(-0.5, len(BUDGETS) - 0.4)
    ax.set_xlabel("Search: simulations per decision", fontsize=9, color=T["ink2"])
    style(ax)
    ax.legend(handles=[Patch(color=T[n], label=n) for n in NETS], loc="upper left", frameon=False, fontsize=9,
              labelcolor=T["ink2"])
    fig.tight_layout()
    save(fig, "019-ladder")


def curves():
    data = {net: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()] for net, p in RUNS.items()}
    panels = [("policy/set_nll", "Policy: loss (lower is better)"),
              ("policy/top1_nonpass", "Policy: picks the player's move"),
              ("replay_attack/acc", "Attacks: accuracy"),
              ("value/auc", "Value: AUC predicting the winner")]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6), facecolor=T["surface"])
    for ax, (key, title) in zip(axs, panels):
        for net, rows in data.items():
            rows = [r for r in rows if r["seen"] > 0]
            # the end-of-run evaluation can fall a few steps after the last scheduled one: keep the last only
            rows = [r for i, r in enumerate(rows) if i + 1 == len(rows) or rows[i + 1]["seen"] - r["seen"] > 2e5]
            x = [r["seen"] / 1e6 for r in rows]
            y = [r[key] for r in rows]
            ax.plot(x, y, "-o", color=T[net], lw=2, ms=3.5, markeredgewidth=0)
            if net == "MLP":
                up = key != "policy/set_nll"
                ax.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), textcoords="offset points", xytext=(0, 6 if up else -7),
                            fontsize=8, color=T["ink2"], ha="center", va="bottom" if up else "top")
            else:
                ax.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), textcoords="offset points", xytext=(5, 0),
                            fontsize=8, color=T["ink2"], ha="left", va="center")
        for e in (10.95, 21.9):
            ax.axvline(e, color=T["axis"], lw=0.8, ls=(0, (3, 3)), zorder=0)
        style(ax)
        ax.set_title(title, loc="left", fontsize=9.5, color=T["ink"])
        ax.set_xlabel("Training examples seen (millions)", fontsize=8.5, color=T["ink2"])
        ax.set_xlim(0, 40)
        ax.margins(y=0.14)
        ax.set_xticks([0, 10, 20, 30])
    axs[0].text(10.95, axs[0].get_ylim()[1], " 1 epoch", fontsize=7.5, color=T["muted"], va="top")
    axs[0].text(21.9, axs[0].get_ylim()[1], " 2 epochs", fontsize=7.5, color=T["muted"], va="top")
    axs[0].legend(handles=[Patch(color=T[n], label=n) for n in RUNS], loc="center right", frameon=False,
                  fontsize=8.5, labelcolor=T["ink2"])
    fig.tight_layout()
    save(fig, "019-curves")


def cards():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.9), facecolor=T["surface"],
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    series = ["Transformer, greedy", "MLP, greedy", "Transformer, sampled"]
    colours = [T["Transformer"], T["MLP"], "#9cc0ec"]
    bw = 0.22
    for gi, (group, vals) in enumerate(CORR.items()):
        c, lo, hi = vals["ceiling"]
        ax.add_patch(plt.Rectangle((gi - 0.45, lo), 0.9, hi - lo, color=T["grid"], zorder=0, lw=0))
        ax.plot([gi - 0.45, gi + 0.45], [c, c], color=T["muted"], lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.text(gi, hi + 0.015, f"noise ceiling {c:.2f} (5–95%: {lo:.2f}–{hi:.2f})", ha="center", va="bottom",
                fontsize=7.8, color=T["muted"])
        for si, (s, col) in enumerate(zip(series, colours)):
            x = gi + (si - 1) * (bw + 0.03)
            ax.bar(x, vals[s], width=bw, color=col, edgecolor=T["surface"], lw=1.5, zorder=2)
            ax.text(x, vals[s] + 0.012, f"{vals[s]:.2f}", ha="center", va="bottom", fontsize=8, color=T["ink2"])
    ax.set_xticks(range(len(CORR)), list(CORR), fontsize=9, color=T["ink"])
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.55, len(CORR) - 0.45)
    ax.set_ylabel("Rank correlation with 17lands (Spearman)", fontsize=9, color=T["ink2"])
    ax.set_title("How well self-play ranks the cards, against what noise allows", loc="left", fontsize=10,
                 color=T["ink"])
    style(ax)
    ax.legend(handles=[Patch(color=c, label=s.replace("\n", " ")) for s, c in zip(series, colours)],
              loc="upper left", frameon=False, fontsize=8, labelcolor=T["ink2"], ncol=3, bbox_to_anchor=(0, 1.0))
    # slope chart: 17lands' rank -> self-play rank
    # the left column spreads 17lands' ranks 1-15 over the height; the right is the self-play rank, 1-90
    right = []
    for i, (name, r) in enumerate(TOP15):
        y0, y1 = (i + 1) * 6, r
        drop = r - (i + 1) >= 25
        col = T["human"] if drop else T["axis"]
        bx.plot([0, 1], [y0, y1], color=col, lw=2 if drop else 1.2, zorder=2 if drop else 1)
        bx.plot([0, 1], [y0, y1], "o", ms=4, color=col, zorder=3)
        bx.text(-0.04, y0, f"{i + 1}. {name}", ha="right", va="center", fontsize=7.8,
                color=T["ink"] if drop else T["ink2"])
        if drop:
            right.append((r, name))
    last = -10
    for r, name in sorted(right):   # nudge labels apart where ranks are close
        y = max(r, last + 4)
        bx.text(1.04, y, f"{r}. {name}", ha="left", va="center", fontsize=7.8, color=T["ink"])
        last = y
    bx.text(1.04, 6, "1–15: the rest\n(Dazzling Angel 1st)", ha="left", va="center", fontsize=7.5, color=T["ink2"])
    bx.set_ylim(93, 0)
    bx.set_xlim(-0.05, 1.6)
    bx.set_xticks([0, 1], ["rank on 17lands", "rank in self-play\n(of 90 commons)"], fontsize=8.5, color=T["ink2"])
    bx.set_yticks([])
    for sp in ("top", "right", "left", "bottom"):
        bx.spines[sp].set_visible(False)
    bx.tick_params(length=0)
    bx.set_facecolor(T["surface"])
    bx.set_title("17lands' 15 best commons: where self-play ranks them", loc="left", fontsize=10, color=T["ink"],
                 x=-0.45)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.45)
    save(fig, "019-cards")


def colour_pairs():
    sim_mean = sum(s for _, s, _ in COLOURS) / len(COLOURS)
    hum_mean = sum(h for _, _, h in COLOURS) / len(COLOURS)
    fig, ax = plt.subplots(figsize=(7.6, 3.9), facecolor=T["surface"])
    for i, (pair, s, h) in enumerate(COLOURS):
        ds, dh = 100 * (s - sim_mean), 100 * (h - hum_mean)
        ax.plot([i, i], [dh, ds], color=T["grid"], lw=2, zorder=1)
        ax.plot(i, dh, "o", ms=8, color=T["human"], zorder=3, markeredgecolor=T["surface"], markeredgewidth=1.5)
        ax.plot(i, ds, "o", ms=8, color=T["Transformer"], zorder=3, markeredgecolor=T["surface"],
                markeredgewidth=1.5)
    ax.axhline(0, color=T["muted"], lw=1, ls=(0, (4, 3)), zorder=0)
    ax.set_xticks(range(len(COLOURS)), [p for p, _, _ in COLOURS], fontsize=9, color=T["ink"])
    ax.set_yticks(range(-10, 11, 5), [f"{v:+d}" if v else "0" for v in range(-10, 11, 5)])
    ax.set_ylabel("Deck win rate, points above\nor below the average", fontsize=9, color=T["ink2"])
    ax.set_xlabel("Deck colours", fontsize=9, color=T["ink2"])
    style(ax)
    ax.legend(handles=[Patch(color=T["Transformer"], label="the transformer's self-play"),
                       Patch(color=T["human"], label="17lands' top players")], loc="lower left", frameon=False,
              fontsize=8.5, labelcolor=T["ink2"])
    fig.tight_layout()
    save(fig, "019-colours")


def games_figure():
    fig, ax = plt.subplots(figsize=(9.5, 4.3), facecolor=T["surface"])
    ys = list(range(len(GAMES)))[::-1]
    for y, (label, n, kind, vlab) in zip(ys, GAMES):
        ax.barh(y, n, left=1, height=0.56, color=T[kind], edgecolor=T["surface"], linewidth=2, zorder=3)
        ax.text(n * 1.25, y, vlab, va="center", ha="left", color=T["ink"], fontsize=9.5, zorder=4)
    ax.axhline(3.5, color=T["axis"], lw=1, zorder=2)
    ax.text(5e10, 3.5 + 0.1, "other game-playing systems, for scale", va="bottom", ha="right", color=T["muted"],
            fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(1_000, 6e10)
    ax.set_ylim(-0.6, len(GAMES) - 0.4)
    ax.set_yticks(ys, [g[0] for g in GAMES])
    ax.set_xticks([1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10], ["1k", "10k", "100k", "1M", "10M", "100M", "1B", "10B"])
    style(ax, grid="x")
    for lab in ax.get_yticklabels():
        lab.set_color(T["ink"])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Games learned from (log scale)", fontsize=9, color=T["ink2"])
    ax.legend(handles=[Patch(color=T["selfplay"], label="self-play games"), Patch(color=T["human"], label="human games")],
              loc="upper right", frameon=False, fontsize=9, labelcolor=T["ink2"])
    fig.tight_layout()
    save(fig, "019-games")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--running", default="", help="the MLP's budgets still being played (drawn hollow), e.g. 1000,3000")
    a = ap.parse_args(argv)
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 9})
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline()
    curves()
    ladder(set(a.running.split(",")) - {""})
    cards()
    colour_pairs()
    games_figure()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
