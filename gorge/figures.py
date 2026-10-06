"""Draw docs/025's figures from the game records under data/gorge/runs/.

    python gorge/figures.py

Writes docs/img/025-ladder-{light,dark}.png (score against gorge's bot, with 95% intervals over deck pairs) and
docs/img/025-gih-{light,dark}.png (games-in-hand win rates in self-play against 17lands'). Colours are slots of the
validated default categorical palette (the same as docs/015's plot), in both modes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import logging

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # Helvetica falls back to DejaVu Sans
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gorge"))
import analyze  # noqa: E402

RUNS = REPO / "data" / "gorge" / "runs"
OUT = REPO / "docs" / "img"

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", search="#2a78d6", net="#eb6834", other="#1baf7a", faint="#c3c2b7"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", search="#3987e5", net="#d95926", other="#199e70", faint="#52514e"),
}

# (label, games file, which side is "ours", colour role). Missing files are skipped.
LADDER = [
    ("az@100 (no network)", "bench/az100_v_bot.jsonl", "A", "search"),
    ("az@25 + gen-1 network", "eval/gen1/full_v_bot.jsonl", "A", "net"),
    ("az@25 (no network)", "bench/az25_v_bot.jsonl", "A", "search"),
    ("az@10 (no network)", "bench/az10_v_bot.jsonl", "A", "search"),
    ("gen-1 network alone", "eval/gen1/prior_v_bot.jsonl", "A", "net"),
    ("random", "bench/bot_v_random.jsonl", "B", "other"),
]


def done(p: Path) -> bool:
    """A run is drawn only once it has finished (dzgorge writes the summary last)."""
    return Path(str(p) + ".summary.json").exists()


def style(t):
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.edgecolor": t["axis"],
        "xtick.color": t["muted"], "xtick.labelcolor": t["ink2"],
        "ytick.color": t["muted"], "ytick.labelcolor": t["ink2"],
    })


def ladder(mode: str) -> Path | None:
    t = THEMES[mode]
    style(t)
    rows = []
    for label, path, side, role in LADDER:
        p = RUNS / path
        if not done(p):
            continue
        r = analyze.score(analyze.load([p]))
        s, lo, hi = r["score"], r["lo"], r["hi"]
        if side == "B":
            s, lo, hi = 1 - s, 1 - hi, 1 - lo
        rows.append((label, s, lo, hi, r["games"], role))
    if not rows:
        return None
    h = 0.62 * len(rows) + 2.1  # figure height, inches
    fig, ax = plt.subplots(figsize=(9, h), dpi=160)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    fig.subplots_adjust(left=0.25, right=0.97, top=1 - 1.25 / h, bottom=0.62 / h)
    for i, (label, s, lo, hi, n, role) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([100 * lo, 100 * hi], [y, y], color=t[role], linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter([100 * s], [y], s=64, color=t[role], edgecolors=t["surface"], linewidths=2, zorder=3)
        ax.text(100 * hi + 1.2, y, f"{100 * s:.1f}%", color=t["ink"], fontsize=10, va="center")
        ax.text(-0.02, y + 0.12, label, transform=ax.get_yaxis_transform(), color=t["ink"], fontsize=10.5,
                ha="right", va="center")
        ax.text(-0.02, y - 0.2, f"{n:,} games", transform=ax.get_yaxis_transform(), color=t["muted"], fontsize=8.5,
                ha="right", va="center")
    ax.axvline(50, color=t["ink2"], linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks(range(0, 101, 10))
    ax.set_xticklabels([f"{x}%" for x in range(0, 101, 10)])
    ax.tick_params(axis="y", length=0, labelleft=False)
    ax.grid(True, axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Score against gorge's bot (paired games on the eval decks, 95% intervals over deck pairs)",
                  color=t["ink2"])
    fig.text(0.02, 1 - 0.22 / h, "Score against gorge's bot", color=t["ink"], fontsize=13, fontweight="bold",
             ha="left", va="top")
    handles = [Line2D([], [], color=t[r], marker="o", linewidth=2, markersize=7, label=l)
               for r, l in (("search", "gorge's search, heuristic leaf"), ("net", "with the trained network"),
                            ("other", "random"))
               if any(row[5] == r for row in rows)]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.015, 1 - 0.5 / h), ncol=len(handles),
                     frameon=False, fontsize=9.5, handlelength=2.2, columnspacing=2.0)
    for text in leg.get_texts():
        text.set_color(t["ink2"])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"025-ladder-{mode}.png"
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)
    return path


GIH_PANELS = [
    ("random against itself", "gih/random.jsonl"),
    ("gorge's bot against itself", "gih/bot.jsonl"),
    ("search (az@10) against itself", "gih/az10.jsonl"),
]


def gih(mode: str) -> Path | None:
    t = THEMES[mode]
    style(t)
    panels = [(title, RUNS / p) for title, p in GIH_PANELS if done(RUNS / p)]
    if not panels:
        return None
    rar = analyze.rarity()
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels) + 0.8, 5.6), dpi=160, squeeze=False,
                             sharex=True, sharey=True)
    fig.patch.set_facecolor(t["surface"])
    fig.subplots_adjust(left=0.07, right=0.985, top=0.76, bottom=0.11, wspace=0.08)
    fig.canvas.draw()
    for ax, (title, p) in zip(axes[0], panels):
        r = analyze.gih_report(analyze.load([p]), 30)
        ax.set_facecolor(t["surface"])
        rows = r["all"]["rows"]
        common = [x for x in rows if rar.get(x[0]) == "common"]
        other = [x for x in rows if rar.get(x[0]) != "common"]
        ax.scatter([100 * x[3] for x in other], [100 * x[2] for x in other], s=14, color=t["faint"],
                   edgecolors="none", label="uncommons, rares and mythics", zorder=2)
        ax.scatter([100 * x[3] for x in common], [100 * x[2] for x in common], s=22, color=t["search"],
                   edgecolors=t["surface"], linewidths=0.8, label="commons", zorder=3)
        ax.set_title(f"{title}\n{r['player_games']:,} player-games · Spearman {r['commons']['spearman']:.2f} commons, "
                     f"{r['all']['spearman']:.2f} all", color=t["ink"], fontsize=10, loc="left")
        ax.set_xlabel("17lands GIH win rate (%)", color=t["ink2"])
        if ax is axes[0][0]:
            ax.set_ylabel("Simulated GIH win rate (%)", color=t["ink2"])
        ax.grid(True, color=t["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax0 = axes[0][0]
    ax0.set_xlim(33, 70)
    ax0.set_ylim(15, 75)
    fig.canvas.draw()
    # Label 17lands' three best commons in every panel, so the eye can follow them across policies. A label that
    # would collide with one already placed is dropped.
    ref = json.loads(analyze.REFERENCE.read_text())["cards"]
    best = sorted((c for c in ref if rar.get(c) == "common"), key=lambda c: -ref[c]["gih_wr"])[:3]
    for ax, (title, p) in zip(axes[0], panels):
        r = analyze.gih_report(analyze.load([p]), 30)
        pts = sorted([(100 * s_, 100 * h, c) for c, _n, s_, h in r["all"]["rows"] if c in best])
        # Labels sit right of the cloud, at least 4 points apart, each joined to its card by a thin line.
        label_y = []
        for y, _x, _c in pts:
            label_y.append(max(y, label_y[-1] + 4) if label_y else y)
        for (y, x, c), ly in zip(pts, label_y):
            ax.scatter([x], [y], s=46, facecolors="none", edgecolors=t["ink"], linewidths=1.2, zorder=4)
            ax.annotate(c, (x, y), xytext=(61.0, ly), textcoords="data", fontsize=8, color=t["ink"],
                        ha="left", va="center",
                        arrowprops=dict(arrowstyle="-", color=t["ink2"], linewidth=0.7, shrinkA=0, shrinkB=4))
    fig.text(0.02, 0.965, "Card win rates when drawn: gorge self-play against 17lands", color=t["ink"], fontsize=13,
             fontweight="bold", ha="left", va="top")
    fig.text(0.02, 0.915, "Each dot is a card with 30+ games in hand. Ringed: 17lands' three best commons (Bake into a "
             "Pie, Burst Lightning, Stab).", color=t["ink2"], fontsize=9.5, ha="left", va="top")
    handles, labels = axes[0][0].get_legend_handles_labels()
    leg = fig.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(0.015, 0.89), ncol=2,
                     frameon=False, fontsize=9, handletextpad=0.3, columnspacing=1.6)
    for text in leg.get_texts():
        text.set_color(t["ink2"])
    path = OUT / f"025-gih-{mode}.png"
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)
    return path


if __name__ == "__main__":
    for mode in THEMES:
        for fn in (ladder, gih):
            p = fn(mode)
            if p:
                print(p)
