"""The games behind each agent (docs/017, figure 1): DraftZero's self-play and human data against
AlphaGo's human games and AlphaZero's chess self-play. Writes docs/img/017-games-{light,dark}.png.

    python tools/imitation_scale/fig_games.py
"""
from __future__ import annotations

from pathlib import Path

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", selfplay="#2a78d6", human="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", selfplay="#3987e5", human="#d95926"),
}

# (label, games, kind, value label); top to bottom
BARS = [
    ("DraftZero #2a: self-play, 18 generations", 2_673, "selfplay", "2.7k"),
    ("DraftZero #2b: human games it was pretrained on", 13_533, "human", "13.5k"),
    ("DraftZero #4: top players' games (train split)", 140_000, "human", "~140k"),
    ("All 17lands FDN Premier Draft games", 791_159, "human", "791k"),
    ("AlphaGo (2016): human games (KGS) for its policy", 160_000, "human", "160k"),
    ("AlphaZero (2018): chess self-play", 44_000_000, "selfplay", "44M"),
]


def main(out_dir: Path = Path("docs/img")) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    out_dir.mkdir(parents=True, exist_ok=True)
    for mode, t in THEMES.items():
        plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                             "font.size": 10})
        fig, ax = plt.subplots(figsize=(9, 4.3), dpi=150)
        fig.patch.set_facecolor(t["surface"])
        ax.set_facecolor(t["surface"])
        ys = list(range(len(BARS)))[::-1]
        for y, (label, n, kind, vlab) in zip(ys, BARS):
            ax.barh(y, n, left=1, height=0.56, color=t[kind], edgecolor=t["surface"], linewidth=2, zorder=3)
            ax.text(n * 1.18, y, vlab, va="center", ha="left", color=t["ink"], fontsize=10, zorder=4)
        ax.axhline(1.5, color=t["axis"], linewidth=1, zorder=2)
        ax.text(2.5e8, 1.5 + 0.08, "other projects, for scale", va="bottom", ha="right", color=t["muted"], fontsize=8.5)
        ax.set_xscale("log")
        ax.set_xlim(1_000, 3e8)
        ax.set_ylim(-0.6, len(BARS) - 0.4)
        ax.set_yticks(ys, [b[0] for b in BARS])
        ax.set_xticks([1e3, 1e4, 1e5, 1e6, 1e7, 1e8], ["1k", "10k", "100k", "1M", "10M", "100M"])
        ax.tick_params(axis="both", colors=t["ink2"], length=0, labelsize=9.5)
        for lab in ax.get_yticklabels():
            lab.set_color(t["ink"])
        ax.grid(axis="x", color=t["grid"], linewidth=0.8, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(t["axis"])
        ax.set_xlabel("Games (log scale)", color=t["ink2"])
        ax.legend(handles=[Patch(color=t["selfplay"], label="self-play games"),
                           Patch(color=t["human"], label="human games")],
                  loc="upper right", frameon=False, fontsize=9, labelcolor=t["ink2"])
        fig.text(0.015, 0.95, "How many games each agent learned from", color=t["ink"], fontsize=13,
                 fontweight="bold", ha="left")
        fig.text(0.015, 0.885, "The 17lands games are free. 140k games of #2a's self-play would take about "
                 "2,300 pod-hours (~$1,100).", color=t["ink2"], fontsize=10, ha="left")
        fig.subplots_adjust(left=0.40, right=0.97, top=0.82, bottom=0.14)
        p = out_dir / f"017-games-{mode}.png"
        fig.savefig(p, facecolor=t["surface"])
        plt.close(fig)
        print(p)


if __name__ == "__main__":
    main()
