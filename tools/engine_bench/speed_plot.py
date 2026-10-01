"""Draw docs/015's plot: each engine's speed relative to XMage, on one laptop core.

The numbers are the medians measured for docs/015 (M1 Pro, one engine at a time, one core each):
XMage, Forge, gorge and mtg-kernel on 2026-09-28, ManaBrew on 2026-09-30. They are copied here,
not recomputed. Run from the repo root:

    python tools/engine_bench/speed_plot.py

Writes docs/img/015-engine-speed-light.png and docs/img/015-engine-speed-dark.png.
One log axis, one baseline: XMage = 1x. Bars grow from it, right for faster and left for slower.
Colours are the first three slots of the validated default categorical palette, in both modes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402
from matplotlib.transforms import IdentityTransform  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "docs" / "img"

THEMES = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7",
        series={"play": "#2a78d6", "copy": "#eb6834", "step": "#1baf7a"},
    ),
    "dark": dict(
        surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835",
        series={"play": "#3987e5", "copy": "#d95926", "step": "#199e70"},
    ),
}

# XMage (MageZero v0.2.0 bundle): 54.67 turns/s of random play on the Burn mirror; a median
# Game.copy() of 164.5 us on a mid-game FDN state (pair A); and 1.24 ms per MageZero MCTS
# simulation at a turn-5 decision of pair A (its search step).
XMAGE = {"play": 54.67, "copy_us": 164.5, "step_us": 1240.0}
# (engine, sub-label, random-play turns/s on the Burn mirror, median copy in us,
#  one search step in us: copy + one action + advance to the next decision)
ENGINES = [
    ("mtg-kernel", "Rust · FDN in progress", 25560.5, 6.625, 17.2),
    ("gorge", "Go · FDN 282 of 286 cards", 2703.0, 14.42, 46.13),
    ("ManaBrew", "Rust port of Forge · FDN 286 of 286 cards", 1240.27, 2.125, 520.1),
    ("Forge", "Java · FDN 286 of 286 cards", 30.8475, 2310.54, 5525.7),
]
SERIES = ("play", "copy", "step")
LABELS = {
    "play": "Random play, turns per second (Burn mirror)",
    "copy": "Copying a mid-game state",
    "step": "One search step",
}

BAR_PX = 22      # bar thickness in output pixels (dpi 160)
GAP_PX = 4       # surface gap between a row's bars
RADIUS_PX = 5    # rounded data end; the baseline end stays square


def ratio_text(r: float) -> str:
    return f"{r:,.0f}×" if r >= 10 else f"{r:.2g}×"


def draw_bar(fig, ax, row: float, offset_px: float, value: float, colour: str, t) -> None:
    """A bar from the 1x baseline to value, in display pixels so the corners stay round."""
    x_base, y_row = ax.transData.transform((1.0, row))
    x_end, _ = ax.transData.transform((value, row))
    y0 = y_row + offset_px - BAR_PX / 2
    left, right = min(x_base, x_end), max(x_base, x_end)
    rounded = FancyBboxPatch((left, y0), right - left, BAR_PX,
                             boxstyle=f"round,pad=0,rounding_size={RADIUS_PX}",
                             transform=IdentityTransform(), facecolor=colour, edgecolor="none")
    # Square off the baseline end by covering it with a plain rectangle.
    sq_left = left if x_end >= x_base else left + RADIUS_PX
    square = Rectangle((sq_left, y0), right - left - RADIUS_PX, BAR_PX,
                       transform=IdentityTransform(), facecolor=colour, edgecolor="none")
    fig.add_artist(rounded)
    fig.add_artist(square)
    pad = 7
    if x_end >= x_base:
        fig.text(x_end + pad, y0 + BAR_PX / 2, ratio_text(value), transform=IdentityTransform(),
                 color=t["ink"], fontsize=10, va="center", ha="left")
    else:
        fig.text(x_end - pad, y0 + BAR_PX / 2, ratio_text(value), transform=IdentityTransform(),
                 color=t["ink"], fontsize=10, va="center", ha="right")


def draw(mode: str) -> Path:
    t = THEMES[mode]
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.edgecolor": t["axis"],
        "xtick.color": t["muted"],
        "xtick.labelcolor": t["ink2"],
    })
    fig, ax = plt.subplots(figsize=(10, 6.4), dpi=160)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    fig.subplots_adjust(left=0.27, right=0.95, top=0.79, bottom=0.17)

    ax.set_xscale("log")
    ax.set_xlim(0.012, 3000)
    ax.set_ylim(-0.6, len(ENGINES) - 0.4)
    ax.set_xticks([0.1, 1, 10, 100, 1000])
    ax.set_xticklabels(["0.1×", "1×", "10×", "100×", "1,000×"])
    ax.tick_params(axis="x", which="minor", length=0)
    ax.tick_params(axis="y", length=0, labelleft=False)
    ax.grid(True, axis="x", which="major", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.set_xlabel("Speed relative to XMage (log scale; right is faster)", color=t["ink2"],
                  labelpad=6)
    ax.axvline(1.0, color=t["ink2"], linewidth=1.2, zorder=1)
    ax.text(1.0, len(ENGINES) - 0.4, "XMage = 1×", color=t["ink2"], fontsize=9, ha="center",
            va="bottom")

    rows = {name: len(ENGINES) - 1 - i for i, (name, *_rest) in enumerate(ENGINES)}
    for name, sub, *_vals in ENGINES:
        y = rows[name]
        ax.text(-0.03, y + 0.1, name, transform=ax.get_yaxis_transform(), color=t["ink"],
                fontsize=11, fontweight="bold", ha="right", va="center")
        ax.text(-0.03, y - 0.14, sub, transform=ax.get_yaxis_transform(), color=t["ink2"],
                fontsize=8.5, ha="right", va="center")

    fig.text(0.03, 0.95, "Speed relative to XMage, one laptop core", color=t["ink"],
             fontsize=13, fontweight="bold", ha="left")
    fig.text(0.03, 0.915, "Random play: turns per second on the Pauper Burn mirror, the only "
             "workload all five engines run. Copy: one copy of a mid-game state.",
             color=t["ink2"], fontsize=10, ha="left")
    fig.text(0.03, 0.89, "Search step: copy, one action, advance to the next decision.",
             color=t["ink2"], fontsize=10, ha="left")
    leg = fig.legend(handles=[Patch(facecolor=t["series"][k], edgecolor="none", label=LABELS[k])
                              for k in SERIES],
                     loc="upper left", bbox_to_anchor=(0.03, 0.875), ncol=3, frameon=False,
                     fontsize=9.5, handlelength=1.4, columnspacing=2.0)
    for text in leg.get_texts():
        text.set_color(t["ink2"])
    fig.text(0.03, 0.05, "Copies and search steps are on mid-game FDN states (pair A), except "
             "mtg-kernel's, which has no FDN yet: a Burn state, and its step is derived (copy + "
             "one step).", color=t["muted"], fontsize=8.5, ha="left")
    fig.text(0.03, 0.025, "ManaBrew's copy is copy-on-write (a full copy is 72 µs), and a game "
             "resumes only at the start of a turn. XMage: 54.7 turns/s, 164 µs per copy, "
             "1.24 ms per step.", color=t["muted"], fontsize=8.5, ha="left")

    fig.canvas.draw()  # fix the layout before placing bars in display pixels
    pitch = BAR_PX + GAP_PX
    offsets = {"play": pitch, "copy": 0.0, "step": -pitch}
    for name, _sub, play, copy_us, step_us in ENGINES:
        y = rows[name]
        ratios = {"play": play / XMAGE["play"], "copy": XMAGE["copy_us"] / copy_us,
                  "step": XMAGE["step_us"] / step_us}
        for k in SERIES:
            draw_bar(fig, ax, y, offsets[k], ratios[k], t["series"][k], t)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"015-engine-speed-{mode}.png"
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)
    return path


if __name__ == "__main__":
    for mode in THEMES:
        print(draw(mode))
