"""Draw docs/012's illustrative plot for the first experiment: agreement against compute.

Every number below is made up. The plot shows the first experiment's output format (docs/012
§2.7), not a result. Run from the repo root:

    python tools/search_bench/dummy_frontier_plot.py

Writes docs/img/012-frontier-dummy-light.png and docs/img/012-frontier-dummy-dark.png.
Two panels share both axes: offline search (the heuristic evaluator) and a trained network.
Colours are the first three slots of the validated default categorical palette, in both modes;
methods that fail the hidden-information test use the muted ink, hollow.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "docs" / "img"

# Pod-wide throughput assumed for the x positions (docs/006, docs/010): about 550 simulations/s
# for offline search and 250 with the network. A method's factor is its cost per simulation
# relative to today's search. All illustrative.
RATE = {"offline": 550.0, "network": 250.0}
FACTOR = {"ismcts": 3.0, "pimc4": 1.1, "pimc1": 1.05, "clairvoyant": 1.0}

THEMES = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7", band="#f0efec",
        series={"pimc4": "#2a78d6", "pimc1": "#eb6834", "ismcts": "#1baf7a"},
    ),
    "dark": dict(
        surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835", band="#262624",
        series={"pimc4": "#3987e5", "pimc1": "#d95926", "ismcts": "#199e70"},
    ),
}

LABELS = {
    "ismcts": "IS-MCTS, a fresh world every iteration",
    "pimc4": "PIMC, 4 sampled worlds",
    "pimc1": "PIMC, 1 sampled world",
    "clairvoyant": "MageZero MCTS today (searches the real hidden cards)",
}
END_LABELS = {"ismcts": "IS-MCTS", "pimc4": "PIMC, 4 worlds", "pimc1": "PIMC, 1 world",
              "clairvoyant": "Clairvoyant MCTS"}

# {panel: {method: {simulations per decision: agreement %}}}
CURVES = {
    "offline": {
        "ismcts": {100: 49.5, 300: 55.5, 1000: 61.5, 3000: 64.5},
        "pimc4": {100: 48.0, 300: 53.0, 1000: 57.0, 3000: 60.0},
        "clairvoyant": {100: 48.5, 300: 52.5, 1000: 55.5, 3000: 57.5},
        "pimc1": {100: 47.5, 300: 51.5, 1000: 54.0, 3000: 55.0},
    },
    "network": {
        "ismcts": {100: 51.5, 300: 57.5, 1000: 64.5, 3000: 67.5},
        "pimc4": {100: 50.0, 300: 55.0, 1000: 59.5, 3000: 62.5},
        "clairvoyant": {100: 50.5, 300: 54.5, 1000: 58.0, 3000: 60.5},
        "pimc1": {100: 49.5, 300: 53.5, 1000: 56.5, 3000: 58.0},
    },
}
# No-search references: (label, panel, pod-seconds per decision, agreement %, marker, eligible)
REFERENCES = [
    ("Rule heuristic (no search)", "offline", 0.0006, 46.0, "D", True),
    ("XMage MAD AI (reads its own next draw)", "offline", 2.5, 46.0, "^", False),
    ("Policy network (no search)", "network", 0.004, 44.0, "s", True),
]
PANEL_TITLES = {
    "offline": "Offline search (a heuristic scores positions)",
    "network": "A trained network from experiment #2",
}
CEILING = (74.0, 80.0)


def pod_seconds(panel: str, method: str, sims: int) -> float:
    return sims * FACTOR[method] / RATE[panel]


def style_axes(ax, t) -> None:
    ax.set_facecolor(t["surface"])
    ax.set_xscale("log")
    ax.set_xlim(0.00025, 200)
    ax.set_ylim(38, 84)
    ax.grid(True, which="major", color=t["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
    ax.tick_params(which="minor", length=0)
    ax.set_xticks([0.001, 0.01, 0.1, 1, 10, 100])
    ax.set_xticklabels(["0.001", "0.01", "0.1", "1", "10", "100"])
    ax.set_yticks(range(40, 81, 10))
    ax.set_yticklabels([f"{y}%" for y in range(40, 81, 10)])
    ax.axhspan(*CEILING, color=t["band"], zorder=0, linewidth=0)


def draw_panel(ax, panel: str, t) -> None:
    style_axes(ax, t)
    ax.set_title(PANEL_TITLES[panel], color=t["ink"], fontsize=11, loc="left", pad=10)
    ax.text(150, CEILING[1] - 1.0, "Ceiling: unknown, below 100% (placeholder band)",
            color=t["ink2"], fontsize=8.5, va="top", ha="right")

    for method, pts in CURVES[panel].items():
        xs = [pod_seconds(panel, method, s) for s in pts]
        ys = list(pts.values())
        if method == "clairvoyant":
            ax.plot(xs, ys, color=t["muted"], linewidth=2, solid_capstyle="round", zorder=2)
            ax.plot(xs, ys, linestyle="none", marker="o", markersize=8,
                    markerfacecolor=t["surface"], markeredgecolor=t["muted"],
                    markeredgewidth=2, zorder=3)
            colour_text = t["ink2"]
        else:
            colour = t["series"][method]
            ax.plot(xs, ys, color=colour, linewidth=2, solid_capstyle="round", zorder=4)
            ax.plot(xs, ys, linestyle="none", marker="o", markersize=8, markerfacecolor=colour,
                    markeredgecolor=t["surface"], markeredgewidth=2, zorder=5)
            colour_text = t["ink"]
        ax.annotate(END_LABELS[method], (xs[-1], ys[-1]), xytext=(9, 0),
                    textcoords="offset points", color=colour_text, fontsize=9, va="center")

    for label, where, x, y, marker, eligible in REFERENCES:
        if where != panel:
            continue
        if eligible:
            ax.plot([x], [y], linestyle="none", marker=marker, markersize=8,
                    markerfacecolor=t["ink2"], markeredgecolor=t["surface"], markeredgewidth=2,
                    zorder=5)
            text = label.replace(" (no search)", "\n(no search)")
            ax.annotate(text, (x, y), xytext=(0, -12), textcoords="offset points",
                        color=t["ink2"], fontsize=8.5, ha="center", va="top")
        else:
            ax.plot([x], [y], linestyle="none", marker=marker, markersize=9,
                    markerfacecolor=t["surface"], markeredgecolor=t["muted"], markeredgewidth=2,
                    zorder=3)
            ax.annotate("MAD AI", (x, y), xytext=(8, -2), textcoords="offset points",
                        color=t["ink2"], fontsize=9, va="center")


def draw(mode: str) -> Path:
    t = THEMES[mode]
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.edgecolor": t["axis"],
        "axes.labelcolor": t["ink2"],
        "xtick.color": t["muted"],
        "ytick.color": t["muted"],
        "xtick.labelcolor": t["ink2"],
        "ytick.labelcolor": t["ink2"],
    })
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), dpi=160, sharex=True, sharey=True)
    fig.patch.set_facecolor(t["surface"])
    for ax, panel in zip(axes, ("offline", "network")):
        draw_panel(ax, panel, t)
        ax.set_xlabel("Pod-seconds per decision (log scale)", color=t["ink2"], labelpad=6)
    axes[0].set_ylabel("Agreement with top 17lands players", color=t["ink2"], labelpad=8)

    # One legend for both panels, in two groups: the hidden-information test is pass/fail.
    def header(text: str) -> Line2D:
        return Line2D([], [], linestyle="none", marker=None, label=text)

    def blank() -> Line2D:
        return Line2D([], [], linestyle="none", marker=None, label=" ")

    passes = [header("Passes the hidden-information test")]
    for method in ("ismcts", "pimc4", "pimc1"):
        colour = t["series"][method]
        passes.append(Line2D([], [], color=colour, linewidth=2, marker="o", markersize=7,
                             markerfacecolor=colour, markeredgecolor=t["surface"],
                             markeredgewidth=1.5, label=LABELS[method]))
    for label, _, _, _, marker, eligible in REFERENCES:
        if eligible:
            passes.append(Line2D([], [], linestyle="none", marker=marker, markersize=7,
                                 markerfacecolor=t["ink2"], markeredgecolor=t["surface"],
                                 markeredgewidth=1.5, label=label))
    fails = [header("Fails it: reads hidden cards")]
    fails.append(Line2D([], [], color=t["muted"], linewidth=2, marker="o", markersize=7,
                        markerfacecolor=t["surface"], markeredgecolor=t["muted"],
                        markeredgewidth=2, label=LABELS["clairvoyant"]))
    fails.append(Line2D([], [], linestyle="none", marker="^", markersize=8,
                        markerfacecolor=t["surface"], markeredgecolor=t["muted"],
                        markeredgewidth=2, label=REFERENCES[1][0]))
    while len(fails) < len(passes):
        fails.append(blank())
    leg = fig.legend(handles=passes + fails, loc="lower center", ncol=2, frameon=False,
                     fontsize=9, handlelength=2.2, columnspacing=3.0, labelspacing=0.45,
                     bbox_to_anchor=(0.5, 0.0))
    for text in leg.get_texts():
        text.set_color(t["ink2"])
        if text.get_text().startswith(("Passes", "Fails")):
            text.set_color(t["ink"])
            text.set_fontweight("bold")

    fig.text(0.06, 0.965, "ILLUSTRATIVE ONLY: the first experiment's plot", color=t["ink"],
             fontsize=13, fontweight="bold", ha="left")
    fig.text(0.06, 0.935, "Dummy numbers that show its format; nothing here is measured. Each "
             "line's points are 100, 300, 1,000 and 3,000 simulations per decision, left to right. "
             "1 pod-second = $0.00014.", color=t["ink2"], fontsize=10, ha="left")
    fig.subplots_adjust(left=0.06, right=0.93, top=0.86, bottom=0.30, wspace=0.12)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"012-frontier-dummy-{mode}.png"
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)
    return path


if __name__ == "__main__":
    for mode in THEMES:
        print(draw(mode))
