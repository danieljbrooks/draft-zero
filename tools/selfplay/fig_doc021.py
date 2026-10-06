"""The figure of docs/021 (the self-play plan), light only (Dan reads docs in Claude, which shows no <picture>):

    docs/img/021-games-per-dollar-light.png   GNN self-play games a dollar at 1,000 simulations (PIMC), by machine

    python tools/selfplay/fig_doc021.py

The anchor is docs/024's measurement: on a Community RTX A4000 ($0.17/h, 13.6 cores of an EPYC 7452, 12 workers,
the GNN on its GPU), the GNN at 100 simulations against the heuristic played ~79 games an hour, almost all of it the
GNN's side, so self-play (both seats the GNN) is ~40 an hour, ~3 a core. Other machines scale that by usable cores
and core speed (Zen 3-5 cores ~1.3-1.6x a Zen 2 core, docs/020); CPU-only machines also pay for serving the GNN on
the CPU (x0.67-0.75). 1,000 simulations cost ~9x 100 (docs/024: 100 -> 300 took 2.6x; flat per simulation beyond).
docs/021 §5 has the table these come from.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

OUT = Path("docs/img")
AT_1000 = 9.0   # games a dollar at 100 simulations / at 1,000

# (label, games a dollar at 100 simulations: low, high, anchored on a measurement)
ROWS = [
    ("RunPod RTX A4000, Community\n$0.17/h, 13.6 cores (EPYC 7452)", 235, 245, True),
    ("RunPod RTX 3070, Community\n$0.13/h, ~18.7 cores (EPYC 7663)", 400, 650, False),
    ("RunPod RTX 3090, Community\n$0.22/h, ~23-27 cores on a good host", 320, 550, False),
    ("RunPod RTX 3090, Secure\n$0.50/h, ~31 cores (EPYC 7763)", 220, 280, False),
    ("Prime Intellect spot CPU (DataCrunch)\n$0.04/h for 8 vCPU (EPYC Turin), GNN on the CPU", 250, 400, False),
    ("Prime Intellect CPU, on demand (DataCrunch)\n$0.10/h for 8 vCPU, GNN on the CPU", 100, 160, False),
    ("Prime Intellect CPU (Nebius)\n$0.20/h for 8 vCPU, GNN on the CPU", 50, 80, False),
    ("Dan's tower (4 cores, GTX 1070)\nelectricity, ~$0.05/h", 160, 280, False),
]

# The reference palette's first slot (blue), as docs/019 and docs/020's figures, validated there in both modes.
LIGHT = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e1e0d9", axis="#c3c2b7", bar="#2a78d6",
             est="#a9c8ef")


def draw() -> None:
    t = LIGHT
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "hatch.linewidth": 0.8})
    rows = sorted(((lab, lo / AT_1000, hi / AT_1000, m) for lab, lo, hi, m in ROWS), key=lambda r: -(r[1] + r[2]) / 2)
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    ys = list(range(len(rows)))[::-1]
    for y, (label, lo, hi, anchored) in zip(ys, rows):
        mid = (lo + hi) / 2
        if anchored:
            ax.barh(y, mid, height=0.56, color=t["bar"], linewidth=0)
            ax.text(mid + 1.2, y, f"~{mid:,.0f}", va="center", ha="left", color=t["ink"], fontsize=9)
        else:
            ax.barh(y, mid, height=0.56, color=t["est"], edgecolor=t["bar"], hatch="////", linewidth=0)
            ax.plot([lo, hi], [y, y], color=t["ink2"], lw=1.4, solid_capstyle="butt")
            for x in (lo, hi):
                ax.plot([x, x], [y - 0.14, y + 0.14], color=t["ink2"], lw=1.4)
            ax.text(hi + 1.2, y, f"~{lo:,.0f}-{hi:,.0f}", va="center", ha="left", color=t["ink2"], fontsize=9)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.3, color=t["ink2"])
    ax.set_xlim(0, 86)
    ax.set_xlabel("Self-play games a dollar at 1,000 simulations (the GNN, PIMC, both seats search)",
                  color=t["ink2"], fontsize=9)
    ax.grid(color=t["grid"], lw=0.8, axis="x")
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["ink2"], labelsize=8.5, length=0)
    ax.legend(handles=[Patch(facecolor=t["bar"], label="anchored on a measurement (docs/024)"),
                       Patch(facecolor=t["est"], edgecolor=t["bar"], hatch="////", label="estimate: cores x speed x price, range")],
              loc="lower right", frameon=False, fontsize=8.5, labelcolor=t["ink2"])
    fig.tight_layout()
    p = OUT / "021-games-per-dollar-light.png"
    fig.savefig(p, dpi=150, facecolor=t["surface"])
    plt.close(fig)
    print(p)


if __name__ == "__main__":
    draw()
