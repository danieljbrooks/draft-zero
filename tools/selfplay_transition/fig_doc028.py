"""docs/028's figure: how far each policy target sits from the network that played the games.

    docs/img/028-targets-light.png   x: KL(target || the network's policy), y: share of decisions whose top option
                                     differs from the network's; visit shares (sharpened by tau) and the network's
                                     policy tilted by the search's option values (scale s)

    python tools/selfplay_transition/fig_doc028.py runs/sp/target_stats_calib.json [--title "..."]

The input is target_stats.py's JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stats", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/img/028-targets-light.png"))
    ap.add_argument("--title", default=None)
    a = ap.parse_args(argv)
    s = json.loads(a.stats.read_text())
    vis = sorted(((float(k.split("tau")[1]), v) for k, v in s.items() if k.startswith("visits_tau")), reverse=True)
    cq = sorted((float(k.split("_s")[1]), v) for k, v in s.items() if k.startswith("cq_s"))

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)

    def series(pts, color, label, lab, origin, off):
        xs = ([0.0] if origin else []) + [v["kl_to_pi"] for _, v in pts]
        ys = ([0.0] if origin else []) + [100 * v["top_differs"] for _, v in pts]
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=7, mec=SURF, mew=2, label=label, zorder=3)
        for (k, v) in pts:
            ax.annotate(lab(k), (v["kl_to_pi"], 100 * v["top_differs"]), textcoords="offset points", xytext=off,
                        fontsize=8.5, color=INK2)

    series(sorted(vis), BLUE, "Visit shares, sharpened to visits^(1/τ)", lambda k: f"τ = {k:g}", False, (-12, 9))
    series(cq, ORANGE, "The network's policy tilted by the search's values", lambda k: f"s = {k:g}", True, (7, -4))
    ax.annotate("the network's own policy", (0, 0), textcoords="offset points", xytext=(6, 6), fontsize=8.5, color=INK2)
    ax.set_xlabel("How far the target is from the network's policy: KL(target ‖ policy), nats", color=INK, fontsize=9.5)
    ax.set_ylabel("Decisions where the target's top move differs (%)", color=INK, fontsize=9.5)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.set_xlim(left=-0.03)
    ax.set_ylim(bottom=-0.8)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK)
    ax.set_title(a.title or f"Policy targets: {s['seat_games'] // 2} self-play games, 100 simulations, "
                 f"{s['rows']:,} decisions", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, facecolor=SURF)
    print(a.out)


if __name__ == "__main__":
    main()
