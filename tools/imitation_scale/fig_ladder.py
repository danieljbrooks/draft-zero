"""Strength by search budget, the stage-3 transformer against the full-data MLP (docs/018): each network's score
against heuristic@100 (win 1, a capped game 0.5, loss 0) with its 95% Wilson interval, the policy alone (greedy) and
IS-MCTS at 100-3,000 simulations; and the median minutes a game takes one worker. A budget's games may be split over
several runs (shards, tails, top-ups): they are pooled, one result per (deck pair, seat swap).

    python tools/imitation_scale/fig_ladder.py --out docs/img/018-ladder [--running 1000,3000]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

G = "runs/exp4/games"
BUDGETS = ["0", "100", "300", "1000", "3000"]
LABELS = {"0": "policy alone\n(greedy)", "100": "100", "300": "300", "1000": "1,000", "3000": "3,000"}
RUNS = {
    "Transformer (stage 3)": {"0": [f"{G}/c1-pvh-t0-s*"], "100": [f"{G}/c2-ilbc100"], "300": [f"{G}/c2-ilbc300"],
                              "1000": [f"{G}/c2-ilbc1000"]},
    "MLP (run 2)": {"0": [f"{G}/mlp-pvh-t0"], "100": [f"{G}/mlp-ilbc100"], "300": [f"{G}/mlp-ilbc300"],
                    "1000": [f"{G}/mlp-ilbc1000-s*", f"{G}/mlp-ilbc1000-top"],
                    "3000": [f"{G}/mlp-ilbc3000-s*", f"{G}/mlp-ilbc3000-top"]},
}
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "series": ["#52514e", "#1baf7a"]},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "series": ["#c3c2b7", "#199e70"]},
}


def games(patterns: list[str]) -> dict:
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
                out[(g["pair"], g["swap"])] = (pts, g["seconds"] / 60)
    return out


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--running", default="", help="budgets still being played (drawn hollow), e.g. 1000,3000")
    ap.add_argument("--note", default="", help="a footnote under the figure")
    a = ap.parse_args(argv)
    running = set(a.running.split(",")) - {""}
    table = {}
    for net, spec in RUNS.items():
        for b, pats in spec.items():
            g = games(pats)
            if not g:
                continue
            s = sum(p for p, _ in g.values())
            n = len(g)
            table[(net, b)] = (s / n, *wilson(s, n), n, statistics.median(m for _, m in g.values()))
    for (net, b), (sc, lo, hi, n, med) in sorted(table.items(), key=lambda kv: (kv[0][0], BUDGETS.index(kv[0][1]))):
        print(f"{net:24s} {b:>5s}: {sc:.3f} [{lo:.2f}, {hi:.2f}] n={n:5d}  median {med:.1f} min")
    for mode, t in THEMES.items():
        fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8), facecolor=t["surface"], gridspec_kw={"width_ratios": [1.6, 1]})
        xs = {b: i for i, b in enumerate(BUDGETS)}
        for k, net in enumerate(RUNS):
            c = t["series"][k]
            off = -0.08 if k == 0 else 0.08
            pts = [(b, table[(net, b)]) for b in BUDGETS if (net, b) in table]
            ax = axs[0]
            ax.plot([xs[b] + off for b, _ in pts], [v[0] for _, v in pts], color=c, lw=2, zorder=2, label=net)
            for b, (sc, lo, hi, n, _) in pts:
                hollow = net.startswith("MLP") and b in running
                ax.errorbar(xs[b] + off, sc, yerr=[[sc - lo], [hi - sc]], color=c, lw=1.4, capsize=4, zorder=3)
                ax.plot(xs[b] + off, sc, "o", ms=8, color=t["surface"] if hollow else c, markeredgecolor=c,
                        markeredgewidth=2, zorder=4)
                ax.annotate(f"{sc:.2f}" + (f"\n(n={n}, running)" if hollow else ""), (xs[b] + off, sc),
                            textcoords="offset points", xytext=(-9, -9) if k == 0 else (9, 9), fontsize=8.5,
                            color=t["ink2"], ha="right" if k == 0 else "left", va="top" if k == 0 else "bottom")
            ax = axs[1]
            ax.plot([xs[b] + off for b, _ in pts], [v[4] for _, v in pts], "o-", color=c, lw=2, ms=7, label=net,
                    markeredgecolor=t["surface"], markeredgewidth=1)
        ax = axs[0]
        ax.axhline(0.5, color=t["muted"], lw=1, ls=(0, (4, 3)), zorder=1)
        ax.text(len(BUDGETS) - 0.6, 0.505, "even with the heuristic", fontsize=8, color=t["muted"], ha="right", va="bottom")
        ax.set_ylim(0.25, 0.92)
        ax.set_title("Score against heuristic@100 [95% CI]", loc="left", fontsize=10.5, color=t["ink"])
        axs[1].set_yscale("log")
        axs[1].set_title("Median minutes a game (one worker)", loc="left", fontsize=10.5, color=t["ink"])
        for ax in axs:
            ax.set_facecolor(t["surface"])
            ax.set_xticks(range(len(BUDGETS)))
            ax.set_xticklabels([LABELS[b] for b in BUDGETS], fontsize=9, color=t["ink2"])
            ax.set_xlim(-0.5, len(BUDGETS) - 0.5)
            ax.set_xlabel("IS-MCTS simulations a decision", fontsize=9, color=t["ink2"])
            ax.grid(color=t["grid"], lw=0.8, axis="y")
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(t["axis"])
            ax.tick_params(colors=t["muted"], labelsize=8.5)
        axs[0].legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=t["ink2"])
        fig.suptitle("Search strength: the stage-3 transformer against the full-data MLP, by simulations a decision",
                     x=0.01, ha="left", fontsize=12, color=t["ink"])
        if a.note:
            fig.text(0.99, 0.015, a.note, ha="right", fontsize=8, color=t["muted"])
        fig.tight_layout(rect=(0, 0.04 if a.note else 0, 1, 0.94))
        p = Path(f"{a.out}-{mode}.png")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=130, facecolor=t["surface"])
        plt.close(fig)
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
