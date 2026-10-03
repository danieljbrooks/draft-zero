"""Games-in-hand win rates of simulated games against 17lands' (docs/018, C3): commons on the left, every
non-basic card by rarity on the right, each with its Spearman correlation. Cards with fewer than --min-games games in
hand are left out; the commons furthest from their 17lands rank are labelled.

    python tools/imitation_scale/fig_gih.py runs/exp4/games/c1-selfplay/games.jsonl --out docs/img/018-gih \\
        [--ceiling-common 0.839 --ceiling-all 0.804]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "common": "#2a78d6", "uncommon": "#1baf7a", "rare": "#eda100", "mythic": "#eb6834"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "common": "#3987e5", "uncommon": "#199e70", "rare": "#c98500", "mythic": "#d95926"},
}


def _gih():
    spec = importlib.util.spec_from_file_location("gih", Path(__file__).with_name("gih.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ranks(x):
    return _gih().ranks(np.asarray(x, float))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-games", type=int, default=30)
    ap.add_argument("--label", type=int, default=6, help="commons to label at each end of the residual")
    ap.add_argument("--ceiling-common", type=float, default=None)
    ap.add_argument("--ceiling-all", type=float, default=None)
    ap.add_argument("--title", default="Games-in-hand win rate: the stage-3 policy's self-play against 17lands")
    a = ap.parse_args(argv)
    gih = _gih()
    from draftzero.stats import load_rarity
    rar = load_rarity()
    games = gih.load_games(a.games)
    counts, seats = gih.gih_counts(games)
    ref = json.loads((REPO / "assets" / "reference" / "FDN_gih.json").read_text())["cards"]
    rows = gih.compare(counts, ref, a.min_games)["rows"]
    for r in rows:
        r["rarity"] = rar.get(r["card"], "")
    com = [r for r in rows if r["rarity"] == "common"]
    rho_c = gih.spearman([r["gih_wr"] for r in com], [r["ref_gih_wr"] for r in com])
    rho_a = gih.spearman([r["gih_wr"] for r in rows], [r["ref_gih_wr"] for r in rows])
    # residual in rank terms, for the labels
    rx, ry = ranks([r["ref_gih_wr"] for r in com]), ranks([r["gih_wr"] for r in com])
    res = ry - rx
    order = np.argsort(res)
    lab = set(order[:a.label]) | set(order[-a.label:])
    for mode, t in THEMES.items():
        fig, axs = plt.subplots(1, 2, figsize=(13.5, 6.2), facecolor=t["surface"])
        for ax in axs:
            ax.set_facecolor(t["surface"])
            ax.grid(color=t["grid"], lw=0.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(t["axis"])
            ax.tick_params(colors=t["muted"], labelsize=8.5)
            ax.set_xlabel("17lands GIH win rate (791k human player-games)", fontsize=9, color=t["ink2"])
            ax.set_ylabel(f"Simulated GIH win rate ({seats:,} player-games)", fontsize=9, color=t["ink2"])
        ax = axs[0]
        xs = [r["ref_gih_wr"] for r in com]
        ys = [r["gih_wr"] for r in com]
        sz = [10 + 50 * np.sqrt(r["games"] / max(x["games"] for x in com)) for r in com]
        ax.scatter(xs, ys, s=sz, color=t["common"], alpha=0.85, edgecolor=t["surface"], linewidth=0.8, zorder=3)
        if len(xs) > 2:
            k, b = np.polyfit(xs, ys, 1)
            gx = np.array([min(xs), max(xs)])
            ax.plot(gx, k * gx + b, color=t["muted"], lw=1.2, ls=(0, (4, 3)), zorder=2)
        for j, i in enumerate(sorted(lab, key=lambda i: (com[i]["gih_wr"], com[i]["ref_gih_wr"]))):
            r = com[i]
            ax.annotate(r["card"], (r["ref_gih_wr"], r["gih_wr"]), textcoords="offset points",
                        xytext=(6, 4 if j % 2 else -10), fontsize=7.5, color=t["ink2"])
        ceil = f", noise ceiling at this game count {a.ceiling_common:.2f}" if a.ceiling_common else ""
        ax.set_title(f"Commons, {len(com)} cards: Spearman {rho_c:.2f}{ceil}", loc="left", fontsize=10.5, color=t["ink"])
        ax = axs[1]
        for rr in ("common", "uncommon", "rare", "mythic"):
            sub = [r for r in rows if r["rarity"] == rr]
            if not sub:
                continue
            ax.scatter([r["ref_gih_wr"] for r in sub], [r["gih_wr"] for r in sub], s=18, color=t[rr], alpha=0.8,
                       edgecolor=t["surface"], linewidth=0.6, label=f"{rr} ({len(sub)})", zorder=3)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=t["ink2"], loc="upper left")
        ceil = f", ceiling {a.ceiling_all:.2f}" if a.ceiling_all else ""
        ax.set_title(f"Every non-basic card, {len(rows)}: Spearman {rho_a:.2f}{ceil}", loc="left", fontsize=10.5,
                     color=t["ink"])
        fig.suptitle(a.title + f" ({len(games):,} games; cards with {a.min_games}+ games in hand)", x=0.01, ha="left",
                     fontsize=12, color=t["ink"])
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        p = Path(f"{a.out}-{mode}.png")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=130, facecolor=t["surface"])
        plt.close(fig)
        print(p)
    print("most under 17lands (commons):", [com[i]["card"] for i in order[:a.label]])
    print("most over 17lands (commons):", [com[i]["card"] for i in order[-a.label:][::-1]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
