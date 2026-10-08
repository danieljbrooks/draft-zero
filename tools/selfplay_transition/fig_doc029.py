"""docs/029's figures (light PNGs).

    curves   stage 1's arms as they train: 17lands NLL, the policy's entropy and the value's log-loss and calibration
             error on held-out self-play games, against training steps        -> <out>/029-curves-light.png
    strength each arm's policy alone against heuristic@100 and against the start's policy alone (paired games)
                                                                              -> <out>/029-strength-light.png

    python tools/selfplay_transition/fig_doc029.py --runs runs/r1/sp --arms visits kl cq10 cq30 [--out docs/img]

--runs holds arms/<arm>/metrics.jsonl and h2h/{ph,p0}-<arm>/games.jsonl (r1's runs/sp, or the laptop's copy).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LABELS = {"visits": "visit counts", "kl": "visits + KL anchor", "cq10": "Q-tilted, s = 10", "cq30": "Q-tilted, s = 30",
          "v0": "start network"}


def style(ax):
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8)


def curves(runs: Path, arms: list[str], out: Path) -> None:
    panels = [("17lands validation set NLL (lower: closer to people)", lambda r: r.get("human", {}).get("policy/set_nll")),
              ("Policy entropy on held-out self-play (nats)", lambda r: r["heldout"]["entropy"]),
              ("Value log-loss on held-out self-play", lambda r: r["heldout"]["value_logloss"]),
              ("Value calibration error on held-out self-play", lambda r: r["heldout"]["value_ece"])]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6), dpi=150)
    fig.patch.set_facecolor(SURF)
    for (title, get), ax in zip(panels, axes.ravel()):
        style(ax)
        for i, arm in enumerate(arms):
            f = runs / "arms" / arm / "metrics.jsonl"
            if not f.exists():
                continue
            rs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
            pts = [(r["epoch"], get(r)) for r in rs if get(r) is not None]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=PALETTE[i], lw=2, marker="o", ms=4, mec=SURF, mew=1, label=LABELS.get(arm, arm))
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("passes over the 1,411 games' decisions", fontsize=8, color=INK2)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Stage 1a: one training step from the start network, four policy targets, the same self-play games",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "029-curves-light.png", facecolor=SURF)
    print(out / "029-curves-light.png")


def strength(runs: Path, arms: list[str], out: Path, p0_arms: list[str] | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), dpi=150, sharey=False)
    fig.patch.set_facecolor(SURF)
    for ax, kind, title, base in ((axes[0], "ph", "Policy alone against heuristic@100", None),
                                  (axes[1], "p0", "Policy alone against the start's policy alone", 0.5)):
        style(ax)
        names, vals, errs = [], [], []
        for arm in ["v0"] + arms if kind == "ph" else (p0_arms if p0_arms is not None else arms):
            d = runs / "h2h" / f"{kind}-{arm}"
            if not (d / "games.jsonl").exists():
                continue
            s = paired.score(paired.load([d]))
            if not s["pairs"]:
                continue
            names.append(f"{LABELS.get(arm, arm)}\n({s['games']:,} games)")
            vals.append(100 * s["score"])
            errs.append(100 * (s["se"] or 0))
        y = list(range(len(names)))[::-1]
        cols = [INK2 if n.startswith("start") else PALETTE[0] for n in names]
        ax.barh(y, vals, xerr=errs, color=cols, height=0.55, error_kw={"ecolor": INK2, "lw": 1, "capsize": 3})
        for yi, v, e in zip(y, vals, errs):
            ax.text(v + e + 0.6, yi, f"{v:.1f}%", va="center", fontsize=8.5, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8, color=INK)
        if base:
            ax.axvline(100 * base, color=INK2, lw=1, ls="--")
        lo = min(vals) - 8 if vals else 30
        ax.set_xlim(max(0, lo), max(v + e for v, e in zip(vals, errs)) + 6 if vals else 70)
        ax.set_xlabel("won (%); bars: one standard error across deck pairs", fontsize=8, color=INK2)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / "029-strength-light.png", facecolor=SURF)
    print(out / "029-strength-light.png")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/img"))
    ap.add_argument("--only", choices=["curves", "strength"])
    ap.add_argument("--p0-arms", nargs="*", help="arms with a valid match against the start (default: --arms)")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.only in (None, "curves"):
        curves(a.runs, a.arms, a.out)
    if a.only in (None, "strength"):
        strength(a.runs, a.arms, a.out, a.p0_arms)


if __name__ == "__main__":
    main()
