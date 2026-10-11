"""docs/031's figures (light theme; docs are read where raw HTML doesn't render).

    python gorge/recipe/figs.py e1    docs/img/031-e1-light.png  data/gorge/runs/r31/e1
    python gorge/recipe/figs.py loops docs/img/031-loops-light.png LOOPS.json

e1: each arm's best checkpoint (by its training holdout) scored on the eval-deck games, against games trained on.
loops: {"title": ..., "series": {label: [[games, score, se], ...]}}, strength against the zero point by total games.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dzg_figures import GREY, INK, INK2, SERIES, SURF, _style  # noqa: E402


def e1(out: str, root: str) -> None:
    rows = {}
    for d in sorted(Path(root).glob("w*-g*k")):
        m = re.match(r"w(\d+)-g(\d+)k", d.name)
        if not m or not (d / "best.pt").exists():
            continue
        best = None
        for line in open(d / "curves.jsonl"):
            r = json.loads(line)
            if r.get("type") == "eval" and (best is None or r["eval"]["holdout"]["total"] < best["eval"]["holdout"]["total"]):
                best = r
        if best:
            rows.setdefault(int(m[1]), []).append((int(m[2]), best["eval"]["evaldecks"]))
    panels = [("value_logloss", "Value log-loss on the eval decks' games (lower is better)"),
              ("value_auc", "Value AUC on the eval decks' games (higher is better)"),
              ("policy_ce", "Policy cross-entropy to the targets (lower is better)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), dpi=150)
    fig.patch.set_facecolor(SURF)
    for ax, (key, title) in zip(axes, panels):
        _style(ax)
        for i, w in enumerate(sorted(rows)):
            pts = sorted(rows[w])
            ax.plot([p[0] for p in pts], [p[1][key] for p in pts], color=SERIES[i], lw=2, marker="o", ms=5,
                    mec=SURF, mew=1, label=f"width {w}")
        if key == "value_logloss" and rows:
            base = next(iter(rows.values()))[0][1]["value_base_logloss"]
            ax.axhline(base, color=GREY, lw=1, ls="--")
            ax.text(5, base, " the base rate alone", color=INK2, fontsize=8, va="bottom")
        ax.set_xscale("log", base=2)
        ax.set_xticks([5, 10, 20, 40, 80, 160])
        ax.set_xticklabels(["5k", "10k", "20k", "40k", "80k", "160k"])
        ax.set_xlabel("games trained on")
        ax.set_title(title, fontsize=9.5, loc="left")
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURF)
    print(out)


def loops(out: str, path: str) -> None:
    r = json.loads(Path(path).read_text())
    fig, ax = plt.subplots(figsize=(7.6, 4.3), dpi=160)
    fig.patch.set_facecolor(SURF)
    _style(ax)
    for i, (label, pts) in enumerate(r["series"].items()):
        ax.errorbar([p[0] / 1000 for p in pts], [100 * p[1] for p in pts], yerr=[100 * p[2] for p in pts],
                    color=SERIES[i], lw=2, marker="o", ms=6, mec=SURF, mew=1.2, capsize=3, label=label)
    ax.axhline(50, color=GREY, lw=1, ls="--")
    ax.set_xlabel("self-play games so far (thousands)")
    ax.set_ylabel("won against the zero point (%)")
    ax.set_title(r.get("title", "Strength by games played"), fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURF)
    print(out)


if __name__ == "__main__":
    {"e1": e1, "loops": loops}[sys.argv[1]](*sys.argv[2:])
