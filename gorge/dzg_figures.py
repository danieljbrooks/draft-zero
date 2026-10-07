"""Figures for docs/027 (dzg networks on gorge): training curves, games an hour, scores in games.

    python gorge/dzg_figures.py curves  OUT.png RUN_DIR[=label] ...   # holdout value and policy loss by epoch
    python gorge/dzg_figures.py speed   OUT.png BENCH_DIR ...           # games an hour by simulations
    python gorge/dzg_figures.py scores  OUT.png SCORES.json             # dot plot of paired-game scores
    python gorge/dzg_figures.py gens    OUT.png GENS.json               # score by generation

Light theme only (docs are read where raw HTML doesn't render). Colours are the dataviz skill's
validated categorical slots, in fixed order; the no-network baseline is neutral grey.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GREY = "#8a8a85"
INK, INK2, GRID, SURF = "#1a1a19", "#5e5d59", "#e6e5df", "#ffffff"


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def _fig(w=9.0, h=3.6, n=1):
    fig, axes = plt.subplots(1, n, figsize=(w, h), dpi=150, facecolor=SURF)
    axes = [axes] if n == 1 else list(axes)
    for ax in axes:
        _style(ax)
    return fig, axes


def _read_curves(run):
    tr, ev = [], []
    with open(os.path.join(run, "curves.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            (ev if r.get("type") == "eval" else tr).append(r)
    return tr, ev


def curves(out, runs, valset="evaldecks"):
    """Two panels: value log loss and policy cross-entropy on held-out positions, by epoch; the
    training value loss dashed beside the held-out one, to show where the value starts to memorise."""
    fig, (av, ap) = _fig(10, 3.8, 2)
    for i, spec in enumerate(runs):
        run, label = (spec.split("=", 1) + [None])[:2] if "=" in spec else (spec, os.path.basename(spec.rstrip("/")))
        tr, ev = _read_curves(run)
        c = SERIES[i % len(SERIES)]
        ex = [e["epoch"] for e in ev]
        vl = [e["eval"][valset]["value_logloss"] for e in ev]
        pc = [e["eval"][valset]["policy_ce"] for e in ev]
        av.plot(ex, vl, color=c, lw=2, marker="o", ms=3.5, label=f"{label}, held out")
        av.plot([t["epoch"] for t in tr], [t["value_logloss"] for t in tr], color=c, lw=1.2, ls="--", alpha=0.8,
                label=f"{label}, training")
        ap.plot(ex, pc, color=c, lw=2, marker="o", ms=3.5, label=label)
        b = min(range(len(ev)), key=lambda j: ev[j]["eval"]["holdout"]["total"])
        av.plot([ex[b]], [vl[b]], marker="o", ms=8, mfc="none", mec=c, mew=1.5)
    e0 = ev[0]["eval"][valset]
    av.axhline(e0["value_base_logloss"], color=GREY, lw=1, ls=":")
    av.text(av.get_xlim()[1], e0["value_base_logloss"], " base rate", color=INK2, fontsize=8, va="center", ha="right")
    ap.axhline(e0["policy_ce_uniform"], color=GREY, lw=1, ls=":")
    ap.text(ap.get_xlim()[1], e0["policy_ce_uniform"], "uniform ", color=INK2, fontsize=8, va="bottom", ha="right")
    ap.axhline(e0["target_entropy"], color=GREY, lw=1, ls=":")
    ap.text(ap.get_xlim()[1], e0["target_entropy"], "targets' entropy (floor) ", color=INK2, fontsize=8, va="bottom",
            ha="right")
    av.set_title("Value: log loss (lower is better)", fontsize=10, loc="left")
    ap.set_title("Policy: cross-entropy with the search's visits", fontsize=10, loc="left")
    for ax in (av, ap):
        ax.set_xlabel("epochs")
    av.legend(fontsize=7.5, frameon=False, ncol=2)
    ap.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURF)
    print("wrote", out)


def speed(out, path):
    """Games an hour by simulations (log-log), one line per (machine, network, mode) row group of
    SPEED.json: [{"series": label, "sims": n, "games_per_hour": x}, ...]. Series keep their order."""
    rows = json.load(open(path))
    order = list(dict.fromkeys(r["series"] for r in rows))
    fig, (ax,) = _fig(7.5, 4.2)
    for i, name in enumerate(order):
        pts = sorted((r["sims"], r["games_per_hour"]) for r in rows if r["series"] == name)
        c = GREY if name.lower().startswith("no network") and i > 3 else SERIES[i % len(SERIES)]
        ls = "--" if "self-play" in name else "-"
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, lw=2, ls=ls, marker="o", ms=5)
        x, y = pts[-1]
        ax.annotate(name, (x, y), xytext=(6, 0), textcoords="offset points", fontsize=8, color=INK2, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("simulations per searched decision")
    ax.set_ylabel("games an hour")
    ax.set_xlim(right=ax.get_xlim()[1] * 6)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURF)
    print("wrote", out)


def scores(out, path, title=""):
    """Dot plot of paired-game scores, SCORES.json: [{"label": ..., "score": 0.535, "games": 600,
    "group": i}, ...], top to bottom; a dashed line at 50%. Labels carry n."""
    rows = json.load(open(path))
    fig, (ax,) = _fig(7.5, 0.42 * len(rows) + 1.0)
    for k, r in enumerate(rows):
        y = len(rows) - 1 - k
        c = SERIES[r.get("group", 0) % len(SERIES)]
        ax.plot([r["score"] * 100], [y], marker="o", ms=8, color=c, mec=SURF, mew=1.5)
        ax.text(r["score"] * 100 + 0.6, y, f"{r['score'] * 100:.1f}%", fontsize=8, va="center", color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['label']}  (n={r['games']:,})" for r in rows][::-1], fontsize=8.5, color=INK)
    ax.axvline(50, color=GREY, lw=1, ls="--")
    ax.set_xlabel("score (%)")
    lo = min(r["score"] for r in rows) * 100
    hi = max(r["score"] for r in rows) * 100
    ax.set_xlim(min(45, lo - 2), max(60, hi + 4))
    ax.grid(axis="y", visible=False)
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURF)
    print("wrote", out)


def gens(out, path):
    """Score against the search without a network by generation, GENS.json:
    {"series": [{"label": ..., "points": [[gen, score, games], ...]}, ...], "ylabel": ...}."""
    d = json.load(open(path))
    fig, (ax,) = _fig(7.5, 4.0)
    for i, s in enumerate(d["series"]):
        pts = sorted(s["points"])
        c = SERIES[i % len(SERIES)]
        ax.plot([p[0] for p in pts], [p[1] * 100 for p in pts], color=c, lw=2, marker="o", ms=6, label=s["label"])
        g, sc, n = pts[-1]
        ax.annotate(s["label"], (g, sc * 100), xytext=(8, 0), textcoords="offset points", fontsize=8.5, color=INK2,
                    va="center")
    ax.axhline(50, color=GREY, lw=1, ls="--")
    ax.set_xlabel("generation")
    ax.set_ylabel(d.get("ylabel", "score against the search without a network (%)"))
    ax.set_xticks(sorted({p[0] for s in d["series"] for p in s["points"]}))
    ax.set_xlim(right=ax.get_xlim()[1] + 0.8)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURF)
    print("wrote", out)


def main(argv):
    cmd, out, rest = argv[1], argv[2], argv[3:]
    if cmd == "curves":
        curves(out, rest)
    elif cmd == "speed":
        speed(out, rest[0])
    elif cmd == "scores":
        scores(out, rest[0], rest[1] if len(rest) > 1 else "")
    elif cmd == "gens":
        gens(out, rest[0])
    else:
        raise SystemExit(f"unknown figure {cmd}")


if __name__ == "__main__":
    main(sys.argv)
