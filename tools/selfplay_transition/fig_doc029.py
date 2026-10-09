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
          "v0": "start network", "b_lam1": "+ λ 1.0 (result only)", "b_lam95": "+ λ 0.95", "b_vwarm": "+ value head first",
          "b_vonly": "value only (policy pinned)", "b_human": "+ 25% human rows", "b_lr1e4": "+ learning rate 1e-4"}


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


def stage1(results: Path, out: Path) -> None:
    """Every stage-1 arm's policy alone: against the start's policy alone, and against heuristic@100, from a results
    JSON ({"p0": {arm: {score, se, pairs}}, "ph": {...}})."""
    r = json.loads(results.read_text())
    order = ["visits", "kl", "cq10", "cq30", "b_lam1", "b_lam95", "b_vwarm", "b_vonly", "b_human", "b_lr1e4"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.4), dpi=150)
    fig.patch.set_facecolor(SURF)
    for ax, kind, title, ref, reflab in ((axes[0], "p0", "Policy alone against the start's policy alone", 50.0, "even"),
                                         (axes[1], "ph", "Policy alone against heuristic@100", None, None)):
        style(ax)
        arms = [a for a in order if a in r[kind]]
        if kind == "ph" and "v0" in r["ph"]:
            ref, reflab = 100 * r["ph"]["v0"]["score"], f"start network {100 * r['ph']['v0']['score']:.1f}%"
        y = list(range(len(arms)))[::-1]
        vals = [100 * r[kind][a]["score"] for a in arms]
        errs = [100 * r[kind][a]["se"] for a in arms]
        cols = [PALETTE[0] if not a.startswith("b_") else PALETTE[1] for a in arms]
        ax.barh(y, vals, xerr=errs, color=cols, height=0.6, error_kw={"ecolor": INK2, "lw": 1, "capsize": 3})
        for yi, v, e, a in zip(y, vals, errs, arms):
            ax.text(v + e + 0.3, yi, f"{v:.1f}% of {2 * r[kind][a]['pairs']:,}", va="center", fontsize=7.5, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS.get(a, a) for a in arms], fontsize=8, color=INK)
        ax.axvline(ref, color=INK2, lw=1, ls="--")
        ax.text(ref, len(arms) - 0.35, f" {reflab}", fontsize=7.5, color=INK2, va="bottom")
        lo = min(vals + [ref]) - 4
        ax.set_xlim(lo, max(v + e for v, e in zip(vals, errs)) + 7)
        ax.set_xlabel("won (%); bars: one standard error across deck pairs", fontsize=8, color=INK2)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    fig.suptitle("Stage 1: blue, the policy targets (1a); orange, value and anchor variants on the Q-tilted s = 10 target (1b)",
                 fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "029-stage1-light.png", facecolor=SURF)
    print(out / "029-stage1-light.png")


def generations(results: Path, out: Path) -> None:
    """The start, generation 1 (cq10) and generation 2, from a results JSON: {"series": {label: [[gen, score, se,
    games], ...]}}; one line per check."""
    r = json.loads(results.read_text())
    fig, ax = plt.subplots(figsize=(7.4, 4.3), dpi=160)
    fig.patch.set_facecolor(SURF)
    style(ax)
    for i, (label, pts) in enumerate(r["series"].items()):
        xs = [p[0] for p in pts]
        ys = [100 * p[1] for p in pts]
        es = [100 * p[2] for p in pts]
        ax.errorbar(xs, ys, yerr=es, color=PALETTE[i], lw=2, marker="o", ms=7, mec=SURF, mew=1.5, capsize=3,
                    label=label)
        for x, y, p in zip(xs, ys, pts):
            if p[3]:
                ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(8, [4, -12, -3][i % 3]), fontsize=7.5,
                            color=INK2)
    ax.axhline(50, color=INK2, lw=1, ls="--")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["start\n(imitation)", "generation 1\n(cq10)", "generation 2"], fontsize=8.5, color=INK)
    ax.set_ylabel("won (%), paired games; bars: one standard error", fontsize=8.5, color=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    ax.set_title(r.get("title", "From imitation to self-play, generation by generation"), fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / "029-generations-light.png", facecolor=SURF)
    print(out / "029-generations-light.png")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/img"))
    ap.add_argument("--only", choices=["curves", "strength", "stage1", "generations"])
    ap.add_argument("--gens", type=Path, help="the generations figure's JSON")
    ap.add_argument("--results", type=Path, help="stage 1's results JSON (for the stage1 figure)")
    ap.add_argument("--p0-arms", nargs="*", help="arms with a valid match against the start (default: --arms)")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.only in (None, "curves"):
        curves(a.runs, a.arms, a.out)
    if a.only in (None, "strength"):
        strength(a.runs, a.arms, a.out, a.p0_arms)
    if a.only in (None, "stage1") and a.results:
        stage1(a.results, a.out)
    if a.only in (None, "generations") and a.gens:
        generations(a.gens, a.out)


if __name__ == "__main__":
    main()
