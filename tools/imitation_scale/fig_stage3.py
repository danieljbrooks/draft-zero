"""Stage 3's learning curves against the same recipe on 10% and 30% of the training games (docs/018): the
validation measures against the training rows seen, so the three runs share an axis.

    python tools/imitation_scale/fig_stage3.py --run runs/exp4/main/evals.jsonl \\
        --ref "30%, one epoch=runs/exp4/sweep_s30/runs/s30-l1-actor3-td99/evals.jsonl" \\
        --ref "10%, one epoch=runs/exp4/sweep_r2x/runs/act3-td99/evals.jsonl" --out docs/img/018-stage3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MEASURES = [("policy/top1_nonpass", "Policy: non-Pass top-1 (higher is better)"),
            ("policy/set_nll", "Policy: set NLL (lower is better)"),
            ("binary/acc", "Attack: accuracy"),
            ("replay_target/top1", "Targets: top-1"),
            ("value/auc", "Value: AUC against the result"),
            ("value/logloss", "Value: log-loss (lower is better)")]
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834", "#1baf7a"]},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "series": ["#3987e5", "#d95926", "#199e70"]},
}


def curve(path: Path) -> list[dict]:
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    return [r for r in rows if r.get("step", 0) > 0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--label", default="all the games (stage 3)")
    ap.add_argument("--ref", action="append", default=[], help="label=evals.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    series = [(a.label, curve(a.run))] + [(r.split("=", 1)[0], curve(Path(r.split("=", 1)[1]))) for r in a.ref]
    epoch_rows = None
    run = series[0][1]
    if run and run[-1].get("epoch"):
        epoch_rows = run[-1]["seen"] / run[-1]["epoch"]
    for mode, t in THEMES.items():
        fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.2), facecolor=t["surface"])
        for ax, (key, title) in zip(axs.flat, MEASURES):
            ax.set_facecolor(t["surface"])
            for si, (label, rows) in enumerate(series):
                xs = [r["seen"] / 1e6 for r in rows if key in r]
                ys = [r[key] for r in rows if key in r]
                ax.plot(xs, ys, color=t["series"][si], lw=2, marker="o", ms=4 if si == 0 else 5, label=label,
                        zorder=3 if si == 0 else 2)
            if epoch_rows:
                for e in (1, 2):
                    ax.axvline(e * epoch_rows / 1e6, color=t["muted"], lw=1, ls=(0, (1, 2)), zorder=1)
            ax.grid(axis="y", color=t["grid"], lw=0.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(t["axis"])
            ax.tick_params(colors=t["muted"], labelsize=8)
            ax.set_title(title, loc="left", fontsize=10, color=t["ink"])
            ax.set_xlabel("Training rows seen (millions); dotted: stage 3's epochs", fontsize=8.5, color=t["muted"])
        axs.flat[0].legend(frameon=False, fontsize=8.5, labelcolor=t["ink2"], loc="lower right")
        fig.suptitle("Experiment #4, stage 3: the recipe on all the training games, against 10% and 30% of them "
                     "(validation rows)", x=0.01, ha="left", fontsize=12, color=t["ink"])
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        p = Path(f"{a.out}-{mode}.png")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=130, facecolor=t["surface"])
        plt.close(fig)
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
