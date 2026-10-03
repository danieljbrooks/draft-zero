"""Learning curves of the MLP follow-ups against their bases and the transformer at the same data (docs/018): the
validation measures against the training rows seen, one figure per data size.

    python tools/imitation_scale/fig_mlp_curves.py --out docs/img/018-mlp-curves-10 --title "10% of the games" \\
        --series "ref:Transformer (1 layer)=runs/exp4/curves/act3-td99.jsonl" \\
        --series "base:MLP base (m4-best)=runs/exp4/curves/m4-best.jsonl" \\
        --series "Table x30 (a2)=runs/exp4/curves/a2-emblr30.jsonl" ...

A series is "label=evals.jsonl", optionally prefixed: "ref:" (grey, solid: a reference model), "base:" (grey,
dashed: the base recipe), "seed:" (the previous series' colour, dashed: a seed repeat), or "c0:".."c3:" (a fixed
colour slot, so one recipe keeps its colour across figures; otherwise slots go in order).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MEASURES = [("policy/set_nll", "Policy: set NLL (lower is better)"),
            ("policy/top1_nonpass", "Policy: non-Pass top-1"),
            ("binary/acc", "Attack: accuracy"),
            ("replay_target/top1", "Targets: top-1"),
            ("value/auc", "Value: AUC against the result"),
            ("value/logloss", "Value: log-loss (lower is better)")]
# categorical slots validated for colour-vision separation (dataviz validate_palette, light and dark surfaces)
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "ref": "#52514e", "series": ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"]},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "ref": "#c3c2b7", "series": ["#3987e5", "#d95926", "#199e70", "#9b72e0"]},
}


def curve(path: Path) -> list[dict]:
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    return [r for r in rows if r.get("step", 0) > 0]


def parse(spec: str) -> tuple[str, str, Path]:
    kind = "run"
    for k in ("ref", "base", "seed", "c0", "c1", "c2", "c3"):
        if spec.startswith(k + ":"):
            kind, spec = k, spec[len(k) + 1:]
    label, path = spec.split("=", 1)
    return kind, label, Path(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", action="append", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    series = [(k, label, curve(p)) for k, label, p in map(parse, a.series)]
    for mode, t in THEMES.items():
        fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.4), facecolor=t["surface"])
        for ax, (key, title) in zip(axs.flat, MEASURES):
            ax.set_facecolor(t["surface"])
            slot, color = -1, None
            for kind, label, rows in series:
                if kind == "ref":
                    c, ls, lw = t["ref"], "-", 1.6
                elif kind == "base":
                    c, ls, lw = t["muted"], (0, (4, 3)), 1.6
                elif kind == "seed":
                    c, ls, lw = color, (0, (4, 3)), 1.6
                else:
                    slot = int(kind[1]) if kind.startswith("c") else slot + 1
                    c, ls, lw = t["series"][slot % len(t["series"])], "-", 2.2
                    color = c
                xs = [r["seen"] / 1e6 for r in rows if r.get(key) is not None]
                ys = [r[key] for r in rows if r.get(key) is not None]
                ax.plot(xs, ys, color=c, ls=ls, lw=lw, marker="o", ms=4, label=label,
                        markeredgecolor=t["surface"], markeredgewidth=0.8, zorder=2 if kind in ("ref", "base") else 3)
            ax.set_title(title, loc="left", fontsize=10, color=t["ink"])
            ax.grid(color=t["grid"], lw=0.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(t["axis"])
            ax.tick_params(colors=t["muted"], labelsize=8.5)
            ax.set_xlabel("training rows seen (millions)", fontsize=8.5, color=t["ink2"])
        h, lab = axs.flat[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=min(len(lab), 4), frameon=False, fontsize=9,
                   labelcolor=t["ink2"], bbox_to_anchor=(0.5, 0.0))
        fig.suptitle(f"MLP learning curves, {a.title}: validation over one epoch", x=0.01, ha="left", fontsize=12,
                     color=t["ink"])
        fig.tight_layout(rect=(0, 0.07, 1, 0.95))
        p = Path(f"{a.out}-{mode}.png")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=130, facecolor=t["surface"])
        plt.close(fig)
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
