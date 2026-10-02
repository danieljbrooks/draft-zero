"""Learning curves of experiment #4's sweep runs (docs/018, stage 2): one row of panels per group of
runs, one column per validation measure, against epochs of the sweep's 10% subset. The reference run
is drawn in grey in every row; each row's other runs take the categorical hues in a fixed order.

    python tools/imitation_scale/fig_sweep.py --runs runs/exp4/sweep/runs --spec round1 \\
        --out docs/img/018-sweep-r1        # writes <out>-light.png and <out>-dark.png

--spec names a group list below, or a JSON file of [{"title", "runs": [[name, label], ...]}, ...] with
a top-level {"reference": [name, label], "groups": [...]}.
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
            ("value/auc", "Value: AUC against the result")]
CHANCE = {"binary/acc": 0.5, "value/auc": 0.5}

SPECS = {
    "round1": {
        "reference": ["default", "default (2 layers, width 512, lr 3e-4)"],
        "groups": [
            {"title": "Shape", "runs": [["layers-1", "1 layer"], ["width-256", "width 256"], ["mlp", "MLP"],
                                        ["width-768", "width 768"], ["layers-4", "4 layers"]]},
            {"title": "Learning rate", "runs": [["lr-1e-4", "lr 1e-4"], ["lr-1e-3", "lr 1e-3"],
                                                ["default-seed1", "default, seed 1"]]},
            {"title": "Value target", "runs": [["value-td", "TD(0.95)"], ["value-aux", "+ turns-left head"],
                                               ["value-td-aux", "TD + turns left"], ["value-weight-0.1", "value weight 0.1"]]},
            {"title": "Initialisation and warm-up", "runs": [["warmup-3k", "3k warm-up steps"],
                                                             ["emb-std-0.02", "embedding std 0.02"], ["pre-ln", "pre-LN"],
                                                             ["modern", "all three"], ["layers-4-modern", "4 layers, all three"]]},
        ],
    },
    "round2": {
        "reference": ["ref", "ref: default shape, lr 1e-4, embedding std 0.02"],
        "groups": [
            {"title": "Noise and shape", "runs": [["ref-seed1", "ref, seed 1"], ["width-256", "width 256"],
                                                  ["layers-4-pre-ln", "4 layers, pre-LN"]]},
            {"title": "1 layer", "runs": [["layers-1", "1 layer"], ["layers-1-lr-3e-4", "1 layer, lr 3e-4"],
                                          ["layers-1-w1024", "1 layer, width 1024"],
                                          ["layers-1-value-tower", "1 layer + value tower"],
                                          ["layers-1-swiglu-attnpool", "1 layer, SwiGLU + attention pooling"]]},
            {"title": "Architecture", "runs": [["value-tower", "value tower (1 layer)"], ["pre-ln", "pre-LN"],
                                               ["swiglu-attnpool", "SwiGLU + attention pooling"],
                                               ["mlp-lr-1e-3", "MLP, lr 1e-3"], ["mlp-wide", "MLP, width 1024, 4 blocks"]]},
            {"title": "Schedule and dropout", "runs": [["lr-2e-4", "lr 2e-4"], ["cosine-2e-4", "cosine from 2e-4"],
                                                       ["pre-ln-lr-5e-4", "pre-LN, lr 5e-4"],
                                                       ["token-dropout-0.1", "token dropout 0.1"],
                                                       ["no-dropout", "no dropout"]]},
        ],
    },
}

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "ref": "#898781",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "ref": "#898781",
             "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]},
}


def read_curve(run_dir: Path) -> list[dict]:
    p = run_dir / "evals.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return [r for r in rows if r.get("epoch", 0) > 0]


def plot(runs: Path, spec: dict, out: Path, title: str) -> list[Path]:
    ref_name, ref_label = spec["reference"]
    ref = read_curve(runs / ref_name)
    groups = [g for g in spec["groups"] if any(read_curve(runs / n) for n, _ in g["runs"])]
    paths = []
    for mode, t in THEMES.items():
        fig, axs = plt.subplots(len(groups), len(MEASURES), figsize=(4.1 * len(MEASURES), 2.9 * len(groups) + 0.8),
                                facecolor=t["surface"], squeeze=False)
        for gi, g in enumerate(groups):
            for mi, (key, mtitle) in enumerate(MEASURES):
                ax = axs[gi][mi]
                ax.set_facecolor(t["surface"])
                if ref:
                    ax.plot([e["epoch"] for e in ref], [e.get(key) for e in ref], color=t["ref"], lw=2, ls=(0, (4, 2)),
                            label=ref_label, zorder=2)
                for si, (name, label) in enumerate(g["runs"]):
                    c = read_curve(runs / name)
                    if not c:
                        continue
                    ax.plot([e["epoch"] for e in c], [e.get(key) for e in c], color=t["series"][si], lw=2,
                            marker="o", ms=4.5, label=label, zorder=3)
                if key in CHANCE:
                    ax.axhline(CHANCE[key], color=t["muted"], lw=1, ls=(0, (1, 2)), zorder=1)
                ax.set_xlim(0, 1.05)
                ax.grid(axis="y", color=t["grid"], lw=0.8)
                ax.set_axisbelow(True)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                for sp in ("left", "bottom"):
                    ax.spines[sp].set_color(t["axis"])
                ax.tick_params(colors=t["muted"], labelsize=8)
                if gi == 0:
                    ax.set_title(mtitle, loc="left", fontsize=10, color=t["ink"])
                if gi == len(groups) - 1:
                    ax.set_xlabel("Epochs of the 10% subset (1.10M rows)", fontsize=8.5, color=t["muted"])
                if mi == 0:
                    ax.set_ylabel(g["title"], fontsize=10, color=t["ink2"])
                if mi == len(MEASURES) - 1:
                    ax.legend(frameon=False, fontsize=7.5, labelcolor=t["ink2"], loc="center left",
                              bbox_to_anchor=(1.02, 0.5))
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=t["ink"])
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        p = Path(f"{out}-{mode}.png")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=130, facecolor=t["surface"])
        plt.close(fig)
        paths.append(p)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True, help="a sweep's runs/ directory")
    ap.add_argument("--spec", default="round1", help="a group list name, or a JSON file")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="Experiment #4 sweep: validation curves over one epoch of the 10% subset "
                                       "(grey dashes: the reference run; dotted: chance)")
    a = ap.parse_args(argv)
    spec = SPECS[a.spec] if a.spec in SPECS else json.loads(Path(a.spec).read_text())
    for p in plot(a.runs, spec, a.out, a.title):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
