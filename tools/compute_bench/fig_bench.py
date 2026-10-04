"""docs/020's figures from the compute benchmark (tools/compute_bench/headline.py --json rows.json).

    python tools/compute_bench/fig_bench.py --rows runs/compute_bench/headline.json --out docs/img/020 \\
        --labels labels.txt --highlight 3090-secure/games_b100 \\
        --tstep "RTX 3090 pod, EPYC 7763 (31 threads)=runs/compute_bench/3090-secure/tstep_cpu.json@31" ...

Writes <out>-selfplay-{light,dark}.png (self-play at 100 simulations: games an hour on every machine, games a dollar
on the rented ones; one hue, the recommended machine emphasised) and <out>-training-{light,dark}.png (training
samples a second by machine and device, log scale; and where a CPU step's time goes, as the trainer is, with fused
AdamW and with the feature table frozen). --labels holds "tag[/run]=display label" lines.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# the categorical slots of docs/018's figures (validated for colour-vision separation on both surfaces)
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "rest": "#c9c8c1", "series": ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"]},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "rest": "#4a4a46", "series": ["#3987e5", "#d95926", "#199e70", "#9b72e0"]},
}
LABELS: dict = {}


def label_of(r: dict) -> str:
    return LABELS.get(f"{r['tag']}/{r.get('run', '')}") or LABELS.get(r["tag"]) or r["tag"]


def style(ax, t, xlabel, title):
    ax.set_facecolor(t["surface"])
    ax.grid(axis="x", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(axis="x", colors=t["muted"], labelsize=8.5)
    ax.tick_params(axis="y", colors=t["ink2"], labelsize=9, length=0)
    ax.set_xlabel(xlabel, fontsize=8.5, color=t["ink2"])
    ax.set_title(title, loc="left", fontsize=10.5, color=t["ink"], pad=10)


def hbars(ax, t, names, vals, hi, fmt="{:,.0f}"):
    y = list(range(len(names)))
    colors = [t["series"][0] if h else t["rest"] for h in hi]
    ax.barh(y, vals, height=0.62, color=colors)
    for yi, v in zip(y, vals):
        ax.text(v, yi, "  " + fmt.format(v), va="center", fontsize=8.5, color=t["ink2"])
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.18)


def fig_selfplay(rows: list[dict], out: str, highlight: str) -> None:
    rows = [r for r in rows if r.get("g100")]
    per_hour = sorted(rows, key=lambda r: -r["g100"])
    paid = sorted([r for r in rows if r.get("g100_per_usd")], key=lambda r: -r["g100_per_usd"])
    key = lambda r: f"{r['tag']}/{r['run']}"  # noqa: E731
    for mode, t in THEMES.items():
        fig, axs = plt.subplots(1, 2, figsize=(13.0, 4.6), facecolor=t["surface"])
        hbars(axs[0], t, [label_of(r) for r in per_hour], [r["g100"] for r in per_hour],
              [key(r) == highlight for r in per_hour])
        style(axs[0], t, "games / hour, one machine", "Games an hour (all machines)")
        hbars(axs[1], t, [label_of(r) for r in paid], [r["g100_per_usd"] for r in paid],
              [key(r) == highlight for r in paid])
        style(axs[1], t, "games / $", "Games a dollar (rented machines)")
        fig.suptitle("Self-play at 100 simulations: the MLP searching for both seats", x=0.01, ha="left",
                     fontsize=12, color=t["ink"])
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        p = Path(f"{out}-selfplay-{mode}.png")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=130, facecolor=t["surface"])
        plt.close(fig)
        print(p)


def fig_training(rows: list[dict], tsteps: list[tuple[str, list]], out: str) -> None:
    bars = []
    for r in rows:
        if r.get("run") != "games_b100":
            continue
        for dev in ("gpu", "cpu"):
            v = r.get(f"train_{dev}")
            if v:
                name = LABELS.get(r["tag"]) or r["tag"]       # the machine, not the games run
                bars.append((name + (" GPU" if dev == "gpu" else " CPU"), v, dev))
    bars.sort(key=lambda b: -b[1])
    for mode, t in THEMES.items():
        fig, axs = plt.subplots(1, 2, figsize=(13.0, 4.8), facecolor=t["surface"],
                                gridspec_kw={"width_ratios": [1, 1.15]})
        ax = axs[0]
        y = list(range(len(bars)))
        ax.barh(y, [b[1] for b in bars], height=0.62,
                color=[t["series"][0] if b[2] == "gpu" else t["series"][1] for b in bars])
        for yi, b in zip(y, bars):
            ax.text(b[1], yi, f"  {b[1]:,.0f}", va="center", fontsize=8.5, color=t["ink2"])
        ax.set_yticks(y)
        ax.set_yticklabels([b[0] for b in bars])
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(20, 40000)
        style(ax, t, "training samples / s (log scale)", "Training the MLP: samples a second")
        h = [plt.Rectangle((0, 0), 1, 1, color=t["series"][0]), plt.Rectangle((0, 0), 1, 1, color=t["series"][1])]
        ax.legend(h, ["GPU", "CPU"], frameon=False, fontsize=8.5, labelcolor=t["ink2"], loc="lower right")
        ax = axs[1]
        labels, parts, steps = [], [], []
        for name, res in tsteps:
            for r in res:
                labels.append(f"{name}: {dict(dense='as it is', fused='fused AdamW', frozen='table frozen')[r['variant']]}")
                parts.append((r["ms_forward"], r["ms_backward"], r["ms_optimizer"]))
                steps.append(r["ms_step"])                 # the median step (the parts' medians don't add up to it)
        left = [0.0] * len(parts)
        yy = list(range(len(parts)))
        for k, (lab, c) in enumerate((("forward + loss", t["series"][0]), ("backward", t["series"][1]),
                                      ("clip + optimizer", t["series"][3]))):
            vals = [p[k] for p in parts]
            ax.barh(yy, vals, left=left, color=c, height=0.62, label=lab, edgecolor=t["surface"], linewidth=1.5)
            left = [a + b for a, b in zip(left, vals)]
        for yi, tot, st in zip(yy, left, steps):
            ax.text(tot, yi, f"  {st:,.0f}", va="center", fontsize=8.5, color=t["ink2"])
        ax.set_yticks(yy)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlim(0, max(left) * 1.15)
        style(ax, t, "ms a training step (median, ~39 rows)", "Where a CPU training step goes")
        ax.legend(frameon=False, fontsize=8.5, labelcolor=t["ink2"], loc="lower right")
        fig.tight_layout()
        p = Path(f"{out}-training-{mode}.png")
        fig.savefig(p, dpi=130, facecolor=t["surface"])
        plt.close(fig)
        print(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--highlight", default="3090-secure/games_b100")
    ap.add_argument("--tstep", action="append", default=[], help="label=tstep_cpu.json@threads")
    a = ap.parse_args(argv)
    if a.labels:
        LABELS.update(dict(ln.split("=", 1) for ln in a.labels.read_text().splitlines() if "=" in ln))
    rows = json.loads(a.rows.read_text())
    tsteps = []
    for spec in a.tstep:
        name, rest = spec.split("=", 1)
        path, _, th = rest.partition("@")
        res = [r for r in json.loads(Path(path).read_text()) if not th or r.get("threads") == int(th)]
        tsteps.append((name, res))
    fig_selfplay(rows, a.out, a.highlight)
    fig_training(rows, tsteps, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
