"""The full-data MLPs against the stage-3 transformer and the 30% MLP (docs/018): validation by decisions seen, and by
position in each run's learning-rate schedule. Run from the repo root with each run's evals.jsonl copied to
runs/exp4/curves/ (mlp_1ep, mlp_full_v1, d1-s30-combined) and runs/exp4/main/evals.jsonl:

    python tools/imitation_scale/fig_mlp_full.py docs/img/018-mlp-run2     # -> ...-rows.png, ...-schedule.png
"""
import json, math, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = sys.argv[1] if len(sys.argv) > 1 else "fig_mlp_full"
def load(p):
    return [r for r in (json.loads(l) for l in open(p) if l.strip()) if r.get("step", 0) > 0]
series = [("Transformer, all the games, 3 epochs (stage 3)", "runs/exp4/main/evals.jsonl", 3, "#52514e", "-"),
          ("MLP run 2: all the games, 1 annealed epoch (done)", "runs/exp4/curves/mlp_1ep.jsonl", 1, "#1baf7a", "-"),
          ("MLP run 1: all the games, 10-epoch schedule, weight decay (stopped)", "runs/exp4/curves/mlp_full_v1.jsonl", 10, "#eb6834", "-"),
          ("MLP, 30% of the games, 1 epoch (d1)", "runs/exp4/curves/d1-s30-combined.jsonl", 1, "#1baf7a", (0, (4, 3)))]
M = [("policy/set_nll", "Policy: set NLL (lower is better)"), ("policy/top1_nonpass", "Policy: non-Pass top-1"),
     ("value/auc", "Value: AUC against the result"), ("value/logloss", "Value: log-loss (lower is better)")]
lr_frac = lambda f: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, f)))
t = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"}
for view in ("rows", "schedule"):
    fig, axs = plt.subplots(1, 4, figsize=(16.5, 4.6), facecolor=t["surface"])
    for ax, (key, title) in zip(axs, M):
        ax.set_facecolor(t["surface"])
        for label, path, ep, col, ls in series:
            rows = load(path)
            xs = [r["seen"] / 1e6 if view == "rows" else r["epoch"] / ep for r in rows if r.get(key) is not None]
            ys = [r[key] for r in rows if r.get(key) is not None]
            ax.plot(xs, ys, color=col, ls=ls, lw=2.4 if "run 2" in label else 1.7, marker="o", ms=3.5, label=label,
                    markeredgecolor=t["surface"], markeredgewidth=0.6)
        if view == "schedule":
            ax2 = ax.twinx(); fx = [i / 100 for i in range(101)]
            ax2.plot(fx, [lr_frac(f) for f in fx], color=t["muted"], lw=1, ls=":")
            ax2.set_ylim(0, 1.05); ax2.tick_params(colors=t["muted"], labelsize=7.5)
            ax2.set_yticks([0.1, 0.5, 1.0]); ax2.set_yticklabels(["10%", "50%", "100%"])
            for sp in ax2.spines.values(): sp.set_visible(False)
        ax.set_title(title, loc="left", fontsize=10, color=t["ink"])
        ax.grid(color=t["grid"], lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(t["axis"])
        ax.tick_params(colors=t["muted"], labelsize=8.5)
        ax.set_xlabel("training decisions seen (millions)" if view == "rows" else "fraction of the run's schedule (learning rate: dotted)", fontsize=8.5, color=t["ink2"])
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=t["ink2"])
    last = load("runs/exp4/curves/mlp_1ep.jsonl")[-1]
    fig.suptitle(("MLP run 2 (finished, epoch %.2f) against run 1, the 30%% MLP and the transformer: " % last["epoch"]) + ("by decisions seen" if view == "rows" else "by position in each run's learning-rate schedule"),
                 x=0.01, ha="left", fontsize=12, color=t["ink"])
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(f"{OUT}-{view}.png", dpi=120, facecolor=t["surface"]); plt.close(fig)
    print(f"{OUT}-{view}.png")
