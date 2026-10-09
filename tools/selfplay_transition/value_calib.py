"""docs/028-029: how well a network's value is calibrated on self-play positions, against the games' results.

For a sample of searched decisions from batch files: the network's predicted chance of winning (sigmoid(2 x)), and the
search's root value as a chance ((1 + q) / 2), binned against the result of the seat's game. Writes a JSON of the bins
and, with --fig, a reliability figure (light PNG).

    python tools/selfplay_transition/value_calib.py --batches runs/sp/batches/d0 --model $M0 [--label start] \
        [--model2 runs/sp/arms/x/final.pt.gz --label2 x] --json out.json --fig docs/img/029-calib-light.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def predictions(model_path, t, dev, dtype):
    import torch
    from draftzero.gameplay import graph_supervised as gs
    from draftzero.gameplay import supervised as sv
    from draftzero.selfplay import gnn_train
    model, _, _, _ = gs.load_checkpoint(model_path, dev)
    out = []
    with torch.no_grad():
        for s in range(0, t.n, 256):
            r = np.arange(s, min(t.n, s + 256))
            bd = gs.to_device(gs.make_batch([t], [(0, r)]), dev)
            with sv._autocast(dev, dtype):
                _, vx = gnn_train.option_logprobs(model, bd)
            out.append(vx.float().cpu().numpy())
    x = np.concatenate(out).astype(np.float64)
    return 1 / (1 + np.exp(-2 * x))


def bins(p, y, k=10):
    b = np.minimum((p * k).astype(int), k - 1)
    rows = []
    for i in range(k):
        m = b == i
        if m.sum():
            rows.append({"bin": i, "n": int(m.sum()), "pred": round(float(p[m].mean()), 4), "obs": round(float(y[m].mean()), 4)})
    ece = sum(abs(r["pred"] - r["obs"]) * r["n"] for r in rows) / max(1, len(p))
    ll = float(-(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))).mean())
    return {"bins": rows, "ece": round(ece, 4), "logloss": round(ll, 4), "mean_pred": round(float(p.mean()), 4),
            "mean_obs": round(float(y.mean()), 4), "n": int(len(p))}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batches", type=Path, nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="start network")
    ap.add_argument("--model2")
    ap.add_argument("--label2", default="trained")
    ap.add_argument("--more", action="append", default=[], help="label=checkpoint (more networks to compare)")
    ap.add_argument("--max-rows", type=int, default=30000)
    ap.add_argument("--heldout-only", action="store_true", help="only the held-out games (10% of deck pairs)")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--fig", type=Path)
    ap.add_argument("--title")
    a = ap.parse_args(argv)

    import torch
    from draftzero.gameplay import graph_supervised as gs
    from draftzero.selfplay import gnn_train, tables
    from sp_train import load_rows

    dev = gnn_train.device_of("auto")
    dtype = torch.bfloat16 if dev.type == "cuda" else None
    rows = load_rows(a.batches, 0.1, None, print)
    keep = np.isfinite(rows["z"]) & (rows["heldout"] if a.heldout_only else True)
    if keep.sum() > a.max_rows:
        rng = np.random.default_rng(0)
        g = np.unique(rows["game"][keep])
        rng.shuffle(g)
        cum = np.cumsum([int((rows["game"] == k).sum()) for k in g])
        keep &= np.isin(rows["game"], g[: int(np.searchsorted(cum, a.max_rows)) + 1])
    rows = tables.select(rows, keep)
    _, vocab, edge_vocab, _ = gs.load_checkpoint(a.model, "cpu")
    t = tables.to_graph_table(rows, vocab, edge_vocab)
    y = (1 + t.z.astype(np.float64)) / 2
    res = {"rows": int(t.n), "seat_games": int(len(np.unique(t.game))),
           a.label: bins(predictions(a.model, t, dev, dtype), y),
           "search root value": bins((1 + t.aux["q_root"].astype(np.float64)) / 2, y)}
    if a.model2:
        res[a.label2] = bins(predictions(a.model2, t, dev, dtype), y)
    extra = [m.split("=", 1) for m in a.more]
    for lab, path in extra:
        res[lab] = bins(predictions(path, t, dev, dtype), y)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "bins"})
                      for k, v in res.items()}, indent=1))
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))
    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.6, 5.0), dpi=160)
        fig.patch.set_facecolor(SURF)
        ax.set_facecolor(SURF)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="perfectly calibrated")
        more_cols = ["#eda100", "#4a3aa7", "#e87ba4"]
        series = ([(a.label, BLUE), ("search root value", ORANGE)] + ([(a.label2, AQUA)] if a.model2 else [])
                  + [(lab, more_cols[i % 3]) for i, (lab, _) in enumerate(extra)])
        for name, col in series:
            r = res[name]
            pts = [b for b in r["bins"] if b["n"] >= 50]
            ax.plot([b["pred"] for b in pts], [b["obs"] for b in pts], color=col, lw=2, marker="o", ms=6, mec=SURF,
                    mew=1.5, label=f"{name}: mean {100 * r['mean_pred']:.1f}% vs {100 * r['mean_obs']:.1f}% won, "
                                  f"ECE {r['ece']:.3f}")
        ax.set_xlabel("Predicted chance that the player to move wins", color=INK, fontsize=9.5)
        ax.set_ylabel("Share of those positions whose player won", color=INK, fontsize=9.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.legend(loc="upper left", frameon=False, fontsize=7.5, labelcolor=INK)
        ax.set_title(a.title or f"Value calibration on self-play: {res['seat_games'] // 2} games, {res['rows']:,} decisions",
                     fontsize=9.5, color=INK, loc="left")
        fig.tight_layout()
        a.fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.fig, facecolor=SURF)
        print(a.fig)


if __name__ == "__main__":
    main()
