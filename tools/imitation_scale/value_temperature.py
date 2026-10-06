"""Temperature-calibrate a graph network's value head (docs/024): P(win) = sigmoid(2x / T), with the one number T
fitted on half of the validation games (split by game) and checked on the other half. T > 1 pulls an overconfident
value toward 50%; dividing by a constant keeps the order, so the AUC is unchanged. T is folded into the value head's
last layer (its weight and bias divided by T), so the calibrated checkpoint is one ordinary network: no inference
change anywhere.

    python tools/imitation_scale/value_temperature.py runs/gnn/full_r1/best_policy.pt.gz \\
        --config configs/gnn_full_r1.yml --tables-dir data/imitation_graph/slim \\
        --out runs/gnn/full_r1/best_policy_calibrated.pt.gz --json runs/gnn/full_r1/value_temperature.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def logloss(x: np.ndarray, won: np.ndarray) -> float:
    p = np.clip(1 / (1 + np.exp(-2 * x)), 1e-7, 1 - 1e-7)
    return float(-(won * np.log(p) + (1 - won) * np.log(1 - p)).mean())


def fit_inverse_temperature(x: np.ndarray, won: np.ndarray) -> float:
    """a = 1/T minimising the log-loss of sigmoid(2 a x): convex in a, so a ternary search."""
    lo, hi = 0.02, 5.0
    for _ in range(200):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if logloss(m1 * x, won) < logloss(m2 * x, won):
            hi = m2
        else:
            lo = m1
    return (lo + hi) / 2


def main(argv=None) -> int:
    from draftzero.gameplay import graph_supervised as gs
    from draftzero.gameplay import supervised as sv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--tables-dir", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    cfg = gs.resolve_config(sv.load_config_file(a.config), {"tables_dir": a.tables_dir})
    dev = sv.pick_device(cfg["device"])
    dtype = sv.amp_dtype(cfg["amp"], dev)
    model, vocab, edge_vocab, meta = gs.load_checkpoint(a.checkpoint, dev)
    model.eval()
    data = gs.load_data(cfg, vocabs=(vocab, edge_vocab), splits=("val",))
    xs, zs, games = [], [], []
    with torch.no_grad():
        for t in data.val:
            if not t.spec["value"]:
                continue
            o = gs.table_outputs(model, t, dev, dtype, cfg["eval_batch_rows"])
            xs.append(o["vx"])
            zs.append(t.z)
            games.append(t.game)
    x, z, g = np.concatenate(xs).astype(np.float64), np.concatenate(zs), np.concatenate(games)
    ok = np.isfinite(z)
    x, won, g = x[ok], (z[ok] > 0).astype(np.float64), g[ok]
    fit = (g % 2) == 0                       # half the validation games fit T, the other half checks it
    inv = fit_inverse_temperature(x[fit], won[fit])
    rep = {"checkpoint": a.checkpoint, "temperature": 1 / inv, "rows_fit": int(fit.sum()), "rows_check": int((~fit).sum())}
    for name, m in (("fit", fit), ("check", ~fit)):
        before, after = sv.value_metrics(x[m], np.where(won[m] > 0, 1.0, -1.0), np.zeros(m.sum())), \
            sv.value_metrics(inv * x[m], np.where(won[m] > 0, 1.0, -1.0), np.zeros(m.sum()))
        rep[name] = {"before": {k: before[k] for k in ("logloss", "ece", "auc")},
                     "after": {k: after[k] for k in ("logloss", "ece", "auc")}}
    last = model.value_head[-2]              # Linear(head_hidden, 1): x = last(...), so scaling it scales x
    with torch.no_grad():
        last.weight.mul_(inv)
        last.bias.mul_(inv)
    info = {**(meta.get("info") or {}), "value_temperature": rep["temperature"],
            "calibrated_from": a.checkpoint}
    gs.save_weights(a.out, model, vocab, edge_vocab, info)
    print(json.dumps(rep, indent=1))
    if a.json:
        a.json.write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
