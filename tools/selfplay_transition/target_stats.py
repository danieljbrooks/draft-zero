"""docs/028: how far each policy target sits from the network that played the games.

For every searched decision in the batch files: the network's own policy pi (T = 1, no noise), the search's visit
shares, the visits sharpened (visits^(1/tau)), and the network's policy tilted by the search's option values
(cq: pi(a) exp(s (Q(a) - Q_root))). Per target: its entropy, KL(target || pi), and how often its top option differs
from the network's. Also the value: the root value's and the network's mean predicted P(win) against the results.

    python tools/selfplay_transition/target_stats.py --batches runs/sp/batches/v0 --model $M0 [--json out.json]
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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batches", type=Path, nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--taus", default="1,0.5,0.25")
    ap.add_argument("--scales", default="3,10,30,100")
    ap.add_argument("--max-rows", type=int, default=40000)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)

    import torch
    from draftzero.gameplay import graph_net as gn
    from draftzero.gameplay import graph_supervised as gs
    from draftzero.gameplay import supervised as sv
    from draftzero.selfplay import gnn_train, tables
    from sp_train import load_rows

    dev = gnn_train.device_of("auto")
    dtype = torch.bfloat16 if dev.type == "cuda" else None
    model, vocab, edge_vocab, _ = gs.load_checkpoint(a.model, dev)
    rows = load_rows(a.batches, 0.1, None, print)
    if tables.n_rows(rows) > a.max_rows:
        keep = np.unique(rows["game"])
        rng = np.random.default_rng(0)
        rng.shuffle(keep)
        g = rows["game"]
        sel = np.zeros(len(g), bool)
        for k in keep:
            sel |= g == k
            if sel.sum() >= a.max_rows:
                break
        rows = tables.select(rows, sel)
    t = tables.to_graph_table(rows, vocab, edge_vocab)
    n = t.n
    logps, vxs = [], []
    with torch.no_grad():
        for s in range(0, n, 256):
            r = np.arange(s, min(n, s + 256))
            bd = gs.to_device(gs.make_batch([t], [(0, r)]), dev)
            with sv._autocast(dev, dtype):
                lp, vx = gnn_train.option_logprobs(model, bd)
            logps.append(lp.float().cpu().numpy())
            vxs.append(vx.float().cpu().numpy())
    logp = np.concatenate(logps)
    vx = np.concatenate(vxs)
    ptr = t.row_opt_ptr
    row = np.repeat(np.arange(n), np.diff(ptr))
    visits = t.aux["opt_p"].astype(np.float64)
    q = t.aux["opt_q"].astype(np.float64)
    qr = t.aux["q_root"].astype(np.float64)[row]
    pi = np.exp(logp.astype(np.float64))

    def norm(x):
        return x / np.maximum(np.bincount(row, weights=x, minlength=n)[row], 1e-300)

    def top(x):
        out = np.zeros(n, np.int64)
        for i in range(n):
            out[i] = np.argmax(x[ptr[i]:ptr[i + 1]])
        return out

    def stats(p):
        p = norm(p)
        ent = -np.bincount(row, weights=np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0), minlength=n)
        kl = np.bincount(row, weights=np.where(p > 0, p * (np.log(np.maximum(p, 1e-300)) - logp), 0), minlength=n)
        return {"entropy": round(float(ent.mean()), 4), "kl_to_pi": round(float(kl.mean()), 4),
                "top_differs": round(float((top(p) != top_pi).mean()), 4)}

    top_pi = top(pi)
    out = {"rows": int(n), "seat_games": int(len(np.unique(t.game))),
           "pi": {"entropy": round(float(-np.bincount(row, weights=pi * logp, minlength=n).mean()), 4)},
           "options_mean": round(float(np.diff(ptr).mean()), 2), "binary_share": round(float((np.diff(ptr) == 2).mean()), 4)}
    for tau in [float(x) for x in a.taus.split(",")]:
        out[f"visits_tau{tau:g}"] = stats(visits ** (1 / tau))
    adv = np.where(np.isfinite(q) & np.isfinite(qr), q - qr, 0.0)
    out["adv_abs_mean"] = round(float(np.abs(adv[np.isfinite(q)]).mean()), 4)
    for s in [float(x) for x in a.scales.split(",")]:
        out[f"cq_s{s:g}"] = stats(pi * np.exp(s * adv))
    z = t.z.astype(np.float64)
    ok = np.isfinite(z)
    pw = 1 / (1 + np.exp(-2 * vx))
    out["value"] = {"results_mean": round(float(((1 + z[ok]) / 2).mean()), 4),
                    "net_mean_p": round(float(pw[ok].mean()), 4),
                    "root_q_mean_p": round(float(((1 + t.aux["q_root"][ok]) / 2).mean()), 4),
                    "net_auc": gnn_train.auc(vx[ok], z[ok]), "root_auc": gnn_train.auc(t.aux["q_root"][ok], z[ok])}
    print(json.dumps(out, indent=1))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
