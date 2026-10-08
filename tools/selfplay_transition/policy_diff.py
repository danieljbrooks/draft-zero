"""docs/029: where trained networks' policies differ from the start's, by kind of decision.

For a sample of self-play decisions: each network's policy over the options, and per decision type (priority: what
to cast or activate, or pass; target: targets, attackers and blockers; yes/no: optional "may" choices, a head the
imitation never trained): the share of decisions, how often each network's top option differs from the start's, its
KL from the start, and how often its top option is the search's most visited.

    python tools/selfplay_transition/policy_diff.py --batches runs/sp/batches/x --start $M0 \
        --model visits=runs/sp/arms/visits/final.pt.gz --model cq10=... [--max-rows 8000] [--json out.json]
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

KINDS = {0: "priority", 3: "target", 5: "yes/no"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batches", type=Path, nargs="+", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--model", action="append", default=[], help="name=checkpoint")
    ap.add_argument("--max-rows", type=int, default=8000)
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
    rows = load_rows(a.batches, 0.1, None, print)
    rows = tables.select(rows, rows["heldout"])
    if tables.n_rows(rows) > a.max_rows:
        rows = tables.select(rows, np.arange(tables.n_rows(rows)) < a.max_rows)
    _, vocab, edge_vocab, _ = gs.load_checkpoint(a.start, "cpu")
    t = tables.to_graph_table(rows, vocab, edge_vocab)
    ptr = t.row_opt_ptr
    row = np.repeat(np.arange(t.n), np.diff(ptr))

    def logprobs(path):
        model, _, _, _ = gs.load_checkpoint(path, dev)
        out = []
        with torch.no_grad():
            for s in range(0, t.n, 256):
                r = np.arange(s, min(t.n, s + 256))
                bd = gs.to_device(gs.make_batch([t], [(0, r)]), dev)
                with sv._autocast(dev, dtype):
                    lp, _ = gnn_train.option_logprobs(model, bd)
                out.append(lp.float().cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    def top(x):
        return np.array([np.argmax(x[ptr[i]:ptr[i + 1]]) for i in range(t.n)])

    base = logprobs(a.start)
    top0 = top(base)
    visits = t.aux["opt_p"].astype(np.float64)
    topv = top(visits)
    gt = t.graph_type
    res = {"rows": int(t.n), "kinds": {KINDS.get(k, str(k)): round(float((gt == k).mean()), 4) for k in np.unique(gt)},
           "start": {KINDS.get(k, str(k)): {"agree_search": round(float((top0 == topv)[gt == k].mean()), 4)}
                     for k in np.unique(gt)}}
    for spec in a.model:
        name, path = spec.split("=", 1)
        lp = logprobs(path)
        tp = top(lp)
        kl = np.bincount(row, weights=np.exp(base) * (base - lp), minlength=t.n)
        res[name] = {KINDS.get(k, str(k)): {"top_differs": round(float((tp != top0)[gt == k].mean()), 4),
                                             "kl_from_start": round(float(kl[gt == k].mean()), 5),
                                             "agree_search": round(float((tp == topv)[gt == k].mean()), 4)}
                     for k in np.unique(gt)}
    print(json.dumps(res, indent=1))
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
