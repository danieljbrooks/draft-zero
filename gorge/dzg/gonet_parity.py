"""Parity inputs for gorge's in-process Go network (cmd/dzgorge/gonet.go): N records of a shard, the
PyTorch model's outputs on them (float32, CPU, eval mode), as JSON that `dzgorge gonet-check` reads.

  python -m dzg.gonet_parity OUT/best.pt SHARD_DIR parity.json -n 200
  python -m dzg.export OUT/best.pt best.dzgw
  dzgorge gonet-check -net best.dzgw -parity parity.json      # max |Go - PyTorch| over logits and values

Records are spread evenly over the shard. Each record is one state as gorge's policynet.State and
Option fields name it (rows and values as [row, value] pairs, ea/eb as 1 + card index), with the
model's option logits (`scores`), its value logit and sigmoid(value logit) (`value`).
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

from . import data as D
from . import models as M
from . import packfmt as pf


def _pairs(rows, vals) -> list:
    return [[int(r), float(v)] for r, v in zip(rows, vals)]


def records(pack: dict) -> list[dict]:
    """A pack (offsets from 0) as per-state dicts in gorge's field names."""
    out = []
    so, co, cro, oo = (np.asarray(pack[k], np.int64) for k in ("sp_off", "card_off", "cr_off", "opt_off"))
    oso, oho = np.asarray(pack["os_off"], np.int64), np.asarray(pack["oh_off"], np.int64)
    for i in range(len(pack["dense"])):
        cards = []
        for c in range(co[i], co[i + 1]):
            cards.append({"group": int(pack["card_group"][c]), "raw": [float(x) for x in pack["card_raw"][c]],
                          "rows": _pairs(pack["cr_row"][cro[c]:cro[c + 1]], pack["cr_val"][cro[c]:cro[c + 1]])})
        opts = []
        for o in range(oo[i], oo[i + 1]):
            opts.append({"slots": _pairs(pack["os_row"][oso[o]:oso[o + 1]], pack["os_val"][oso[o]:oso[o + 1]]),
                         "hashed": _pairs(pack["oh_row"][oho[o]:oho[o + 1]], pack["oh_val"][oho[o]:oho[o + 1]]),
                         "dense": [float(x) for x in pack["opt_dense"][o]], "bot": int(pack["opt_bot"][o]),
                         "ea": int(pack["opt_ea"][o]), "eb": int(pack["opt_eb"][o])})
        out.append({"dense": [float(x) for x in pack["dense"][i]],
                    "sparse": _pairs(pack["sp_row"][so[i]:so[i + 1]], pack["sp_val"][so[i]:so[i + 1]]),
                    "cards": cards, "options": opts})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dzg.gonet_parity", description=__doc__.split("\n\n")[0])
    ap.add_argument("ckpt")
    ap.add_argument("shard", help="a shard directory (or a pack output directory: its first shard)")
    ap.add_argument("out", help="the JSON to write")
    ap.add_argument("-n", type=int, default=200, help="records (spread evenly over the shard)")
    a = ap.parse_args(argv)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model, ck = M.load(a.ckpt, device="cpu")
    model = model.float().eval()
    pack, meta = pf.read_shard(pf.shard_dirs(a.shard)[0])
    n = min(a.n, meta["n"])
    sel = np.unique(np.linspace(0, meta["n"] - 1, n).round().astype(np.int64))
    sub = D.take(pack, sel, targets=False)
    with torch.inference_mode():
        scores, vlogit = model(D.to_batch(sub, targets=False))
    scores, vlogit = scores.float().numpy(), vlogit.float().numpy()
    recs = records(sub)
    oo = np.asarray(sub["opt_off"], np.int64)
    for i, r in enumerate(recs):
        r["index"] = int(sel[i])
        r["scores"] = [float(x) for x in scores[oo[i]:oo[i + 1]]]
        r["value_logit"] = float(vlogit[i])
        r["value"] = float(1 / (1 + np.exp(-np.float64(vlogit[i]))))
    with open(a.out, "w") as f:
        json.dump({"ckpt": a.ckpt, "arch": ck["arch"], "config": ck["config"], "shard": str(a.shard),
                   "records": recs}, f)
    no = int(oo[-1])
    ne = sum(1 for r in recs for o in r["options"] if o["eb"] > 0)
    print(f"dzg.gonet_parity: {a.out}: {len(recs)} states, {no} options ({ne} with eb), "
          f"{sum(len(r['cards']) for r in recs)} cards")
    return 0


if __name__ == "__main__":
    sys.exit(main())
