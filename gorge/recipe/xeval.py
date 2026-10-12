"""Score several networks on the held-out games of one or more packs (the games no training ever used: dzg's holdout
split, crc32(game id) % 20 == 0), with visit-count targets and the result as the value's label.

    python gorge/recipe/xeval.py --packs PACK [...] --nets NAME=best.pt [...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dzg import data as D  # noqa: E402
from dzg import models as M  # noqa: E402
from dzg import train as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--packs", nargs="+", required=True)
ap.add_argument("--nets", nargs="+", required=True)
ap.add_argument("--device", default="auto")
a = ap.parse_args()
dev = M.pick_device(a.device)
ds = D.ShardSet(a.packs, in_ram=True)
_, ho = ds.split()
args = T.parse_args(["--train", "x", "--out", "x", "--policy-target", "visits", "--value-blend", "0"])
print(f"{len(ho):,} held-out records from {', '.join(a.packs)}")
for spec in a.nets:
    name, _, path = spec.partition("=")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = M.build(ck["arch"], ck["config"])
    m.load_state_dict(ck["state_dict"])
    m.to(dev)
    r = T.evaluate(m, ds, ho, args, dev)
    print(json.dumps({"net": name, "value_logloss": round(r["value_logloss"], 4), "value_auc": round(r["value_auc"], 4),
                      "policy_ce": round(r["policy_ce"], 4), "uniform": round(r["policy_ce_uniform"], 4),
                      "top1_visits": round(r["top1_visits"], 3), "target_entropy": round(r["target_entropy"], 4)}))
