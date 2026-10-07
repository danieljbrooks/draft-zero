"""Rust/PyTorch parity fixture: PyTorch outputs of a dzk-net-v1 network on real dumped decisions, for `dzk parity`
(and the `nn_parity` test, which checks tests/fixtures/parity_v2/).

    uv run --no-project --python 3.12 --with torch --with numpy python mtgkernel/py/parity.py \\
        --net runs/x/g0/train/net.dzkn --data runs/x/g0/data [more data dirs] --min-decisions 300 \\
        --out mtgkernel/tests/fixtures/parity_v2 [--dzk mtgkernel/target/release/dzk]

The sample is stratified so that rare decisions are covered, not just the first games in shard order: up to
--per-family decisions of every decision family (priority, target, attack, block, damage, mulligan, bottom,
trigger order, legend, other), up to --per-kind decisions containing each action kind, the --largest decisions by
action count, and then whole games in shard order until --min-decisions. The data is scanned once, keeping only the
selected decisions. They are written as one shard (OUT/decisions.dzd.gz; one member per source game, holding just
its selected decisions), the network is copied (OUT/net.dzkn), and PyTorch's logits and values, computed one
decision at a time in float32, go to OUT/expected.json. With --dzk, runs `dzk parity` on the result.
"""

import argparse
import gzip
import heapq
import json
import os
import shutil
import subprocess
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D  # noqa: E402
import model as M  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--min-decisions", type=int, default=300)
    ap.add_argument("--per-family", type=int, default=40)
    ap.add_argument("--per-kind", type=int, default=8)
    ap.add_argument("--largest", type=int, default=4)
    ap.add_argument("--max-actions", type=int, default=6000, help="never select decisions with more actions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dzk", help="dzk binary: run `dzk parity` on the fixture")
    a = ap.parse_args()
    torch.set_num_threads(4)
    net, header = M.load_dzkn(a.net)
    net.eval()
    os.makedirs(a.out, exist_ok=True)

    games = []                 # (game index, header)
    chosen = {}                # (game, record index) -> Decision
    fam_n, kind_n = Counter(), Counter()
    largest = []               # heap of (n_act, game, i)
    fill = 0
    scanned = 0
    for gi, (h, recs) in enumerate(D.iter_dir(*a.data)):
        games.append(h)
        whole = fill < a.min_decisions
        for i, r in enumerate(recs):
            scanned += 1
            if r.n_act > a.max_actions:
                continue
            take = whole
            f = r.family
            if fam_n[f] < a.per_family:
                fam_n[f] += 1
                take = True
            for k in set(int(x) for x in r.act_kind):
                if kind_n[k] < a.per_kind:
                    kind_n[k] += 1
                    take = True
            if len(largest) < a.largest:
                heapq.heappush(largest, (r.n_act, gi, i))
                take = True
            elif r.n_act > largest[0][0]:
                heapq.heapreplace(largest, (r.n_act, gi, i))
                take = True
            if take:
                chosen[(gi, i)] = r.copy()  # (frees the member's buffer)
                fill += int(whole)
        # (a decision taken only for the largest-menu heap and pushed out later stays: harmless)
    keys = sorted(chosen)
    decs = [chosen[k] for k in keys]
    members = []
    for gi in sorted({g for g, _ in keys}):
        sel = [chosen[k] for k in keys if k[0] == gi]
        members.append(D.member_bytes(dict(games[gi], parity_subset=True), sel))
    with open(os.path.join(a.out, "decisions.dzd.gz"), "wb") as f:
        for m in members:
            f.write(gzip.compress(m, compresslevel=6, mtime=0))
    if os.path.abspath(a.net) != os.path.abspath(os.path.join(a.out, "net.dzkn")):
        shutil.copyfile(a.net, os.path.join(a.out, "net.dzkn"))
    out = []
    with torch.no_grad():
        for d in decs:  # one at a time: the same reduction shapes as a single Rust decision
            b = D.collate([d])
            logit, v = net(b)
            out.append({"value": float(v[0]), "logits": [float(x) for x in logit]})
    families = Counter(D.FAMILIES[d.family] for d in decs)
    kinds = Counter(int(k) for d in decs for k in set(int(x) for x in d.act_kind))
    exp = {"net_sha256": header["blob_sha256"], "torch": torch.__version__, "games": len(members),
           "decisions": out, "max_actions": max(d.n_act for d in decs), "max_objects": max(len(d.obj_card) for d in decs),
           "families": dict(families), "action_kinds": dict(sorted(kinds.items())), "scanned": scanned}
    json.dump(exp, open(os.path.join(a.out, "expected.json"), "w"))
    print(f"{len(decs)} decisions from {len(members)} games of {scanned} scanned (max {exp['max_actions']} actions, "
          f"{exp['max_objects']} objects); families {dict(families)}; action kinds {sorted(kinds)} -> {a.out}")
    missing = [f for f in D.FAMILIES if f not in families]
    if missing:
        print(f"note: no decision of these families in the data: {missing}")
    if a.dzk:
        r = subprocess.run([a.dzk, "parity", "--net", os.path.join(a.out, "net.dzkn"), "--data",
                            os.path.join(a.out, "decisions.dzd.gz"), "--expected", os.path.join(a.out, "expected.json")],
                           capture_output=True, text=True)
        print(r.stdout.strip())
        if r.returncode != 0:
            print(r.stderr.strip())
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
