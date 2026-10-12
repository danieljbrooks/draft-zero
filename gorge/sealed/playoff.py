"""
Test-time playoff: build a few candidate decks per pool, play each against a fixed gauntlet, keep the best.

The last step of the human process ("build a few promising decks, pick the one you like"), with games instead of
taste. For each pool the builder's top candidates (the refined decks of its best colour configurations) each play
paired games against a gauntlet of the same builder's decks from train pools. The choice is the candidate with the
best record after shrinking toward the network's own preference: a Bradley-Terry posterior, with the network's score
difference as the prior mean and the games as evidence, so a few lucky games don't overturn a clear network verdict.

Self-play only: the gauntlet is built from train pools; no human deck is involved.

  python -m sealed.playoff OUT_DIR --ens ENS.pt --pools 500 --cands 3 --opps 40 --sims 0 --workers 4
  python -m sealed.evalbuild OUT_DIR --builder playoff --pools 500 --sims 100   # reuses OUT_DIR/built.jsonl
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

from . import build as Bd
from . import decks as D
from . import games as G
from . import model as M

C = D.C


def _cands(args):
    ens_path, pools, k, seed = args
    torch.set_num_threads(1)
    ens = M.Ensemble.load(ens_path)
    q = np.mean([m.quality().detach().numpy() for m in ens.members], 0)
    obj = Bd.Objective(ens, "mean")
    lands = ens.meta.get("lands")
    out = []
    for p in pools:
        cs = Bd.build(p, obj, q, n_refine=max(k, 4), return_all=True, lands=lands)
        uniq = []
        for d, s in cs:
            if all(np.abs(d - u).sum() > 0 for u, _ in uniq):
                uniq.append((d, s))
        out.append(uniq[:k])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--ens", required=True)
    ap.add_argument("--kind", default="sealed")
    ap.add_argument("--split", default="val")
    ap.add_argument("--pools", type=int, default=500)
    ap.add_argument("--cands", type=int, default=3)
    ap.add_argument("--opps", type=int, default=40, help="gauntlet decks; each candidate plays each twice (seats)")
    ap.add_argument("--sims", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--prior", type=float, default=1.0, help="weight of the network's score difference (0 = games only)")
    a = ap.parse_args()
    import multiprocessing as mp
    os.makedirs(a.out, exist_ok=True)
    pools = D.load_pools(a.kind, a.split)[: a.pools]
    train = D.load_pools("sealed", "train")
    rng = np.random.default_rng(7)
    gidx = rng.choice(len(train), a.opps, replace=False)
    chunks = [[p["pool_v"] for p in pools[i::a.procs]] for i in range(a.procs)]
    gch = [[train[k]["pool_v"] for k in gidx[i::a.procs]] for i in range(a.procs)]
    with mp.get_context("spawn").Pool(a.procs) as pool:
        res = pool.map(_cands, [(a.ens, c, a.cands, i) for i, c in enumerate(chunks)])
        gres = pool.map(_cands, [(a.ens, c, 1, i) for i, c in enumerate(gch)])
    cands = [None] * len(pools)
    for i in range(a.procs):
        for j, cs in enumerate(res[i]):
            cands[i + j * a.procs] = cs
    gaunt = [cs[0][0] for r in gres for cs in r]
    decks = {f"o{k}": d for k, d in enumerate(gaunt)}
    pairs = []
    for i, cs in enumerate(cands):
        for c, (d, _) in enumerate(cs):
            decks[f"p{i}c{c}"] = d
            if len(cs) == 1:
                continue
            for o in range(len(gaunt)):
                s = int(rng.integers(1, 2**62))
                pairs += [(f"p{i}c{c}", f"o{o}", s), (f"o{o}", f"p{i}c{c}", s)]
    D.write_decks(os.path.join(a.out, "playoff_decks.jsonl"), decks)
    with open(os.path.join(a.out, "playoff_pairs.tsv"), "w") as f:
        for x, y, s in pairs:
            f.write(f"{x}\t{y}\t{s}\n")
    print(f"{len(pools)} pools, {sum(len(c) for c in cands)} candidates, {len(pairs)} games", file=sys.stderr)
    gp = os.path.join(a.out, "playoff_games.jsonl")
    with open(os.path.join(a.out, "playoff.log"), "a") as pl:
        G.play(os.path.join(a.out, "playoff_decks.jsonl"), os.path.join(a.out, "playoff_pairs.tsv"), gp, sims=a.sims,
               workers=a.workers, chunk=max(a.workers * 200, 3000) if a.sims <= 0 else None, log=pl)
    rec = {}
    for g in G.results(gp):
        if g.get("winner", -1) not in (0, 1):
            continue
        for s in (0, 1):
            n = g["decks"][s]
            if n.startswith("p"):
                w, t = rec.get(n, (0, 0))
                rec[n] = (w + (g["winner"] == s), t + 1)
    built, changed, report = [], 0, []
    for i, cs in enumerate(cands):
        best, best_v = 0, -1e9
        for c, (d, s) in enumerate(cs):
            w, t = rec.get(f"p{i}c{c}", (0, 0))
            # posterior log-odds against the gauntlet: the network's score (relative to the first candidate) as a
            # prior with the weight of `prior` x 10 games, updated by this candidate's record
            prior_logit = a.prior * (s - cs[0][1])
            n0 = 10.0 * a.prior
            p0 = 1 / (1 + np.exp(-prior_logit))
            v = (w + n0 * p0) / (t + n0) if t + n0 > 0 else 0.5
            if v > best_v:
                best, best_v = c, v
        built.append(cs[best][0])
        changed += best != 0
        report.append({"pool": i, "chosen": best, "records": [rec.get(f"p{i}c{c}", (0, 0)) for c in range(len(cs))],
                       "scores": [float(s) for _, s in cs]})
    with open(os.path.join(a.out, "built.jsonl"), "w") as f:
        for d in built:
            f.write(json.dumps(C.dict(d)) + "\n")
    json.dump(report, open(os.path.join(a.out, "playoff_report.json"), "w"))
    print(f"playoff changed the network's choice in {changed} of {len(pools)} pools", file=sys.stderr)


if __name__ == "__main__":
    main()
