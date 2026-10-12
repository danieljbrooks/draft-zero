"""
Card audit: does a card the builder skips actually help or hurt with this pilot? A paired, controlled test.

For each of N builder decks that can play the card (its colours are among the deck's), a variant swaps the card in:
a spell replaces the deck's weakest (by the network's card quality) creature of mana value 2-3, a land replaces one
basic of the deck's most-played colour. Base and variant both play the same human decks from other val pools on the
same deals (both seats), so the win-rate difference is the card's effect for this pilot, free of pool and deal luck.

  python -m sealed.cardaudit OUT_DIR --built EVAL_DIR/built.jsonl --ens ENS.pt --card "Giada, Font of Hope" \
      --decks 120 --opps 2 --sims 100 --workers 15
"""
import argparse
import json
import math
import os
import sys

import numpy as np

from . import decks as D
from . import games as G
from . import model as M

C = D.C


def variant(d, card, q):
    c = C.index[card]
    e = d.copy()
    if C.is_land[c]:
        basics = [i for i in D.BASIC_IDX if e[i] > 0]
        out = max(basics, key=lambda i: e[i])
    else:
        cr = [i for i in np.flatnonzero(e) if C.is_creature[i] and 2 <= C.mv[i] <= 3 and i != c]
        if not cr:
            cr = [i for i in np.flatnonzero(e) if C.spell[i] and i != c]
        out = min(cr, key=lambda i: q[i])
    e[out] -= 1
    e[c] += 1
    return e, C.names[out]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--built", required=True)
    ap.add_argument("--ens", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--decks", type=int, default=120)
    ap.add_argument("--opps", type=int, default=2)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--workers", type=int, default=15)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ens = M.Ensemble.load(a.ens)
    q = np.mean([m.quality().detach().numpy() for m in ens.members], 0)
    built = [C.vec(json.loads(l)) for l in open(a.built)]
    pools = D.load_pools("sealed", "val")[: len(built)]
    c = C.index[a.card]
    need = C.makes[c] > 0 if C.is_land[c] else C.colors[c]
    picked = []
    for k, d in enumerate(built):
        if d[c] > 0:
            continue
        pips = (d[:, None] * C.pips * C.spell[:, None]).sum(0) > 0
        if C.is_land[c] and need.sum() >= 2:
            ok = (need & pips).sum() >= 2                     # a dual: the deck plays both its colours
        else:
            ok = not (need & ~pips).any()
        if ok:
            picked.append(k)
        if len(picked) >= a.decks:
            break
    decks, rows, swapped = {}, [], []
    rng = np.random.default_rng(31)
    for k in picked:
        v, out = variant(built[k], a.card, q)
        decks[f"b{k}"], decks[f"v{k}"] = built[k], v
        swapped.append(out)
        for j in range(1, a.opps + 1):
            o = (k + 17 * j) % len(pools)
            decks[f"h{o}"] = pools[o]["ref_v"]
            s = int(rng.integers(1, 2**62))
            for x in ("b", "v"):
                rows += [(f"{x}{k}", f"h{o}", s), (f"h{o}", f"{x}{k}", s)]
    D.write_decks(os.path.join(a.out, "decks.jsonl"), decks)
    with open(os.path.join(a.out, "pairs.tsv"), "w") as f:
        for x, y, s in rows:
            f.write(f"{x}\t{y}\t{s}\n")
    print(f"{a.card}: {len(picked)} decks, {len(rows)} games; swapped out {dict(zip(*np.unique(swapped, return_counts=True)))}",
          file=sys.stderr)
    gp = os.path.join(a.out, "games.jsonl")
    with open(os.path.join(a.out, "play.log"), "a") as pl:
        G.play(os.path.join(a.out, "decks.jsonl"), os.path.join(a.out, "pairs.tsv"), gp, sims=a.sims, workers=a.workers,
               log=pl)
    res = {}
    for g in G.results(gp):
        if g.get("winner", -1) not in (0, 1):
            continue
        for s in (0, 1):
            n = g["decks"][s]
            if n[0] in "bv":
                res.setdefault(int(n[1:]), {}).setdefault(n[0], []).append(1.0 if g["winner"] == s else 0.0)
    diffs = [np.mean(r["v"]) - np.mean(r["b"]) for r in res.values() if "v" in r and "b" in r]
    out = {"card": a.card, "decks": len(diffs), "games": sum(len(x) for r in res.values() for x in r.values()),
           "base": float(np.mean([np.mean(r["b"]) for r in res.values() if "b" in r])),
           "with_card": float(np.mean([np.mean(r["v"]) for r in res.values() if "v" in r])),
           "effect": float(np.mean(diffs)), "se": float(np.std(diffs, ddof=1) / math.sqrt(len(diffs)))}
    json.dump(out, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    print(json.dumps(out), file=sys.stderr)


if __name__ == "__main__":
    main()
