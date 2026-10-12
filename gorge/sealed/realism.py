"""
The realism ladder: how human-like is play at each search depth? (Dan, 12 October.)

The pilot alone, no deck building. Human reference decks from sealed train pools, paired at random across pools (one
fixed list of pairings and seeds), are played at several search depths. Per level:

- **card win rates:** each card's games-in-hand win rate in these games against 17lands' (rank correlation over the
  spells drawn at least `min_drawn` times);
- **colour win rates:** each main-colour pair's win rate against the pair's win rate in 17lands' sealed games;
- **rewarding better building:** whether decks built by stronger players (17lands' player win-rate bucket) win more.

  python -m sealed.realism setup OUT_DIR --games 2000            # decks.jsonl, pairs.tsv (seed 3333)
  python -m sealed.realism analyse OUT_DIR/s25 OUT_DIR/s100 ...   # one games.jsonl per level directory
"""
import argparse
import collections
import json
import math
import os
import sys

import numpy as np
from scipy.stats import spearmanr

from . import decks as D

C = D.C


def setup(out, n_games, seed=3333):
    """n_games/2 pairings of human decks from different train pools, each played twice with the seats swapped."""
    os.makedirs(out, exist_ok=True)
    pools = D.load_pools("sealed", "train")
    rng = np.random.default_rng(seed)
    decks, rows, meta = {}, [], {}
    for _ in range(n_games // 2):
        i, j = (int(x) for x in rng.choice(len(pools), 2, replace=False))
        for k in (i, j):
            decks[f"h{k}"] = pools[k]["ref_v"]
            ref = next(b for b in pools[k]["builds"] if b["build_index"] == pools[k]["ref"])
            meta[f"h{k}"] = {"user_wr": pools[k]["user_wr"],
                             "main": "".join(sorted(ref["main_colors"] or "", key="WUBRG".index))}
        s = int(rng.integers(1, 2**62))
        rows += [(f"h{i}", f"h{j}", s), (f"h{j}", f"h{i}", s)]
    D.write_decks(os.path.join(out, "decks.jsonl"), decks)
    with open(os.path.join(out, "pairs.tsv"), "w") as f:
        for a, b, s in rows:
            f.write(f"{a}\t{b}\t{s}\n")
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"))
    print(f"{len(decks)} decks, {len(rows)} games -> {out}", file=sys.stderr)


def colour_reference():
    """Each main-colour pair's game win rate in 17lands' sealed data (all builds, Bo1 and Bo3)."""
    w, n = collections.Counter(), collections.Counter()
    for line in open(os.path.join(D.cardsmod.REPO, "data/sealed/sealed.jsonl")):
        p = json.loads(line)
        for b in p["builds"]:
            k = "".join(sorted(b["main_colors"] or "", key="WUBRG".index))
            w[k] += b["wins"]; n[k] += b["games"]
    return {k: w[k] / n[k] for k in n if n[k] >= 500}


def analyse(dirs, min_drawn=60, limit=None):
    gih = json.load(open(os.path.join(D.cardsmod.REPO, "assets/reference/FDN_card_stats.json")))["cards"]
    ref = np.array([gih.get(x, {}).get("gih_wr", np.nan) for x in C.names])
    cref = colour_reference()
    out = []
    for d in dirs:
        meta = json.load(open(os.path.join(os.path.dirname(d.rstrip("/")), "meta.json")))
        games = [json.loads(l) for l in open(os.path.join(d, "games.jsonl")) if l.strip()]
        games = sorted(games, key=lambda g: g.get("line", 0))[:limit] if limit else games
        w = np.zeros(len(C)); n = np.zeros(len(C))
        cw, cn = collections.Counter(), collections.Counter()
        uw = collections.defaultdict(list)
        turns, decided = [], 0
        for g in games:
            if g.get("winner", -1) not in (0, 1):
                continue
            decided += 1; turns.append(g.get("turns", 0))
            for s in (0, 1):
                won = g["winner"] == s
                dk = g["decks"][s]
                for c in set(g.get("in_hand", [[], []])[s]):
                    i = C.index.get(c)
                    if i is not None:
                        n[i] += 1; w[i] += won
                m = meta.get(dk, {})
                cw[m.get("main", "")] += won; cn[m.get("main", "")] += 1
                if m.get("user_wr") is not None:
                    uw[m["user_wr"] >= 0.6].append(won)
        ok = C.spell & (n >= min_drawn) & ~np.isnan(ref)
        rho = spearmanr(w[ok] / n[ok], ref[ok]).correlation
        # noise ceiling: split-half reliability of the card win rates (pairings split by index parity, both seats of
        # a pairing in one half), Spearman-Brown corrected; a correlation with any reference is bounded by sqrt(rel)
        hw = np.zeros((2, len(C))); hn = np.zeros((2, len(C)))
        for g in games:
            if g.get("winner", -1) not in (0, 1):
                continue
            h = (g.get("line", 0) // 2) % 2
            for s in (0, 1):
                for c in set(g.get("in_hand", [[], []])[s]):
                    i = C.index.get(c)
                    if i is not None:
                        hn[h, i] += 1; hw[h, i] += g["winner"] == s
        ok2 = ok & (hn.min(0) >= min_drawn / 2)
        r_half = spearmanr(hw[0, ok2] / hn[0, ok2], hw[1, ok2] / hn[1, ok2]).correlation
        rel = 2 * r_half / (1 + r_half) if r_half > -1 else float("nan")
        ceiling = math.sqrt(max(rel, 0))
        keys = [k for k in cref if len(k) == 2 and cn[k] >= 100]
        crho = spearmanr([cw[k] / cn[k] for k in keys], [cref[k] for k in keys]).correlation
        strong, weak = np.mean(uw[True]), np.mean(uw[False])
        se = math.sqrt(strong * (1 - strong) / len(uw[True]) + weak * (1 - weak) / len(uw[False]))
        row = {"level": os.path.basename(d.rstrip("/")), "games": decided, "mean_turns": round(float(np.mean(turns)), 1),
               "card_rho": round(float(rho), 3), "cards": int(ok.sum()), "card_ceiling": round(float(ceiling), 3), "colour_rho": round(float(crho), 3),
               "colour_pairs": len(keys), "strong_minus_weak": round(float(strong - weak), 4),
               "strong_minus_weak_se": round(se, 4),
               "colour_wr": {k: round(cw[k] / cn[k], 3) for k in keys}}
        out.append(row)
        print(json.dumps(row))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["setup", "analyse"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--min-drawn", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None, help="analyse only the first N games of each level (to match n)")
    a = ap.parse_args()
    if a.what == "setup":
        setup(a.paths[0], a.games)
    else:
        analyse(a.paths, a.min_drawn, a.limit)


if __name__ == "__main__":
    main()
