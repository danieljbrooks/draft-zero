"""
Scoring a deck builder against the humans' decks from the same pools.

**Agreement** (no games): per pool, the cards the builder's deck shares with the human reference deck (multiset
intersection, out of 40), the non-land cards shared, whether the main colours match, and the land count.

**Win rate** (games in gorge): pools are paired (i, j); the builder's deck from pool i plays the human deck from pool
j and the builder's deck from pool j plays the human deck from pool i, each twice with the seats swapped on one seed:
4 games a pool pair. A builder that builds exactly like the humans scores exactly 50%. The standard error is taken
over pool pairs.
"""
import json
import math
import os

import numpy as np

from . import decks as D

C = D.C


def agreement(built, pools):
    """built: list of count vectors, aligned with pools (rows of load_pools)."""
    ov, sp, col, lands, hl, ncol = [], [], [], [], [], []
    for d, p in zip(built, pools):
        h = p["ref_v"]
        ov.append(D.overlap(d, h)); sp.append(D.overlap(d, h, spells_only=True))
        a, b = D.deck_colors(d), D.deck_colors(h)
        col.append(set(a.split("+")[0]) == set(b.split("+")[0]))
        lands.append(d[C.is_land].sum()); hl.append(h[C.is_land].sum())
        ncol.append(len(a.split("+")[0]))
    n = len(ov)
    return {"pools": n, "overlap": float(np.mean(ov)), "overlap_se": float(np.std(ov) / math.sqrt(n)),
            "spells": float(np.mean(sp)), "human_spells": float(np.mean([p["ref_v"][C.spell].sum() for p in pools])),
            "main_colours_match": float(np.mean(col)), "lands_mean": float(np.mean(lands)),
            "lands_hist": {int(k): int(v) for k, v in zip(*np.unique(lands, return_counts=True))},
            "main_colours_hist": {int(k): int(v) for k, v in zip(*np.unique(ncol, return_counts=True))},
            "overlap_hist": {int(k): int(v) for k, v in zip(*np.unique(np.round(ov), return_counts=True))}}


def write_match(out_dir, built, pools, tag="m", seed=3232):
    """Decks (tag<i> = the builder's, h<i> = the human's) and the crossed pairs for a win-rate match."""
    os.makedirs(out_dir, exist_ok=True)
    decks = {}
    for k, (d, p) in enumerate(zip(built, pools)):
        decks[f"{tag}{k}"] = d; decks[f"h{k}"] = p["ref_v"]
    D.write_decks(os.path.join(out_dir, "decks.jsonl"), decks)
    rng = np.random.default_rng(seed)
    with open(os.path.join(out_dir, "pairs.tsv"), "w") as f:
        for i in range(0, len(pools) - 1, 2):
            j = i + 1
            for a, b in ((i, j), (j, i)):
                s = int(rng.integers(1, 2**62))
                f.write(f"{tag}{a}\th{b}\t{s}\nh{b}\t{tag}{a}\t{s}\n")


def winrate(games_path, tag="m"):
    """The builder's decks' share of wins against the human decks, SE over pool pairs (decided games only)."""
    by = {}
    n_games = 0
    for l in open(games_path):
        g = json.loads(l)
        if g.get("winner", -1) not in (0, 1):
            continue
        a, b = g["decks"]
        mine = 0 if a.startswith(tag) else 1
        won = 1.0 if g["winner"] == mine else 0.0
        k = int((a if mine == 0 else b)[len(tag):])
        by.setdefault(k // 2, []).append(won)
        n_games += 1
    per = np.array([np.mean(v) for v in by.values()])
    if not len(per):
        return {"winrate": None, "games": 0}
    return {"winrate": float(per.mean()), "se": float(per.std(ddof=1) / math.sqrt(len(per))) if len(per) > 1 else None,
            "games": n_games, "pool_pairs": len(per)}
