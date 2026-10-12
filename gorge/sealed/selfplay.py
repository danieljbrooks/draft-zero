"""
Self-play data for the deck builder: decks built from train pools play each other in gorge.

A generation directory holds decks.jsonl (gorge deck files), pairs.tsv (deckA, deckB, seed: every pairing is played
twice on one seed with the seats swapped) and games.jsonl (results). Deck names are `<gen>.<pool>.<variant>`.

  python -m sealed.selfplay decks GEN_DIR --builder random --pools 3000 --variants 2 --games 20000
"""
import argparse
import json
import os
import sys

import numpy as np

from . import decks as D

C = D.C


def pair_up(names, pool_of, n_games, rng):
    """n_games/2 pairings of decks from different pools, each played twice (seats swapped, one seed)."""
    names = list(names)
    out = []
    while len(out) < n_games:
        a, b = rng.choice(len(names), 2, replace=False)
        if pool_of[names[a]] == pool_of[names[b]]:
            continue
        s = int(rng.integers(1, 2**62))
        out += [(names[a], names[b], s), (names[b], names[a], s)]
    return out


def write_gen(gen_dir, decks, pool_of, n_games, seed):
    os.makedirs(gen_dir, exist_ok=True)
    D.write_decks(os.path.join(gen_dir, "decks.jsonl"), decks)
    rng = np.random.default_rng(seed)
    with open(os.path.join(gen_dir, "pairs.tsv"), "w") as f:
        for a, b, s in pair_up(decks, pool_of, n_games, rng):
            f.write(f"{a}\t{b}\t{s}\n")
    json.dump({"decks": len(decks), "games": n_games, "seed": seed}, open(os.path.join(gen_dir, "meta.json"), "w"))


def load_decks(paths):
    """name -> count vector, from one or more decks.jsonl files."""
    out = {}
    for p in paths:
        for l in open(p):
            d = json.loads(l)
            out[d["name"]] = C.vec({c["name"]: c["count"] for c in d["cards"]})
    return out


def load_games(paths):
    out = []
    for p in paths:
        if os.path.exists(p):
            out += [json.loads(l) for l in open(p) if l.strip()]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gen_dir")
    ap.add_argument("--pools", type=int, default=3000)
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    train = D.load_pools("sealed", "train")
    pick = rng.choice(len(train), min(a.pools, len(train)), replace=False)
    tag = os.path.basename(a.gen_dir.rstrip("/"))
    decks, pool_of = {}, {}
    for k in pick:
        for v in range(a.variants):
            n = f"{tag}.{k}.r{v}"
            decks[n] = D.random_build(train[k]["pool_v"], rng)
            pool_of[n] = k
    write_gen(a.gen_dir, decks, pool_of, a.games, a.seed)
    print(f"{len(decks)} decks, {a.games} games -> {a.gen_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
