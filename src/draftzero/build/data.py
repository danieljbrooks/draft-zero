"""
Sealed (and draft) pools with the decks humans built from them, from 17lands' public game data (CC BY 4.0).

A 17lands game row lists the player's deck (`deck_*`) and the rest of their pool (`sideboard_*`); basic lands appear
in decks only. A pool is the deck's non-basic cards plus the sideboard. A player may rebuild a pool several times
(`build_index`); every build is kept with its games and wins, and the build with the most games is the pool's
*reference deck* (ties go to the earlier build).

Splits are fixed by a hash of the pool id: for sealed, 1,000 val and 1,000 test pools, the rest train; for draft
(the secondary test), 1,000 val and 1,000 test pools sampled from Premier Draft. Only val and test decks are used
to evaluate; the self-play builder never reads any deck, only train pools' contents.

  python -m draftzero.build.data        # -> data/sealed/{sealed,draft}.jsonl
"""
import argparse
import hashlib
import json
import os
import sys

import pandas as pd

from draftzero.data.paths import REPO as _REPO

REPO = str(_REPO)
BASICS = ("Plains", "Island", "Swamp", "Mountain", "Forest")
DATA = os.path.expanduser("~/Desktop/Data")
SOURCES = {
    "sealed": [f"{DATA}/Sealed_FDN/game_data_public.FDN.Sealed.csv.gz",
               f"{DATA}/Sealed_FDN/game_data_public.FDN.TradSealed.csv.gz"],
    "draft": [f"{DATA}/Premier_FDN/game_data_public.FDN.PremierDraft.csv.gz"],
}


def h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:12], 16)


def builds(path: str):
    hdr = pd.read_csv(path, nrows=0).columns
    deck = [c for c in hdr if c.startswith("deck_")]
    sb = [c for c in hdr if c.startswith("sideboard_")]
    meta = ["event_type", "draft_id", "build_index", "won", "main_colors", "splash_colors",
            "user_game_win_rate_bucket", "user_n_games_bucket"]
    for chunk in pd.read_csv(path, usecols=meta + deck + sb, chunksize=200_000):
        g = chunk.groupby(["draft_id", "build_index"], sort=False)
        agg = g.agg(games=("won", "size"), wins=("won", "sum"))
        first = g.head(1).set_index(["draft_id", "build_index"])
        for key, row in first.iterrows():
            d = {c[5:]: int(row[c]) for c in deck if row[c] > 0}
            s = {c[10:]: int(row[c]) for c in sb if row[c] > 0 and c[10:] not in BASICS}
            yield {
                "event": row["event_type"], "pool_id": key[0], "build_index": int(key[1]),
                "deck": d, "side": s,
                "games": int(agg.loc[key, "games"]), "wins": int(agg.loc[key, "wins"]),
                "main_colors": row["main_colors"] if isinstance(row["main_colors"], str) else "",
                "splash_colors": row["splash_colors"] if isinstance(row["splash_colors"], str) else "",
                "user_wr": None if pd.isna(row["user_game_win_rate_bucket"]) else float(row["user_game_win_rate_bucket"]),
                "user_games": None if pd.isna(row["user_n_games_bucket"]) else int(row["user_n_games_bucket"]),
            }


def pools(kind: str):
    by = {}
    for path in SOURCES[kind]:
        for b in builds(path):
            p = by.setdefault(b["pool_id"], {"pool_id": b["pool_id"], "event": b["event"], "builds": {}})
            # a build seen in two chunks: add its games
            o = p["builds"].get(b["build_index"])
            if o:
                o["games"] += b["games"]; o["wins"] += b["wins"]
            else:
                p["builds"][b["build_index"]] = b
    out = []
    for p in by.values():
        bl = sorted(p["builds"].values(), key=lambda b: b["build_index"])
        ref = max(bl, key=lambda b: (b["games"], -b["build_index"]))
        pool = {c: n for c, n in ref["deck"].items() if c not in BASICS}
        for c, n in ref["side"].items():
            pool[c] = pool.get(c, 0) + n
        out.append({
            "pool_id": p["pool_id"], "event": p["event"], "pool": dict(sorted(pool.items())),
            "ref": ref["build_index"], "user_wr": ref["user_wr"], "user_games": ref["user_games"],
            "builds": [{k: b[k] for k in ("build_index", "deck", "games", "wins", "main_colors", "splash_colors")}
                       for b in bl],
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "data/sealed"))
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--test", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for kind in ("sealed", "draft"):
        ps = sorted(pools(kind), key=lambda p: h(kind + p["pool_id"]))
        for i, p in enumerate(ps):
            p["split"] = "test" if i < args.test else "val" if i < args.test + args.val else "train"
        if kind == "draft":   # the secondary test: keep only the sampled val and test pools
            ps = [p for p in ps if p["split"] != "train"]
        with open(os.path.join(args.out, f"{kind}.jsonl"), "w") as f:
            for p in ps:
                f.write(json.dumps(p) + "\n")
        n = {s: sum(p["split"] == s for p in ps) for s in ("train", "val", "test")}
        sizes = pd.Series([sum(p["pool"].values()) for p in ps])
        print(f"{kind}: {len(ps):,} pools {n}; pool size median {sizes.median():.0f} "
              f"(min {sizes.min()}, max {sizes.max()})", file=sys.stderr)


if __name__ == "__main__":
    main()
