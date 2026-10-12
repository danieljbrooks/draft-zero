"""
Score one builder on held-out pools: card agreement with the human decks, and (with --sims) the win rate against
them in gorge.

  python -m sealed.evalbuild OUT_DIR --builder ens:data/sealed/A/gen9/ens.pt --kind sealed --split val --pools 500 \
      --sims 100 --workers 15

Builders: `random`, `gih` (17lands games-in-hand win rate, best two colours, top 23: uses human data), `ens:PATH`
(a deck value ensemble with the search builder; `ens:PATH:cautious` builds for the mean minus one standard deviation).
"""
import argparse
import json
import os
import sys

import numpy as np

from . import decks as D
from . import evaluate as E
from . import games as G
from . import loop as L

C = D.C


def build_all(builder, pools, procs=4, seed=5):
    if builder == "random":
        rng = np.random.default_rng(seed)
        return [D.random_build(p["pool_v"], rng) for p in pools]
    if builder == "gih":
        gih = json.load(open(os.path.join(D.cardsmod.REPO, "assets/reference/FDN_card_stats.json")))["cards"]
        r = np.array([gih.get(n, {}).get("gih_wr", np.nan) for n in C.names])
        return [D.rating_build(p["pool_v"], r) for p in pools]
    if builder.startswith("ens:"):
        path = builder[4:]
        kind = "b"
        if path.endswith(":cautious"):
            path, kind = path[: -len(":cautious")], "c"
        built = L.build_many(path, [(f"v{k}", p["pool_v"], kind) for k, p in enumerate(pools)], procs, seed)
        return [built[f"v{k}.{kind}"] for k in range(len(pools))]
    raise ValueError(builder)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--builder", required=True)
    ap.add_argument("--kind", default="sealed")
    ap.add_argument("--split", default="val")
    ap.add_argument("--pools", type=int, default=500)
    ap.add_argument("--sims", type=int, default=None, help="play the win-rate match at this search level")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pools = D.load_pools(a.kind, a.split)[: a.pools]
    dp = os.path.join(a.out, "built.jsonl")
    if os.path.exists(dp):
        built = [C.vec(json.loads(l)) for l in open(dp)]
    else:
        built = build_all(a.builder, pools, a.procs)
        with open(dp, "w") as f:
            for d in built:
                f.write(json.dumps(C.dict(d)) + "\n")
    res = {"builder": a.builder, "kind": a.kind, "split": a.split, "agreement": E.agreement(built, pools)}
    if a.sims is not None:
        E.write_match(a.out, built, pools)
        with open(os.path.join(a.out, "play.log"), "a") as pl:
            G.play(os.path.join(a.out, "decks.jsonl"), os.path.join(a.out, "pairs.tsv"),
                   os.path.join(a.out, "games.jsonl"), sims=a.sims, workers=a.workers, log=pl)
        res["sims"] = a.sims
        res["vs_human"] = E.winrate(os.path.join(a.out, "games.jsonl"))
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    ag = res["agreement"]
    print(f"{a.builder} {a.kind}/{a.split}: overlap {ag['overlap']:.1f}/40, spells {ag['spells']:.1f}, "
          f"colours {ag['main_colours_match']:.2f}, lands {ag['lands_mean']:.1f}"
          + (f"; vs human at {a.sims}: {res['vs_human']['winrate']:.3f} ± {res['vs_human']['se']:.3f} of "
             f"{res['vs_human']['games']}" if "vs_human" in res else ""), file=sys.stderr)


if __name__ == "__main__":
    main()
