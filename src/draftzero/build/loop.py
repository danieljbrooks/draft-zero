"""
The self-play loop for the deck builder. Each generation:

1. **Build** decks for a sample of train pools with the current ensemble (generation 0: random deck-shaped builds,
   wide: 1-5 colours, 12-21 lands). Four decks a pool:
   - `b`, the best deck under the ensemble's mean (exploit);
   - `t`, the best deck under one random member (Thompson sampling: explore where the members disagree);
   - `p`, `b` with 1-3 random swaps (explore around the best);
   - `r`, a wide random build with probability `--random-share`, else another `t`.
2. **Play:** decks from different pools paired at random, each pairing twice with the seats swapped on one seed, at
   the loop's search level (`--sims`; 0 is gorge's heuristic bot).
3. **Train** a new ensemble on the last `--window` generations' games.
4. **Score:** agreement with the human decks on val pools (no games) and, every `--eval-every` generations, the
   win rate of the `b` decks against the human decks on val pools at the loop's search level.

  python -m draftzero.build.loop OUT_DIR --gens 10 --sims 0 --pools 800 --games 40000 --workers 4 --procs 4
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

from . import build as Bd
from . import decks as D
from . import evaluate as E
from . import games as G
from . import model as M
from . import selfplay as S
from . import train as T

C = D.C


def _build_worker(args):
    ens_path, jobs, seed, random_share = args
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    ens = M.Ensemble.load(ens_path)
    q = np.mean([m.quality().detach().numpy() for m in ens.members], 0)
    mean = Bd.Objective(ens, "mean")
    lands = ens.meta.get("lands")
    out = {}
    for name, pool, kinds in jobs:
        best = None
        for kind in kinds:
            if kind == "b":
                best = d = Bd.build(pool, mean, q, lands=lands)
            elif kind == "c":     # cautious: the mean less one standard deviation across members
                d = Bd.build(pool, Bd.Objective(ens, "cautious", k=1.0), q, lands=lands)
            elif kind == "k":     # the best deck in one random colour pair: a colour-diverse league of opponents
                pairs = [c for c in Bd.CONFIGS if c.sum() == 2]
                d = Bd.build(pool, mean, q, n_refine=1, lands=lands, configs=[pairs[int(rng.integers(len(pairs)))]])
            elif kind == "l":     # the best deck with 1-3 lands more or fewer: explore the land count's edges
                b0 = best if best is not None else Bd.build(pool, mean, q, lands=lands)
                d = D.shift_lands(b0, pool, int(rng.choice([-3, -2, -1, 1, 2, 3])), q)
            elif kind == "t" or (kind == "r" and rng.random() >= random_share):
                k = int(rng.integers(len(ens.members)))
                qk = ens.members[k].quality().detach().numpy()
                d = Bd.build(pool, Bd.Objective(ens, member=k), qk, n_refine=2, lands=lands)
            elif kind == "p":
                d = D.perturb(best if best is not None else Bd.build(pool, mean, q, lands=lands), pool, rng,
                              int(rng.integers(1, 4)))
            else:
                d = D.random_build(pool, rng, wide=True)
            out[f"{name}.{kind}"] = d
    return out


def build_many(ens_path, items, procs, seed, random_share=0.15):
    """items: (name, pool vector, kinds) -> {name.kind: deck}, built in `procs` processes."""
    chunks = [items[i::procs] for i in range(procs)]
    with mp.get_context("spawn").Pool(procs) as pool:
        res = pool.map(_build_worker, [(ens_path, c, seed + i, random_share) for i, c in enumerate(chunks)])
    out = {}
    for r in res:
        out.update(r)
    return out


def log_to(path):
    f = open(path, "a")

    def log(s):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {s}"
        print(line, file=sys.stderr, flush=True)
        f.write(line + "\n"); f.flush()
    return log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--sims", type=int, default=0)
    ap.add_argument("--pools", type=int, default=800)
    ap.add_argument("--games", type=int, default=40000)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4, help="gorge game workers")
    ap.add_argument("--procs", type=int, default=4, help="deck-building processes")
    ap.add_argument("--random-share", type=float, default=0.15)
    ap.add_argument("--start", default=None, help="an ensemble to start from (a curriculum's previous phase)")
    ap.add_argument("--prior-gens", nargs="*", default=[], help="generation dirs from a previous phase for the window")
    ap.add_argument("--warm", action="store_true", help="train each generation's members from the previous ensemble "
                    "(a lower learning rate) instead of from scratch")
    ap.add_argument("--warm-lr", type=float, default=1e-3)
    ap.add_argument("--draws", action="store_true", help="train with the games-in-hand draw term (model.loss_fn)")
    ap.add_argument("--castable", action="store_true", help="weight each spell by how castable the deck's mana makes it "
                    "(model.DeckNet.castability)")
    ap.add_argument("--kinds", default="btpr", help="deck variants per pool: b best, t Thompson, p perturbed, "
                    "r random (or t), l land-shifted, c cautious, k best in a random colour pair")
    ap.add_argument("--support", type=float, default=0.02, help="land counts with at least this share of the "
                    "window's deck-games are the builder's range (model.land_support); 0 = no range")
    ap.add_argument("--val-pools", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--eval-pools", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--remotes", default=None, help="JSON list of remote game machines (games.play_distributed); "
                    "re-read every generation, so machines can join or leave")
    a = ap.parse_args()
    torch.set_num_threads(a.procs)
    os.makedirs(a.out, exist_ok=True)
    log = log_to(os.path.join(a.out, "loop.log"))
    log(f"start: {vars(a)}")
    train = D.load_pools("sealed", "train")
    val = D.load_pools("sealed", "val")
    rng = np.random.default_rng(a.seed)
    gens = [g for g in a.prior_gens]
    ens_path = a.start
    for gen in range(a.gens):
        gd = os.path.join(a.out, f"gen{gen}")
        met_path = os.path.join(gd, "metrics.json")
        if os.path.exists(met_path):       # resume
            gens.append(gd); ens_path = os.path.join(gd, "ens.pt"); continue
        t0 = time.time()
        pick = rng.choice(len(train), a.pools, replace=False)
        if not os.path.exists(os.path.join(gd, "pairs.tsv")):
            if ens_path is None:
                r = np.random.default_rng(a.seed * 1000 + gen)
                decks = {f"g{gen}.{k}.r{v}": D.random_build(train[k]["pool_v"], r, wide=True) for k in pick for v in range(4)}
            else:
                items = [(f"g{gen}.{k}", train[k]["pool_v"], a.kinds) for k in pick]
                decks = build_many(ens_path, items, a.procs, a.seed * 1000 + gen, a.random_share)
            pool_of = {n: n.split(".")[1] for n in decks}
            S.write_gen(gd, decks, pool_of, a.games, a.seed * 1000 + gen)
        log(f"gen {gen}: {a.pools} pools built in {time.time() - t0:.0f}s; playing {a.games} games at sims={a.sims}")
        t1 = time.time()
        remotes = json.load(open(a.remotes)) if a.remotes and os.path.exists(a.remotes) else []
        with open(os.path.join(gd, "play.log"), "a") as pl:
            G.play_distributed(os.path.join(gd, "decks.jsonl"), os.path.join(gd, "pairs.tsv"),
                               os.path.join(gd, "games.jsonl"), sims=a.sims, workers=a.workers, remotes=remotes, log=pl,
                               chunk=max(a.workers * 200, 3000) if a.sims == 0 else None)
        log(f"gen {gen}: games in {time.time() - t1:.0f}s")
        gens.append(gd)
        t2 = time.time()
        if a.warm and ens_path:
            ens, rep = T.fit(gens[-a.window:], a.members, log=log, init=M.Ensemble.load(ens_path), lr=a.warm_lr,
                             draws=a.draws, castable=a.castable)
        else:
            ens, rep = T.fit(gens[-a.window:], a.members, log=log, draws=a.draws, castable=a.castable)
        if a.support:
            ens.meta["lands"] = M.land_support(S.load_games([os.path.join(g, "games.jsonl") for g in gens[-a.window:]]),
                                               S.load_decks([os.path.join(g, "decks.jsonl") for g in gens[-a.window:]]),
                                               a.support)
            log(f"gen {gen}: builder's land range {ens.meta['lands']}")
        ens.save(os.path.join(gd, "ens.pt"))
        ens_path = os.path.join(gd, "ens.pt")
        log(f"gen {gen}: trained in {time.time() - t2:.0f}s")
        # agreement on val pools (no games)
        vp = val[: a.val_pools]
        built = build_many(ens_path, [(f"v{k}", p["pool_v"], "b") for k, p in enumerate(vp)], a.procs, 7)
        vb = [built[f"v{k}.b"] for k in range(len(vp))]
        met = {"gen": gen, "sims": a.sims, "train": rep, "agreement": E.agreement(vb, vp)}
        if a.eval_every and (gen + 1) % a.eval_every == 0:
            ep = val[: a.eval_pools]
            built = build_many(ens_path, [(f"v{k}", p["pool_v"], "b") for k, p in enumerate(ep)], a.procs, 7)
            md = os.path.join(gd, "vs_human")
            E.write_match(md, [built[f"v{k}.b"] for k in range(len(ep))], ep)
            with open(os.path.join(md, "play.log"), "a") as pl:
                G.play(os.path.join(md, "decks.jsonl"), os.path.join(md, "pairs.tsv"), os.path.join(md, "games.jsonl"),
                       sims=a.sims, workers=a.workers, chunk=max(a.workers * 200, 3000) if a.sims == 0 else None, log=pl)
            met["vs_human"] = E.winrate(os.path.join(md, "games.jsonl"))
        json.dump(met, open(met_path, "w"), indent=1)
        ag = met["agreement"]
        log(f"gen {gen}: overlap {ag['overlap']:.1f}/40, spells {ag['spells']:.1f}, colours {ag['main_colours_match']:.2f}, "
            f"lands {ag['lands_mean']:.1f}, held-out loss {rep['logloss']:.4f} (base {rep['base_logloss']:.4f})"
            + (f", vs human {met['vs_human']['winrate']:.3f} of {met['vs_human']['games']}" if "vs_human" in met else "")
            + f"; {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
