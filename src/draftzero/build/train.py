"""
Train the deck value ensemble on self-play generations.

  python -m draftzero.build.train OUT.pt GEN_DIR [GEN_DIR ...] [--members 5]

5% of pairings (by seed) are held out for early stopping and reported: the log-loss and accuracy of the predicted
winner against a model that knows only who played first.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

from . import model as M
from . import selfplay as S

C = M.C


def split_games(games, frac=0.05):
    tr, va = [], []
    for g in games:
        (va if (g.get("seed", 0) * 2654435761 % 1000) < frac * 1000 else tr).append(g)
    return tr, va


def evaluate(ens, games, decks):
    """Held-out log-loss and accuracy of the ensemble's mean prediction, and the log-loss of knowing only who played
    first; games 0 (and None for the rest) without a decided game."""
    if not any(g.get("winner", -1) in (0, 1) and all(d in decks for d in g["decks"]) for g in games):
        return {"games": 0, "logloss": None, "base_logloss": None, "acc": None}
    A, B, P, Y = M.games_tensors(games, decks)
    with torch.no_grad():
        s = np.mean([(m(A) - m(B) + m.beta * P).numpy() for m in ens.members], 0)
    p = 1 / (1 + np.exp(-s))
    y = Y.numpy()
    ll = float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))
    pf = np.clip(np.mean(y[P.numpy() > 0]) if (P.numpy() > 0).any() else 0.5, 0.01, 0.99)
    base_p = np.where(P.numpy() > 0, pf, 1 - pf)
    base = float(-np.mean(y * np.log(base_p) + (1 - y) * np.log(1 - base_p)))
    return {"games": int(len(y)), "logloss": ll, "base_logloss": base, "acc": float(np.mean((p > 0.5) == (y > 0.5)))}


def fit(gen_dirs, members=5, epochs=40, init=None, log=print, weights=None, draws=False, castable=False, **kw):
    decks = S.load_decks([os.path.join(g, "decks.jsonl") for g in gen_dirs])
    games = S.load_games([os.path.join(g, "games.jsonl") for g in gen_dirs])
    return fit_games(decks, games, members, epochs, init=init, log=log, draws=draws, castable=castable, **kw)


def fit_games(decks, games, members=5, epochs=40, init=None, log=print, draws=False, castable=False, **kw):
    """An ensemble of `members` DeckNets on `games` (dicts with decks, winner, starter, seed) between `decks`
    (name -> count vector), each on a bootstrap resample when members > 1, early-stopped on the held-out 5% of
    pairings (split_games); member k starts from init's member k mod its size. Only decided games between known
    decks count (a draw, stall or skipped game says nothing here). -> (Ensemble, held-out report)."""
    games = [g for g in games if g.get("winner", -1) in (0, 1) and all(d in decks for d in g.get("decks", ()))]
    tr, va = split_games(games)
    if not tr:
        raise ValueError(f"no decided game to train on ({len(va)} held out)")
    log(f"{len(games):,} games ({len(tr):,} train, {len(va):,} held out), {len(decks):,} decks")
    ms = []
    for k in range(members):
        t0 = time.time()
        m = M.train(tr, decks, va, seed=k, epochs=epochs, bootstrap=members > 1,
                    init=init.members[k % len(init.members)] if init else None, draws=draws, castable=castable, **kw)
        log(f"member {k}: held-out loss {m.val_loss:.4f} ({time.time() - t0:.0f}s)")
        ms.append(m)
    ens = M.Ensemble(ms)
    rep = evaluate(ens, va, decks)
    log(json.dumps(rep))
    return ens, rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out"); ap.add_argument("gens", nargs="+")
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--reg", type=float, default=1e-3)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
    ens, rep = fit(a.gens, a.members, a.epochs, lr=a.lr, reg=a.reg, log=lambda s: print(s, file=sys.stderr, flush=True))
    ens.save(a.out)
    json.dump(rep, open(a.out + ".json", "w"))


if __name__ == "__main__":
    main()
