"""
Building a deck from a pool with a deck value network, the way a person does:

1. **Colours:** for every colour configuration (5 mono, 10 pairs, 10 triples) a quick deck: the configuration's
   castable spells ranked by the network's card quality, its non-basic lands, basics split by pips. Each is scored
   by the full network.
2. **Candidates:** the best few configurations are refined by steepest-ascent swaps: any card out (basics included),
   any card in (a pool card not yet used up, or any basic), keeping 40 cards, until no swap raises s by more than
   `eps`. Swaps can splash a card, change the land count or swap a basic's colour. This is the iterative "rank every
   card in the context of the current deck" step, repeated to convergence.
3. **Choice:** the candidate with the highest s.

The objective is a function of the ensemble's scores: the mean (`objective="mean"`), the mean minus k standard
deviations (cautious: avoid decks the networks disagree on), or one member (Thompson sampling for exploration).
"""
import itertools

import numpy as np

from . import decks as D

C = D.C
BASIC_IDX = D.BASIC_IDX
CONFIGS = [np.array([k in c for k in range(5)]) for r in (1, 2, 3) for c in itertools.combinations(range(5), r)]


class Objective:
    def __init__(self, ens, kind="mean", k=0.0, member=None):
        self.ens, self.kind, self.k, self.member = ens, kind, k, member

    def __call__(self, decks):
        s = self.ens.scores(decks)
        if self.member is not None:
            return s[self.member]
        if self.kind == "mean":
            return s.mean(0)
        return s.mean(0) - self.k * s.std(0)


def quick_deck(pool, cols, quality, n_lands=17):
    ok = D.castable(cols) & C.spell & (pool > 0)
    cand = np.repeat(np.flatnonzero(ok), pool[ok].astype(int))
    if len(cand) < 40 - n_lands:
        return None
    top = cand[np.argsort(-quality[cand], kind="stable")][: 40 - n_lands]
    deck = np.zeros(len(C), np.float32)
    np.add.at(deck, top, 1)
    lands = np.flatnonzero((pool > 0) & C.is_land & ~C.is_basic & ((C.makes[:, cols] > 0).any(1)))
    for l in lands[np.argsort(-quality[lands])][:3]:
        deck[l] += 1
    deck += D.basics_split(deck, 40 - int(deck.sum()), cols)
    return deck


def neighbours(deck, pool):
    """All 1-for-1 swaps of a 40-card deck within the pool (basics unlimited)."""
    outs = np.flatnonzero(deck > 0)
    room = pool - deck
    room[BASIC_IDX] = 99
    ins = np.flatnonzero(room > 0)
    o, i = np.meshgrid(outs, ins, indexing="ij")
    o, i = o.ravel(), i.ravel()
    keep = o != i
    o, i = o[keep], i[keep]
    nb = np.repeat(deck[None, :], len(o), 0)
    nb[np.arange(len(o)), o] -= 1
    nb[np.arange(len(o)), i] += 1
    return nb


def refine(deck, pool, obj, eps=1e-4, max_steps=60):
    s = obj(deck[None])[0]
    for step in range(max_steps):
        nb = neighbours(deck, pool)
        v = obj(nb)
        j = int(np.argmax(v))
        if v[j] <= s + eps:
            break
        deck, s = nb[j], v[j]
    return deck, float(s)


def build(pool, obj, quality, n_refine=4, return_all=False):
    """The best deck for `pool` (a count vector over non-basic cards) under objective `obj`."""
    pool = pool.copy()
    pool[BASIC_IDX] = 0
    starts = []
    for cols in CONFIGS:
        d = quick_deck(pool, cols, quality)
        if d is not None:
            starts.append(d)
    if not starts:   # a tiny pool: everything castable
        starts = [quick_deck(pool, np.ones(5, bool), quality, n_lands=max(17, 40 - int((pool * C.spell).sum())))]
    sc = obj(np.array(starts))
    order = np.argsort(-sc)[:n_refine]
    cands = [refine(starts[i], pool, obj) for i in order]
    cands.sort(key=lambda t: -t[1])
    if return_all:
        return cands
    return cands[0][0]
