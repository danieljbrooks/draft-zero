"""
The decks a builder version builds from one pool: the exploit deck and the exploration decks around it (docs/032 §4;
formerly build/loop.py's _build_worker, without its process pool).

A builder version is an ensemble of deck value networks (model.Ensemble), or None for an untrained version 0, which
builds `UNTRAINED` wide random deck-shaped decks a pool (decks.random_build: 1-5 colours, 12-21 lands), whatever
the kinds, as docs/032's generation 0 did. With an ensemble, one deck per kind letter:

    b  the best deck under the ensemble's mean (build.build, inside the ensemble's land range)
    t  the best deck under one random member (Thompson sampling: explore where the members disagree)
    p  b with 1-3 random swaps
    l  b with 1-3 lands more or fewer (decks.shift_lands)
    r  a wide random build with probability random_share, else another t
    c  cautious: the mean less one standard deviation across members
    k  the best deck in one random colour pair

An unknown letter is an error (docs/032's loop turned it into a random build without a word).
"""
from __future__ import annotations

import numpy as np

from . import build as Bd
from . import decks as D

KINDS = "btplrck"
UNTRAINED = 4                   # decks a pool from an untrained version 0 (r0..r3)


class Builder:
    """Deck building with one builder version: `ens` an Ensemble, or None (untrained)."""

    def __init__(self, ens=None):
        self.ens = ens
        if ens is not None:
            self.q = np.mean([m.quality().detach().numpy() for m in ens.members], 0)
            self.mean = Bd.Objective(ens, "mean")
            self.lands = ens.meta.get("lands")

    @property
    def trained(self) -> bool:
        return self.ens is not None

    def best(self, pool: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """The exploit deck (kind b); an untrained builder's is a random deck-shaped one (needs rng)."""
        if self.ens is None:
            return D.random_build(pool, rng if rng is not None else np.random.default_rng(0), wide=False)
        return Bd.build(pool, self.mean, self.q, lands=self.lands)

    def decks(self, pool: np.ndarray, kinds: str, rng: np.random.Generator, random_share: float = 0.15
              ) -> dict[str, np.ndarray]:
        """{kind: deck} for one pool (count vector, basics ignored): one deck per letter of `kinds` (a letter twice
        gives one deck), or r0..r3 from an untrained builder."""
        bad = [k for k in kinds if k not in KINDS]
        if bad:
            raise ValueError(f"deck kinds {bad}: want letters of {KINDS}")
        if self.ens is None:
            return {f"r{i}": D.random_build(pool, rng, wide=True) for i in range(UNTRAINED)}
        out: dict[str, np.ndarray] = {}
        best = None

        def b0():
            nonlocal best
            if best is None:
                best = Bd.build(pool, self.mean, self.q, lands=self.lands)
            return best

        for kind in dict.fromkeys(kinds):
            if kind == "b":
                d = b0()
            elif kind == "c":
                d = Bd.build(pool, Bd.Objective(self.ens, "cautious", k=1.0), self.q, lands=self.lands)
            elif kind == "k":
                pairs = [c for c in Bd.CONFIGS if c.sum() == 2]
                d = Bd.build(pool, self.mean, self.q, n_refine=1, lands=self.lands,
                             configs=[pairs[int(rng.integers(len(pairs)))]])
            elif kind == "l":
                d = D.shift_lands(b0(), pool, int(rng.choice([-3, -2, -1, 1, 2, 3])), self.q)
            elif kind == "p":
                d = D.perturb(b0(), pool, rng, int(rng.integers(1, 4)))
            elif kind == "r" and rng.random() < random_share:
                d = D.random_build(pool, rng, wide=True)
            else:                                        # t, or r without the random build
                k = int(rng.integers(len(self.ens.members)))
                qk = self.ens.members[k].quality().detach().numpy()
                d = Bd.build(pool, Bd.Objective(self.ens, member=k), qk, n_refine=2, lands=self.lands)
            out[kind] = d
        return out
