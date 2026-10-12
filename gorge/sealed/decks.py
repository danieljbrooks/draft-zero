"""
Deck construction helpers shared by every builder: basic-land splits, simple builders and gorge deck files.

A deck is a count vector over cards.load()'s 286 cards (basics included); a pool is the same over its non-basic
cards (basics are unlimited).
"""
import json
import os

import numpy as np

from . import cards as cardsmod

C = cardsmod.load()
BASIC_IDX = np.array([C.index[b] for b in cardsmod.BASICS])     # W U B R G order
COLORS = cardsmod.COLORS


def castable(colors: np.ndarray) -> np.ndarray:
    """Cards whose cost needs only `colors` (bool[5]); colourless cards always."""
    return ~(C.colors & ~colors[None, :]).any(1)


def basics_split(deck: np.ndarray, n: int, colors=None, floor=0) -> np.ndarray:
    """n basic lands split over the deck's coloured pips (largest remainder), less what its other lands already
    make. `colors` limits the basics to a colour set; `floor` gives each pipped colour at least that many."""
    pips = (deck[:, None] * C.pips).sum(0)
    if colors is not None:
        pips = pips * colors
    if pips.sum() == 0:
        pips = np.ones(5) if colors is None else colors.astype(float)
    src = (deck[:, None] * (C.makes * (C.is_land & ~C.is_basic)[:, None])).sum(0)
    want = np.maximum(pips / pips.sum() * (n + src.sum()) - src, 0)
    if want.sum() == 0:
        want = pips
    want = want / want.sum() * n
    out = np.floor(want)
    if floor:
        need = (pips > 0) & (out < floor)
        out[need] = floor
    while out.sum() > n:
        out[np.argmax(out - want)] -= 1
    while out.sum() < n:
        out[np.argmax(want - out)] += 1
    v = np.zeros(len(C), np.float32)
    v[BASIC_IDX] = out
    return v


def deck_colors(deck: np.ndarray) -> str:
    """The colours of a deck's spells by pips, main colours first (17lands-like: a colour is 'main' with > 25% of
    the pips, else a splash)."""
    pips = (deck[:, None] * C.pips * C.spell[:, None]).sum(0)
    if pips.sum() == 0:
        return ""
    share = pips / pips.sum()
    main = "".join(c for c, s in zip(COLORS, share) if s > 0.25)
    spl = "".join(c for c, s in zip(COLORS, share) if 0 < s <= 0.25)
    return main + ("+" + spl if spl else "")


def random_build(pool: np.ndarray, rng: np.random.Generator, wide: bool = False) -> np.ndarray:
    """Generation 0's builder: a random but deck-shaped choice, no knowledge of card quality.
    (`wide` widens the colours and land counts below.)

    - 1, 2 or 3 main colours (10% / 70% / 20%), chosen at random among the colours the pool has cards in;
    - with probability 0.3, a splash of 1-3 cards of one more colour;
    - 14-19 lands (uniform), on-colour non-basic lands first with probability 0.5 each;
    - the spells uniform among the castable ones; basics split by pips.
    """
    have = np.array([(pool * C.colors[:, k] * C.spell).sum() for k in range(5)])
    if wide:   # also the decks that shouldn't work: up to five colours, 12-21 lands
        k = rng.choice([1, 2, 3, 4, 5], p=[0.1, 0.5, 0.2, 0.1, 0.1])
        n_lands = int(rng.integers(12, 22))
    else:
        k = rng.choice([1, 2, 3], p=[0.1, 0.7, 0.2])
        n_lands = int(rng.integers(14, 20))
    k = min(k, int((have > 0).sum()))
    cols = np.zeros(5, bool)
    cols[rng.choice(5, size=k, replace=False, p=have / have.sum())] = True
    deck = np.zeros(len(C), np.float32)
    while (pool * C.spell * castable(cols)).sum() < 40 - n_lands and not cols.all():   # too few spells: add a colour
        cols[rng.choice(np.flatnonzero(~cols))] = True
    ok = castable(cols)
    spells = np.repeat(np.arange(len(C)), pool.astype(int))
    spells = spells[C.spell[spells] & ok[spells]]
    n_spells = 40 - n_lands
    if rng.random() < 0.3:
        other = np.flatnonzero(~cols & (have > 0))
        if len(other):
            s = rng.choice(other)
            spl = np.repeat(np.arange(len(C)), pool.astype(int))
            spl = spl[C.spell[spl] & C.colors[spl, s] & castable(cols | (np.arange(5) == s))[spl] & ~ok[spl]]
            take = rng.permutation(spl)[: int(rng.integers(1, 4))]
            np.add.at(deck, take, 1)
            cols = cols.copy(); cols[s] = True
    rest = rng.permutation(spells)[: n_spells - int(deck.sum())]
    np.add.at(deck, rest, 1)
    lands = np.repeat(np.arange(len(C)), pool.astype(int))
    lands = lands[C.is_land[lands] & ~C.is_basic[lands]]
    for l in rng.permutation(lands):
        if deck[C.spell].sum() + deck[C.is_land].sum() >= 40:
            break
        if ((C.makes[l] > 0) & cols).sum() >= min(2, cols.sum()) and rng.random() < 0.5:
            deck[l] += 1
    deck += basics_split(deck, 40 - int(deck.sum()), cols)
    return deck


def rating_build(pool: np.ndarray, rating: np.ndarray, n_spells=23, splash=False) -> np.ndarray:
    """The classic heuristic: the best two colours by the sum of their top `n_spells` ratings, those spells,
    on-colour duals, and basics by pips. `rating` is per card (higher is better); NaN excludes a card."""
    best, best_s = None, -np.inf
    r = np.nan_to_num(rating, nan=-1e9)
    for a in range(5):
        for b in range(a + 1, 5):
            cols = np.zeros(5, bool); cols[[a, b]] = True
            ok = castable(cols) & C.spell & (pool > 0)
            cand = np.repeat(np.flatnonzero(ok), pool[ok].astype(int))
            top = cand[np.argsort(-r[cand], kind="stable")][:n_spells]
            s = r[top].sum() if len(top) == n_spells else -1e12 + r[top].sum()
            if s > best_s:
                best, best_s = (cols, top), s
    cols, top = best
    deck = np.zeros(len(C), np.float32)
    np.add.at(deck, top, 1)
    duals = np.flatnonzero((pool > 0) & C.is_land & ~C.is_basic & ((C.makes[:, cols] > 0).sum(1) >= 2))
    for l in duals:
        if deck.sum() < 40 - 15:
            deck[l] += 1
    deck += basics_split(deck, 40 - int(deck.sum()), cols)
    return deck


def gorge_deck(name: str, deck: np.ndarray, note: str = "") -> str:
    """One line of a -deckfile."""
    return json.dumps({"name": name, "format": "limited", "notes": note,
                       "cards": [{"name": c, "count": n} for c, n in sorted(C.dict(deck).items())]})


def write_decks(path: str, decks: dict):
    """decks: {name: count vector} -> a -deckfile (JSONL)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for n, d in decks.items():
            f.write(gorge_deck(n, d) + "\n")


def overlap(a: np.ndarray, b: np.ndarray, spells_only=False) -> float:
    """Cards two decks share (multiset intersection), over all cards or the non-land ones."""
    m = np.minimum(a, b)
    return float(m[C.spell].sum() if spells_only else m.sum())


def load_pools(kind="sealed", split=None):
    """data/sealed/{kind}.jsonl rows, with `pool_v` and `ref_v` (the reference deck) as count vectors."""
    path = os.path.join(cardsmod.REPO, "data/sealed", f"{kind}.jsonl")
    out = []
    for line in open(path):
        p = json.loads(line)
        if split and p["split"] != split:
            continue
        p["pool_v"] = C.vec(p["pool"])
        ref = next(b for b in p["builds"] if b["build_index"] == p["ref"])
        p["ref_v"] = C.vec(ref["deck"])
        out.append(p)
    return out


def shift_lands(deck: np.ndarray, pool: np.ndarray, k: int, quality: np.ndarray) -> np.ndarray:
    """The deck with k more lands (k < 0: fewer): the lowest-quality spells swapped for basics split by pips, or
    basics swapped for the best unused castable spells."""
    d = deck.copy()
    cols = (d[:, None] * C.pips).sum(0) > 0
    for _ in range(abs(k)):
        if k > 0:
            sp = [i for i in np.argsort(quality) if d[i] > 0 and C.spell[i]]
            if not sp:
                break
            d[sp[0]] -= 1
            d += basics_split(d, 1, cols if cols.any() else None)
        else:
            b = [i for i in BASIC_IDX if d[i] > 0]
            add = [i for i in np.argsort(-quality) if C.spell[i] and pool[i] - d[i] > 0 and castable(cols)[i]]
            if not b or not add:
                break
            d[max(b, key=lambda i: d[i])] -= 1
            d[add[0]] += 1
    return d


def perturb(deck: np.ndarray, pool: np.ndarray, rng: np.random.Generator, k: int) -> np.ndarray:
    """k random 1-for-1 swaps within the pool (basics unlimited): exploration around a built deck."""
    d = deck.copy()
    for _ in range(k):
        out = rng.choice(np.flatnonzero(d > 0))
        room = pool - d
        room[BASIC_IDX] = 99
        room[out] = 0
        d[out] -= 1
        d[rng.choice(np.flatnonzero(room > 0))] += 1
    return d
