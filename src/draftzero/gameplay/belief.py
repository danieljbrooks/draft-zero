"""
Opponent belief: sample the opponent's hidden decklist and hand from 17lands decks, fill a partial
StateSpec with them (determinization), and measure how good the samples are on mirrored games.

Neither 17lands nor Arena shows the opponent's deck or hand, but a coach (and a hidden-information
search) needs concrete cards there. The 17lands users' own decks are the population the opponents
come from, so the models here are built from those decks (game_data_public: one row per game, the
same games as the replay file; each distinct deck of a draft is kept once and weighted by the
games it played, which is how often an opponent brings it).

Models (all share `sample(...) -> (deck, hand)` and `marginal(...)`):
  OpponentModel     (deck-based) posterior over real decks:
                      P(D | colours, seen) ∝ games(D) · P(colours revealed | D's colours)
                                             · Π_seen copies (max(D_c - j, 0) + α·f_g(c))
                    where f_g(c) is the mean number of copies of c in decks of D's colour group
                    (a smoothed "the opponent swapped a card in" term, so decks missing a seen
                    card stay possible but lose weight). A sampled deck is forced to contain every
                    seen / zone / lost card, replacing its least-likely cards (lowest group
                    frequency) of the same broad type (land / creature / other) and nearest mana
                    value; the hidden hand is then drawn from the deck minus the cards the spec
                    places in known zones.
  FrequencyModel    baseline: the colour-conditioned card frequencies (the same prior, no seen-card
                    likelihood); the deck is the forced cards plus a fill drawn by frequency
  UniformPoolModel  baseline: every card of the revealed colours equally likely

Hand draw: uniform over the deck's hidden cards by default. With a `HandRetention` table
(`retention=`), each hidden card is weighted by the odds that a card of its kind and mana value is
still in hand after the owner's t turns (learned from 17lands users' own hands, where the hand is
known: at the end of turn 5 about 5% of the hidden lands but 25% of the hidden 5-drops are in hand).
It is optional because it changes the hand's land/spell mix: see the evaluation.

Cards in no known zone: an opponent card that was seen but is in none of the spec's zones (it left
the battlefield for an unrecorded zone: bounced, tucked or exiled unseen) is somewhere hidden, so it
can be drawn into the hand like any hidden card. reconstruct.opp_location_unknown(g, n, split=True)
separates those most likely back in the hand (they left as the opponent's hand count rose with no
recorded reason: 84% were in the real hand on 800 mirrored pairs) from the rest (6% there, mostly
exiled). Passed as `lost=`, each 'likely in hand' copy is in the hand with probability
P_LIKELY_IN_HAND; the rest are drawn like any hidden card. Held out, K=16, user turns 3/5/7: on 300
pairs recall@K of the deck model went from 0.132 to 0.134 (log-lik per card -4.309 to -4.299); on
the 151 views of 1,500 pairs that have such cards, from 0.079 to 0.212 (-4.60 to -4.06), and the
likely copies (90 of 93 in the real hand there) went from 4% to 84% of the sampled hands.

Colours: pass the 17lands `opp_colors` (the colours the opponent revealed over the whole game:
always a subset of its deck colours, equal to them in 82% of mirrored games) or None to infer them
from the seen cards only. P(colours revealed | deck) treats each main colour as revealed with
probability 0.97 and each splash colour with 0.46 (fit on the 108,276 mirrored-game perspectives
of FDN Premier: e.g. a 2-colour deck without splash shows both colours in 93.9% of games).

`determinize(spec, model, k, seed)` returns k full copies of a partial spec with B.decklist
(decklistSource "belief") and B.hand sampled (handUnknown -> 0) and `provenance.sampled` set; A is
untouched. An "exact" B decklist (a mirrored pair) is kept and only the hand is drawn.

Evaluation (`main`): mirrored pairs (`pairs.py`) give the TRUE opponent hand and deck. For
2,500 pairs x both perspectives at user turns 3, 5 and 7 it reports recall@K (share of the true
hand's cards in a sampled hand, mean of K=16 samples), coverage@K (in any of the K hands), the
per-card log-likelihood of the true hand under the model's single-card marginal and the sampled
deck's overlap with the true deck. The deck pool and the retention table leave out every draft of
the evaluated pairs (both players) and of a small dev set used to pick α; an "oracle" row draws the
hand from the true deck (the ceiling for any deck model with that hand draw).

Usage:
  python -m draftzero.gameplay.belief                      # evaluation -> data/gameplay/belief_eval.json
  python -m draftzero.gameplay.belief --pairs 500 --turns 5 --alpha 1   # quick run

  from draftzero.gameplay.belief import OpponentModel, determinize, game_evidence
  model = OpponentModel.load()                     # builds data/gameplay/deckpool_FDN_PremierDraft.npz once
  spec, seen, colors = game_evidence(game, 5)      # a 17lands user turn
  lost = reconstruct.opp_location_unknown(game, 5, split=True)
  specs = determinize(spec, model, k=8, seed=1, opp_colors=colors, seen=seen, lost=lost)

Leakage on 17lands rows: the full pool holds every 17lands user's deck, so for one half of a
mirrored pair it holds the opponent's REAL deck (at user turn 7 about 10% of the posterior mass lands
on it, measured on 150 pairs). Build the model with `exclude_drafts=` both rows' draft ids
(`pairs.partner_map` finds the other row) when the belief must not know it, and note that
game_evidence's colours are whole-game (hindsight) colours.

17lands public data is CC BY 4.0 (https://www.17lands.com/public_datasets).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import sys
import time
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from draftzero.gameplay import replay
from draftzero.gameplay.ids import COLORS, DATA_DIR, REPO, Ids
from draftzero.gameplay.statespec import SEATS, StateSpec

OUT_DIR = REPO / "data" / "gameplay"
P_MAIN_SEEN = 0.97        # P(a main colour shows up in opp_colors); fit on FDN Premier mirrored games
P_SPLASH_SEEN = 0.46      # the same for a splash colour
ALPHA = 1.0               # smoothing of the deck likelihood: best of 0.1..10 on the evaluation's dev set
F_FLOOR = 1e-3            # smallest per-group card frequency: a card no deck of the group played
LL_FLOOR = 1e-3           # the log-likelihood metric mixes every marginal with this much uniform mass
# P(a 'likely in hand' card is in the opponent's hand): reconstruct.opp_location_unknown's first
# group, measured on 800 mirrored pairs (every decision state of both halves): 148/177
P_LIKELY_IN_HAND = 0.84
KIND_LAND, KIND_CREATURE, KIND_OTHER = 0, 1, 2
_BIT = {c: 1 << i for i, c in enumerate(COLORS)}


def color_bits(s: str | None) -> int:
    return sum(_BIT[c] for c in set((s or "").upper()) if c in _BIT)


def _popcount(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint8)
    return sum(((x >> i) & 1) for i in range(5)).astype(np.int8)


def _wsum(w: np.ndarray, counts: np.ndarray, rows: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """w @ counts[rows] in float64 without materialising the whole float matrix."""
    out = np.zeros(counts.shape[1], np.float64)
    for s in range(0, len(rows), chunk):
        out += w[s:s + chunk] @ counts[rows[s:s + chunk]].astype(np.float64)
    return out


def as_counter(cards) -> Counter:
    if cards is None:
        return Counter()
    return Counter(cards) if not isinstance(cards, Counter) else cards


def split_lost(lost) -> tuple[Counter, Counter]:
    """`lost` as (likely in hand, the rest): the pair reconstruct.opp_location_unknown(g, n,
    split=True) returns, or one multiset (all of it 'the rest'), or None."""
    if lost is None:
        return Counter(), Counter()
    if isinstance(lost, tuple):
        return as_counter(lost[0]), as_counter(lost[1])
    return Counter(), as_counter(lost)


# =============================================================================================
# Deck pool, card facts, hand retention
# =============================================================================================

@dataclass
class DeckPool:
    """Distinct 17lands maindecks (one per draft and build) as a count matrix over the set's cards."""
    cards: list[str]           # card names (the file's deck_ columns)
    counts: np.ndarray         # [n_decks, n_cards] uint8 copies
    games: np.ndarray          # [n_decks] games played with the deck: its prior weight
    main: np.ndarray           # [n_decks] main-colour bits
    splash: np.ndarray         # [n_decks] splash-colour bits
    draft_key: np.ndarray      # [n_decks] crc32 of the draft id: only to hold drafts out

    def __post_init__(self):
        self.index = {c: i for i, c in enumerate(self.cards)}

    def __len__(self) -> int:
        return len(self.games)

    @classmethod
    def build(cls, set_code: str = "FDN", fmt: str = "PremierDraft", data_dir=None) -> "DeckPool":
        """From game_data_public (60 MB, fast) or, without it, the replay file's deck_ columns."""
        d = Path(data_dir or DATA_DIR)
        path = d / f"game_data_public.{set_code}.{fmt}.csv.gz"
        if not path.exists():
            path = replay.replay_path(set_code, fmt, data_dir)
        with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
            cols = next(csv.reader([f.readline()]))
            ix = {c: i for i, c in enumerate(cols)}
            deck_cols = [(c[5:], i) for i, c in enumerate(cols) if c.startswith("deck_")]
            names = [n for n, _ in deck_cols]
            idx = [i for _, i in deck_cols]
            step = idx[1] - idx[0] if len(idx) > 1 else 1
            regular = idx == list(range(idx[0], idx[0] + step * len(idx), step))
            seen: dict[tuple, int] = {}
            rows, games, main, splash, dkey = [], [], [], [], []
            for line in f:
                p = line.rstrip("\r\n").split(",")
                if len(p) != len(cols):
                    p = next(csv.reader([line]))
                vals = p[idx[0]:idx[-1] + 1:step] if regular else [p[i] for i in idx]
                key = (p[ix["draft_id"]], ",".join(vals))
                j = seen.get(key)
                if j is not None:
                    games[j] += 1
                    continue
                v = np.array([int(float(x or 0)) for x in vals], dtype=np.int16)
                if not 40 <= v.sum() <= 60:
                    continue
                seen[key] = len(rows)
                rows.append(np.minimum(v, 255).astype(np.uint8))
                games.append(1)
                main.append(color_bits(p[ix["main_colors"]]))
                splash.append(color_bits(p[ix["splash_colors"]]))
                dkey.append(zlib.crc32(p[ix["draft_id"]].encode()))
        return cls(names, np.stack(rows), np.array(games, np.int32), np.array(main, np.uint8),
                   np.array(splash, np.uint8), np.array(dkey, np.uint32))

    @classmethod
    def load(cls, set_code: str = "FDN", fmt: str = "PremierDraft", data_dir=None, cache_dir=None,
             rebuild: bool = False) -> "DeckPool":
        """Cached build: data/gameplay/deckpool_<SET>_<FMT>.npz (gitignored): the decks, their colours
        and games, and a crc32 of each deck's draft id (to hold drafts out; no draft ids are stored)."""
        cache = Path(cache_dir or OUT_DIR) / f"deckpool_{set_code}_{fmt}.npz"
        if cache.exists() and not rebuild:
            z = np.load(cache, allow_pickle=False)
            return cls([str(x) for x in z["cards"]], z["counts"], z["games"], z["main"], z["splash"], z["draft_key"])
        pool = cls.build(set_code, fmt, data_dir)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, cards=np.array(pool.cards), counts=pool.counts, games=pool.games,
                            main=pool.main, splash=pool.splash, draft_key=pool.draft_key)
        return pool

    def without_drafts(self, draft_ids: Iterable[str]) -> "DeckPool":
        """The pool minus every deck of these drafts (held-out evaluation)."""
        keys = np.array(sorted({zlib.crc32(d.encode()) for d in draft_ids if d}), dtype=np.uint32)
        keep = ~np.isin(self.draft_key, keys)
        return DeckPool(self.cards, self.counts[keep], self.games[keep], self.main[keep], self.splash[keep],
                        self.draft_key[keep])

    @classmethod
    def from_decks(cls, decks: list[tuple[Counter, str, str]], cards: list[str] | None = None,
                   games: list[int] | None = None) -> "DeckPool":
        """A small pool from (deck Counter, main colours, splash colours) triples (tests, custom pools)."""
        cards = cards or sorted({c for d, _, _ in decks for c in d})
        ix = {c: i for i, c in enumerate(cards)}
        m = np.zeros((len(decks), len(cards)), np.uint8)
        for r, (d, _, _) in enumerate(decks):
            for c, k in d.items():
                m[r, ix[c]] = k
        return cls(cards, m, np.array(games or [1] * len(decks), np.int32),
                   np.array([color_bits(mc) for _, mc, _ in decks], np.uint8),
                   np.array([color_bits(sc) for _, _, sc in decks], np.uint8),
                   np.arange(len(decks), dtype=np.uint32))


@dataclass
class CardMeta:
    """Per-card arrays over a pool's card axis."""
    kind: np.ndarray       # KIND_LAND / KIND_CREATURE / KIND_OTHER
    mv: np.ndarray
    colors: np.ndarray     # colour bits: mana-cost colours for spells, produced colours for lands
    basic: np.ndarray      # bool

    @classmethod
    def from_ids(cls, cards: list[str], ids: Ids | None = None) -> "CardMeta":
        ids = ids or Ids.load()
        by_name: dict[str, dict] = {}
        home = (ids.set_code, "SPG")
        for r in ids.cards17.values():
            if r.get("rarity") == "token":
                continue
            cur = by_name.get(r["name"])
            if cur is None or (r.get("expansion") in home and cur.get("expansion") not in home):
                by_name[r["name"]] = r
        n = len(cards)
        kind, mv = np.full(n, KIND_OTHER, np.int8), np.zeros(n, np.int8)
        cb, basic = np.zeros(n, np.uint8), np.zeros(n, bool)
        for i, c in enumerate(cards):
            r = by_name.get(c) or {}
            types = r.get("types", "")
            info = ids.info(c)
            kind[i] = KIND_LAND if "Land" in types else KIND_CREATURE if "Creature" in types else KIND_OTHER
            try:
                mv[i] = int(float(r.get("mana_value") or 0))
            except ValueError:
                pass
            basic[i] = "Basic Land" in types
            if kind[i] == KIND_LAND:
                cb[i] = color_bits(info.land_mana or r.get("color_identity", ""))
            else:
                cb[i] = color_bits("".join(info.pips) or r.get("color_identity", ""))
        return cls(kind, mv, cb, basic)

    @classmethod
    def simple(cls, cards: list[str], kinds: dict[str, int], mvs: dict[str, int], colors: dict[str, str],
               basics: Iterable[str] = ()) -> "CardMeta":
        basics = set(basics)
        return cls(np.array([kinds.get(c, KIND_OTHER) for c in cards], np.int8),
                   np.array([mvs.get(c, 0) for c in cards], np.int8),
                   np.array([color_bits(colors.get(c, "")) for c in cards], np.uint8),
                   np.array([c in basics for c in cards], bool))


class HandRetention:
    """P(a hidden card is in its owner's hand rather than the library | card kind, mana value, the
    owner's completed turns t), from 17lands users' own hands: at the end of each user turn the hand
    is known and the hidden cards are the deck minus the battlefield, graveyard and exile (the
    reconstruct history walk). Stored as in-hand / hidden counts per cell; rate = (in+1)/(all+2)."""
    MAX_MV, MAX_T = 6, 10

    def __init__(self, cells: dict[tuple[int, int, int], list[int]]):
        self.cells = cells
        self._odds: dict = {}

    @classmethod
    def key(cls, kind: int, mv: int, t: int) -> tuple[int, int, int]:
        return int(kind), min(int(mv), cls.MAX_MV), max(0, min(int(t), cls.MAX_T))

    def rate(self, kind: int, mv: int, t: int) -> float:
        a, b = self.cells.get(self.key(kind, mv, t), (0, 0))
        return (a + 1) / (b + 2)

    def odds(self, meta: CardMeta, t: int) -> np.ndarray:
        """Per-card sampling weights (odds of 'in hand') after t completed own turns."""
        ck = (id(meta), min(max(int(t), 0), self.MAX_T))
        hit = self._odds.get(ck)
        # the entry keeps its meta alive and is checked by identity: a bare id() key would hand a
        # new CardMeta that reuses a freed one's address the old card axis's odds
        if hit is None or hit[0] is not meta:
            r = np.clip([self.rate(k, m, t) for k, m in zip(meta.kind, meta.mv)], 1e-3, 1 - 1e-3)
            hit = self._odds[ck] = (meta, r / (1 - r))
        return hit[1]

    @classmethod
    def learn(cls, games: Iterable[replay.Game], ids: Ids, index: dict[str, int], meta: CardMeta) -> "HandRetention":
        from draftzero.gameplay import reconstruct as rc
        cells: dict = {}
        for g in games:
            ana = rc.analyze(g, ids)
            deck = Counter(g.deck)
            for t in g.turns:
                if t.side != "user" or not t.played or t.terminal:
                    continue
                st = ana.states[t.seq]
                vis = Counter(i.name for i in st.bf["user"] if not i.token) + st.gy["user"] + st.exile["user"]
                hand = Counter(ids.name(c) for c in t.L("eot_user_cards_in_hand"))
                for name, k in (deck - vis).items():
                    i = index.get(name)
                    if i is None:
                        continue
                    cell = cells.setdefault(cls.key(meta.kind[i], meta.mv[i], t.n), [0, 0])
                    cell[0] += min(hand[name], k)
                    cell[1] += k
        return cls(cells)

    def to_json(self) -> str:
        return json.dumps([[*k, *v] for k, v in sorted(self.cells.items())])

    @classmethod
    def from_json(cls, s: str) -> "HandRetention":
        return cls({tuple(r[:3]): list(r[3:]) for r in json.loads(s)})

    @classmethod
    def load(cls, index: dict[str, int], meta: CardMeta, ids: Ids, set_code: str = "FDN",
             fmt: str = "PremierDraft", every: int = 97, limit: int = 6000, exclude_drafts: Iterable[str] = (),
             cache: bool = True, rebuild: bool = False) -> "HandRetention":
        """Learn from every `every`-th replay row (default ~6,000 games, ~40 s), skipping held-out
        drafts; cached in data/gameplay/hand_retention_<SET>_<FMT>.json unless drafts are excluded."""
        exclude = set(exclude_drafts)
        path = OUT_DIR / f"hand_retention_{set_code}_{fmt}.json"
        if cache and not exclude and path.exists() and not rebuild:
            return cls.from_json(path.read_text())
        games = replay.iter_games(replay.replay_path(set_code, fmt), limit, every,
                                  predicate=(lambda g: g.meta.get("draft_id") not in exclude) if exclude else None)
        r = cls.learn(games, ids, index, meta)
        if cache and not exclude:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(r.to_json() + "\n")
        return r


# =============================================================================================
# Models
# =============================================================================================

class _Base:
    """Shared plumbing: card vectors, candidate decks by colour, forced cards and hand draws."""
    name = "base"

    def __init__(self, pool: DeckPool, meta: CardMeta | None = None, ids: Ids | None = None,
                 retention: HandRetention | None = None):
        self.pool = pool
        self.meta = meta or CardMeta.from_ids(pool.cards, ids)
        self.retention = retention
        self.cards = pool.cards
        self.index = pool.index
        self.C = len(pool.cards)
        self.gbits = (pool.main | pool.splash).astype(np.uint8)
        self.size = pool.counts.sum(1, dtype=np.int32)
        # per colour group (main|splash bits): games-weighted mean copies of every card
        self.group_freq = np.full((32, self.C), 0.0, np.float32)
        w = pool.games.astype(np.float64)
        for g in np.unique(self.gbits):
            m = np.nonzero(self.gbits == g)[0]
            self.group_freq[g] = _wsum(w[m], pool.counts, m) / max(w[m].sum(), 1.0)
        self._cand_cache: dict = {}
        self.last_replaced = 0         # forced copies the last sampled deck had to swap in

    # --- vectors ------------------------------------------------------------------------------

    def vec(self, cards) -> tuple[np.ndarray, Counter]:
        """Counter / list of names -> (count vector over the pool's cards, names outside it)."""
        v = np.zeros(self.C, np.int16)
        extra = Counter()
        for c, k in as_counter(cards).items():
            i = self.index.get(c)
            if i is None:
                extra[c] += k
            else:
                v[i] += k
        return v, extra

    def names(self, v: np.ndarray, extra: Counter | None = None) -> list[str]:
        out = [self.cards[i] for i in np.repeat(np.arange(self.C), np.maximum(v, 0))]
        return sorted(out + list((extra or Counter()).elements()))

    def seen_colors(self, seen_vec: np.ndarray) -> int:
        """Colours of the seen spells and basic lands (duals and colourless cards say nothing)."""
        m = (seen_vec > 0) & ((self.meta.kind != KIND_LAND) | self.meta.basic)
        return int(np.bitwise_or.reduce(self.meta.colors[m])) if m.any() else 0

    def hand_odds(self, turn: int | None) -> np.ndarray | None:
        """Per-card hand weights (None = uniform draw)."""
        return None if self.retention is None or turn is None else self.retention.odds(self.meta, turn)

    # --- candidates -----------------------------------------------------------------------------

    def candidates(self, opp_colors: str | None, seen_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(deck indices, log prior) for the colour information given. With opp_colors: decks whose
        colours contain them, weighted by games x P(these colours revealed | deck colours). Without:
        decks containing the seen cards' colours, weighted by games."""
        if opp_colors is not None:
            key = ("row", color_bits(opp_colors))
        else:
            key = ("seen", self.seen_colors(seen_vec))
        hit = self._cand_cache.get(key)
        if hit is not None:
            return hit
        mode, S = key
        idx = np.nonzero((S & ~self.gbits) == 0)[0]
        if len(idx) == 0:                                # e.g. 5 colours seen: nothing contains them
            idx = np.arange(len(self.pool))
        logp = np.log(self.pool.games[idx].astype(np.float64))
        if mode == "row":
            main, splash = self.pool.main[idx], self.pool.splash[idx] & ~self.pool.main[idx]
            s = np.uint8(S)
            logp += (_popcount(main & s) * math.log(P_MAIN_SEEN) + _popcount(main & ~s) * math.log(1 - P_MAIN_SEEN)
                     + _popcount(splash & s) * math.log(P_SPLASH_SEEN) + _popcount(splash & ~s) * math.log(1 - P_SPLASH_SEEN))
        hit = (idx, logp)
        self._cand_cache[key] = hit
        return hit

    def color_freq(self, opp_colors: str | None, seen_vec: np.ndarray) -> np.ndarray:
        """Prior-weighted mean copies per card over the candidate decks (the card-frequency view)."""
        key = ("freq", opp_colors is None, color_bits(opp_colors) if opp_colors is not None else self.seen_colors(seen_vec))
        f = self._cand_cache.get(key)
        if f is None:
            idx, logp = self.candidates(opp_colors, seen_vec)
            w = np.exp(logp - logp.max())
            f = _wsum(w, self.pool.counts, idx) / w.sum()
            self._cand_cache[key] = f
        return f

    # --- forced cards and the hand ------------------------------------------------------------

    def force_include(self, d: np.ndarray, forced: np.ndarray, freq: np.ndarray) -> tuple[np.ndarray, int]:
        """Add the forced copies a deck lacks, each replacing the deck's least frequent unforced card
        of the same kind (land / creature / other) and nearest mana value. Returns (deck, #replaced)."""
        d = d.astype(np.int16).copy()
        miss = forced - d
        replaced = 0
        for c in np.nonzero(miss > 0)[0]:
            for _ in range(int(miss[c])):
                spare = (d - forced) > 0
                if not spare.any():
                    d[c] += 1                    # nothing left to swap out: the deck grows
                    continue
                score = ((self.meta.kind != self.meta.kind[c]) * 1e6
                         + np.abs(self.meta.mv.astype(np.int16) - self.meta.mv[c]) * 1e3 + freq)
                r = int(np.argmin(np.where(spare, score, np.inf)))
                d[r] -= 1
                d[c] += 1
                replaced += 1
        return d, replaced

    @staticmethod
    def fit_room(d: np.ndarray, known: np.ndarray, hand_count: int, filler: int) -> np.ndarray:
        """At least one library card must remain after the hand is drawn from d - known (an empty
        library loses the game at the next draw): pad with `filler` (a basic) when the zones
        over-count."""
        short = hand_count + 1 - int((d - known).clip(0).sum())
        if short > 0:
            d = d.copy()
            d[filler] += short
        return d

    def draw_hand(self, d: np.ndarray, known: np.ndarray, hand_count: int, rng, turn: int | None = None,
                  likely: np.ndarray | None = None) -> np.ndarray:
        """hand_count cards from the deck's hidden part, d - known (known: the cards the spec places
        in a known zone). Each `likely` copy (a card that most likely went back to the hand) is in
        the hand with probability P_LIKELY_IN_HAND; the rest of the hand comes from the other hidden
        cards: uniform, or weighted by the retention odds (Efraimidis-Spirakis weighted sampling
        without replacement)."""
        hidden = np.maximum(d - known, 0)
        h = np.zeros(self.C, np.int16)
        if hand_count <= 0 or not hidden.any():
            return h
        if likely is not None and likely.any():
            lk = np.minimum(np.maximum(likely, 0), hidden)
            copies = np.repeat(np.arange(self.C), lk)
            take = copies[rng.random(len(copies)) < P_LIKELY_IN_HAND]
            if len(take) > hand_count:
                take = rng.choice(take, size=hand_count, replace=False)
            np.add.at(h, take, 1)
            hand_count -= len(take)
            hidden = hidden - lk                 # a likely copy left out is in the library (or exile)
        rest = np.repeat(np.arange(self.C), hidden)
        if hand_count <= 0 or not len(rest):
            return h
        k = min(hand_count, len(rest))
        o = self.hand_odds(turn)
        if o is None:
            np.add.at(h, rng.choice(rest, size=k, replace=False), 1)
        else:
            keys = np.log(rng.random(len(rest))) / o[rest]
            np.add.at(h, rest[np.argsort(-keys)[:k]], 1)
        return h

    def filler(self, colors: int) -> int:
        """Index of a basic land of these colours (the most played one), else any basic."""
        b = np.nonzero(self.meta.basic)[0]
        if len(b) == 0:
            return 0
        ok = [i for i in b if self.meta.colors[i] & colors] or list(b)
        return int(max(ok, key=lambda i: self.group_freq[:, i].sum()))

    # --- public API ---------------------------------------------------------------------------

    def sample(self, opp_colors: str | None, seen_cards, hand_count: int, known_zone_cards=None,
               rng=None, turn: int | None = None, lost=None) -> tuple[list[str], list[str]]:
        """(decklist, hidden hand) for one determinization. seen_cards: every opponent card revealed
        so far (a multiset, max simultaneous copies); known_zone_cards: the cards now in the
        opponent's known zones (battlefield cards it owns, graveyard, exile, known hand, library
        top, its spells on the stack); lost: its cards that left the battlefield for an unrecorded
        zone, as reconstruct.opp_location_unknown(g, n, split=True) gives them (likely in hand, the
        rest). The deck contains all of them; the hand (hand_count cards) is drawn from the deck
        minus the known zones, so a seen card that is in no known zone (bounced, tucked, exiled
        unseen) can be in it, and a 'likely in hand' card is, with probability P_LIKELY_IN_HAND.
        turn: the opponent's completed turns (used only with a retention table)."""
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        d, h, _, extra = self._draw(opp_colors, as_counter(seen_cards), as_counter(known_zone_cards),
                                    hand_count, rng, turn, lost)
        return self.names(d, extra), self.names(h)

    def sample_vec(self, opp_colors, seen: Counter, zones: Counter, hand_count: int, rng, turn: int | None = None,
                   lost=None):
        """sample() as count vectors: (deck, hand, forced)."""
        d, h, forced, _ = self._draw(opp_colors, seen, zones, hand_count, rng, turn, lost)
        return d, h, forced

    def _draw(self, opp_colors, seen: Counter, zones: Counter, hand_count: int, rng, turn, lost):
        sv, forced, extra = self.forced(seen, zones, lost)
        zv, _ = self.vec(zones)
        d = self._sample_deck(opp_colors, sv, forced, zv, extra, hand_count, rng)
        return d, self.draw_hand(d, zv, hand_count, rng, turn, self.vec(split_lost(lost)[0])[0]), forced, extra

    def forced(self, seen: Counter, zones: Counter, lost=None) -> tuple[np.ndarray, np.ndarray, Counter]:
        """(seen vector, forced vector, names outside the pool). Forced: the copies a sampled deck
        must hold, the per-card max of the seen cards and of the cards the spec places (zones) plus
        the lost ones, which are in no zone of the spec (split_lost)."""
        likely, rest = split_lost(lost)
        sv, se = self.vec(seen)
        pv, pe = self.vec(zones + likely + rest)
        return sv, np.maximum(sv, pv), se | pe

    def _sample_deck(self, opp_colors, sv: np.ndarray, forced: np.ndarray, zv: np.ndarray, extra: Counter,
                     hand_count: int, rng) -> np.ndarray:
        """A deck count vector holding the forced copies, with room for the hand in d - zv."""
        raise NotImplementedError

    def marginal(self, opp_colors: str | None, seen_cards, known_zone_cards=None, turn: int | None = None,
                 lost=None, hand_count: int | None = None) -> np.ndarray:
        """P(one hidden hand card is card i) over the pool's cards (the first draw of the hand),
        drawn as sample() draws it; with hand_count, the 'likely in hand' copies of `lost` enter
        with their own probability (the mean share of the hand that is card i)."""
        raise NotImplementedError

    def _known_and_likely(self, zones: Counter, lost, hand_count: int | None) -> tuple[np.ndarray, np.ndarray]:
        """(zone vector, likely vector) for a marginal; likely is empty without a hand count."""
        zv, _ = self.vec(zones)
        lk = self.vec(split_lost(lost)[0])[0] if hand_count else np.zeros(self.C, np.int16)
        return zv, lk

    def _weighted(self, x: np.ndarray, turn: int | None) -> np.ndarray:
        o = self.hand_odds(turn)
        x = x if o is None else x * o
        return x / max(x.sum(), 1e-12)

    @staticmethod
    def _mix_likely(p: np.ndarray, lk: np.ndarray, hand_count: int | None) -> np.ndarray:
        """The one-card marginal of a hand of hand_count cards whose `lk` copies are each in it with
        probability P_LIKELY_IN_HAND (scaled down when they would overfill it), the rest drawn by p."""
        if not hand_count or not lk.any():
            return p
        e = P_LIKELY_IN_HAND * lk.astype(np.float64)
        e *= min(1.0, hand_count / e.sum())
        return (e + (hand_count - e.sum()) * p) / hand_count


class OpponentModel(_Base):
    """Deck-based belief: a real 17lands deck of matching colours, weighted by how well it explains
    the seen cards, forced to contain them (see the module docstring)."""
    name = "deck"

    def __init__(self, pool: DeckPool, meta: CardMeta | None = None, ids: Ids | None = None, alpha: float = ALPHA,
                 retention: HandRetention | None = None):
        super().__init__(pool, meta, ids, retention)
        self.alpha = alpha
        self._post_key = None
        self._post = None
        self._cdf = None

    @classmethod
    def load(cls, set_code: str = "FDN", fmt: str = "PremierDraft", alpha: float = ALPHA, ids: Ids | None = None,
             exclude_drafts: Iterable[str] = (), retention: bool = False) -> "OpponentModel":
        """The model over the cached deck pool; retention=True also loads (or learns) the hand table."""
        ids = ids or Ids.load(set_code)
        pool = DeckPool.load(set_code, fmt)
        if exclude_drafts:
            pool = pool.without_drafts(exclude_drafts)
        meta = CardMeta.from_ids(pool.cards, ids)
        ret = HandRetention.load(pool.index, meta, ids, set_code, fmt, exclude_drafts=exclude_drafts) if retention else None
        return cls(pool, meta, alpha=alpha, retention=ret)

    def posterior(self, opp_colors: str | None, seen_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(candidate deck indices, normalised weights)."""
        key = (opp_colors, seen_vec.tobytes(), self.alpha)
        if key == self._post_key:
            return self._post
        idx, logp = self.candidates(opp_colors, seen_vec)
        lw = logp.copy()
        cols = np.nonzero(seen_vec > 0)[0]
        if len(cols):
            D = self.pool.counts[np.ix_(idx, cols)].astype(np.float32)
            smooth = self.alpha * np.maximum(self.group_freq[:, cols][self.gbits[idx]], F_FLOOR)
            for j in range(int(seen_vec[cols].max())):
                use = seen_vec[cols] > j                        # the (j+1)-th copy of each seen card
                lw += np.log(np.maximum(D[:, use] - j, 0) + smooth[:, use]).sum(1)
            lw -= int(seen_vec.sum()) * np.log(self.size[idx].astype(np.float64))
        w = np.exp(lw - lw.max())
        w /= w.sum()
        self._post_key, self._post, self._cdf = key, (idx, w), np.cumsum(w)
        return idx, w

    def _sample_deck(self, opp_colors, sv, forced, zv, extra, hand_count: int, rng) -> np.ndarray:
        idx, _ = self.posterior(opp_colors, sv)
        j = int(idx[min(int(np.searchsorted(self._cdf, rng.random() * self._cdf[-1], side="right")), len(idx) - 1)])
        d, self.last_replaced = self.force_include(self.pool.counts[j], forced, self.group_freq[self.gbits[j]])
        return self.fit_room(d, zv, hand_count, self.filler(int(self.gbits[j])))

    def marginal(self, opp_colors, seen_cards, known_zone_cards=None, turn: int | None = None, lost=None,
                 hand_count: int | None = None, tail: float = 1e-4) -> np.ndarray:
        # Over the most likely candidate decks holding all but `tail` of the posterior mass. A deck's
        # hidden cards are max(D, forced) - zones: force_include's swap-ins are counted, the cards
        # they replace are not taken out. The likely copies are left out here and mixed in by
        # _mix_likely with their own probability.
        zones = as_counter(known_zone_cards)
        sv, forced, _ = self.forced(as_counter(seen_cards), zones, lost)
        zv, lk = self._known_and_likely(zones, lost, hand_count)
        idx, w = self.posterior(opp_colors, sv)
        order = np.argsort(-w)
        n = int(np.searchsorted(np.cumsum(w[order]), 1 - tail)) + 1
        top, wt = idx[order[:n]], w[order[:n]]
        o = self.hand_odds(turn)
        o = np.ones(self.C, np.float32) if o is None else o.astype(np.float32)
        cols = np.nonzero(forced > 0)[0]
        dc = self.pool.counts[np.ix_(top, cols)].astype(np.float32)
        # the forced columns' hidden copies minus the deck's own (forced >= zones + likely, so >= 0)
        adj = np.maximum(dc, forced[cols]) - (zv[cols] + lk[cols]) - dc
        mass = np.zeros(len(top), np.float64)             # each deck's (weighted) hidden size
        for s in range(0, len(top), 8192):
            mass[s:s + 8192] = self.pool.counts[top[s:s + 8192]].astype(np.float32) @ o
        mass += adj @ o[cols]
        coef = (wt / np.maximum(mass, 1e-6)).astype(np.float32)
        p = np.zeros(self.C, np.float64)
        for s in range(0, len(top), 8192):
            p += coef[s:s + 8192] @ self.pool.counts[top[s:s + 8192]].astype(np.float32)
        p[cols] += coef @ adj
        p = np.maximum(p, 0) * o
        return self._mix_likely(p / max(p.sum(), 1e-12), lk, hand_count)


class FrequencyModel(_Base):
    """Baseline: colour-conditioned card frequencies; the deck is the forced cards plus a fill drawn
    in proportion to each card's expected copies beyond the forced ones."""
    name = "freq"

    def _sample_deck(self, opp_colors, sv, forced, zv, extra, hand_count: int, rng) -> np.ndarray:
        x = np.maximum(self.color_freq(opp_colors, sv) - forced, 0)
        n = max(40 - int(forced.sum()) - sum(extra.values()), hand_count + 1)
        self.last_replaced = 0
        return forced.astype(np.int16) + _weighted_fill(x, n, rng)

    def marginal(self, opp_colors, seen_cards, known_zone_cards=None, turn: int | None = None, lost=None,
                 hand_count: int | None = None) -> np.ndarray:
        # hidden: the forced copies outside the known zones (and the likely ones) plus the fill
        zones = as_counter(known_zone_cards)
        sv, forced, extra = self.forced(as_counter(seen_cards), zones, lost)
        zv, lk = self._known_and_likely(zones, lost, hand_count)
        x = np.maximum(self.color_freq(opp_colors, sv) - forced, 0)
        n = max(40 - int(forced.sum()) - sum(extra.values()), (hand_count or 0) + 1)
        fill = x * (n / x.sum()) if x.sum() > 0 else x
        return self._mix_likely(self._weighted(np.maximum(forced - zv - lk, 0) + fill, turn), lk, hand_count)


class UniformPoolModel(_Base):
    """Baseline: every card of the revealed colours (and colourless ones) equally likely."""
    name = "uniform"

    def pool_mask(self, opp_colors, seen_vec) -> np.ndarray:
        S = color_bits(opp_colors) if opp_colors is not None else self.seen_colors(seen_vec)
        if S == 0:
            return np.ones(self.C, bool)
        return (self.meta.colors & ~np.uint8(S)) == 0

    def _sample_deck(self, opp_colors, sv, forced, zv, extra, hand_count: int, rng) -> np.ndarray:
        m = np.nonzero(self.pool_mask(opp_colors, sv))[0]
        n = max(40 - int(forced.sum()) - sum(extra.values()), hand_count + 1)
        d = forced.astype(np.int16).copy()
        np.add.at(d, rng.choice(m, size=n, replace=True), 1)
        self.last_replaced = 0
        return d

    def marginal(self, opp_colors, seen_cards, known_zone_cards=None, turn: int | None = None, lost=None,
                 hand_count: int | None = None) -> np.ndarray:
        zones = as_counter(known_zone_cards)
        sv, forced, extra = self.forced(as_counter(seen_cards), zones, lost)
        zv, lk = self._known_and_likely(zones, lost, hand_count)
        mask = self.pool_mask(opp_colors, sv).astype(np.float64)
        n = max(40 - int(forced.sum()) - sum(extra.values()), (hand_count or 0) + 1)
        fill = mask * (n / max(mask.sum(), 1.0))
        return self._mix_likely(self._weighted(np.maximum(forced - zv - lk, 0) + fill, turn), lk, hand_count)


def _weighted_fill(x: np.ndarray, n: int, rng) -> np.ndarray:
    """n cards drawn without replacement from 'copy units': card c has ceil(x_c) units, the last one
    fractional, so the expected copies track x (Efraimidis-Spirakis weighted sampling)."""
    reps = np.ceil(x).astype(np.int64)
    card = np.repeat(np.arange(len(x)), reps)
    out = np.zeros(len(x), np.int16)
    if len(card) == 0:
        return out
    j = np.arange(len(card)) - np.repeat(np.cumsum(reps) - reps, reps)
    wt = np.clip(x[card] - j, 1e-9, 1.0)
    keys = np.log(rng.random(len(card))) / wt
    take = card[np.argsort(-keys)[:n]]
    np.add.at(out, take, 1)
    if len(take) < n:                  # the frequencies ran out: top up in proportion to them
        p = x / x.sum() if x.sum() > 0 else np.full(len(x), 1 / len(x))
        np.add.at(out, rng.choice(len(x), size=n - len(take), p=p), 1)
    return out


# =============================================================================================
# StateSpec determinization
# =============================================================================================

def zone_cards(spec: StateSpec, seat: str = "B") -> Counter:
    """The named cards of a seat's deck that the spec places in a known zone, counted as
    StateSpec.validate and the bridge take them out of its decklist: its graveyard, exile, known
    hand and known library top, the cards (not tokens) it owns on either battlefield (a permanent
    is listed under its controller, with Perm.owner when the other seat owns it: a stolen or
    reanimated card) and its spells on the stack."""
    p = spec.players[seat]
    c = Counter(p.hand + p.graveyard + p.exile + p.libraryTop)
    for s, q in spec.players.items():
        for perm in q.battlefield:
            if perm.name and not (perm.token or perm.tokenClass) and (perm.owner or s) == seat:
                c[perm.name] += perm.count
    c.update(x.card for x in spec.stack if x.controller == seat and x.card)
    return c


def turns_done(spec: StateSpec, seat: str) -> int:
    """How many of its own turns `seat` has completed at this spec (its current turn counts once
    the END phase is reached)."""
    # the starting player takes the odd global turns
    start = spec.startingPlayer or (spec.activePlayer if spec.turn % 2 else
                                    SEATS[1 - SEATS.index(spec.activePlayer)])
    k = (spec.turn + 1) // 2 if start == seat else spec.turn // 2
    if spec.activePlayer == seat and spec.phase != "END":
        k -= 1
    return max(k, 0)


def determinize(spec: StateSpec, model: _Base, k: int, seed: int = 0, opp_colors: str | None = None,
                seen=None, lost=None) -> list[StateSpec]:
    """k copies of `spec` with the opponent's (B's) hidden cards sampled: B.decklist from the model
    (decklistSource "belief"; an "exact" decklist is kept) and B.hand = known + sampled cards
    (handUnknown 0). `seen` is the multiset of B's cards revealed so far (default: its zone cards;
    A's cards that B controls, Perm.owner "A", are taken out of it).
    The hidden hand is drawn from the deck minus the cards the spec places (zone_cards), so a seen
    card in no zone of the spec can be in it. `lost`: B's cards that left the battlefield for an
    unrecorded zone, reconstruct.opp_location_unknown(g, n, split=True) = (likely in hand, the
    rest), at the snapshot the spec starts from (after=True for state_after_user_turn); a likely
    one is in the hand with probability P_LIKELY_IN_HAND (84% measured, against 6% of the rest).
    Determinization i uses the RNG seeded by (seed, i), so the first j of k are the same for any k.
    A is untouched."""
    B = spec.players["B"]
    zones = zone_cards(spec, "B")
    # the seen multiset (reconstruct's revealed cards) counts every card seen on B's side, A's cards
    # that B controls included (listed under B with Perm.owner "A"): those are not B's
    theirs = Counter(p.name for p in B.battlefield if p.owner not in (None, "B") and p.name
                     and not (p.token or p.tokenClass) for _ in range(p.count))
    seen = (as_counter(seen) - theirs) | zones
    turn = turns_done(spec, "B")
    likely = model.vec(split_lost(lost)[0])[0]
    out = []
    for i in range(k):
        rng = np.random.default_rng([seed, i])
        s = StateSpec.from_dict(spec.to_dict())
        sb = s.players["B"]
        sampled = ["B.hand"] if B.handUnknown > 0 else []
        if B.decklistSource == "exact" and len(B.decklist) >= 40:
            deck = list(B.decklist)
            # cards of an exact list outside the model's card pool stay in the library
            dv, _ = model.vec(deck)
            zv, _ = model.vec(zones)
            known = np.minimum(zv, dv)
            short = B.handUnknown + 1 - int((dv - known).sum())
            if short > 0:
                # the known zones (or cards outside the pool) leave too few cards for the hidden
                # hand plus a library card: pad with basics rather than silently shrink the hand;
                # the list is then no longer exactly the real one (0 of 10,371 FDN pair specs)
                f = model.filler(model.seen_colors(dv))
                dv = model.fit_room(dv, known, B.handUnknown, f)
                deck = sorted(deck + [model.cards[f]] * short)
                sb.decklistSource = "belief"
                sampled.insert(0, "B.decklist")
                s.provenance.flags.append("B_decklist_padded")
            hand = model.names(model.draw_hand(dv, known, B.handUnknown, rng, turn, likely))
        else:
            deck, hand = model.sample(opp_colors, seen, B.handUnknown, zones, rng, turn, lost)
            sb.decklistSource = "belief"
            sampled.insert(0, "B.decklist")
        sb.decklist = deck
        sb.hand = sorted(sb.hand + hand)
        sb.handUnknown = 0
        s.provenance.sampled = sorted(set(s.provenance.sampled) | set(sampled))
        s.provenance.flags = [f for f in s.provenance.flags if f != "opp_decklist_placeholder"]
        s.comment = (s.comment + " | " if s.comment else "") + f"B sampled by belief.{model.name} (seed {seed}, #{i})"
        out.append(s)
    return out


def game_evidence(g: replay.Game, n: int, ids: Ids | None = None, entry: str = "eot_rollover"):
    """(StateSpec at user turn n, the opponent's revealed cards so far, its 17lands opp_colors or
    None when the row has none: 0.3% of rows, mostly games that ended before a colour showed).
    opp_colors is HINDSIGHT: the colours revealed over the whole game, not by turn n. For a
    hindsight-free belief (coaching, the 'seen-colours' rows of the evaluation) pass None instead."""
    from draftzero.gameplay import reconstruct as rc
    ids = ids or Ids.load(g.meta.get("expansion") or "FDN")
    spec = rc.state_at_user_turn(g, n, entry, ids=ids)
    p = g.prev_slot(n)
    seen = Counter(rc.analyze(g, ids).states[p.seq].revealed) if p is not None else Counter()
    return spec, seen, g.meta.get("opp_colors") or None


# =============================================================================================
# Evaluation on mirrored pairs
# =============================================================================================

@dataclass
class View:
    """One user turn of one game half, with the other half's truth."""
    turn: int                 # user turn
    opp_turns: int            # the opponent's completed turns at the snapshot
    spec: StateSpec
    colors: str | None
    seen: Counter
    zones: Counter
    hand_count: int
    true_hand: Counter
    true_deck: Counter
    # reconstruct.opp_location_unknown(split=True) at the snapshot: (likely in hand, the rest)
    lost: tuple = field(default_factory=lambda: (Counter(), Counter()))


def pair_views(g: replay.Game, other: replay.Game, turns, ids: Ids) -> list[View]:
    """Views of game g (user = g's user) at the given user turns, the truth read from `other`, the
    same game recorded by the opponent. The true hand is the other row's own end-of-turn hand of
    the half-turn whose snapshot starts user turn n."""
    from draftzero.gameplay import reconstruct as rc
    out = []
    for n in turns:
        if n not in g.decision_turns():
            continue
        p = g.prev_slot(n)
        if p is None:
            continue
        t = other.user_slot(p.n)
        if t is None:
            continue
        spec, seen, colors = game_evidence(g, n, ids)
        truth = Counter(ids.name(c) for c in t.L("eot_user_cards_in_hand") if c != -1 and ids.cards17.get(c))
        out.append(View(n, p.n, spec, colors, seen, zone_cards(spec, "B"), spec.players["B"].handUnknown,
                        truth, Counter(other.deck), rc.opp_location_unknown(g, n, ids, split=True)))
    return out


def _overlap(a: np.ndarray, b: np.ndarray, mask=None) -> tuple[int, int]:
    if mask is not None:
        a, b = a[mask], b[mask]
    return int(np.minimum(a, b).sum()), int(b.sum())


class _Acc:
    def __init__(self):
        self.s = Counter()

    def add(self, **kw):
        for k, v in kw.items():
            self.s[k] += v

    def result(self) -> dict:
        s = self.s
        r = lambda a, b: round(s[a] / s[b], 4) if s[b] else None
        return {"views": s["views"], "true_cards": s["true"], "recall@K": r("hit", "trueK"),
                "recall_nonland@K": r("hit_nl", "true_nlK"), "coverage@K": r("cov", "true"),
                "ll_per_card": r("ll", "true"), "ll_per_nonland_card": r("ll_nl", "true_nl"),
                "hand_lands_sampled": r("lands_s", "samples"), "hand_lands_true": r("lands_t", "views"),
                "deck_overlap": r("deck_hit", "deck_n"), "deck_overlap_nonland": r("deck_hit_nl", "deck_n_nl"),
                "swaps_per_deck": r("swapped", "samples"),
                "lost_in_true_hand": s["true_lost"], "recall_lost@K": r("hit_lost", "true_lostK")}


def run_model(model: _Base, views: list[View], K: int, seed: int, colors: str = "row",
              oracle: bool = False) -> dict:
    """Metrics per user turn (and 'all') for one model over the views. oracle: the true deck
    instead of a sampled one (the model supplies only the hand draw)."""
    acc: dict = {}
    nl = model.meta.kind != KIND_LAND
    floor = 1.0 / model.C
    for vi, v in enumerate(views):
        if not v.true_hand:
            continue
        oc = v.colors if colors == "row" else None
        tv, _ = model.vec(v.true_hand)
        dv, _ = model.vec(v.true_deck)
        rng = np.random.default_rng([seed, vi])
        _, forced, _ = model.forced(v.seen, v.zones, v.lost)
        zv, _ = model.vec(v.zones)
        lk, _ = model.vec(v.lost[0])
        lost_t = np.minimum(tv, model.vec(v.lost[0] + v.lost[1])[0])   # true hand cards that were lost
        hits = hits_nl = hits_lost = swapped = lands = 0
        best = np.zeros(model.C, np.int16)
        dh = dhn = dn = dnn = 0
        od = model.fit_room(np.maximum(dv, forced), zv, v.hand_count, model.filler(0))
        for _ in range(K):
            if oracle:
                d, h = od, model.draw_hand(od, zv, v.hand_count, rng, v.opp_turns, lk)
            else:
                d, h, _ = model.sample_vec(oc, v.seen, v.zones, v.hand_count, rng, v.opp_turns, v.lost)
                swapped += model.last_replaced
            hits += int(np.minimum(h, tv).sum())
            hits_lost += int(np.minimum(h, lost_t).sum())
            hits_nl += int(np.minimum(h, tv)[nl].sum())
            lands += int(h[~nl].sum())
            best = np.maximum(best, h)
            a, b = _overlap(d, dv)
            dh, dn = dh + a, dn + b
            a, b = _overlap(d, dv, nl)
            dhn, dnn = dhn + a, dnn + b
        if oracle:
            p = model._mix_likely(model._weighted(np.maximum(od - zv - lk, 0).astype(np.float64), v.opp_turns),
                                  lk, v.hand_count)
        else:
            p = model.marginal(oc, v.seen, v.zones, v.opp_turns, v.lost, v.hand_count)
        lp = np.log((1 - LL_FLOOR) * p + LL_FLOOR * floor) * tv
        for key in (v.turn, "all"):
            acc.setdefault(key, _Acc()).add(
                views=1, true=int(tv.sum()), true_nl=int(tv[nl].sum()), trueK=int(tv.sum()) * K,
                true_nlK=int(tv[nl].sum()) * K, hit=hits, hit_nl=hits_nl, cov=int(np.minimum(best, tv).sum()),
                ll=float(lp.sum()), ll_nl=float(lp[nl].sum()), lands_s=lands, lands_t=int(tv[~nl].sum()),
                deck_hit=dh, deck_n=dn, deck_hit_nl=dhn, deck_n_nl=dnn, samples=K, swapped=swapped,
                true_lost=int(lost_t.sum()), true_lostK=int(lost_t.sum()) * K, hit_lost=hits_lost)
    return {str(k): a.result() for k, a in acc.items()}


def tune_alpha(model: OpponentModel, views: list[View], grid=(0.1, 0.3, 1.0, 3.0, 10.0)) -> tuple[float, dict]:
    """α with the best mean per-card log-likelihood of the true hands on the dev views."""
    scores = {}
    floor = 1.0 / model.C
    for a in grid:
        model.alpha = a
        ll = n = 0.0
        for v in views:
            if not v.true_hand:
                continue
            tv, _ = model.vec(v.true_hand)
            p = (1 - LL_FLOOR) * model.marginal(v.colors, v.seen, v.zones, None, v.lost, v.hand_count) + LL_FLOOR * floor
            idx = np.nonzero(tv)[0]
            ll += float((np.log(p[idx]) * tv[idx]).sum())
            n += int(tv.sum())
        scores[a] = round(ll / max(n, 1), 4)
    best = max(scores, key=scores.get)
    model.alpha = best
    return best, scores


def evaluate(n_pairs: int = 2500, n_dev: int = 300, turns=(3, 5, 7), K: int = 16, seed: int = 0,
             alpha: float | None = None, set_code: str = "FDN", fmt: str = "PremierDraft",
             pairs_file=None, replay_file=None, retention: bool = True, log=print) -> dict:
    from draftzero.gameplay.pairs import load_pairs
    t0 = time.time()
    ids = Ids.load(set_code)
    path = replay_file or replay.replay_path(set_code, fmt)
    pairs = load_pairs(pairs_file, set_code, fmt)
    rnd = random.Random(seed)
    chosen = rnd.sample(pairs, min(len(pairs), n_pairs + n_dev))
    ev, dev = chosen[:n_pairs], chosen[n_pairs:]
    games = replay.read_games({r for q in chosen for r in (q.row_a, q.row_b)}, path)
    log(f"read {len(games)} rows of {len(chosen)} pairs in {time.time() - t0:.0f} s")
    drafts = {g.meta.get("draft_id") or "" for g in games.values()}
    pool_all = DeckPool.load(set_code, fmt)
    pool = pool_all.without_drafts(drafts)
    log(f"deck pool: {len(pool_all)} distinct decks ({int(pool_all.games.sum())} games); "
        f"{len(pool_all) - len(pool)} decks of the {len(drafts)} evaluated drafts held out")
    meta = CardMeta.from_ids(pool.cards, ids)
    ret = None
    if retention:
        t1 = time.time()
        ret = HandRetention.load(pool.index, meta, ids, set_code, fmt, exclude_drafts=drafts, cache=False)
        log(f"hand retention table: {len(ret.cells)} cells from {sum(c[1] for c in ret.cells.values()):,} hidden "
            f"cards (held-out drafts skipped) in {time.time() - t1:.0f} s")

    def views_of(qs):
        out = []
        for q in qs:
            a, b = games[q.row_a], games[q.row_b]
            out += pair_views(a, b, turns, ids) + pair_views(b, a, turns, ids)
        return out
    t1 = time.time()
    V, Vdev = views_of(ev), views_of(dev)
    agree = sum(v.hand_count == sum(v.true_hand.values()) for v in V)
    log(f"{len(V)} eval views, {len(Vdev)} dev views built in {time.time() - t1:.0f} s; "
        f"hand count == true hand size in {agree}/{len(V)}")
    deck = OpponentModel(pool, meta)
    if alpha is None:
        alpha, scores = tune_alpha(deck, Vdev)
        log(f"alpha tuned on dev: {alpha} (per-card log-lik {scores})")
    else:
        deck.alpha, scores = alpha, {}
    bad = 0
    for i, v in enumerate(V):                   # every view's spec must determinize into a valid full spec
        s = determinize(v.spec, deck, 1, seed + i, v.colors, v.seen, v.lost)[0]
        bad += bool(s.validate()) or s.is_partial()
    log(f"determinize: {bad} of {len(V)} specs invalid or partial")
    freq, unif = FrequencyModel(pool, meta), UniformPoolModel(pool, meta)
    runs = [("deck", deck, "row", False), ("freq", freq, "row", False), ("uniform", unif, "row", False),
            ("deck/seen-colours", deck, "seen", False), ("freq/seen-colours", freq, "seen", False),
            ("oracle-deck", deck, "row", True)]
    if ret is not None:
        deck_r = OpponentModel(pool, meta, alpha=alpha, retention=ret)
        freq_r = FrequencyModel(pool, meta, retention=ret)
        runs += [("deck+retention", deck_r, "row", False), ("freq+retention", freq_r, "row", False),
                 ("oracle-deck+retention", deck_r, "row", True)]
    results = {}
    for name, m, cmode, orc in runs:
        t2 = time.time()
        results[name] = run_model(m, V, K, seed, cmode, orc)
        log(f"  {name}: {time.time() - t2:.0f} s")
    return {
        "set": set_code, "format": fmt, "pairs": len(ev), "dev_pairs": len(dev), "views": len(V),
        "turns": list(turns), "K": K, "seed": seed, "alpha": alpha,
        "alpha_dev_scores": {str(k): v for k, v in scores.items()},
        "pool_decks": len(pool), "held_out_drafts": len(drafts), "hand_count_agrees": agree,
        "determinize_invalid": bad, "colour_reveal": {"main": P_MAIN_SEEN, "splash": P_SPLASH_SEEN},
        "ll_floor": LL_FLOOR, "results": results, "seconds": round(time.time() - t0, 1),
        "notes": ["views = user turns of both halves of each pair; truth = the other half's end-of-turn hand",
                  "colours 'row' = 17lands opp_colors (whole-game revealed colours); 'seen' = from cards seen so far",
                  "oracle-deck draws the hand from the TRUE deck: the ceiling for any deck sampler",
                  "ll_per_card: mean natural log P(card) of the true hand's cards under the model's one-card marginal",
                  "+retention: hand drawn with odds by card kind x mana value x opponent turns (users' own hands)"],
    }


def markdown(r: dict) -> str:
    turns = [str(t) for t in r["turns"]] + ["all"]
    lines = [f"Opponent belief on {r['pairs']:,} held-out mirrored pairs ({r['views']:,} views), K={r['K']}, "
             f"alpha={r['alpha']}, pool {r['pool_decks']:,} decks",
             "", "| model | turn | views | recall@K | recall nonland | coverage@K | log-lik/card | log-lik nonland "
             "| lands in hand (sampled/true) | deck overlap | nonland overlap |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, res in r["results"].items():
        for t in turns:
            x = res.get(t)
            if not x:
                continue
            lines.append(f"| {name} | {t} | {x['views']} | {x['recall@K']} | {x['recall_nonland@K']} | {x['coverage@K']} | "
                         f"{x['ll_per_card']} | {x['ll_per_nonland_card']} | {x['hand_lands_sampled']}/{x['hand_lands_true']} | "
                         f"{x['deck_overlap']} | {x['deck_overlap_nonland']} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m draftzero.gameplay.belief",
                                 description="Evaluate opponent belief models on held-out mirrored 17lands pairs.")
    ap.add_argument("--set", default="FDN")
    ap.add_argument("--format", default="PremierDraft")
    ap.add_argument("--pairs", type=int, default=2500, help="evaluated pairs (both halves each)")
    ap.add_argument("--dev", type=int, default=300, help="dev pairs for picking alpha")
    ap.add_argument("--turns", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("-K", type=int, default=16, help="samples per view")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=None, help="deck-likelihood smoothing (default: tune on dev)")
    ap.add_argument("--no-retention", action="store_true", help="skip the retention-weighted hand rows")
    ap.add_argument("--out", default=None, help="results json (default data/gameplay/belief_eval.json)")
    ap.add_argument("--rebuild-pool", action="store_true")
    a = ap.parse_args(argv)
    if a.rebuild_pool:
        DeckPool.load(a.set, a.format, rebuild=True)
    r = evaluate(a.pairs, a.dev, tuple(a.turns), a.K, a.seed, a.alpha, a.set, a.format, retention=not a.no_retention)
    out = Path(a.out or OUT_DIR / "belief_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1) + "\n")
    print(markdown(r))
    print(f"\nwrote {out} ({r['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
