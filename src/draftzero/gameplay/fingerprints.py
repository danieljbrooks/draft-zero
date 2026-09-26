"""
Behavioural fingerprints: how a player plays FDN limited at the turn level, computed the same way
for 17lands humans (by skill bucket) and for MageZero games, so a MageZero run can be checked
against human play without mapping a single state into XMage (experiment E0 of the gameplay-data
plan; the literature review ranked it first among the uses of 17lands data).

Exp #1's documented failure is this kind of thing: switched on at gen 3, the self-play priority
prior "halved the number of spells cast" (docs/003-fdn-generalist-report.md §3.5). A fingerprint
shows such a regression in one table: spells per own turn by turn number, land drops, mana use,
attacks and blocks, next to the same numbers for 17lands players.

    17lands replay row ── records_from_game ─────────┐
    MageZero JVM log   ── records_from_jvm_log ──────┤
    ProbePlayer JSONL  ── records_from_probe ────────┼──► per-turn records ──► fingerprint_from_records
    GAME_SUMMARY lines ── records_from_game_summaries┤                          (JSON summary)
    any other logger   ── the record format below ───┘

Per-turn record format (a plain dict per half-turn, from the TRACKED player's point of view;
the records of one game are contiguous and chronological; a new `game` value starts a new game).
Every field except the first four is optional: None or absent means "unknown", and each metric
only counts the turns whose inputs are known.

  game               hashable game id
  side               "me" (the tracked player is active) | "opp" (the opponent is active)
  turn               the ACTIVE player's own turn number, 1-based
  on_play            bool: the tracked player took the first turn
  terminal           bool: the game ended during this half-turn (its events can be incomplete)
  mulligans, won     game-level (read from whichever record of the game carries them)
 side == "me":
  lands_played       lands played this turn
  creatures_cast     creature spells cast
  noncreatures_cast  non-creature permanent spells cast (artifacts, enchantments, planeswalkers)
  instants_sorceries_cast
  cast_mv            total mana value of those spells
  activations        activated non-mana abilities (MageZero's priority actions besides Cast/Play)
  mana_spent         mana spent this turn (17lands: includes activated abilities)
  lands_in_play      the tracked player's lands at the end of the turn
  attack_eligible    creatures that could have attacked: on the battlefield since the turn began,
                     no Defender, no "can't attack" aura, not removed before combat without attacking
  attackers          how many of those attacked
  attackers_new      attackers that entered this turn (haste), not in the rate
  land_in_hand       held a land at some point of the turn
  castable_mv        held a non-land card with mana value <= the lands in play that make mana this
                     turn (not a land played tapped this turn, not Evolving Wilds)
  castable           ... whose coloured pips those lands can also pay (lands only; see below)
  castable_engine    XMage offered at least one "Cast" action this turn (MageZero logs only)
 side == "opp":
  opp_attackers      creatures the opponent attacked with
  blockers           the tracked player's creatures that blocked
  potential_blockers the tracked player's creatures that could have blocked (humans: untapped
                     creatures at the start of that turn; MageZero: creatures XMage asked about)
  instants_cast      instants the tracked player cast during the opponent's turn
  flash_cast         flash permanents cast during the opponent's turn (17lands does not list them:
                     inferred from the snapshots, a card that left the hand for the battlefield)

Metric definitions (by_turn keys: own turn number 1..10, "11+", and "all" pooled; opponent-turn
metrics, prefixed opp_, are keyed by the opponent's turn number). Per-turn metrics skip the game's
terminal half-turn (the game ended during it, often by concession); per-game totals include it.

  land_drop          share of own turns with a land played
  land_miss          share of own turns with a land in hand but no land played
  spells             spells cast per own turn (creature + non-creature permanent + instant/sorcery)
  creature_spells    creature spells per own turn
  activations        activated non-mana abilities per own turn (17lands: resolved ability ids
                     whose category is "activated", see ids.AbilityRef.is_action)
  cast_mv            mana value cast per own turn
  mana_spent         mana spent per own turn (humans only: 17lands counts activations too)
  lands_in_play      lands at the end of the own turn (MageZero logs: lands played so far)
  mana_eff           sum(mana_spent) / sum(lands_in_play): mana efficiency
  cast_mv_eff        sum(cast_mv) / sum(lands_in_play): the same from spells alone (both sources)
  idle               share of own turns in which the player held a castable card (`castable`) and
                     cast nothing, neither that turn nor on the following opponent turn (instants
                     and flash permanents).
                     APPROXIMATE: castability is judged from mana value and coloured pips against
                     the lands in play after the land drop; it ignores mana creatures, Treasures,
                     cost reductions, X costs (X = 0), additional costs and whether a target
                     exists, and a held combat trick or counterspell can be the right play
  idle_given         the same share among the turns in which a castable card was held
  idle_mv, idle_mv_given          castability by mana value only
  idle_engine, idle_engine_given  castability as XMage saw it (MageZero only)
  attack_rate        sum(attackers) / sum(attack_eligible), over turns with >= 1 eligible creature
  attack_any         share of those turns with at least one attacker
  attackers          attackers per own turn (including hasty newcomers)
  opp_blockers_per_attacker  sum(blockers) / sum(opp_attackers) over opponent turns with an attack
  opp_block_any      share of opponent attacks (with >= 1 potential blocker) the player blocked
  opp_block_rate     sum(blockers) / sum(potential_blockers) over opponent attacks
  opp_instants       instants cast per opponent turn
 game-level: games, win_rate, mulligan_rate (>= 1 mulligan), mulligans_mean, turns_global_mean
  and _median (both players' turns, XMage's turn number at the end), own_turns_mean,
  spells_per_game, offturn_instants_per_game, offturn_casts_per_game (instants + flash permanents
  on opponent turns), cast_mix (shares of all casts by family),
  first_spell_turn (own turn of the first spell cast on an own turn: median, mean over games with
  one, share of games with none, histogram).

Skill groups (17lands `user_game_win_rate_bucket`, 2-point steps): lt50, 50to58, ge58; and "top"
(bucket >= 0.60 with user_n_games_bucket >= 100) and "lt50_n100" (bucket < 0.50, n >= 100). The
bucket is computed over games that include the game itself, so for users with few games it leaks
the outcome (critique N3): the n >= 100 groups are the clean skill contrast. Every group is split
by on_play ("play", "draw") and pooled ("all").

Only the 17lands user's side is used: its actions on its own turns are complete and its hand is
known, while the opponent has no skill bucket, a hidden hand (no land_miss or idle) and plays that
17lands does not record on the user's turns (draws, flash creatures; replay_empirics §3.3). The
user's own flash permanents on opponent turns are recovered from the user's snapshots (flash_cast).
The fast attack eligibility here agrees with labels.turn_label (full reconstruction) on 98.6% of
12,512 sampled user turns (every 200th game); the rest are creatures under a "hostile" aura that
can still attack, which labels excludes and this module counts.

MageZero logs. The data-generation JVM log (the fork logs every decision at INFO) carries almost
everything: records_from_jvm_log reads lands, casts, activations, attack and block questions,
the actions XMage offered at each search and every upkeep hand. Missing: mana spent (cast_mv
stands in), lands in play (lands played so far stands in), mulligans (off in the harness), and
Pass choices (never logged as "chose action"; no metric needs them). Attack eligibility there is
the creatures XMage asked about at declare attackers: untapped ones, hasty newcomers included,
while the human count also holds creatures tapped before combat (mana creatures, tap costs), so
MageZero's attack_rate has a slightly smaller denominator by construction.
A cheaper and sturdier source would be one JSON line per player per turn from the fork, e.g.
`TURN_SUMMARY {"game", "seat", "turn", <the record fields above>}`, which the "records" reader
takes as is (as JSONL); GAME_SUMMARY alone gives only game length.

Usage:
  python -m draftzero.gameplay.fingerprints --every 16            # 1/16 sample
  python -m draftzero.gameplay.fingerprints                       # full pass: 791,159 games in
                                                                  # 518 s, 110 MB (M1 Pro, 1 process)
  python -m draftzero.gameplay.fingerprints --human data/gameplay/fingerprints_FDN.json \\
      --mz pilot_gen1=bench_local/exp2_pilot/podA/runs/<run>/jvm_gen1_00_self.log
      # compare MageZero games (a JVM log, ProbePlayer JSONL, GAME_SUMMARY log/games.jsonl or a
      # JSONL of records in the format above) with the human groups

Writes data/gameplay/fingerprints_FDN.json (humans) and, with --mz, fingerprints_magezero.json,
and prints markdown tables. 17lands public data is CC BY 4.0 (https://www.17lands.com/public_datasets).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator

from draftzero.gameplay.ids import BASIC_MANA, COLORS, REPO, Ids
from draftzero.gameplay.replay import Game, TurnRecord, iter_games, replay_path

OUT_DIR = REPO / "data" / "gameplay"
TB_MAX = 11                  # per-turn tables: own turns 1..10, then "11+"
GROUPS = ("all", "lt50", "50to58", "ge58", "top", "lt50_n100")
# Auras whose host can neither attack nor stay untapped. FDN's other "hostile" auras (Eaten by
# Piranhas, Witness Protection) leave a creature that can still attack, and Imprisoned in the
# Moon turns it into a land, so it leaves the creature list by itself.
CANT_ATTACK_AURAS = frozenset({"Pacifism"})


# --- card facts ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CardFacts:
    kind: str                 # land | creature | instant_sorcery | noncreature | unknown
    mv: int
    need: tuple               # ((colour bit, pips), ...) for the coloured part of the cost
    land_mask: int            # colours a land can tap for (0: colourless or none)
    keywords: frozenset
    doesnt_untap: bool
    cant_attack_aura: bool
    mana_land: bool = False   # a land that taps for mana (Evolving Wilds does not)
    etb_tapped: bool = False  # enters the battlefield tapped (no mana the turn it is played)


def _mask(colours: str) -> int:
    return sum(1 << COLORS.index(c) for c in set(colours) if c in COLORS)


class _Facts:
    """CardFacts by grpId (17lands) or by card name (XMage logs), cached per Ids."""

    def __init__(self, ids: Ids):
        self.ids = ids
        self.by_grp: dict[int, CardFacts] = {}
        self.by_name: dict[str, CardFacts] = {}
        self._name_grp: dict[str, int] | None = None
        self.actions: dict[int, bool] = {}

    def is_action(self, aid: int) -> bool:
        a = self.actions.get(aid)
        if a is None:
            a = self.actions[aid] = self.ids.ability(aid).is_action
        return a

    def _make(self, name: str, types: str, mv: int) -> CardFacts:
        info = self.ids.info(name)
        if "Land" in types:
            kind = "land"
        elif "Creature" in types:
            kind = "creature"
        elif "Instant" in types or "Sorcery" in types:
            kind = "instant_sorcery"
        elif types:
            kind = "noncreature"
        else:
            kind = "unknown"
        need = tuple(sorted((1 << COLORS.index(c), k) for c, k in info.pips.items()))
        produces = BASIC_MANA.get(name) or info.land_mana or ""
        return CardFacts(kind, mv, need, _mask(produces), frozenset(info.keywords),
                         "doesnt_untap" in info.features,
                         name in CANT_ATTACK_AURAS or info.attach.startswith("aura:freeze"),
                         kind == "land" and bool(produces), "etb_tapped" in info.features)

    def grp(self, g: int) -> CardFacts:
        f = self.by_grp.get(g)
        if f is None:
            f = self.by_grp[g] = self._make(self.ids.name(g), self.ids.types(g), self.ids.mv(g))
        return f

    def name(self, name: str) -> CardFacts:
        f = self.by_name.get(name)
        if f is None:
            if self._name_grp is None:
                # prefer the set's own printing; other printings carry the same types and mana value
                self._name_grp = {}
                for g, r in self.ids.cards17.items():
                    if r["name"] not in self._name_grp or r.get("expansion") == self.ids.set_code:
                        self._name_grp[r["name"]] = g
            g = self._name_grp.get(name)
            f = self.grp(g) if g is not None else self._make(name, "", 0)
            if g is None and name in BASIC_MANA:   # a basic land missing from the table
                f = CardFacts("land", 0, (), f.land_mask, f.keywords, False, False, True, False)
            self.by_name[name] = f
        return f


_FACTS: dict[int, _Facts] = {}


def _facts(ids: Ids) -> _Facts:
    f = _FACTS.get(id(ids))
    if f is None or f.ids is not ids:
        f = _FACTS[id(ids)] = _Facts(ids)
    return f


@lru_cache(maxsize=200_000)
def payable(land_masks: tuple, mv: int, need: tuple) -> bool:
    """Can these lands (one mana each, of any colour in its mask) pay a cost of mana value `mv`
    with the coloured pips `need`? Hall's condition on the pips-to-lands matching is exact here:
    every set of needed colours must be covered by at least as many lands as it has pips."""
    if len(land_masks) < mv:
        return False
    k = len(need)
    for s in range(1, 1 << k):
        m = p = 0
        for j in range(k):
            if s >> j & 1:
                m |= need[j][0]
                p += need[j][1]
        if sum(1 for lm in land_masks if lm & m) < p:
            return False
    return True


def _castable(hand: Iterable[CardFacts], lands: list[CardFacts], played: Iterable[CardFacts] = ()) \
        -> tuple[bool, bool]:
    """(castable by mana value, castable by mana value and colours) for any non-land card in
    `hand`, paying with `lands` (after the land drop) minus the ones that make no mana this turn:
    lands without a mana ability, and lands `played` this turn that enter tapped."""
    tapped = Counter(f for f in played if f.etb_tapped)
    avail = []
    for f in lands:
        if tapped[f]:
            tapped[f] -= 1
        elif f.mana_land:
            avail.append(f)
    n = len(avail)
    masks = tuple(sorted(f.land_mask for f in avail))
    by_mv = by_col = False
    for f in hand:
        if f.kind == "land" or f.mv > n:
            continue
        by_mv = True
        if payable(masks, f.mv, f.need):
            return True, True
    return by_mv, by_col


# --- 17lands -> records ----------------------------------------------------------------------------------

def skill_groups(meta: dict) -> list[str]:
    """The GROUPS a 17lands game belongs to (see the module docstring)."""
    out = ["all"]
    b = meta.get("user_game_win_rate_bucket")
    if b is None:
        return out
    pct, n = round(b * 100), int(meta.get("user_n_games_bucket") or 0)
    out.append("lt50" if pct < 50 else "50to58" if pct < 58 else "ge58")
    if pct >= 60 and n >= 100:
        out.append("top")
    if pct < 50 and n >= 100:
        out.append("lt50_n100")
    return out


def records_from_game(g: Game, ids: Ids) -> list[dict]:
    """Per-turn records of the 17lands user in one game (format in the module docstring)."""
    F = _facts(ids)
    recs = []
    snap: TurnRecord | None = None       # the last played slot: its snapshot is the current state
    last_user: TurnRecord | None = None
    for t in g.turns:
        if not t.played:                 # an empty "hole" slot: no events and no snapshot
            continue
        r = {"game": g.row_index, "side": "me" if t.side == "user" else "opp", "turn": t.n,
             "on_play": g.on_play, "terminal": t.terminal}
        if t.side == "user":
            r.update(_user_turn(g, t, snap, F))
            last_user = t
        else:
            r.update(_oppo_turn(t, last_user, snap, F))
        recs.append(r)
        snap = t
    if recs:
        recs[0]["mulligans"] = int(g.meta.get("num_mulligans") or 0)
        recs[0]["won"] = g.won
    return recs


def _user_turn(g: Game, t: TurnRecord, snap: TurnRecord | None, F: _Facts) -> dict:
    L = t.L
    crea, nonc, inst = L("creatures_cast"), L("non_creatures_cast"), L("user_instants_sorceries_cast")
    lands = [F.grp(c) for c in L("eot_user_lands_in_play")]
    # the hand over the turn: the previous snapshot (or the kept hand) plus every card drawn
    if snap is None:
        hand = Counter(g.opening_hand)
        hand.subtract(g.bottomed)
        hand = +hand
    else:
        hand = Counter(snap.L("eot_user_cards_in_hand"))
    hand.update(L("cards_drawn"))
    hand.update(L("cards_tutored"))
    held = [F.grp(c) for c in hand]
    by_mv, by_col = _castable(held, lands, [F.grp(c) for c in L("lands_played")])
    # attack eligibility: creatures on the battlefield when the turn began
    board = Counter(c for c in (snap.L("eot_user_creatures_in_play") if snap else ())
                    if "Defender" not in F.grp(c).keywords)
    att = Counter(L("creatures_attacked"))
    from_board = att & board
    stay = board - from_board
    excl = sum(k for c, k in stay.items() if F.grp(c).doesnt_untap)   # may have started tapped
    stay = Counter({c: k for c, k in stay.items() if not F.grp(c).doesnt_untap})
    gone = Counter(L("user_creatures_killed_non_combat")) & stay        # removed before it could attack
    left = sum(stay.values()) - sum(gone.values())
    pacified = sum(1 for c in (snap.L("eot_oppo_non_creatures_in_play") if snap else ()) if F.grp(c).cant_attack_aura)
    excl += sum(gone.values()) + min(pacified, left)
    return {
        "lands_played": len(L("lands_played")),
        "creatures_cast": len(crea), "noncreatures_cast": len(nonc), "instants_sorceries_cast": len(inst),
        "cast_mv": sum(F.grp(c).mv for c in crea + nonc + inst),
        "activations": sum(1 for a in L("user_abilities") if F.is_action(a)),
        "mana_spent": int(t.num("user_mana_spent")),
        "lands_in_play": len(lands),
        "attack_eligible": sum(board.values()) - excl,
        "attackers": sum(from_board.values()),
        "attackers_new": sum((att - board).values()),
        # a land played was held even when the hand model missed it (0.13% of user turns)
        "land_in_hand": bool(L("lands_played")) or any(f.kind == "land" for f in held),
        "castable_mv": by_mv, "castable": by_col,
    }


def _oppo_turn(t: TurnRecord, last_user: TurnRecord | None, snap: TurnRecord | None, F: _Facts) -> dict:
    pot = 0
    if last_user is not None:
        # the user's attackers (without vigilance) stay tapped until the user's next untap step
        board = Counter(last_user.L("eot_user_creatures_in_play"))
        tapped = Counter(c for c in last_user.L("creatures_attacked") if "Vigilance" not in F.grp(c).keywords)
        pot = sum(board.values()) - sum((board & tapped).values())
    return {"opp_attackers": len(t.L("creatures_attacked")), "blockers": len(t.L("creatures_blocking")),
            "potential_blockers": pot, "instants_cast": len(t.L("user_instants_sorceries_cast")),
            "flash_cast": _flash_cast(t, snap)}


def _flash_cast(t: TurnRecord, prev: TurnRecord | None) -> int:
    """User permanents cast on the opponent's turn. 17lands lists only the non-active player's
    instants and sorceries, but such a cast shows in the snapshots: a card that left the user's
    hand and is now on the user's battlefield (replay_empirics §4.3,
    "flash_creature_on_opp_turn_uncast": 0.26-0.28 per game, against 0.90 instants), or died there
    this turn (flash blockers: 0.07 more). 97% of them are cards with the Flash keyword (1/40
    sample). A lower bound otherwise: a second copy drawn the same turn, or a flash permanent that
    left by exile or bounce, hides it."""
    if prev is None:
        return 0
    left = Counter(prev.L("eot_user_cards_in_hand")) - Counter(t.L("eot_user_cards_in_hand"))
    if not left:
        return 0
    before = Counter(prev.L("eot_user_creatures_in_play")) + Counter(prev.L("eot_user_non_creatures_in_play"))
    after = (Counter(t.L("eot_user_creatures_in_play")) + Counter(t.L("eot_user_non_creatures_in_play"))
             + Counter(t.L("user_creatures_killed_combat")) + Counter(t.L("user_creatures_killed_non_combat")))
    return sum((left & (after - before)).values())


# --- aggregation ---------------------------------------------------------------------------------------

@dataclass
class GameContrib:
    """One game's contribution to a fingerprint (computed once, added to every group it is in)."""
    on_play: bool
    won: bool | None = None
    mulligans: int | None = None
    turns_global: int = 0
    own_turns: int = 0
    first_spell: int | str | None = None      # own turn number, "never", or None (no spell data)
    casts: Counter = field(default_factory=Counter)
    cast_known: bool = False
    items: list = field(default_factory=list)  # (metric, turn bucket, numerator, denominator)


def game_contrib(recs: list[dict]) -> GameContrib:
    """Turn one game's records into its metric contributions (definitions in the module docstring)."""
    on_play = bool(recs[0].get("on_play"))
    c = GameContrib(on_play)
    add = c.items.append
    for k, r in enumerate(recs):
        if r.get("mulligans") is not None:
            c.mulligans = int(r["mulligans"])
        if r.get("won") is not None:
            c.won = bool(r["won"])
        me = r["side"] == "me"
        n = int(r["turn"])
        c.turns_global = max(c.turns_global, 2 * n - 1 if me == on_play else 2 * n)
        tb = min(n, TB_MAX)
        term = bool(r.get("terminal"))
        if me:
            c.own_turns += 1
            fam = (r.get("creatures_cast"), r.get("noncreatures_cast"), r.get("instants_sorceries_cast"))
            spells = None if all(x is None for x in fam) else sum(x or 0 for x in fam)
            if spells is not None:
                c.cast_known = True
                c.casts.update({"creature": fam[0] or 0, "noncreature": fam[1] or 0, "instant_sorcery": fam[2] or 0})
                if spells and c.first_spell is None:
                    c.first_spell = n
            if term:
                continue
            nxt = recs[k + 1] if k + 1 < len(recs) and recs[k + 1]["side"] == "opp" else None
            off = (nxt.get("instants_cast") or 0) + (nxt.get("flash_cast") or 0) if nxt else 0
            lp = r.get("lands_played")
            if lp is not None:
                add(("land_drop", tb, lp > 0, 1))
                if r.get("land_in_hand"):
                    add(("land_miss", tb, lp == 0, 1))
            if spells is not None:
                add(("spells", tb, spells, 1))
                add(("creature_spells", tb, fam[0] or 0, 1))
                quiet = spells == 0 and off == 0
                for key, m in (("castable", "idle"), ("castable_mv", "idle_mv"), ("castable_engine", "idle_engine")):
                    held = r.get(key)
                    if held is not None:
                        add((m, tb, bool(held) and quiet, 1))
                        if held:
                            add((m + "_given", tb, quiet, 1))
            if r.get("activations") is not None:
                add(("activations", tb, r["activations"], 1))
            ms, lip, cmv = r.get("mana_spent"), r.get("lands_in_play"), r.get("cast_mv")
            if ms is not None:
                add(("mana_spent", tb, ms, 1))
            if cmv is not None:
                add(("cast_mv", tb, cmv, 1))
            if lip is not None:
                add(("lands_in_play", tb, lip, 1))
                if lip > 0 and ms is not None:
                    add(("mana_eff", tb, ms, lip))
                if lip > 0 and cmv is not None:
                    add(("cast_mv_eff", tb, cmv, lip))
            el, at, new = r.get("attack_eligible"), r.get("attackers"), r.get("attackers_new")
            if el:
                add(("attack_rate", tb, min(at or 0, el), el))
                add(("attack_any", tb, (at or 0) > 0, 1))
            if at is not None:
                add(("attackers", tb, at + (new or 0), 1))
        else:
            ic, fl = r.get("instants_cast"), r.get("flash_cast")
            if ic is not None or fl is not None:
                c.casts.update({"offturn_instant": ic or 0, "offturn_flash": fl or 0})
            if term:
                continue
            if ic is not None:
                add(("opp_instants", tb, ic, 1))
            oa, bl, pot = r.get("opp_attackers"), r.get("blockers") or 0, r.get("potential_blockers")
            if oa:
                add(("opp_blockers_per_attacker", tb, bl, oa))
                if pot is None or pot > 0:
                    add(("opp_block_any", tb, bl > 0, 1))
                if pot:
                    add(("opp_block_rate", tb, min(bl, pot), pot))
    if c.cast_known and c.first_spell is None:
        c.first_spell = "never"
    return c


def _median(h: Counter):
    n = sum(h.values())
    if not n:
        return None
    k = 0
    for v in sorted(h):
        k += h[v]
        if 2 * k >= n:
            return v


def _mean(h: Counter):
    n = sum(h.values())
    return sum(v * k for v, k in h.items()) / n if n else None


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


class _Acc:
    """Running sums for one population and one of play/draw."""

    def __init__(self):
        self.games = 0
        self.sums: dict = defaultdict(lambda: [0.0, 0.0])
        self.hist: dict[str, Counter] = defaultdict(Counter)
        self.casts = Counter()
        self.cast_games = 0
        self.won = [0, 0]           # wins, games with a known result
        self.mull = [0, 0, 0]       # games with >= 1 mulligan, total mulligans, games known

    def add(self, c: GameContrib) -> None:
        self.games += 1
        self.hist["turns_global"][c.turns_global] += 1
        self.hist["own_turns"][c.own_turns] += 1
        if c.won is not None:
            self.won[0] += c.won
            self.won[1] += 1
        if c.mulligans is not None:
            self.mull[0] += c.mulligans > 0
            self.mull[1] += c.mulligans
            self.mull[2] += 1
        if c.cast_known:
            self.cast_games += 1
            self.casts.update(c.casts)
            self.hist["first_spell"][c.first_spell] += 1
        for m, tb, num, den in c.items:
            for key in ((m, tb), (m, "all")):
                s = self.sums[key]
                s[0] += num
                s[1] += den

    def merge(self, o: "_Acc") -> "_Acc":
        out = _Acc()
        for a in (self, o):
            out.games += a.games
            for k, h in a.hist.items():
                out.hist[k].update(h)
            out.casts.update(a.casts)
            out.cast_games += a.cast_games
            out.won = [x + y for x, y in zip(out.won, a.won)]
            out.mull = [x + y for x, y in zip(out.mull, a.mull)]
            for k, s in a.sums.items():
                t = out.sums[k]
                t[0] += s[0]
                t[1] += s[1]
        return out

    def summary(self) -> dict:
        own = sum(v for k, v in self.casts.items() if k in ("creature", "noncreature", "instant_sorcery"))
        allc = sum(self.casts.values())
        fs = self.hist["first_spell"]
        fs_num = Counter({k: v for k, v in fs.items() if k != "never"})
        s = {
            "games": self.games,
            "win_rate": _r(self.won[0] / self.won[1]) if self.won[1] else None,
            "mulligan_rate": _r(self.mull[0] / self.mull[2]) if self.mull[2] else None,
            "mulligans_mean": _r(self.mull[1] / self.mull[2]) if self.mull[2] else None,
            "turns_global_mean": _r(_mean(self.hist["turns_global"]), 3),
            "turns_global_median": _median(self.hist["turns_global"]),
            "own_turns_mean": _r(_mean(self.hist["own_turns"]), 3),
            "spells_per_game": _r(own / self.cast_games, 3) if self.cast_games else None,
            "offturn_instants_per_game": _r(self.casts["offturn_instant"] / self.cast_games, 3) if self.cast_games else None,
            "offturn_casts_per_game": _r((self.casts["offturn_instant"] + self.casts["offturn_flash"]) / self.cast_games, 3)
                                      if self.cast_games else None,
            "cast_mix": {k: _r(self.casts[k] / allc) for k in ("creature", "noncreature", "instant_sorcery",
                                                              "offturn_instant", "offturn_flash")} if allc else {},
            "first_spell_turn": {
                "median": _median(fs_num), "mean": _r(_mean(fs_num), 3),
                "never": _r(fs["never"] / self.cast_games) if self.cast_games else None,
                "hist": {_tb_key(k): v for k, v in sorted(_capped(fs_num).items())},
            },
            "by_turn": {}, "n": {},
        }
        for (m, tb), (num, den) in sorted(self.sums.items(), key=lambda x: (x[0][0], str(x[0][1]).zfill(3))):
            key = _tb_key(tb)
            s["by_turn"].setdefault(m, {})[key] = _r(num / den) if den else None
            s["n"].setdefault(m, {})[key] = int(den)
        return s


def _tb_key(tb) -> str:
    return "all" if tb == "all" else ("11+" if tb >= TB_MAX else str(tb))


def _capped(h: Counter) -> Counter:
    out = Counter()
    for k, v in h.items():
        out[min(k, TB_MAX)] += v
    return out


class Fingerprint:
    """A population's fingerprint, split by play/draw. `add_records(recs)` adds one game."""

    def __init__(self):
        self.parts = {"play": _Acc(), "draw": _Acc()}

    def add(self, c: GameContrib) -> None:
        self.parts["play" if c.on_play else "draw"].add(c)

    def add_records(self, recs: list[dict]) -> None:
        if recs:
            self.add(game_contrib(recs))

    def summary(self) -> dict:
        return {"all": self.parts["play"].merge(self.parts["draw"]).summary(),
                "play": self.parts["play"].summary(), "draw": self.parts["draw"].summary()}


def group_games(records: Iterable[dict]) -> Iterator[list[dict]]:
    """Split a record stream into games (a change of `game` starts a new one)."""
    cur, key = [], object()
    for r in records:
        if r["game"] != key and cur:
            yield cur
            cur = []
        key = r["game"]
        cur.append(r)
    if cur:
        yield cur


def fingerprint_from_records(records: Iterable) -> dict:
    """The fingerprint ({"all", "play", "draw"} summaries) of an iterable of per-turn records in
    the format of the module docstring, or of an iterable of per-game record lists."""
    fp = Fingerprint()
    records = iter(records)
    first = next(records, None)
    if first is None:
        return fp.summary()
    if isinstance(first, dict):
        games = group_games(_chain(first, records))
    else:
        games = _chain(first, records)
    for recs in games:
        fp.add_records(list(recs))
    return fp.summary()


def _chain(first, rest):
    yield first
    yield from rest


def human_fingerprints(path=None, every: int = 1, limit: int | None = None, start: int = 0,
                       ids: Ids | None = None, progress: bool = False) -> dict:
    """One pass over a 17lands replay file -> {"meta": ..., "groups": {group: {all, play, draw}}}."""
    ids = ids or Ids.load()
    fps = {g: Fingerprint() for g in GROUPS}
    t0 = time.time()
    n = 0
    for g in iter_games(path, limit=limit, every=every, start=start):
        recs = records_from_game(g, ids)
        if not recs:
            continue
        c = game_contrib(recs)
        for name in skill_groups(g.meta):
            fps[name].add(c)
        n += 1
        if progress and n % 50_000 == 0:
            print(f"  {n} games, {n / (time.time() - t0):.0f} games/s", file=sys.stderr, flush=True)
    meta = {"source": str(path or replay_path()), "every": every, "start": start, "limit": limit,
            "games": n, "seconds": round(time.time() - t0, 1),
            "attribution": "17lands public replay data, CC BY 4.0 (https://www.17lands.com/public_datasets)"}
    return {"meta": meta, "groups": {k: fp.summary() for k, fp in fps.items()}}


# --- MageZero logs -> records ----------------------------------------------------------------------------

# "PRECOMBAT_MAIN0pool= actions: [...]", or with the stack: "UPKEEP1 (top: stack ability (...))pool= ..."
POOL_RE = re.compile(r"^([A-Z_]+)\d*(?: \(top: .*\))?pool= actions: (.*)$")
POOL_ITEM_RE = re.compile(r"\[(.+?) score: \S+ count: \d+\]")
HAND_RE = re.compile(r"^\[(\d+):[^\]]*\](Player[AB]) hand: : (.*)$")
BLOCK_PROMPT = "base choose target choose which creature to block"
SUMMARY_TAG = "GAME_SUMMARY "
CAST_PREFIXES = ("Cast ", "Flashback ")


@dataclass
class _Turnlog:
    """What one player did in one global turn of an XMage game (from any MageZero log)."""
    plays: list = field(default_factory=list)       # land names
    casts: list = field(default_factory=list)       # (card name, cast from hand)
    activations: int = 0                            # other non-Pass priority actions
    attacks: list = field(default_factory=list)     # one bool per attack question ("attack with X?")
    blocks: list = field(default_factory=list)      # one bool per block question ("block with X?")
    block_options: int = 0                          # most attackers offered to one blocker
    offered_cast: bool = False                      # a search had a Cast/Flashback child
    offered_land: bool = False                      # a search had a Play child
    hand: list | None = None                        # a hand seen during the turn (probe logs)
    hand_early: bool = False                        # ... seen in upkeep, before the draw


def _cast_name(action: str, fb: dict) -> tuple[str, bool]:
    """'Cast X' -> (X, from hand); 'Flashback {2}{U}' -> (the card with that flashback key, False)."""
    if action.startswith("Flashback "):
        return fb.get(action, ""), False
    return action[5:], True


def _seat_records(game_id, seat: str, first: str, turns: int, log: dict, F: _Facts, won: bool | None,
                  hands_at: dict | None = None, opp_logged: bool = True) -> list[dict]:
    """Records of `seat` ("A"/"B") from per-(global turn, seat) _Turnlogs.

    hands_at[(turn, seat)] is the hand at that turn's upkeep (JVM logs); without it, the hand a
    probe log saw during the turn is used. opp_logged: the opponent's decisions are in the log too
    (JVM logs), so its attacks are known even when the tracked player had nothing to block with."""
    on_play = first == seat
    other = "B" if seat == "A" else "A"
    second = other if on_play else seat
    recs, lands = [], []
    for gt in range(1, turns + 1):
        active = first if gt % 2 else second
        r = {"game": game_id, "side": "me" if active == seat else "opp", "turn": (gt + 1) // 2,
             "on_play": on_play, "terminal": gt == turns}
        t = log.get((gt, seat), _Turnlog())
        if active == seat:
            lands += [F.name(x) for x in t.plays]
            kinds = Counter(F.name(n).kind for n, _ in t.casts)
            if hands_at is not None and (gt, seat) in hands_at:
                # upkeep hand (before the draw) | end-of-turn hand + what left it: catches the draw
                played = Counter(t.plays) + Counter(n for n, from_hand in t.casts if from_hand)
                hand = Counter(hands_at[(gt, seat)]) | (Counter(hands_at.get((gt + 1, seat), ())) + played)
            else:
                hand = Counter(t.hand) if t.hand is not None else None
            held = [F.name(x) for x in hand] if hand is not None else None
            by_mv, by_col = (_castable(held, lands, [F.name(x) for x in t.plays]) if held is not None
                             else (None, None))
            if held is None and not opp_logged and (gt, seat) not in log:
                # a probe turn with no logged search: XMage offered nothing but Pass
                by_col = False
            r.update({
                "lands_played": len(t.plays), "creatures_cast": kinds["creature"],
                "noncreatures_cast": kinds["noncreature"] + kinds["unknown"] + kinds["land"],
                "instants_sorceries_cast": kinds["instant_sorcery"],
                "cast_mv": sum(F.name(n).mv for n, _ in t.casts), "activations": t.activations,
                "lands_in_play": len(lands),
                "attack_eligible": len(t.attacks), "attackers": sum(t.attacks),
                "land_in_hand": bool(t.plays) or t.offered_land or any(f.kind == "land" for f in held or ()),
                "castable_mv": by_mv, "castable": by_col,
                "castable_engine": t.offered_cast or bool(t.casts),
            })
        else:
            inst = sum(F.name(n).kind == "instant_sorcery" for n, _ in t.casts)
            if opp_logged:
                opp_att, pot = sum(log.get((gt, other), _Turnlog()).attacks), len(t.blocks)
            elif t.blocks:           # a lower bound: the attackers this blocker could block
                opp_att, pot = t.block_options, len(t.blocks)
            else:
                opp_att = pot = None
            r.update({"opp_attackers": opp_att, "blockers": sum(t.blocks), "potential_blockers": pot,
                      "instants_cast": inst, "flash_cast": len(t.casts) - inst})
        recs.append(r)
    if recs:
        recs[0]["won"] = won
    return recs


def records_from_jvm_log(path, ids: Ids, seats: str = "AB") -> Iterator[list[dict]]:
    """Per-game record lists from a MageZero data-generation JVM log. The XMage fork logs every
    decision at INFO; the parse follows magezero.metrics.parse_jvm_log (the harness's own log
    parser): one state machine per game thread, since game threads interleave. Both seats are
    MageZero in self-play; pass seats="A" for games against a baseline.

    From the log: lands and spells chosen ("chose action:"), attack questions ("use attack with:
    X?: true"), block questions ("choose which creature to block for X" -> "Targeting Y"), the
    root actions of each search ("pool= actions:", i.e. castability as XMage saw it), each player's
    hand at every upkeep, and GAME_SUMMARY (first player, winner, turns). Mulligans are off in the
    harness, mana spent is not logged (cast_mv is), and lands_in_play counts the lands played so
    far (fetch and ramp ignored)."""
    from magezero.metrics import ACTOR_RE, CHOSE_RE, LINE_RE, SIM_RE, TURN_RE, USE_ATTACK_RE
    F = _facts(ids)
    fb = {i.flashback_key: n for n, i in ids.infos.items() if i.flashback_key}
    open_games: dict[str, dict] = {}
    done = 0

    def active(g):
        return g["first"] if g["turn"] % 2 else ("B" if g["first"] == "A" else "A")

    def tl(g, turn, seat):
        return g["log"].setdefault((turn, seat), _Turnlog())

    with open(path, errors="replace") as f:
        for raw in f:
            m = LINE_RE.match(raw.rstrip("\n"))
            if not m:
                continue
            _, msg, thread, _ = m.groups()
            if msg.startswith("Player ") and msg.endswith("won the die roll"):
                open_games[thread] = {"first": msg[7], "turn": 0, "log": {}, "hands": {}, "sim": None,
                                      "pending": None, "block": False}
                continue
            g = open_games.get(thread)
            if g is None:
                continue
            if msg.startswith(SUMMARY_TAG):
                s = json.loads(msg[len(SUMMARY_TAG):])
                del open_games[thread]
                done += 1
                first, turns = s.get("first") or g["first"], int(s.get("turns") or g["turn"])
                for seat in seats:
                    won = None if s.get("winner") not in ("A", "B") else s["winner"] == seat
                    yield _seat_records(f"{Path(path).name}#{s.get('game', done)}{seat}", seat, first, turns,
                                        g["log"], F, won, g["hands"])
                continue
            if msg.startswith(("A game simulation failed", "Caught an internal AI/Game exception",
                               "Worker thread failed")):
                del open_games[thread]
                continue
            if (tm := TURN_RE.match(msg)):
                g["turn"] = int(tm.group(1))
            if (hm := HAND_RE.match(msg)):
                g["hands"][(int(hm.group(1)), hm.group(2)[-1])] = [x for x in re.split(r",(?! )", hm.group(3)) if x]
            elif (cm := CHOSE_RE.match(msg)):
                g["pending"] = (int(cm.group(1)), cm.group(3).strip())
            elif (am := ACTOR_RE.match(msg)):
                if g["pending"]:                 # the chooser prints its board right after choosing
                    turn, action = g["pending"]
                    g["pending"] = None
                    t = tl(g, turn, am.group(1)[-1])
                    if action.startswith("Play "):
                        t.plays.append(action[5:])
                    elif action.startswith(CAST_PREFIXES):
                        t.casts.append(_cast_name(action, fb))
                    elif action != "Pass":
                        t.activations += 1
            elif (sm := SIM_RE.match(msg)):
                g["sim"] = sm.group(1)[-1]
            elif (pm := POOL_RE.match(msg)):
                if g["sim"]:
                    t = tl(g, g["turn"], g["sim"])
                    labels = POOL_ITEM_RE.findall(pm.group(2))
                    t.offered_cast |= any(x.startswith(CAST_PREFIXES) for x in labels)
                    t.offered_land |= any(x.startswith("Play ") for x in labels)
                    if g["block"]:
                        t.block_options = max(t.block_options, sum(x != "Stop Choosing" for x in labels))
            elif (um := USE_ATTACK_RE.match(msg)):
                tl(g, g["turn"], active(g)).attacks.append(um.group(2) == "true")
            elif msg.startswith("base choose target "):
                g["block"] = msg.startswith(BLOCK_PROMPT)
            elif msg.startswith("Targeting ") and g["block"]:
                g["block"] = False
                tl(g, g["turn"], "B" if active(g) == "A" else "A").blocks.append(msg != "Targeting Stop Choosing")


def records_from_probe(path, ids: Ids) -> Iterator[list[dict]]:
    """Per-game record lists (seat A) from the research ProbePlayer JSONL (CoachProbe `selfplay`,
    offline MCTS: one JSON line per PlayerA search with >= 2 root children, then one game_over
    line per game). The file must hold one game at a time: the research batch
    (research/out/selfplay.jsonl, and selfplay_all.jsonl which contains it) interleaves concurrent
    JVMs whose game tags repeat, so this reader cannot split it and raises ValueError on it
    (selfplay_ugbr.jsonl and selfplay_test.jsonl are sequential).

    A turn with no logged search is one in which XMage offered PlayerA nothing but Pass (no land to
    play, nothing castable). The hand is the one at the first search after the draw step began.
    The opponent's decisions are not logged: its attacks are known only through PlayerA's block
    questions. Mulligans were off."""
    F = _facts(ids)
    fb = {i.flashback_key: n for n, i in ids.infos.items() if i.flashback_key}
    dec: list[dict] = []
    k = 0
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if not d.get("game_over"):
                if "decision" in d:
                    dec.append(d)
                continue
            k += 1
            # one game's decisions carry its tag and count up from 1; interleaved JVM output
            # (research/out/selfplay.jsonl) breaks both and would silently mix games
            nos = [x.get("decision_no", 0) for x in dec]
            if any(x.get("tag") != d.get("tag") for x in dec) or any(a >= b for a, b in zip(nos, nos[1:])):
                raise ValueError(f"{path}: game {k} ({d.get('tag')}) mixes the decisions of several games; "
                                 "the file interleaves concurrent JVMs, split it per JVM first")
            act = [x for x in dec if x.get("active")]
            if act:
                first = "A" if act[0]["turn"] % 2 else "B"
            elif dec:
                first = "B" if dec[0]["turn"] % 2 else "A"
            else:
                continue
            log: dict = {}
            for x in dec:
                t = log.setdefault((x["turn"], "A"), _Turnlog())
                played = x.get("played") or ""
                kids = [c["action"] for c in x.get("children", [])]
                if t.hand is None or (t.hand_early and x.get("step") not in ("UNTAP", "UPKEEP")):
                    t.hand = list(x.get("hand") or [])
                    t.hand_early = x.get("step") in ("UNTAP", "UPKEEP")
                if x["decision"] == "PRIORITY":
                    t.offered_cast |= any(a.startswith(CAST_PREFIXES) for a in kids)
                    t.offered_land |= any(a.startswith("Play ") for a in kids)
                    if played.startswith("Play "):
                        t.plays.append(played[5:])
                    elif played.startswith(CAST_PREFIXES):
                        t.casts.append(_cast_name(played, fb))
                    elif played and played != "Pass":
                        t.activations += 1
                elif x["decision"] == "CHOOSE_USE" and (x.get("message") or "").startswith("attack with: "):
                    t.attacks.append(played == "use:true")
                elif x["decision"] == "CHOOSE_TARGET" and "which creature to block" in (x.get("message") or ""):
                    t.blocks.append(played != "Stop Choosing")
                    t.block_options = max(t.block_options, sum(a != "Stop Choosing" for a in kids))
            winner = d.get("winner")
            dec = []
            yield _seat_records(f"{Path(path).name}#{k}", "A", first, int(d["turns"]), log, F,
                                None if winner not in ("A", "B") else winner == "A", opp_logged=False)


def records_from_game_summaries(path, seats: str = "A") -> Iterator[list[dict]]:
    """Game-length-only records from GAME_SUMMARY lines (a JVM log) or a harness games.jsonl: the
    fork's summary has the first player, the winner and the turn count, nothing per turn."""
    with open(path, errors="replace") as f:
        for i, line in enumerate(f):
            j = line.find(SUMMARY_TAG)
            try:
                s = json.loads(line[j + len(SUMMARY_TAG):].rsplit(" =>[", 1)[0]) if j >= 0 else json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(s, dict) or "turns" not in s or "first" not in s:
                continue
            T = int(s["turns"])
            for seat in seats:
                won = None if s.get("winner") not in ("A", "B") else s["winner"] == seat
                recs = [{"game": f"{i}{seat}", "turn": (gt + 1) // 2, "on_play": s["first"] == seat,
                         "side": "me" if (gt % 2 == 1) == (s["first"] == seat) else "opp", "terminal": gt == T}
                        for gt in range(1, T + 1)]
                if recs:
                    recs[0]["won"] = won
                    yield recs


def read_records(path) -> Iterator[dict]:
    """Records from a JSONL file in the module's record format."""
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def detect_format(path) -> str:
    """jvm | probe | records | summaries, from the first JSON line or the log's decision lines."""
    with open(path, errors="replace") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line.startswith("{"):
                d = json.loads(line)
                if "decision" in d or "game_over" in d:
                    return "probe"
                if "side" in d and "turn" in d:
                    return "records"
                if "turns" in d and "first" in d:
                    return "summaries"
            elif "won the die roll" in line or "chose action:" in line:
                return "jvm"
            elif SUMMARY_TAG in line:
                return "summaries"
    return "jvm"


def magezero_fingerprint(paths: list[str], ids: Ids, seats: str = "AB") -> dict:
    """Fingerprint of MageZero games from one or more logs (formats detected per file)."""
    fp = Fingerprint()
    files = []
    for p in paths:
        fmt = detect_format(p)
        if fmt == "jvm":
            games = records_from_jvm_log(p, ids, seats)
        elif fmt == "probe":
            games = records_from_probe(p, ids)
        elif fmt == "records":
            games = group_games(read_records(p))
        else:
            games = records_from_game_summaries(p, seats)
        n = 0
        for recs in games:
            fp.add_records(recs)
            n += 1
        files.append({"path": str(p), "format": fmt, "games": n})
        if not n:   # e.g. a WARN-level JVM log, or a run without GAME_SUMMARY lines: say so
            print(f"warning: {p} ({fmt}) gave no finished games", file=sys.stderr)
    out = fp.summary()
    out["files"] = files
    return out


# --- reporting -----------------------------------------------------------------------------------------

HEADLINE = [
    ("games", "games", "{:.0f}"),
    ("win_rate", "win rate", "{:.3f}"),
    ("mulligan_rate", "mulligan rate", "{:.3f}"),
    ("turns_global_mean", "game length (turns, both players)", "{:.1f}"),
    ("own_turns_mean", "own turns per game", "{:.1f}"),
    ("spells_per_game", "spells cast per game (own turns)", "{:.2f}"),
    ("first_spell_turn.median", "first spell: own turn (median)", "{}"),
    ("first_spell_turn.mean", "first spell: own turn (mean)", "{:.2f}"),
    ("first_spell_turn.never", "games with no spell", "{:.3f}"),
    ("by_turn.spells.all", "spells per own turn", "{:.3f}"),
    ("by_turn.activations.all", "activated abilities per own turn", "{:.3f}"),
    ("by_turn.land_drop.all", "land drop rate", "{:.3f}"),
    ("by_turn.land_miss.all", "no land drop with a land in hand", "{:.3f}"),
    ("by_turn.mana_eff.all", "mana spent / lands", "{:.3f}"),
    ("by_turn.cast_mv_eff.all", "mana value cast / lands", "{:.3f}"),
    ("by_turn.idle.all", "idle with a castable card (approx.)", "{:.3f}"),
    ("by_turn.idle_given.all", "  ... given a castable card", "{:.3f}"),
    ("by_turn.idle_engine.all", "idle with a Cast offered (XMage)", "{:.3f}"),
    ("by_turn.attack_rate.all", "attack rate (attackers / eligible)", "{:.3f}"),
    ("by_turn.attack_any.all", "turns with an attack (given eligible)", "{:.3f}"),
    ("by_turn.opp_blockers_per_attacker.all", "blockers per opponent attacker", "{:.3f}"),
    ("by_turn.opp_block_any.all", "opponent attacks blocked", "{:.3f}"),
    ("by_turn.opp_block_rate.all", "blockers / potential blockers", "{:.3f}"),
    ("offturn_instants_per_game", "instants on opponent turns per game", "{:.2f}"),
    ("offturn_casts_per_game", "instant-speed casts on opponent turns per game (+ flash)", "{:.2f}"),
    ("cast_mix.creature", "creature share of casts", "{:.3f}"),
    ("cast_mix.instant_sorcery", "instant/sorcery share (own turn)", "{:.3f}"),
]
PER_TURN = [("spells", "spells per own turn", "{:.2f}"), ("land_drop", "land drop rate", "{:.2f}"),
            ("land_miss", "no land drop with a land in hand", "{:.2f}"),
            ("cast_mv_eff", "mana value cast / lands", "{:.2f}"), ("mana_eff", "mana spent / lands", "{:.2f}"),
            ("attack_rate", "attack rate", "{:.2f}"), ("idle", "idle with a castable card", "{:.2f}")]
TURN_KEYS = [str(i) for i in range(1, TB_MAX)] + ["11+"]


def _get(d: dict, dotted: str):
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _fmt(v, f: str) -> str:
    if v is None:
        return "–"
    try:
        return f.format(v)
    except (ValueError, TypeError):
        return str(v)


def markdown(columns: dict[str, dict], per_turn: bool = True, min_n: int = 1) -> str:
    """Markdown tables: headline metrics, then per-turn tables; one column per summary
    (each an {"all", "play", "draw"}-level dict, e.g. fingerprint["all"]). Cells whose
    denominator is below `min_n` are dropped."""
    names = list(columns)
    out = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key, label, f in HEADLINE:
        vals = []
        for n in names:
            v = _get(columns[n], key)
            if key.startswith("by_turn.") and v is not None:
                den = _get(columns[n], "n." + key[len("by_turn."):])
                v = v if (den or 0) >= min_n else None
            vals.append(_fmt(v, f))
        if any(v != "–" for v in vals):
            out.append(f"| {label} | " + " | ".join(vals) + " |")
    if per_turn:
        for m, label, f in PER_TURN:
            if not any(_get(columns[n], f"by_turn.{m}") for n in names):
                continue
            out += ["", f"**{label}, by own turn**", "",
                    "| group | " + " | ".join(TURN_KEYS) + " |", "|---|" + "---|" * len(TURN_KEYS)]
            for n in names:
                row = _get(columns[n], f"by_turn.{m}") or {}
                cnt = _get(columns[n], f"n.{m}") or {}
                out.append(f"| {n} | " + " | ".join(_fmt(row.get(t) if cnt.get(t, 0) >= min_n else None, f)
                                                    for t in TURN_KEYS) + " |")
    return "\n".join(out)


def _dump(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.part")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m draftzero.gameplay.fingerprints",
                                 description="Turn-level behavioural fingerprints: 17lands humans by skill vs MageZero.")
    ap.add_argument("--path", default=None, help="17lands replay csv.gz (default: FDN PremierDraft)")
    ap.add_argument("--every", type=int, default=1, help="use every Nth game (default: all)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_DIR / "fingerprints_FDN.json"))
    ap.add_argument("--human", default=None, help="load this human fingerprint JSON instead of running the pass")
    ap.add_argument("--mz", action="append", default=[], metavar="NAME=PATH[,PATH...]",
                    help="MageZero logs to fingerprint and compare (JVM log, ProbePlayer JSONL, "
                         "GAME_SUMMARY log or games.jsonl, records JSONL); repeatable")
    ap.add_argument("--seats", default="AB", help="seats of a JVM log to count as MageZero (default AB)")
    ap.add_argument("--mz-out", default=str(OUT_DIR / "fingerprints_magezero.json"))
    ap.add_argument("--side", default="all", choices=("all", "play", "draw"))
    ap.add_argument("--groups", default="all,lt50,50to58,ge58,top,lt50_n100")
    a = ap.parse_args(argv)
    ids = Ids.load()
    if a.human:
        human = json.loads(Path(a.human).read_text())
    else:
        human = human_fingerprints(a.path, a.every, a.limit, a.start, ids, progress=True)
        _dump(human, Path(a.out))
        print(f"wrote {a.out}: {human['meta']['games']} games in {human['meta']['seconds']} s", file=sys.stderr)
    cols = {g: human["groups"][g][a.side] for g in a.groups.split(",") if g in human["groups"]}
    if a.mz:
        mz = {}
        for spec in a.mz:
            name, _, paths = spec.partition("=")
            mz[name] = magezero_fingerprint(paths.split(","), ids, a.seats)
        _dump({"human_source": a.human or a.out, "magezero": mz}, Path(a.mz_out))
        print(f"wrote {a.mz_out}", file=sys.stderr)
        keep = [g for g in ("all", "top", "lt50_n100") if g in cols]
        cols = {**{f"human {g}": cols[g] for g in keep}, **{f"MZ {k}": v[a.side] for k, v in mz.items()}}
    print(markdown(cols))
    return 0


if __name__ == "__main__":
    sys.exit(main())
