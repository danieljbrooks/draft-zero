"""
Rebuild the state at the start of a 17lands user turn as a StateSpec (statespec.py).

`state_at_user_turn(game, n)` returns the state at the START of user turn n AFTER the draw, in
one of two entries:

  eot_rollover (default, recommended by the injection research): the previous half-turn's
      end-of-turn snapshot at END_TURN / PRIORITY_FRESH with the opponent (B) active, and
      A.libraryTop = [the card the user drew in turn n's draw step]. The engine then plays
      cleanup, the untap, upkeep and draw itself, so untapping, summoning sickness and "until end
      of turn" effects come out right without being modelled here. User turn 1 on the play has no
      previous snapshot: it is built as main1 from the opening hand minus the bottomed cards.
  main1: PRECOMBAT_MAIN / PRIORITY_FRESH of user turn n with the draw already applied.

What a 17lands row pins down (docs/008 replay_empirics.md §8): the user's hand, both battlefields
(tokens by id), life totals, the opponent's hand size, whose turn it is. Filled in by inference,
with a flag where it is not certain:
  - the opponent's tapped permanents (its attackers without vigilance, the sources of its {T}
    abilities, lands that entered tapped, and `oppo_mana_spent` more lands chosen by the colours of
    what it cast), summoning sickness, +1/+1 counters from single-source trigger counts and
    "enters with N counters", planeswalker loyalty from loyalty-ability ids, Aura/Equipment
    hosts (unique candidate, else a heuristic pick), graveyards and exile (best effort)
  - the opponent's decklist: `opp_decklist` if given (e.g. from a belief model or a mirrored
    pair), else a placeholder of the cards revealed so far plus basics of `opp_colors`
Never known: library order, the opponent's hand contents.

Provenance: source "17lands", ref "<SET>_<FMT>:row=<i>:user_turn=<n>", flags = the
replay_empirics.md §8 flag names that hold for this state (plus a few spec-level notes), and
tier = the statespec grade (T0 best):
  T3  a visible identity is uncertain: >=2 cards drawn this turn (natural_draw_multi), an
      unexplained zone change earlier in the game (history_unexplained), an unknown card id, or
      the half-turn before is an empty slot in the data (prev_snapshot_missing)
  T2  an attachment host is a heuristic pick, an exile-until-leaves link, a possible token copy
  T1  tapped lands, {T}-ability sources or counters/loyalty inferred, or a graveyard uncertain
      (the user's: surveil/mill/unknown destinations; the opponent's: unseen milled cards)
  T0  none of the above
The research's cumulative ladder (its "T0..T3" = share of states determined up to a rung, a
different and opposite-ordered scale) is `re_ladder(flags)`; `seventeenlands stats` reports both.

Also here: the conservation check of the research (previous snapshot + recorded events = next
snapshot, with every residual classified), used for history flags, graveyard tracking and the
regression statistics.

Usage:
  from draftzero.gameplay import replay, reconstruct
  g = replay.read_games([4])[4]
  spec = reconstruct.state_at_user_turn(g, 5)               # eot_rollover
  spec = reconstruct.state_at_user_turn(g, 5, entry="main1", labels=True)

  python -m draftzero.gameplay.reconstruct --row 4 --turn 5 [--entry main1]
"""
from __future__ import annotations

import argparse
import itertools
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace

from draftzero.gameplay.ids import BASIC_MANA, COLORS, Ids, cost_taps_source
from draftzero.gameplay.replay import Game, TurnRecord
from draftzero.gameplay.statespec import Perm, PlayerState, Provenance, StateSpec

ENTRIES = ("eot_rollover", "main1")

# ============================================================================================
# Conservation (port of the research conservation.py; category names and tiers unchanged)
# ============================================================================================

ZONES_U = ("user_hand", "user_lands", "user_crea", "user_nonc")
ZONES_O = ("oppo_lands", "oppo_crea", "oppo_nonc")
ALL_ID = ZONES_U + ZONES_O
BF_KIND = {"lands": "eot_{w}_lands_in_play", "crea": "eot_{w}_creatures_in_play",
           "nonc": "eot_{w}_non_creatures_in_play"}

# benign: mechanism identified and the resulting state determined; dest_unknown: a card left a
# visible zone for an unrecorded one; unexplained: no mechanism found
BENIGN = {
    "mulligan_bottom", "timing_shift", "token_created", "token_left_unrecorded",
    "flash_creature_on_opp_turn_uncast", "from_hand_without_cast_record", "hand_to_bf_without_cast_record",
    "bounce_bf_to_hand", "bounced_to_hand", "type_change_between_bf_lists", "control_change",
    "land_put_onto_bf(ramp)", "land_put_onto_bf(nonbasic)", "returned_from_graveyard", "graveyard_to_hand",
    "cast_from_graveyard(flashback)", "evolving_wilds_sacrificed", "died_and_returned_same_slot",
    "killed_listed_but_never_on_bf(flash/uncast creature died in slot)", "token_copy_of_card(probable)",
    "draw_on_opp_turn_unrecorded", "left_hand_on_opp_turn_unrecorded(discard)",
    "oppo_creature_appeared_on_user_turn(flash)", "oppo_permanent_appeared_on_user_turn(flash)",
    "from_opp_bf_to_hand", "cast_or_played_not_from_hand(exile/opp card)",
}
UNEXPLAINED = {"appeared_unexplained", "returned_to_bf(flicker/exile-return?)", "unexplained_add_own_turn",
               "left_hand_on_own_turn_unrecorded", "hand_removed_first_slot_excess", "last_slot",
               "land_left_bf", "left_unexplained"}
COPY_MAKERS = {"Homunculus Horde", "Self-Reflection", "Electroduplicate", "Rite of Replication",
               "Extravagant Replication", "Phantasmal Image", "Polyraptor", "Kalamax, the Stormsire",
               "Chandra, Flameshaper", "Abyssal Harvester", "Kindred Charge", "Pyromancer's Goggles"}
EVENT_FIELDS = ("cards_drawn", "cards_tutored", "cards_discarded", "lands_played", "creatures_cast",
                "non_creatures_cast", "user_instants_sorceries_cast", "oppo_instants_sorceries_cast",
                "user_creatures_killed_combat", "oppo_creatures_killed_combat",
                "user_creatures_killed_non_combat", "oppo_creatures_killed_non_combat")


def residual_tier(cat: str) -> str:
    if cat in BENIGN:
        return "benign"
    if cat in UNEXPLAINED:
        return "unexplained"
    return "dest_unknown"


def snapshot(t: TurnRecord) -> dict:
    return {
        "user_hand": Counter(t.L("eot_user_cards_in_hand")),
        "user_lands": Counter(t.L("eot_user_lands_in_play")),
        "user_crea": Counter(t.L("eot_user_creatures_in_play")),
        "user_nonc": Counter(t.L("eot_user_non_creatures_in_play")),
        "oppo_lands": Counter(t.L("eot_oppo_lands_in_play")),
        "oppo_crea": Counter(t.L("eot_oppo_creatures_in_play")),
        "oppo_nonc": Counter(t.L("eot_oppo_non_creatures_in_play")),
        "oppo_hand_n": int(t.num("eot_oppo_cards_in_hand")),
        "user_life": t.num("eot_user_life"),
        "oppo_life": t.num("eot_oppo_life"),
    }


def model(t: TurnRecord) -> dict:
    """Expected signed change per zone from the recorded events of slot t."""
    m = {z: Counter() for z in ALL_ID}
    if t.side == "user":
        for c in t.L("cards_drawn") + t.L("cards_tutored"):
            m["user_hand"][c] += 1
        for f in ("lands_played", "creatures_cast", "non_creatures_cast", "cards_discarded"):
            for c in t.L(f):
                m["user_hand"][c] -= 1
        dst = "user"
    else:
        dst = "oppo"
    for c in t.L("lands_played"):
        m[f"{dst}_lands"][c] += 1
    for c in t.L("creatures_cast"):
        m[f"{dst}_crea"][c] += 1
    for c in t.L("non_creatures_cast"):
        m[f"{dst}_nonc"][c] += 1
    for c in t.L("user_instants_sorceries_cast"):
        m["user_hand"][c] -= 1
    for c in t.L("user_creatures_killed_combat") + t.L("user_creatures_killed_non_combat"):
        m["user_crea"][c] -= 1
    for c in t.L("oppo_creatures_killed_combat") + t.L("oppo_creatures_killed_non_combat"):
        m["oppo_crea"][c] -= 1
    if t.side == "oppo":
        dn = int(t.num("cards_drawn_or_tutored"))
        out = sum(len(t.L(f)) for f in ("lands_played", "creatures_cast", "non_creatures_cast", "cards_discarded"))
    else:
        dn, out = 0, 0
    out += len(t.L("oppo_instants_sorceries_cast"))
    m["oppo_hand_n"] = dn - out
    return m


def _signed_residual(prev: Counter, cur: Counter, mod: Counter) -> dict:
    r = Counter(cur)
    r.subtract(prev)
    r.subtract(mod)
    return {k: v for k, v in r.items() if v != 0}


def _event_ids(t: TurnRecord) -> set:
    out = set()
    for f in EVENT_FIELDS:
        out.update(t.L(f))
    return out


def categorise(ids: Ids, t: TurnRecord, s: int, z: str, c: int, sign: str, prev: dict, cur: dict,
               plus: dict, minus: dict, gy: dict, ever_bf: dict, pos: str) -> str:
    """Mechanism behind one residual unit of card c in zone z (see conservation())."""
    name = ids.name(c)
    ty = ids.types(c)
    tok = ids.is_token(c)
    a = t.side
    owner = z.split("_")[0]
    other = "oppo" if owner == "user" else "user"
    bf_zones = [f"{owner}_lands", f"{owner}_crea", f"{owner}_nonc"]
    if z == "user_hand":
        if sign == "+":
            if c in t.L("user_instants_sorceries_cast") and gy["user"][c] > 0:
                return "cast_from_graveyard(flashback)"
            if any(minus[b][c] > 0 for b in bf_zones):
                return "bounce_bf_to_hand"
            if any(minus[f"{other}_{k}"][c] > 0 for k in ("lands", "crea", "nonc")):
                return "from_opp_bf_to_hand"
            if gy["user"][c] > 0:
                return "graveyard_to_hand"
            if a == "oppo":
                return "draw_on_opp_turn_unrecorded"
            if c in t.L("lands_played") or c in t.L("creatures_cast") or c in t.L("non_creatures_cast") \
                    or c in t.L("user_instants_sorceries_cast"):
                return "cast_or_played_not_from_hand(exile/opp card)"
            return "unexplained_add_own_turn"
        if pos == "first" and s == 0:
            return "mulligan_bottom"
        if any(plus[b][c] > 0 for b in bf_zones):
            if "Creature" in ty and a == "oppo":
                return "flash_creature_on_opp_turn_uncast"
            return "hand_to_bf_without_cast_record"
        if pos == "last":
            return "last_slot"
        if a == "oppo":
            return "left_hand_on_opp_turn_unrecorded(discard)"
        return "left_hand_on_own_turn_unrecorded"
    kind = z.split("_")[1]
    other_kinds = [k for k in ("lands", "crea", "nonc") if k != kind]
    if sign == "+":
        if tok:
            return "token_created"
        if any(minus[f"{owner}_{k}"][c] > 0 for k in other_kinds):
            return "type_change_between_bf_lists"
        if any(minus[f"{other}_{k}"][c] > 0 for k in ("lands", "crea", "nonc")):
            return "control_change"
        if owner == "user" and minus["user_hand"][c] > 0:
            return "from_hand_without_cast_record"
        if kind == "crea":
            killed = t.L(f"{owner}_creatures_killed_combat") + t.L(f"{owner}_creatures_killed_non_combat")
            if c in killed:
                if prev[z][c] == 0 and cur[z][c] == 0:
                    return "killed_listed_but_never_on_bf(flash/uncast creature died in slot)"
                return "died_and_returned_same_slot"
        if gy[owner][c] > 0:
            return "returned_from_graveyard"
        if name in COPY_MAKERS or any(ids.name(x) in COPY_MAKERS for x in cur[z]):
            return "token_copy_of_card(probable)"
        if kind == "lands":
            return "land_put_onto_bf(ramp)" if "Basic" in ty else "land_put_onto_bf(nonbasic)"
        if ever_bf[owner][c] > 0:
            return "returned_to_bf(flicker/exile-return?)"
        if owner == "oppo" and a == "user":
            return ("oppo_creature_appeared_on_user_turn(flash)" if kind == "crea"
                    else "oppo_permanent_appeared_on_user_turn(flash)")
        return "appeared_unexplained"
    if any(plus[f"{owner}_{k}"][c] > 0 for k in other_kinds):
        return "type_change_between_bf_lists"
    if any(plus[f"{other}_{k}"][c] > 0 for k in ("lands", "crea", "nonc")):
        return "control_change"
    if owner == "user" and plus["user_hand"][c] > 0:
        return "bounced_to_hand"
    if tok:
        return "token_left_unrecorded"
    if kind == "crea":
        cast_here = t.L("creatures_cast") if t.side == owner else []
        if c in cast_here and cur[z][c] < prev[z][c] + cast_here.count(c):
            return "cast_but_not_on_bf(countered/removed same slot)"
        return "creature_left_not_killed(exile/bounce/library)"
    if kind == "nonc":
        if "Aura" in ty:
            return "aura_left"
        if "Equipment" in ty:
            return "equipment_left"
        if "Planeswalker" in ty:
            return "planeswalker_left"
        if c in (t.L("non_creatures_cast") if t.side == owner else []):
            return "noncreature_cast_but_not_on_bf"
        return "noncreature_permanent_left"
    if kind == "lands":
        return "evolving_wilds_sacrificed" if name == "Evolving Wilds" else "land_left_bf"
    return "left_unexplained"


@dataclass
class SlotCheck:
    """Conservation result for one slot: residual units and per-zone verdicts."""
    cats: list = field(default_factory=list)       # (zone, sign, category, card id)
    strict: dict = field(default_factory=dict)     # zone -> no residual at all (bottoms excepted)
    benign: dict = field(default_factory=dict)     # zone -> only benign residuals
    nounexp: dict = field(default_factory=dict)    # zone -> no unexplained residual
    oppo_hand_resid: int = 0
    user_life_resid: float = 0.0
    oppo_life_resid: float = 0.0

    def slot_ok(self, level: str, scope: str) -> bool:
        d = {"strict": self.strict, "benign": self.benign, "nounexp": self.nounexp}[level]
        u = all(d[z] for z in ZONES_U)
        if scope == "user_zones":
            return u
        o = all(d[z] for z in ZONES_O) and (self.strict["oppo_hand_n"] if level == "strict" else True)
        return u and o


def conservation(g: Game, ids: Ids) -> list[SlotCheck]:
    """Previous snapshot + recorded events vs next snapshot, for every slot; every residual unit
    classified (research conservation.py, passes 1-3)."""
    mull = int(g.meta.get("num_mulligans") or 0)
    omull = int(g.meta.get("opp_num_mulligans") or 0)
    prev = {"user_hand": Counter(g.opening_hand), "oppo_hand_n": 7 - omull, "user_life": 20.0, "oppo_life": 20.0}
    for z in ALL_ID:
        prev.setdefault(z, Counter())
    n = len(g.turns)
    snaps, residuals, scal = [], [], []
    for t in g.turns:
        cur = snapshot(t)
        mod = model(t)
        residuals.append({z: _signed_residual(prev[z], cur[z], mod[z]) for z in ALL_ID})
        scal.append((cur["oppo_hand_n"] - (prev["oppo_hand_n"] + mod["oppo_hand_n"]),
                     cur["user_life"] - (prev["user_life"] - t.num("user_combat_damage_taken")),
                     cur["oppo_life"] - (prev["oppo_life"] - t.num("oppo_combat_damage_taken"))))
        snaps.append((prev, cur))
        prev = cur
    ev = [_event_ids(t) for t in g.turns]
    cats: list[list] = [[] for _ in range(n)]
    # pass 2: an event listed one slot off shows as opposite-sign residuals in adjacent slots
    for s in range(n - 1):
        for z in ALL_ID:
            r0, r1 = residuals[s][z], residuals[s + 1][z]
            for c in list(r0):
                v0, v1 = r0.get(c, 0), r1.get(c, 0)
                if ids.is_token(c) or not (c in ev[s] or c in ev[s + 1]):
                    continue
                if v0 and v1 and (v0 > 0) != (v1 > 0):
                    k = min(abs(v0), abs(v1))
                    sg0, sg1 = ("+", "-") if v0 > 0 else ("-", "+")
                    for _ in range(k):
                        cats[s].append((z, sg0, "timing_shift", c))
                        cats[s + 1].append((z, sg1, "timing_shift", c))
                    r0[c] = v0 - k * (1 if v0 > 0 else -1)
                    r1[c] = v1 - k * (1 if v1 > 0 else -1)
                    if r0[c] == 0:
                        del r0[c]
                    if r1[c] == 0:
                        del r1[c]
    gy = {"user": Counter(), "oppo": Counter()}
    ever = {"user": Counter(), "oppo": Counter()}
    out = []
    for s, t in enumerate(g.turns):
        pos = "first" if s == 0 else ("last" if s == n - 1 else "mid")
        pv, cur = snaps[s]
        res = residuals[s]
        plus = {z: Counter({c: v for c, v in res[z].items() if v > 0}) for z in ALL_ID}
        minus = {z: Counter({c: -v for c, v in res[z].items() if v < 0}) for z in ALL_ID}
        removed_first = 0
        for z in ALL_ID:
            for c, v in res[z].items():
                sign = "+" if v > 0 else "-"
                for _ in range(abs(v)):
                    cat = categorise(ids, t, s, z, c, sign, pv, cur, plus, minus, gy, ever, pos)
                    if cat == "mulligan_bottom":
                        removed_first += 1
                        if removed_first > mull:
                            cat = "hand_removed_first_slot_excess"
                    cats[s].append((z, sign, cat, c))
        chk = SlotCheck(cats=cats[s])
        for z in ALL_ID:
            chk.strict[z] = chk.benign[z] = chk.nounexp[z] = True
        for z, sign, cat, c in cats[s]:
            tr = residual_tier(cat)
            if cat != "mulligan_bottom":
                chk.strict[z] = False
            if tr != "benign":
                chk.benign[z] = False
            if tr == "unexplained":
                chk.nounexp[z] = False
        chk.oppo_hand_resid, chk.user_life_resid, chk.oppo_life_resid = scal[s]
        chk.strict["oppo_hand_n"] = chk.oppo_hand_resid == 0
        chk.strict["user_life"] = chk.user_life_resid == 0
        chk.strict["oppo_life"] = chk.oppo_life_resid == 0
        out.append(chk)
        for c in t.L("user_instants_sorceries_cast"):
            gy["user"][c] += 1
        for c in t.L("oppo_instants_sorceries_cast"):
            gy["oppo"][c] += 1
        gy[t.side].update(t.L("cards_discarded"))
        for w in ("user", "oppo"):
            for c in t.L(f"{w}_creatures_killed_combat") + t.L(f"{w}_creatures_killed_non_combat"):
                if not ids.is_token(c):
                    gy[w][c] += 1
            for z in (f"{w}_lands", f"{w}_crea", f"{w}_nonc"):
                ever[w].update(cur[z])
    return out


# ============================================================================================
# replay_empirics.md §8 decision-state flags (port of the research reconstruct.py)
# ============================================================================================

RE_FLAGS = ("natural_draw_multi", "t1_mulligan_bottoms", "counters_possible", "counter_card_on_bf",
            "planeswalker_present", "attachment_ambiguous", "aura_or_equipment_present",
            "opp_land_tap_ambiguous", "opp_mana_gt_lands", "exile_link_present", "token_present",
            "token_copy_possible", "gy_uncertain_user", "history_unexplained")
# the research's cumulative rungs: a state "is at" a rung when none of the rung's flags hold
RE_LADDER = {
    "T0": ("natural_draw_multi",),
    "T1": ("natural_draw_multi", "opp_mana_gt_lands"),
    "T1b": ("natural_draw_multi", "opp_mana_gt_lands", "opp_land_tap_ambiguous"),
    "T2": ("natural_draw_multi", "opp_mana_gt_lands", "counter_card_on_bf", "attachment_ambiguous",
           "exile_link_present", "token_copy_possible"),
    "T2b": ("natural_draw_multi", "opp_mana_gt_lands", "counters_possible", "attachment_ambiguous",
            "exile_link_present", "token_copy_possible"),
    "T3": ("natural_draw_multi", "opp_mana_gt_lands", "counter_card_on_bf", "attachment_ambiguous",
           "exile_link_present", "token_copy_possible", "gy_uncertain_user", "history_unexplained"),
}


def re_ladder(flags: dict) -> dict[str, bool]:
    return {rung: not any(flags.get(f) for f in fl) for rung, fl in RE_LADDER.items()}


def re_flags(g: Game, ids: Ids, checks: list[SlotCheck]) -> dict[int, dict]:
    """{user turn n: {flag: bool}} with the replay_empirics.md §8 definitions."""
    feats = lambda c: ids.info_id(c).features
    out: dict[int, dict] = {}
    counters_seen = copy_seen = gy_unc = hist_unexp = False
    prev = None
    for s, t in enumerate(g.turns):
        if t.side == "user":
            fl = {}
            dr = t.L("cards_drawn")
            fl["natural_draw_multi"] = len(dr) >= 2
            fl["t1_mulligan_bottoms"] = s == 0 and int(g.meta.get("num_mulligans") or 0) > 0
            snap = prev
            bf_user = (snap.L("eot_user_creatures_in_play") + snap.L("eot_user_non_creatures_in_play")) if snap else []
            bf_oppo = (snap.L("eot_oppo_creatures_in_play") + snap.L("eot_oppo_non_creatures_in_play")) if snap else []
            crea_user = snap.L("eot_user_creatures_in_play") if snap else []
            crea_oppo = snap.L("eot_oppo_creatures_in_play") if snap else []
            allp = bf_user + bf_oppo
            pw = any("Planeswalker" in ids.types(c) for c in allp)
            cnt_perm = any("counters" in feats(c) for c in allp)
            fl["planeswalker_present"] = pw
            fl["counters_possible"] = bool(allp) and (counters_seen or cnt_perm or pw)
            fl["counter_card_on_bf"] = cnt_perm or pw
            auras = [c for c in allp if "Aura" in ids.types(c)]
            eq_u = [c for c in bf_user if "Equipment" in ids.types(c)]
            eq_o = [c for c in bf_oppo if "Equipment" in ids.types(c)]
            fl["attachment_ambiguous"] = bool((auras and len(crea_user) + len(crea_oppo) >= 2)
                                              or (eq_u and crea_user) or (eq_o and crea_oppo))
            fl["aura_or_equipment_present"] = bool(auras or eq_u or eq_o)
            if snap is not None and snap.side == "oppo":
                k = int(snap.num("oppo_mana_spent"))
                lands = snap.L("eot_oppo_lands_in_play")
                names = {ids.name(c) for c in lands}
                fl["opp_land_tap_ambiguous"] = 0 < k < len(lands) and len(names) > 1
                fl["opp_mana_gt_lands"] = k > len(lands)
            else:
                fl["opp_land_tap_ambiguous"] = fl["opp_mana_gt_lands"] = False
            fl["exile_link_present"] = any("exile_until_leaves" in feats(c) for c in allp)
            fl["token_present"] = any(ids.is_token(c) for c in allp)
            fl["token_copy_possible"] = copy_seen
            fl["gy_uncertain_user"] = gy_unc
            fl["history_unexplained"] = hist_unexp
            out[t.n] = fl
        for w in ("user", "oppo"):
            for ab in t.L(f"{w}_abilities"):
                tf = ids.ability(ab).text_flags
                if "counter_on" in tf:
                    counters_seen = True
                if w == "user" and "mill_surveil" in tf:
                    gy_unc = True
        for f in ("creatures_cast", "non_creatures_cast", "user_instants_sorceries_cast", "oppo_instants_sorceries_cast"):
            for c in t.L(f):
                fe = feats(c)
                if "p1p1_counters" in fe or "counters" in fe:
                    counters_seen = True
                if "copy" in fe:
                    copy_seen = True
                if ("mill" in fe or "scry_surveil" in fe) and (f.startswith("user") or t.side == "user"):
                    gy_unc = True
        for z, sign, cat, c in checks[s].cats:
            tr = residual_tier(cat)
            if tr == "unexplained":
                hist_unexp = True
            if z.startswith("user") and tr == "dest_unknown":
                gy_unc = True
        prev = t
    return out


# ============================================================================================
# History walk: permanents as instances, graveyards/exile, counters, loyalty, attachments
# ============================================================================================

@dataclass
class Inst:
    """One permanent on a battlefield, followed across snapshots by id multiset diffs (copies are
    indistinguishable in the data, so which copy left is a guess; see `exact` fields)."""
    iid: int
    grp: int
    name: str
    side: str                     # controller: "user" / "oppo"
    entered: int                  # slot index it (last) entered the battlefield
    token: bool = False
    token_class: str | None = None
    kind: str = "crea"            # which eot list it is in: lands / crea / nonc
    p1p1: int = 0
    counters_exact: bool = True
    loyalty: int | None = None
    loyalty_exact: bool = True
    host: int | None = None       # iid of the permanent this Aura/Equipment is attached to
    host_exact: bool = True
    attach_issue: str = ""        # "", "no_candidate", "player", "host_left"


@dataclass
class SlotState:
    """Everything tracked after one slot."""
    bf: dict                      # side -> list[Inst]
    gy: dict                      # side -> Counter of names (bottom->top order not known)
    exile: dict                   # side -> Counter of names
    revealed: Counter             # opponent card names seen, max simultaneous multiplicity
    gy_unknown: dict              # side -> milled / surveiled cards of unknown identity (count)


@dataclass
class Analysis:
    checks: list                  # SlotCheck per slot
    flags: dict                   # user turn n -> RE §8 flags
    states: list                  # SlotState per slot


def analyze(g: Game, ids: Ids) -> Analysis:
    """Conservation, RE flags and the history walk for one game (cached on the Game)."""
    ana = g.__dict__.get("_analysis")
    if ana is None:
        checks = conservation(g, ids)
        ana = Analysis(checks, re_flags(g, ids, checks), _walk(g, ids, checks))
        g.__dict__["_analysis"] = ana
    return ana


_KIND_OF = (("lands", "eot_{w}_lands_in_play"), ("crea", "eot_{w}_creatures_in_play"),
            ("nonc", "eot_{w}_non_creatures_in_play"))
# where a permanent that left without a record most likely went: the graveyard for these (Auras
# whose host died, planeswalkers, sacrificed Wilds, countered spells) ...
_GY_DEST = {"cast_but_not_on_bf(countered/removed same slot)", "aura_left", "equipment_left", "planeswalker_left",
            "noncreature_permanent_left", "noncreature_cast_but_not_on_bf", "evolving_wilds_sacrificed"}
# ... and an "unknown destination" sink for these (exile, bounce, library): exile, which is inert
_SINK_DEST = {"creature_left_not_killed(exile/bounce/library)", "land_left_bf", "left_unexplained"}


def _walk(g: Game, ids: Ids, checks: list[SlotCheck]) -> list[SlotState]:
    bf = {"user": [], "oppo": []}
    gy = {"user": Counter(), "oppo": Counter()}
    ex = {"user": Counter(), "oppo": Counter()}
    unk = {"user": 0, "oppo": 0}
    revealed = Counter()
    next_iid = itertools.count(1)
    states: list[SlotState] = []
    hand_prev = Counter(g.opening_hand)
    for s, t in enumerate(g.turns):
        if not t.played and states:
            # an empty "hole" slot inside the game (0.12% of games): no snapshot, carry the state
            last = states[-1]
            states.append(SlotState({w: [replace(i) for i in last.bf[w]] for w in last.bf}, dict(last.gy),
                                    dict(last.exile), Counter(last.revealed), dict(last.gy_unknown)))
            continue
        cats = checks[s].cats
        type_changed = {(z.split("_")[0], c) for z, sign, cat, c in cats if cat == "type_change_between_bf_lists"}
        control_in = {(z.split("_")[0], c) for z, sign, cat, c in cats if cat == "control_change" and sign == "+"}
        departed: dict[str, list[Inst]] = {"user": [], "oppo": []}
        # --- 1. battlefield instances --------------------------------------------------------
        for w in ("user", "oppo"):
            cur, kind_of = Counter(), {}
            for kind, col in _KIND_OF:
                for c in t.L(col.format(w=w)):
                    cur[c] += 1
                    kind_of[c] = kind
            killed = Counter(c for c in t.L(f"{w}_creatures_killed_combat") + t.L(f"{w}_creatures_killed_non_combat"))
            by_grp = defaultdict(list)
            for inst in bf[w]:
                by_grp[inst.grp].append(inst)
            keep: list[Inst] = []
            for grp in set(by_grp) | set(cur):
                have = sorted(by_grp.get(grp, []), key=lambda i: (i.entered, i.iid))
                want = cur.get(grp, 0)
                drop = max(0, len(have) - want)
                reborn = min(max(0, killed.get(grp, 0) - drop), len(have) - drop)   # died and came back
                n_out = drop + reborn
                # killed creatures: the oldest copies are the likelier combatants; otherwise newest
                out = have[:n_out] if killed.get(grp) else have[len(have) - n_out:]
                stay = [i for i in have if i not in out]
                if out and stay and any((o.p1p1, o.host, o.entered) != (stay[0].p1p1, stay[0].host, stay[0].entered)
                                        for o in out):
                    for i in stay:                     # which copy left is a guess
                        i.counters_exact = i.counters_exact and all(o.p1p1 == i.p1p1 for o in out)
                        i.host_exact = False if (i.host or any(o.host for o in out)) else i.host_exact
                departed[w].extend(out)
                for i in stay:
                    i.kind = kind_of.get(grp, i.kind)
                keep.extend(stay)
                for _ in range(want - len(stay)):
                    info = ids.info_id(grp)
                    card = ids.card(grp)
                    inst = Inst(next(next_iid), grp, card.name, w, s, card.is_token, card.token_class,
                                kind_of.get(grp, "crea"))
                    if info.counter_mode == "untracked":
                        inst.counters_exact = False
                    if info.counter_mode == "loyalty":
                        inst.loyalty = info.loyalty
                    if info.etb_counters:
                        # without them a 0/0 Hydra is built dead: XMage's state-based actions bury it
                        inst.p1p1 = _etb_x(t, w, grp, ids) if info.etb_counters == "X" else int(info.etb_counters)
                        if info.etb_counters == "X":
                            inst.counters_exact = False
                    keep.append(inst)
            bf[w] = keep
            # Giada, Font of Hope gives each other Angel entering under its controller extra +1/+1
            # counters (not in the replay): those counts are unknown here
            if any(i.name == "Giada, Font of Hope" for i in keep):
                for i in keep:
                    if i.entered == s and i.name != "Giada, Font of Hope" and "Angel" in ids.types(i.grp):
                        i.counters_exact = False
        # --- 2. graveyards and exile ---------------------------------------------------------
        hand = Counter(hand_prev)
        if t.side == "user":
            hand.update(t.L("cards_drawn") + t.L("cards_tutored"))
            hand.subtract(t.L("lands_played") + t.L("creatures_cast") + t.L("non_creatures_cast"))
        for c in t.L("user_instants_sorceries_cast"):
            name = ids.name(c)
            if hand[c] > 0:
                hand[c] -= 1
                gy["user"][name] += 1
            elif gy["user"][name] > 0 and ids.info(name).flashback_key:
                gy["user"][name] -= 1               # flashback: the card is exiled as it resolves
                ex["user"][name] += 1
            else:
                gy["user"][name] += 1
        for c in t.L("oppo_instants_sorceries_cast"):
            name = ids.name(c)
            if gy["oppo"][name] > 0 and ids.info(name).flashback_key:
                gy["oppo"][name] -= 1               # the opponent's hand is unknown: assume flashback
                ex["oppo"][name] += 1
            else:
                gy["oppo"][name] += 1
        for c in t.L("cards_discarded"):
            gy[t.side][ids.name(c)] += 1
        for w in ("user", "oppo"):
            for c in t.L(f"{w}_creatures_killed_combat") + t.L(f"{w}_creatures_killed_non_combat"):
                if not ids.is_token(c):
                    gy[w][ids.name(c)] += 1
        # the other side exiled something this slot (Banishing Light-style ETB, exile spell)
        exiler = {w: any("exile" in ids.info(src).features for aid in t.L(f"{w}_abilities")
                         for src in ids.ability(aid).source_cards[:1])
                  or any("exile" in ids.info_id(c).features for c in _cast_ids(t, w)) for w in ("user", "oppo")}
        for z, sign, cat, c in cats:
            w = z.split("_")[0]
            name = ids.name(c)
            other = "oppo" if w == "user" else "user"
            if cat == "left_hand_on_opp_turn_unrecorded(discard)":
                gy["user"][name] += 1
            elif sign == "-" and not ids.is_token(c) and z != "user_hand" and exiler[other] \
                    and cat in _GY_DEST | _SINK_DEST and cat != "evolving_wilds_sacrificed":
                ex[w][name] += 1
            elif cat in _GY_DEST and sign == "-" and not ids.is_token(c):
                gy[w][name] += 1
            elif cat in _SINK_DEST and sign == "-" and not ids.is_token(c):
                ex[w][name] += 1
            elif cat in ("graveyard_to_hand", "returned_from_graveyard", "died_and_returned_same_slot"):
                if gy[w][name] > 0:
                    gy[w][name] -= 1
            elif cat == "returned_to_bf(flicker/exile-return?)":
                if ex[w][name] > 0:
                    ex[w][name] -= 1
        for w in ("user", "oppo"):
            unk[w] += sum("mill_surveil" in ids.ability(ab).text_flags for ab in t.L(f"{w}_abilities"))
        # --- 3. counters, loyalty, attachments from this slot's ability events ---------------
        for w in ("user", "oppo"):
            other = "oppo" if w == "user" else "user"
            for aid in t.L(f"{w}_abilities"):
                a = ids.ability(aid)
                if a.self_p1p1 or a.self_double or a.loyalty is not None:
                    cand = [i for i in bf[w] + departed[w] if i.name in a.source_cards]
                    live = [i for i in cand if i in bf[w]]
                    if a.self_p1p1 or a.self_double:
                        if len(cand) == 1:
                            cand[0].p1p1 = cand[0].p1p1 * 2 if a.self_double else cand[0].p1p1 + a.self_p1p1
                        else:
                            for i in live:
                                i.counters_exact = False
                    if a.loyalty is not None:
                        if len(cand) == 1 and cand[0].loyalty is not None:
                            cand[0].loyalty += a.loyalty
                        else:
                            for i in live:
                                i.loyalty_exact = False
                if a.counter_other:
                    for i in bf[w]:
                        if i.kind == "crea":
                            i.counters_exact = False
                if a.xmage_key and a.xmage_key.startswith("Equip"):
                    eq = [i for i in bf[w] if i.name in a.source_cards]
                    if eq:
                        e = min(eq, key=lambda i: (i.host is not None, -i.entered))
                        _attach(e, bf[w], t, ids, friendly=True, exact_equipment=len(eq) == 1)
            # instants and sorceries that place counters (their targets are unknown); permanents
            # that do so act through ability ids, handled above
            for c in t.L(f"{w}_instants_sorceries_cast"):
                if ids.info_id(c).features & {"p1p1_counters", "counters"}:
                    for i in bf[w]:
                        if i.kind == "crea":
                            i.counters_exact = False
            # planeswalker damage is not recorded: an attack by the other side makes loyalty an estimate
            if t.side == other and t.L("creatures_attacked"):
                for i in bf[w]:
                    if i.loyalty is not None:
                        i.loyalty_exact = False
        for w in ("user", "oppo"):
            other = "oppo" if w == "user" else "user"
            for i in bf[w]:
                if i.entered != s or i.token:
                    continue
                att = ids.info(i.name).attach
                if att.startswith("aura"):
                    _, mode, what = att.split(":")
                    if what == "player":
                        i.attach_issue = "player"
                        continue
                    if mode == "control":
                        pool = [x for x in bf[w] if (w, x.grp) in control_in and x is not i]
                        pool = pool or [x for x in bf[other]]
                    else:
                        side = w if mode == "friendly" else other
                        pool = bf[side]
                    if what == "creature":
                        pool = [x for x in pool if x.kind == "crea" or (x.side, x.grp) in type_changed]
                    elif what == "land":
                        pool = [x for x in pool if x.kind == "lands"]
                    pool = [x for x in pool if x is not i]
                    # a creature that became a land in the slot the Aura arrived is its host
                    # (Imprisoned in the Moon); the power heuristic below would miss it
                    moved = [x for x in pool if (x.side, x.grp) in type_changed and x.kind == "lands" and x.entered < s]
                    if len(moved) == 1:
                        i.host, i.host_exact = moved[0].iid, True
                        continue
                    _pick_host(i, pool, t, ids, hostile=mode != "friendly")
                elif att == "equipment:etb":
                    _pick_host(i, [x for x in bf[w] if x.kind == "crea"], t, ids, hostile=False)
        # hosts that left: Auras go with them (if one is still listed its host is unknown),
        # Equipment stays unattached; Equipment also falls off a host that stopped being a creature
        live = {i.iid: i for side in ("user", "oppo") for i in bf[side]}
        for w in ("user", "oppo"):
            for i in bf[w]:
                if i.host is None:
                    continue
                host = live.get(i.host)
                is_aura = ids.info(i.name).attach.startswith("aura")
                if host is None and is_aura:
                    i.host, i.host_exact, i.attach_issue = None, False, "host_left"
                elif host is None or (not is_aura and host.kind != "crea"):
                    i.host, i.host_exact = None, True
        # --- 4. revealed opponent cards ------------------------------------------------------
        seen = Counter(i.name for i in bf["oppo"] if not i.token) + gy["oppo"] + ex["oppo"]
        for name, k in seen.items():
            if k > revealed[name]:
                revealed[name] = k
        states.append(SlotState({w: [replace(i) for i in bf[w]] for w in bf}, {w: +gy[w] for w in gy},
                                {w: +ex[w] for w in ex}, Counter(revealed), dict(unk)))
        hand_prev = Counter(t.L("eot_user_cards_in_hand"))
    return states


def _cast_ids(t: TurnRecord, w: str) -> list[int]:
    out = t.L(f"{w}_instants_sorceries_cast")
    if t.side == w:
        out = out + t.L("creatures_cast") + t.L("non_creatures_cast")
    return out


def _etb_x(t: TurnRecord, w: str, grp: int, ids: Ids) -> int:
    """X of an "enters with X +1/+1 counters" creature (Wildwood Scourge) cast in slot t: the mana
    its controller spent there minus the other casts' mana values and its own (X counts 0). An
    upper bound (abilities also cost mana), and at least 1: a 0/0 on the battlefield had counters."""
    casts = list(_cast_ids(t, w))
    if grp not in casts:
        return 1
    casts.remove(grp)
    spent = int(t.num(f"{w}_mana_spent"))
    return max(1, spent - sum(ids.mv(c) for c in casts) - ids.mv(grp))


def _power(ids: Ids, inst: Inst) -> int:
    if inst.token:
        return 0
    p = ids.info(inst.name).power
    return p if p is not None else 0


def _pick_host(inst: Inst, pool: list[Inst], t: TurnRecord, ids: Ids, hostile: bool) -> None:
    """Attach an Aura / ETB-attaching Equipment: unique candidate, else a heuristic pick."""
    if not pool:
        inst.host, inst.host_exact, inst.attach_issue = None, False, "no_candidate"
        return
    if len(pool) == 1:
        inst.host, inst.host_exact = pool[0].iid, True
        return
    if not hostile:
        attackers = set(t.L("creatures_attacked")) if t.side == inst.side else set()
        att = [x for x in pool if x.grp in attackers]
        if len(att) == 1:
            inst.host, inst.host_exact = att[0].iid, False
            return
    best = max(pool, key=lambda x: (_power(ids, x), x.entered, x.iid))
    inst.host, inst.host_exact = best.iid, False


def _attach(equip: Inst, own: list[Inst], t: TurnRecord, ids: Ids, friendly: bool, exact_equipment: bool) -> None:
    pool = [x for x in own if x.kind == "crea" and x is not equip]
    _pick_host(equip, pool, t, ids, hostile=not friendly)
    if not exact_equipment:
        equip.host_exact = False


# ============================================================================================
# StateSpec builder
# ============================================================================================

def _slug(name: str) -> str:
    words = re.sub(r"[^A-Za-z0-9 ]", "", name.replace("-", " ")).split()
    return "".join(w[0].upper() + w[1:] for w in words) or "x"


def alias(inst: Inst) -> str:
    """Stable per-game permanent alias, shared by specs and labels (e.g. "GleamingBarrier_3")."""
    return f"{_slug(inst.name)}_{inst.iid}"


SEAT = {"user": "A", "oppo": "B"}


def choose_tapped_lands(lands: list[str], k: int, spells: list[str], ids: Ids,
                        entered_tapped: list[str] | None = None) -> tuple[Counter, bool]:
    """Which of `lands` (names) are tapped after spending k mana on `spells`: every feasible choice
    must pay the spells' coloured pips. Returns (tapped names, ambiguous). With several feasible
    choices the pick prefers lands of the spells' colours and keeps dual lands untapped.
    `entered_tapped`: lands that came in tapped this turn (gain lands, Guildgates, fetched basics):
    tapped whatever was spent, and they paid for none of it."""
    forced = Counter(entered_tapped or ()) & Counter(lands)
    if forced:
        rest = list((Counter(lands) - forced).elements())
        tapped, amb = choose_tapped_lands(rest, k, spells, ids)
        return tapped + forced, amb
    n = len(lands)
    if k <= 0:
        return Counter(), False
    if k >= n:
        return Counter(lands), False
    pool = Counter(lands)
    if len(pool) == 1:
        return Counter({lands[0]: k}), False
    pips = Counter()
    for sp in spells:
        pips.update(ids.info(sp).pips)
    names = sorted(pool)
    feasible = []
    for combo in itertools.combinations_with_replacement(names, k):
        c = Counter(combo)
        if any(c[x] > pool[x] for x in c):
            continue
        if _pays(c, pips, ids):
            feasible.append(c)
    if not feasible:
        feasible = [Counter(combo) for combo in itertools.combinations_with_replacement(names, k)
                    if all(Counter(combo)[x] <= pool[x] for x in Counter(combo))]
    spell_colors = set(pips)
    def score(c: Counter):
        on = sum(v for x, v in c.items() if set(_land_colors(x, ids)) & spell_colors)
        duals = sum(v for x, v in c.items() if len(_land_colors(x, ids)) > 1)
        return (-on, duals, sorted(c.elements()))
    feasible.sort(key=score)
    return feasible[0], len(feasible) > 1


def _land_colors(name: str, ids: Ids) -> str:
    return ids.info(name).land_mana or BASIC_MANA.get(name, "")


def _pays(tapped: Counter, pips: Counter, ids: Ids) -> bool:
    """Can the tapped lands produce the coloured pips (one pip per land)?"""
    need = [c for c, v in pips.items() for _ in range(v)]
    if len(need) > sum(tapped.values()):
        return False
    lands = [_land_colors(x, ids) for x, v in tapped.items() for _ in range(v)]
    used = [False] * len(lands)

    def rec(i: int) -> bool:
        if i == len(need):
            return True
        for j, cols in enumerate(lands):
            if not used[j] and need[i] in cols:
                used[j] = True
                if rec(i + 1):
                    return True
                used[j] = False
        return False
    return rec(0)


def _tapped_attackers(slot: TurnRecord | None, side: str, insts: list[Inst], ids: Ids) -> set[int]:
    """iids of `side`'s creatures that attacked in `slot` and are still tapped (no vigilance)."""
    if slot is None or slot.side != side:
        return set()
    out = set()
    att = Counter(slot.L("creatures_attacked"))
    for grp, k in att.items():
        cand = sorted([i for i in insts if i.grp == grp and i.kind == "crea"], key=lambda i: (i.entered, i.iid))
        if not cand or "Vigilance" in ids.info(cand[0].name).keywords:
            continue
        # a copy that entered this turn could only attack with haste: prefer the older ones
        older = [i for i in cand if i.entered < slot.seq] + [i for i in cand if i.entered >= slot.seq]
        out.update(i.iid for i in older[:k])
    return out


def _tapped_by_abilities(slot: TurnRecord | None, side: str, insts: list[Inst], ids: Ids,
                         busy: set[int]) -> tuple[set[int], bool]:
    """iids of `side`'s non-land permanents that paid a {T} cost in `slot` (its own turn: Strix
    Lookout, Fanatical Firebrand, Krenko...); they stay tapped through the other player's turn.
    Mana abilities are left out: 17lands shares their ids with basic lands. Returns (iids, guess)
    with guess True when copies make the tapped one a pick. `busy`: already tapped (attackers)."""
    if slot is None or slot.side != side:
        return set(), False
    out, guess = set(), False
    for aid, k in Counter(slot.L(f"{side}_abilities")).items():
        a = ids.ability(aid)
        if a.category != "activated" or not cost_taps_source(a):
            continue
        cand = [i for i in insts if i.name in a.source_cards and i.kind != "lands"
                and i.iid not in out and i.iid not in busy]
        # a creature that entered this turn cannot pay {T} (no haste in the data): prefer older ones
        cand.sort(key=lambda i: (i.kind == "crea" and i.entered >= slot.seq, i.entered, i.iid))
        guess = guess or len(cand) > k
        out.update(i.iid for i in cand[:k])
    return out, guess


def _entered_tapped_lands(slot: TurnRecord | None, side: str, checks: list[SlotCheck], ids: Ids) -> list[str]:
    """Names of `side`'s lands that entered the battlefield tapped in `slot`: tapped lands played
    (gain lands, Guildgates, Temples) and basics fetched by Evolving Wilds and the like (every FDN
    fetch puts the land in tapped except Grow from the Ashes)."""
    if slot is None or not slot.played:
        return []
    out = []
    if slot.side == side:
        out = [ids.name(c) for c in slot.L("lands_played") if "etb_tapped" in ids.info_id(c).features]
    if not any(ids.name(c) == "Grow from the Ashes" for c in _cast_ids(slot, side)):
        out += [ids.name(c) for z, sign, cat, c in checks[slot.seq].cats
                if z == f"{side}_lands" and sign == "+" and cat.startswith("land_put_onto_bf")]
    return out


def _casts_by(slot: TurnRecord | None, who: str, ids: Ids) -> list[str]:
    if slot is None:
        return []
    out = [ids.name(c) for c in slot.L(f"{who}_instants_sorceries_cast")]
    if slot.side == who:
        out += [ids.name(c) for c in slot.L("creatures_cast") + slot.L("non_creatures_cast")]
    return out


def _perms(insts: list[Inst], tapped: set[int], tapped_lands: Counter, sick_after: int,
           notes: set, hosts: set[int]) -> list[Perm]:
    """Battlefield entries for one seat. Lands are grouped by (name, tapped) unless something is
    attached to them; everything else is one entry per permanent with its alias."""
    perms: list[Perm] = []
    land_groups: Counter = Counter()
    tl = Counter(tapped_lands)
    for i in sorted(insts, key=lambda i: (i.kind != "lands", i.entered, i.iid)):
        if i.token and not i.token_class:
            notes.add("token_unmapped")
            continue
        sick = i.entered >= sick_after and i.kind == "crea"
        if i.kind == "lands" and not i.token and i.iid not in hosts and i.host is None:
            t = tl[i.name] > 0
            if t:
                tl[i.name] -= 1
            land_groups[(i.name, t)] += 1
            continue
        p = Perm(name=None if i.token else i.name, tokenClass=i.token_class if i.token else None,
                 id=alias(i), tapped=i.iid in tapped, sick=sick)
        if i.kind == "lands" and tl[i.name] > 0 and not i.token:
            p.tapped, tl[i.name] = True, tl[i.name] - 1
        if i.p1p1:
            p.counters["P1P1"] = i.p1p1
        if i.loyalty is not None:
            if i.loyalty < 1:                  # it would have died: the count missed something
                notes.add("loyalty_estimated")
            p.counters["LOYALTY"] = max(i.loyalty, 1)
        perms.append(p)
    for (name, t), k in sorted(land_groups.items()):
        perms.insert(0, Perm(name=name, count=k, tapped=t))
    return perms


def _expand(c: Counter) -> list[str]:
    return [n for n in sorted(c) for _ in range(c[n])]


def _basics_for(colors: str) -> list[str]:
    names = {v: k for k, v in BASIC_MANA.items() if v in COLORS}
    return [names[c] for c in COLORS if c in (colors or "")] or ["Plains"]


def state_at_user_turn(g: Game, n: int, entry: str = "eot_rollover", opp_decklist: list[str] | None = None,
                       ids: Ids | None = None, labels: bool = False, opp_decklist_source: str = "belief",
                       future_draws: bool = False, opp_hand: list[str] | None = None) -> StateSpec:
    """StateSpec for the start of user turn n after the draw (see the module docstring).

    opp_decklist / opp_decklist_source: the opponent's 40 (e.g. "exact" from the mirrored row of
    the same game, or "belief" from a sampler); default a placeholder. opp_hand: card names known
    to be in the opponent's hand at that point (the mirrored row's end-of-turn hand); the rest
    stays `handUnknown`. future_draws: also put the user's later draws of turn n on libraryTop
    (hindsight: for replaying the turn, not for decisions)."""
    if entry not in ENTRIES:
        raise ValueError(f"entry {entry!r} not in {ENTRIES}")
    ids = ids or Ids.load(g.meta.get("expansion") or "FDN")
    u = g.user_slot(n)
    if u is None:
        raise ValueError(f"row {g.row_index} has no user turn {n}")
    ana = analyze(g, ids)
    p = g.prev_slot(n)
    notes: set[str] = set()
    fl = ana.flags.get(n, {})
    drawn = [ids.name(c) for c in u.L("cards_drawn") if c != -1 and ids.cards17.get(c)]
    if len(drawn) < len(u.L("cards_drawn")):
        notes.add("draw_unknown_card")              # a corrupt id (-1): the draw is not known
    if not drawn and p is not None and u.played:
        notes.add("draw_unknown_card")              # no draw recorded (0.01% of turns): the engine draws blind
    A, B = PlayerState(name="A"), PlayerState(name="B")
    spec = StateSpec(players={"A": A, "B": B}, startingPlayer="A" if g.on_play else "B")
    if p is None:
        # user turn 1 on the play: no snapshot before it; main1 from the kept hand
        if entry == "eot_rollover":
            notes.add("entry_main1_t1_on_play")
        entry_used = "main1"
        hand = Counter(g.opening_hand)
        hand.subtract(g.bottomed)
        A.hand = sorted(ids.name(c) for c in (+hand).elements() if c != -1)
        A.handUnknown = sum(1 for c in (+hand).elements() if c == -1)
        B.handUnknown = 7 - int(g.meta.get("opp_num_mulligans") or 0)
        if not g.bottomed_exact:
            notes.add("bottomed_inexact")
        spec.turn, spec.activePlayer = u.global_turn, "A"
        st = SlotState({"user": [], "oppo": []}, {"user": Counter(), "oppo": Counter()},
                       {"user": Counter(), "oppo": Counter()}, Counter(), {"user": 0, "oppo": 0})
    else:
        entry_used = entry
        st = ana.states[p.seq]
        snap = p
        if not p.played:
            # the half-turn before is an empty slot: use the last snapshot before it
            notes.add("prev_snapshot_missing")
            snap = next((t for t in reversed(g.turns[:p.seq]) if t.played), p)
        A.life, B.life = int(snap.num("eot_user_life", 20.0)), int(snap.num("eot_oppo_life", 20.0))
        hand = [c for c in snap.L("eot_user_cards_in_hand")]
        A.handUnknown = sum(1 for c in hand if c == -1 or not ids.cards17.get(c))
        A.hand = sorted(ids.name(c) for c in hand if c != -1 and ids.cards17.get(c))
        B.handUnknown = int(snap.num("eot_oppo_cards_in_hand"))
        if A.handUnknown:
            notes.add("hand_unknown_card")
    # --- battlefields, tapped state, sickness ---------------------------------------------------
    a_insts, b_insts = st.bf["user"], st.bf["oppo"]
    if p is None:
        a_tap, b_tap, a_lands, b_lands = set(), set(), Counter(), Counter()
        a_sick = b_sick = 10 ** 9
    else:
        # B's permanents tapped on its own last turn stay tapped through ours: its attackers, the
        # sources of its {T} abilities, the lands it tapped for mana and lands that came in tapped
        b_tap = _tapped_attackers(p, "oppo", b_insts, ids)
        b_abil, guess = _tapped_by_abilities(p, "oppo", b_insts, ids, b_tap)
        b_tap |= b_abil
        if guess:
            notes.add("opp_tap_ability_source_guess")
        b_land_names = [i.name for i in b_insts if i.kind == "lands" and not i.token]
        b_lands, amb = choose_tapped_lands(b_land_names, int(p.num("oppo_mana_spent")), _casts_by(p, "oppo", ids), ids,
                                           entered_tapped=_entered_tapped_lands(p, "oppo", ana.checks, ids))
        if amb:
            notes.add("opp_tapped_lands_ambiguous")
        b_sick = p.seq                                  # entered during B's own last turn
        pu = g.turns[p.seq - 1] if p.seq > 0 else None    # the user's previous turn
        a_tap = _tapped_attackers(pu, "user", a_insts, ids)
        a_tap |= _tapped_by_abilities(pu, "user", a_insts, ids, a_tap)[0]
        # Slumbering Cerberus skips the untap step; its Morbid trigger untaps it at an end step
        # after a creature died (pu or p): tapped from its attack only if nothing died since
        died = any(t is not None and (t.L("user_creatures_killed_combat") + t.L("user_creatures_killed_non_combat")
                                      + t.L("oppo_creatures_killed_combat") + t.L("oppo_creatures_killed_non_combat"))
                   for t in (pu, p))
        stuck = {i.iid for i in a_insts if "doesnt_untap" in ids.info(i.name).features and i.iid in a_tap and not died}
        if any("doesnt_untap" in ids.info(i.name).features for i in a_insts + b_insts):
            notes.add("doesnt_untap_inferred")
        if entry_used == "eot_rollover":
            a_tap = {i for i in a_tap if i not in {x.iid for x in a_insts if "doesnt_untap" in ids.info(x.name).features}} | stuck
            k = int(p.num("user_mana_spent")) + (int(pu.num("user_mana_spent")) if pu is not None and pu.side == "user" else 0)
            a_lands, _ = choose_tapped_lands([i.name for i in a_insts if i.kind == "lands" and not i.token], k,
                                             _casts_by(pu, "user", ids) + _casts_by(p, "user", ids), ids,
                                             entered_tapped=_entered_tapped_lands(pu, "user", ana.checks, ids)
                                             + _entered_tapped_lands(p, "user", ana.checks, ids))
            a_sick = pu.seq if pu is not None and pu.side == "user" else p.seq
        else:
            # after the untap step nothing is new or tapped, bar a "doesn't untap" creature
            a_tap, a_lands, a_sick = stuck, Counter(), 10 ** 9
    hosts = {i.host for i in a_insts + b_insts if i.host is not None}
    # a "doesn't untap" Aura (Starlight Snare) tapped its host when it entered, and it stays tapped
    frozen = {i.host for i in a_insts + b_insts if i.host is not None and ids.info(i.name).attach.startswith("aura:freeze")}
    a_tap, b_tap = set(a_tap) | frozen, set(b_tap) | frozen
    A.battlefield = _perms(a_insts, a_tap, a_lands, a_sick, notes, hosts)
    B.battlefield = _perms(b_insts, b_tap, b_lands, b_sick, notes, hosts)
    for insts, seat_perm in ((a_insts, A), (b_insts, B)):
        for i in insts:
            if i.host is not None:
                host = next((h for h in a_insts + b_insts if h.iid == i.host), None)
                if host is not None:
                    perm = next(x for x in seat_perm.battlefield if x.id == alias(i))
                    perm.attachTo = f"{SEAT[host.side]}:{alias(host)}"
                if not i.host_exact:
                    notes.add("attach_heuristic")
            elif i.attach_issue in ("no_candidate", "host_left"):
                notes.add("attach_no_host")
            elif i.attach_issue == "player":
                notes.add("aura_on_player_unrepresented")
            if not i.counters_exact and i.kind == "crea":
                notes.add("counters_inexact")
            if i.loyalty is not None and not i.loyalty_exact:
                notes.add("loyalty_estimated")
    # --- graveyards, exile, entry specifics ------------------------------------------------------
    A.graveyard, B.graveyard = _expand(st.gy["user"]), _expand(st.gy["oppo"])
    A.exile, B.exile = _expand(st.exile["user"]), _expand(st.exile["oppo"])
    if opp_hand:
        B.hand = sorted(opp_hand)
        if len(opp_hand) != B.handUnknown:
            notes.add("opp_hand_count_mismatch")
        B.handUnknown = max(0, B.handUnknown - len(opp_hand))
    if st.gy_unknown["oppo"]:
        notes.add("opp_gy_incomplete")             # the opponent milled / surveiled unseen cards
    # cards the user draws before its first main phase: the draw step's, after one per upkeep draw
    # trigger it controls (Scrawling Crawler, Phyrexian Arena draw first, so they take cards_drawn[0])
    pre = 1 + sum("upkeep_draw" in ids.info(i.name).features for i in a_insts)
    if p is not None and entry_used == "eot_rollover":
        spec.turn, spec.activePlayer = p.global_turn, "B"
        spec.phase, spec.step = "END", "END_TURN"
        A.libraryTop = drawn[:pre] + (drawn[pre:] if future_draws else [])
        B.landsPlayed = len(p.L("lands_played"))
    elif p is not None:
        spec.turn, spec.activePlayer = u.global_turn, "A"
        A.hand = sorted(A.hand + drawn[:pre])
        A.libraryTop = drawn[pre:] if future_draws else []
        if not drawn:
            A.handUnknown += 1                     # the draw step's card is not in the row
        # main1 skips the untap-to-draw triggers the engine would play from eot_rollover: life
        # (Scrawling Crawler's drain on the draw) and tokens can be off
        if any("turn_start_trigger" in ids.info(i.name).features for i in a_insts) \
                or any("opp_draw_trigger" in ids.info(i.name).features for i in b_insts):
            notes.add("main1_turn_start_triggers_skipped")
    spec.priorityPlayer = spec.activePlayer
    # --- decklists -------------------------------------------------------------------------------
    deck = Counter(g.deck)
    if sum(deck.values()) < 40:
        notes.add("A_decklist_padded")
        basics = _basics_for((g.meta.get("main_colors") or "") + (g.meta.get("splash_colors") or ""))
        for j in range(40 - sum(deck.values())):
            deck[basics[j % len(basics)]] += 1
    _fit_zones(A, deck, notes, "A")
    if opp_decklist is not None:
        B.decklistSource = opp_decklist_source
        _fit_zones(B, Counter(opp_decklist), notes, "B")
    else:
        B.decklistSource = "placeholder"
        notes.add("opp_decklist_placeholder")
        B.decklist = _placeholder_deck(B, st.revealed, g.meta.get("opp_colors") or "", ids)
    # --- provenance ------------------------------------------------------------------------------
    flags = sorted(k for k, v in fl.items() if v) + sorted(notes)
    if u.terminal:
        flags.append("terminal_slot")
    ref = f"{g.meta.get('expansion') or ids.set_code}_{g.meta.get('event_type') or ''}:row={g.row_index}:user_turn={n}"
    spec.provenance = Provenance(source="17lands", ref=ref, tier=spec_tier(fl, notes), flags=flags)
    spec.comment = (f"17lands {ref.split(':')[0]} row {g.row_index}: start of user turn {n} after the draw "
                    f"({entry_used}{' from global turn ' + str(spec.turn) if entry_used == 'eot_rollover' else ''})")
    if labels:
        from draftzero.gameplay.labels import turn_label
        spec.labels = turn_label(g, n, ids)
    return spec


def _fit_zones(P: PlayerState, deck: Counter, notes: set, seat: str) -> None:
    """Every card named in P's zones must come out of the decklist: trim over-counted exile /
    graveyard entries (tracking guesses) first, then extend the decklist (stolen cards, token
    copies recorded under a card id) and flag it."""
    def used() -> Counter:
        return Counter(P.hand + P.graveyard + P.exile + P.libraryTop
                       + [x.name for x in P.battlefield if x.name and not (x.token or x.tokenClass) for _ in range(x.count)])
    over = used() - deck
    for name, k in over.items():
        for zone in (P.exile, P.graveyard):
            while k > 0 and name in zone:
                zone.remove(name)
                k -= 1
                notes.add("zone_overflow_trimmed")
    over = used() - deck
    if over:
        notes.add(f"{seat}_decklist_extended")
        deck = deck + over
    P.decklist = _expand(deck)


def _placeholder_deck(B: PlayerState, revealed: Counter, opp_colors: str, ids: Ids) -> list[str]:
    """Revealed opponent cards (max simultaneous count seen) + everything in B's zones, filled to
    40 with basics of the opponent's colours."""
    need = Counter(B.hand + B.graveyard + B.exile + [x.name for x in B.battlefield if x.name for _ in range(x.count)])
    deck = Counter(revealed) | need
    basics = _basics_for(opp_colors)
    j = 0
    # at least one card left after the hidden hand is drawn: an empty library would lose B the
    # game at its next draw (long games, where the sink zones over-count)
    while sum(deck.values()) < max(40, sum(need.values()) + B.handUnknown + 1):
        deck[basics[j % len(basics)]] += 1
        j += 1
    return _expand(deck)


_T3 = {"natural_draw_multi", "history_unexplained", "hand_unknown_card", "token_unmapped", "bottomed_inexact",
       "prev_snapshot_missing", "draw_unknown_card"}
_T2 = {"attach_heuristic", "attach_no_host", "exile_link_present", "token_copy_possible",
       "A_decklist_extended", "B_decklist_extended", "aura_on_player_unrepresented"}
_T1 = {"opp_mana_gt_lands", "opp_tapped_lands_ambiguous", "counters_inexact", "loyalty_estimated",
       "gy_uncertain_user", "opp_gy_incomplete", "zone_overflow_trimmed", "opp_tap_ability_source_guess",
       "doesnt_untap_inferred", "main1_turn_start_triggers_skipped"}


def spec_tier(re_fl: dict, notes: set) -> str:
    """statespec grade from the RE flags that hold and the builder's own notes (module docstring)."""
    on = {k for k, v in re_fl.items() if v} | set(notes)
    if on & _T3:
        return "T3"
    if on & _T2:
        return "T2"
    if on & _T1:
        return "T1"
    return "T0"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m draftzero.gameplay.reconstruct",
                                 description="Print the StateSpec at the start of one user turn.")
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--turn", type=int, required=True)
    ap.add_argument("--entry", choices=ENTRIES, default="eot_rollover")
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--path", default=None)
    a = ap.parse_args(argv)
    from draftzero.gameplay.replay import read_games
    g = read_games([a.row], a.path)[a.row]
    spec = state_at_user_turn(g, a.turn, a.entry, labels=a.labels)
    print(spec.to_json(indent=1))
    errs = spec.validate()
    if errs:
        print("validate:", errs, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
