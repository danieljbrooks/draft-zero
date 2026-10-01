"""
What the 17lands user did around one turn, as MageZero-shaped labels.

`turn_label(game, n)` covers user turn n and the opponent half-turn right after it (the user's
blocks and instant-speed plays happen there). 17lands records actions as per-turn multisets with no
order, phase, targets or mana payment, so every list here has set semantics (sorted, not ordered):

  lands[]              {"name", "key": "Play X", "idx"}
  casts[]              {"name", "key", "idx", "zone", "family"}; key "Flashback {cost}" when the card
                       was cast from the graveyard (no copy left in hand, one in the graveyard)
  activations[]        activated non-mana abilities only (the MCTS priority actions under autoTap):
                       {"id", "key", "idx", "source", "source_unique", "vocab_exact"}
  attacks              {"A:<alias>": bool} over the attack-eligible creatures at the start of the
                       turn ("new:<name>" for a creature that entered this turn and attacked); not
                       eligible: Defender, a Pacifism or Starlight Snare host
  blocks               [[blocker, attacker or None, exact]] for every potential blocker in the next
                       opponent turn; `block_pairing` is unique | ambiguous | inconsistent |
                       too_many | no_attack | none (research pairing by printed P/T, evasion and
                       deaths, `block_assignments`; the flying rule is lifted when the user had a
                       flying grant, `flying_granted`). exact is False for a guessed pairing (an
                       ambiguous one, an inconsistent one's legal guess), and for "did not block"
                       when the creature left the battlefield outside combat that turn (maybe
                       before blocks)
  offturn_instants[]   instants cast on the following opponent turn (auxiliary target)
  offturn_flash[]      flash permanents inferred on that turn (hand -> battlefield, no cast record)
  deduped[]            the user turn's lists (creatures_attacked, user_abilities) that 17lands
                       recorded twice over and the labels count once (`recorded_once`);
                       offturn_deduped[] the same for the opponent turn's (creatures_attacked,
                       creatures_blocked, creatures_blocking, *_killed_combat, user_abilities)
  mulligan, bottomed   user turn 1 only: {"kept_after": k, "hands": [[names], ...]}, bottomed names

`after_turn_label(game, n)` is the opponent-turn part alone (blocks, block_pairing, offturn_*),
plus `attacked` (the opponent's attackers by name): the labels of
`reconstruct.state_after_user_turn(game, n)`, whose declare_attackers entry is where the blocks are
decided. A potential blocker is a creature of the user's at the end of its turn that is still
untapped in that spec (it did not attack without vigilance or pay a {T} cost there, no Starlight
Snare, a Slumbering Cerberus that untapped), is not a land it animated, can block (not Vampire
Soulcaller or a Pacifism host; Brazen Borrower only against flyers) and was not taken by the
opponent for its turn; flash blockers that entered during the opponent's turn are keyed from the
later state (they are not in the declare_attackers spec, which is flagged offturn_timing_unknown).

Aliases ("A:GleamingBarrier_4") are the permanent ids of the StateSpec that
`reconstruct.state_at_user_turn(game, n)` (or `state_after_user_turn`) builds, so a bridge can
apply the labels to that spec. Labels of a game's last slot (`terminal`) can be incomplete: the
game ended during it.

  python -m draftzero.gameplay.labels --row 0 --turn 3 [--after]
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from collections import Counter

from draftzero.gameplay.ids import AbilityRef, Ids
from draftzero.gameplay.reconstruct import COPY_MAKERS, Inst, _user_tapped_after, alias, analyze
from draftzero.gameplay.replay import Game, TurnRecord


def _key_entry(name: str, key: str, idx: int, exact: bool, **kw) -> dict:
    return {"name": name, "key": key, "idx": idx, "vocab_exact": exact, **kw}


def _casts(t: TurnRecord, who: str, hand: Counter, gy: Counter, ids: Ids) -> list[dict]:
    """Casts by `who` in slot t, flashback-aware. Permanents first (they come from hand), then
    instants/sorceries one by one: a copy cast from hand lands in the graveyard, so a second cast of
    Think Twice in the same turn is its flashback."""
    out = []
    fams = [("creatures_cast", "creature"), ("non_creatures_cast", "noncreature")] if t.side == who else []
    for f, fam in fams + [(f"{who}_instants_sorceries_cast", "instant_sorcery")]:
        for c in t.L(f):
            name = ids.name(c)
            key, idx, exact, zone = ids.cast_key(name, hand, gy)
            if zone == "hand":
                hand[name] -= 1
                if fam == "instant_sorcery":
                    gy[name] += 1
            elif zone == "graveyard":
                gy[name] -= 1
            out.append(_key_entry(name, key, idx, exact, zone=zone, family=fam))
    return sorted(out, key=lambda d: (d["family"], d["name"], d["key"]))


def _activations(aids: list[int], board: set, ids: Ids) -> list[dict]:
    out = []
    for aid in aids:
        a = ids.ability(aid)
        if not a.is_action:
            continue
        key, idx, exact, unique = ids.ability_key(aid, board)
        src = next((s for s, k, _, _ in a.keys_by_source if s in board), a.source_cards[0] if a.source_cards else None)
        out.append({"id": aid, "key": key, "idx": idx, "vocab_exact": exact, "source": src,
                    "source_unique": unique})
    return sorted(out, key=lambda d: (str(d["key"]), d["id"]))


def _match(grps: Counter, insts: list[Inst]) -> tuple[list[Inst], Counter]:
    """Instances for an id multiset (oldest copies first) and the ids left unmatched."""
    by = {}
    for i in sorted(insts, key=lambda i: (i.entered, i.iid)):
        by.setdefault(i.grp, []).append(i)
    got, left = [], Counter()
    for g, k in grps.items():
        pool = by.get(g, [])
        got.extend(pool[:k])
        if k > len(pool):
            left[g] = k - len(pool)
    return got, left


# --- lists recorded twice over -------------------------------------------------------------------------

def _doubled(xs: list) -> bool:
    k = len(xs) // 2
    return k > 0 and len(xs) == 2 * k and xs[:k] == xs[k:]


def _act_cost(a: AbilityRef) -> tuple[int | None, bool]:
    """(mana in an activated ability's cost, None when unknown or X; at most once per source and
    turn: a {T}, sacrifice-itself or loyalty cost), read off its XMage key."""
    k = a.xmage_key
    if not k:
        return None, False
    if a.loyalty is not None:
        return 0, True
    cost = k[6:].split(" ", 1)[0] if k.startswith("Equip ") else k.split(":", 1)[0]
    mana: int | None = 0
    for s in re.findall(r"\{([^}]+)\}", cost):
        if s == "X":
            mana = None
        elif mana is not None and s not in ("T", "this"):
            mana += int(s) if s.isdigit() else 1
    return mana, "{T}" in cost or "Sacrifice {this}" in cost


# FDN cards that can cost less than their mana value (a reduction, affinity or improvise of their
# own: Claws Out with a Cat, Tolarian Terror ...), and those that make other spells cost less or
# nothing while on the battlefield, read off the XMage source: with one cast, or on the caster's
# battlefield, mana spent is no measure of the activations paid (row 123680)
_COST_REDUCED = frozenset({
    "Arcane Epiphany", "Blasphemous Edict", "Bolt Bend", "Claws Out", "Dargo, the Shipwrecker", "Embercleave",
    "Frogmite", "Ghalta, Primal Hunger", "Luminous Rebuke", "Sanguine Indulgence", "Thoughtcast", "Tolarian Terror",
    "Whir of Invention"})
_COST_REDUCERS = frozenset({"Archmage of Runes", "Ballyrush Banneret", "Dragonlord's Servant", "Mocking Sprite",
                            "Omniscience"})


def _abilities_doubled(t: TurnRecord, who: str, xs: list[int], board: Counter, ids: Ids,
                       sources_known: bool = True) -> bool:
    """A doubled ability list the record cannot explain: `who`'s mana spent in t pays for its casts
    and one copy of the activations, not two (no X, no cost reduction in play: `_COST_REDUCED`);
    or a once-per-source ability ({T}, sacrifice itself, loyalty) was activated more often than
    `board` (card names) has sources (when `sources_known`: no copy maker)."""
    half = xs[:len(xs) // 2]
    acts = [ids.ability(x) for x in half if ids.ability(x).is_action]
    if not acts:
        return False
    fams = ("creatures_cast", "non_creatures_cast") if t.side == who else ()
    casts = [c for f in fams + (f"{who}_instants_sorceries_cast",) for c in t.L(f)]
    costs = [_act_cost(a) for a in acts]
    reduced = any(ids.name(c) in _COST_REDUCED for c in casts) or any(board[x] for x in _COST_REDUCERS)
    if (all(m is not None for m, _ in costs) and not reduced
            and not any("{X}" in ids.info_id(c).mana_cost for c in casts)):
        once_mana = sum(m for m, _ in costs)
        if once_mana > 0 and t.num(f"{who}_mana_spent") == sum(ids.mv(c) for c in casts) + once_mana:
            return True
    if not sources_known:
        return False
    tokens = ids.token_names()
    for a, k in Counter(a for a, (_, once) in zip(acts, costs) if once).items():
        sources = sum(board[s] for s in set(a.source_cards))
        # a token source (Food) made and used up within t is on no snapshot: no evidence
        if 0 < sources < 2 * k and not tokens & set(a.source_cards):
            return True
    return False


def recorded_once(g: Game, t: TurnRecord, ids: Ids) -> dict[str, list[int]]:
    """The lists of slot t that 17lands recorded twice over, once each: {field: first half} for
    the fields it halves (usually none). In 3.3% of attacking turns the attacker list is exactly
    two copies of itself, often with the unblocked, blocked and blocking lists of that combat
    (2-5% of ability lists too). Most of them are two copies of a card that both attacked, so a
    doubled list is halved only when the board cannot explain the second copy:

      creatures_attacked    more copies of a card than the active player had on the battlefield
                            at the start of t plus those that entered during t (cast, created,
                            returned, flashed in); tokens are no evidence (one made and killed
                            within t is in no snapshot and no killed list), nor is the board of a
                            player with a copy maker (Chandra, Flameshaper's hasty copy carries
                            the card's id)
      creatures_blocked     more than the attackers (a blocked creature attacked)
      creatures_blocking    more copies of a card than the defender had on the battlefield (the
                            same way), or the blocked list of that combat was halved
      *_killed_combat       more than that side's attackers or blockers
      *_abilities           see `_abilities_doubled` (mana spent, once-per-source costs)

    Row 116964, user turn 5: [Skyknight Squire, Cat Collector] x2 with one of each on the
    battlefield. A slot that holds an extra turn (Temporal Manipulation: 17lands records it in the
    same slot, two land drops) can repeat a combat for real (row 2280, the opponent's turn 5: the
    lifelinking Hawk gained 2); halving it keeps the first turn's, the one the labels describe."""
    ana = analyze(g, ids)
    start = ana.states[t.seq - 1] if t.seq > 0 else None
    board: dict[str, Counter | None] = {}
    raw: dict[str, Counter] = {}
    for w in ("user", "oppo"):
        b = Counter(i.name for i in (start.bf[w] if start else []))
        b.update(ids.name(c) for z, sign, _, c in ana.checks[t.seq].cats
                 if sign == "+" and z in (f"{w}_lands", f"{w}_crea", f"{w}_nonc"))
        if w == t.side:
            b.update(ids.name(c) for c in t.L("creatures_cast") + t.L("non_creatures_cast"))
        # a copy (Chandra, Flameshaper's hasty token copy, Self-Reflection) made and gone within t
        # carries the copied card's id and is on no snapshot: that battlefield is no evidence
        copies = any(x in COPY_MAKERS for x in b) or any(
            ids.name(c) in COPY_MAKERS for c in t.L(f"{w}_instants_sorceries_cast"))
        raw[w], board[w] = b, (None if copies else b)
    out: dict[str, list[int]] = {}

    def once(field: str, bound: Counter | None, cards_only: bool = False, same_record: bool = False) -> list[int]:
        xs = t.L(field)
        n = Counter(ids.name(c) for c in xs if not (cards_only and ids.is_token(c)))
        if _doubled(xs) and (same_record or (bound is not None and any(k > bound[x] for x, k in n.items()))):
            out[field] = xs = xs[:len(xs) // 2]
        return xs
    att = once("creatures_attacked", board[t.side], cards_only=True)
    names = Counter(ids.name(c) for c in att)
    once("creatures_blocked", names)
    # doubled blockers of a combat whose blocked list was recorded twice over: the same double
    blk = Counter(ids.name(c) for c in once("creatures_blocking", board[t.other], cards_only=True,
                                             same_record="creatures_blocked" in out))
    once(f"{t.side}_creatures_killed_combat", names)
    once(f"{t.other}_creatures_killed_combat", blk)
    for who in ("user", "oppo"):
        xs = t.L(f"{who}_abilities")
        if _doubled(xs) and _abilities_doubled(t, who, xs, raw[who], ids, sources_known=board[who] is not None):
            out[f"{who}_abilities"] = xs[:len(xs) // 2]
    return out


# --- blocker -> attacker pairing (port of research semantics2.py) -----------------------------------

# the FDN token classes (mage.game.permanent.token.*) with Flying, and those without Flying, Reach
# or Menace, read off the XMage source; a token of any other class is a wildcard (as its P/T is)
_FLYING_TOKENS = frozenset({"BirdIllusionToken", "DragonToken", "DragonToken2", "DrakeToken", "FaerieToken",
                            "InsectBlackGreenFlyingToken", "SpiritWhiteToken"})
_GROUND_TOKENS = frozenset({
    "BeastToken", "BeastToken2", "CatBeastToken", "CatToken", "CatToken2", "CatToken3", "ClueArtifactToken",
    "EldraziScionToken", "ElfWarriorToken", "FishNoAbilityToken", "FoodToken", "GoblinToken", "GolemToken",
    "HumanToken", "Knight33Token", "KomasCoilToken", "NinjaToken2", "PhyrexianGoblinToken", "RabbitToken",
    "RaccoonToken", "RatCantBlockToken", "RatToken", "RhinoToken", "ScionOfTheDeepToken", "SoldierToken",
    "TreasureToken", "WhiteDogToken", "ZombieToken"})
# creatures whose Flying or Menace in the card-facts table is not printed but granted under a
# condition (Skyknight Squire with three counters, Kitesail Corsair while attacking, Courageous
# Goblin attacking beside a 4-power creature ...): the table's keyword regex matches any mention
_CONDITIONAL_EVASION = frozenset({"Battlesong Berserker", "Courageous Goblin", "Dropkick Bomber",
                                  "Elenda, Saint of Dusk", "Kargan Dragonrider", "Kitesail Corsair",
                                  "Skyknight Squire"})


def _evasion(ids: Ids, grp: int) -> frozenset | None:
    """The printed Flying / Reach / Menace of a card id, plus "blocks_only_flyers" (Brazen
    Borrower); None when not known (a token of an unlisted class, a card without facts, one whose
    evasion is conditional)."""
    if ids.is_token(grp):
        cls = ids.card(grp).token_class
        return frozenset({"Flying"}) if cls in _FLYING_TOKENS else frozenset() if cls in _GROUND_TOKENS else None
    name = ids.name(grp)
    if name not in ids.infos or name in _CONDITIONAL_EVASION:
        return None
    info = ids.infos[name]
    return (info.keywords & {"Flying", "Reach", "Menace"}) | (
        {"blocks_only_flyers"} if "blocks_only_flyers" in info.features else set())


def _can_block(ids: Ids, blocker: int, attacker: int, blockers_may_fly: bool = False) -> bool:
    """Printed evasion allows this block: a flyer is blocked only by Flying or Reach (by anyone
    when the blockers may have been given flying, `flying_granted`), and a creature that blocks
    only flyers blocks only those (unknowns allow it)."""
    a, b = _evasion(ids, attacker), _evasion(ids, blocker)
    if a is None or b is None:
        return True
    return (("Flying" not in a or blockers_may_fly or bool(b & {"Flying", "Reach"}))
            and ("blocks_only_flyers" not in b or "Flying" in a))


# a creature that gives other creatures flying (Dropkick Bomber, its Goblins); the non-creature
# cards that do (Fleeting Flight, Angelic Destiny, Celestial Armor with flash, Akroma's Memorial,
# Hallowed Haunting, Valkyrie's Call's Angels) are those whose facts list Flying or Reach, less
# planeswalkers (Ajani's -3 is sorcery-speed and lasts the turn: never on a blocker) and Vehicles
# (Skysovereign's Flying is its own)
_FLYING_GRANT_CREATURES = frozenset({"Dropkick Bomber"})


def flying_granted(ids: Ids, names) -> bool:
    """Whether a player with these cards (its battlefield, the instants it cast that turn) may
    have given a creature flying or reach, so that a printed ground creature of its blocked a
    flyer: `block_assignments(blockers_may_fly=...)`. Row 238760, the opponent's turn 11: a
    Soldier token under Fleeting Flight blocked Flamewake Phoenix."""
    for x in names:
        info = ids.infos.get(x)
        if x in _FLYING_GRANT_CREATURES or (info is not None and info.power is None and info.loyalty is None
                                            and "vehicle_crew" not in info.features
                                            and info.keywords & {"Flying", "Reach"}):
            return True
    return False


def _pt(ids: Ids, grp: int):
    if ids.is_token(grp):
        return None                       # tokens are wildcards, as in the research
    info = ids.info_id(grp)
    return (info.power, info.toughness) if info.power is not None else None


def _outcomes(ids: Ids, a: int, blockers: list[tuple[int, int]]):
    """Possible (attacker dies, killed blocker indices) for one blocked attacker; printed P/T,
    deathtouch, first/double strike, indestructible; no pumps or counters."""
    pa = _pt(ids, a)
    bl = [(i, _pt(ids, b), ids.info_id(b).keywords) for i, b in blockers]
    if pa is None or any(p is None for _, p, _ in bl):
        return None
    P, T = pa
    ka = ids.info_id(a).keywords
    outs = set()
    a_dt = "Deathtouch" in ka
    a_fs = "FirstStrike" in ka or "DoubleStrike" in ka
    for r in range(len(bl) + 1):
        for subset in itertools.combinations(range(len(bl)), r):
            need = sum(1 if a_dt else max(bl[j][1][1], 0) for j in subset)
            if need > P * (2 if "DoubleStrike" in ka else 1) or (P <= 0 and subset):
                continue
            dmg, dt = 0, False
            for j, (i, (bp, bt), bk) in enumerate(bl):
                if a_fs and j in subset and not ("FirstStrike" in bk or "DoubleStrike" in bk):
                    continue
                dmg += max(bp, 0)
                dt |= "Deathtouch" in bk and bp > 0
            outs.add(((dmg >= T or dt) and "Indestructible" not in ka, frozenset(bl[j][0] for j in subset)))
    return outs


def block_assignments(ids: Ids, blocked: list[int], blockers: list[int], dead_att: list[int],
                      dead_blk: list[int], cap: int = 50000,
                      blockers_may_fly: bool = False) -> tuple[str, list[tuple[int, ...]]]:
    """(status, assignments): each assignment gives, per blocker index, the index of the blocked
    attacker it blocked. status: unique | ambiguous | inconsistent | too_many | none.
    With several blocked attackers an assignment must be legal by printed evasion (`_can_block`:
    a flyer is blocked by Flying or Reach, or by anyone when `blockers_may_fly` (the defender had
    a flying grant, `flying_granted`), Brazen Borrower blocks only flyers; a Menace attacker
    has two or more blockers) and explain the combat deaths by printed P/T: the unique or
    ambiguous ones. When none explains the deaths (pumps, counters, a trick), the status is
    inconsistent with one legal guess in which every blocked attacker is blocked: the blockers
    spread in turn over the blocked attackers if that is legal, else the first legal one (none
    when no assignment is legal: an evasion the record does not show). One blocked attacker is
    the pairing whatever the keywords say: the record settles it."""
    nb, nk = len(blocked), len(blockers)
    if nb == 0 or nk == 0:
        return "none", []
    if nk < nb:
        return "inconsistent", []
    if nb == 1:
        return "unique", [tuple([0] * nk)]
    legal = [[ai for ai in range(nb) if _can_block(ids, b, blocked[ai], blockers_may_fly)] for b in blockers]
    n = 1
    for x in legal:
        n *= len(x)
    if n > cap:
        return "too_many", []
    menace = [ai for ai, a in enumerate(blocked) if "Menace" in (_evasion(ids, a) or ())]

    def allowed(assign) -> bool:
        return len(set(assign)) == nb and all(assign.count(ai) >= 2 for ai in menace)
    dA, dB = Counter(dead_att), Counter(dead_blk)
    seen, good, guess = set(), [], None
    for assign in itertools.product(*legal):
        if not allowed(assign):
            continue
        canon = tuple(sorted((blocked[assign[k]], blockers[k]) for k in range(nk)))
        if canon in seen:
            continue
        seen.add(canon)
        per = []
        for ai, a in enumerate(blocked):
            bl = [(k, blockers[k]) for k in range(nk) if assign[k] == ai]
            o = _outcomes(ids, a, bl)
            if o is None:
                o = {(d, frozenset(s)) for d in (False, True) for r in range(len(bl) + 1)
                     for s in itertools.combinations([k for k, _ in bl], r)}
            per.append((a, o))

        def rec(i, da, dk):
            if i == len(per):
                return da == dA and dk == dB
            a, outs = per[i]
            for dies, killed in outs:
                da2 = da + Counter([a]) if dies else da
                dk2 = dk + Counter(blockers[k] for k in killed)
                if any(da2[x] > dA[x] for x in da2) or any(dk2[x] > dB[x] for x in dk2):
                    continue
                if rec(i + 1, da2, dk2):
                    return True
            return False
        if rec(0, Counter(), Counter()):
            good.append(assign)
        elif guess is None:
            guess = assign
    if not good:
        spread = tuple(k % nb for k in range(nk))
        if all(spread[k] in legal[k] for k in range(nk)) and allowed(spread):
            guess = spread
        return "inconsistent", [guess] if guess is not None else []
    return ("unique" if len(good) == 1 else "ambiguous"), good


# --- the label ---------------------------------------------------------------------------------------

def turn_label(g: Game, n: int, ids: Ids | None = None) -> dict:
    ids = ids or Ids.load(g.meta.get("expansion") or "FDN")
    ana = analyze(g, ids)
    u = g.user_slot(n)
    if u is None:
        raise ValueError(f"row {g.row_index} has no user turn {n}")
    p = g.prev_slot(n)
    start = ana.states[p.seq] if p is not None else None
    a_start = start.bf["user"] if start else []
    lab: dict = {"user_turn": n, "global_turn": u.global_turn, "terminal": u.terminal,
                 "mana_spent": int(u.num("user_mana_spent"))}
    # --- own turn: lands, casts, activations ----------------------------------------------------
    lab["lands"] = sorted((_key_entry(ids.name(c), *ids.play_key(ids.name(c))) for c in u.L("lands_played")),
                          key=lambda d: d["name"])
    if p is None:
        hand = Counter(g.opening_hand)
        hand.subtract(g.bottomed)
    else:
        # an empty "hole" slot has no snapshot: the last one before it (as the spec builder does)
        snap = p if p.played else next((t for t in reversed(g.turns[:p.seq]) if t.played), p)
        hand = Counter(snap.L("eot_user_cards_in_hand"))
    hand.update(u.L("cards_drawn") + u.L("cards_tutored"))
    hand = Counter({ids.name(c): k for c, k in (+hand).items()})
    gy = Counter(start.gy["user"]) if start else Counter()
    for c in u.L("lands_played"):
        hand[ids.name(c)] -= 1
    lab["casts"] = _casts(u, "user", hand, gy, ids)
    once = recorded_once(g, u, ids)
    att_list = once.get("creatures_attacked", u.L("creatures_attacked"))
    lab["deduped"] = sorted(f for f in once if f in ("creatures_attacked", "user_abilities"))
    board = {i.name for i in a_start} | {ids.name(c) for c in _eot_bf(u, "user")}
    lab["activations"] = _activations(once.get("user_abilities", u.L("user_abilities")), board, ids)
    # --- attacks -----------------------------------------------------------------------------------
    b_start = start.bf["oppo"] if start else []
    pacified = _pacified(b_start, ids)
    eligible, notes = [], []
    if "creatures_attacked" in once:
        notes.append("creatures_attacked is recorded twice over (the battlefield has too few copies): counted once")
    left_early = Counter(u.L("user_creatures_killed_non_combat"))
    attacked_grps = Counter(att_list)
    for i in a_start:
        if i.kind != "crea":
            continue
        if "Defender" in ids.info(i.name).keywords:
            continue
        if i.iid in pacified:
            notes.append(f"A:{alias(i)} enchanted by a hostile Aura: excluded")
            continue
        if "doesnt_untap" in ids.info(i.name).features and not attacked_grps.get(i.grp):
            # Slumbering Cerberus may have started the turn tapped: "did not attack" is no choice
            notes.append(f"A:{alias(i)} may not have untapped: excluded")
            continue
        eligible.append(i)
    attacked, new = _match(attacked_grps, eligible)
    att_ids = {i.iid for i in attacked}
    lab["attacks"] = {}
    for i in sorted(eligible, key=lambda i: i.iid):
        if i.iid not in att_ids and left_early.get(i.grp):
            notes.append(f"A:{alias(i)} died outside combat this turn: excluded")
            continue
        lab["attacks"][f"A:{alias(i)}"] = i.iid in att_ids
    for g_, k in new.items():
        for j in range(k):
            lab["attacks"][f"new:{ids.name(g_)}" + (f"#{j + 1}" if k > 1 else "")] = True
    dup = [g_ for g_, k in Counter(i.grp for i in eligible).items()
           if k > 1 and 0 < attacked_grps.get(g_, 0) < k]
    if dup:
        notes.append("which copy attacked is a guess for: " + ", ".join(sorted(ids.name(x) for x in dup)))
    lab["attack_notes"] = notes
    # --- the next opponent turn: blocks and instant-speed plays ------------------------------------
    lab.update(_offturn(g, u, g.next_slot(n), ana, ids))
    # --- mulligan ----------------------------------------------------------------------------------
    if n == 1:
        lab["mulligan"] = {"kept_after": int(g.meta.get("num_mulligans") or 0),
                           "hands": [sorted(ids.name(c) for c in h) for h in g.candidate_hands]}
        lab["bottomed"] = sorted(ids.name(c) for c in g.bottomed)
        lab["bottomed_exact"] = g.bottomed_exact
    return lab


def _offturn(g: Game, u: TurnRecord, q: TurnRecord | None, ana, ids: Ids) -> dict:
    """The user's blocks and instant-speed plays in the opponent half-turn q right after its turn u."""
    lab: dict = {}
    lab["blocks"], lab["block_pairing"] = [], "none"
    lab["offturn_instants"], lab["offturn_flash"], lab["offturn_activations"] = [], [], []
    lab["offturn_deduped"] = []
    if q is not None and q.side == "oppo":
        end_u = ana.states[u.seq]
        once = recorded_once(g, q, ids)
        lab["offturn_deduped"] = sorted(f for f in once if f != "oppo_abilities")
        rec = {f: once.get(f, q.L(f)) for f in ("creatures_attacked", "creatures_blocked", "creatures_blocking",
                                                  "oppo_creatures_killed_combat", "user_creatures_killed_combat")}
        # still tapped from the user's turn (attacked, paid {T}, a freezing Aura): can't block; the
        # same set `state_after_user_turn` taps, so the potential blockers are its untapped creatures
        tapped = _user_tapped_after(g, u, end_u.bf["user"], end_u.bf["oppo"], ids)
        att = rec["creatures_attacked"]
        # Brazen Borrower blocks only flyers (a token attacker may fly: keep it then)
        flyers = any(ids.is_token(c) or "Flying" in ids.info_id(c).keywords for c in att)
        cant = _pacified(end_u.bf["oppo"], ids)
        # a land the user animated on its turn (Soulstone Sanctuary) is a land again: it blocks only if
        # animated again, an off-turn activation
        pot = [i for i in end_u.bf["user"] if i.kind == "crea" and i.iid not in tapped and not ids.is_land(i.grp)
               and i.iid not in cant and not _cant_block(i, flyers, ids)]
        # a creature the opponent took for its turn (a Threaten effect) does not block for the user
        for c in [c for z, sign, cat, c in ana.checks[q.seq].cats if z == "user_crea" and sign == "-"
                  and cat == "control_change"]:
            gone = next((i for i in pot if i.grp == c), None)
            if gone is not None:
                pot.remove(gone)
        # creatures that left the battlefield during that turn outside combat (removal, bounce, a
        # Moon): gone before or after blocks is not recorded, so "did not block" is not certain
        left = Counter(q.L("user_creatures_killed_non_combat"))
        left.update(c for z, sign, cat, c in ana.checks[q.seq].cats if z == "user_crea" and sign == "-"
                    and cat not in ("timing_shift", "control_change"))
        if att:
            # a user creature given flying (Fleeting Flight, Celestial Armor ...) may block a flyer
            may_fly = flying_granted(ids, {i.name for i in end_u.bf["user"] + ana.states[q.seq].bf["user"]}
                                     | {ids.name(c) for c in q.L("user_instants_sorceries_cast")})
            lab["block_pairing"], lab["blocks"] = _blocks(rec, pot, end_u.bf["oppo"] + ana.states[q.seq].bf["oppo"],
                                                          ana.states[q.seq].bf["user"], ids, left, may_fly)
        else:
            lab["block_pairing"] = "no_attack"
        hand_q = Counter({ids.name(c): k for c, k in Counter(u.L("eot_user_cards_in_hand")).items()})
        lab["offturn_instants"] = _casts(q, "user", hand_q, Counter(end_u.gy["user"]), ids)
        flash = [c for z, sign, cat, c in ana.checks[q.seq].cats if sign == "+" and z in ("user_crea", "user_nonc")
                 and cat in ("from_hand_without_cast_record", "flash_creature_on_opp_turn_uncast")]
        lab["offturn_flash"] = sorted((_key_entry(ids.name(c), *ids.cast_key(ids.name(c))[:3]) for c in flash),
                                      key=lambda d: d["name"])
        lab["offturn_activations"] = _activations(once.get("user_abilities", q.L("user_abilities")),
                                                  {i.name for i in end_u.bf["user"]}, ids)
    return lab


def opponent_turn_label(g: Game, n: int, ids: Ids | None = None) -> dict:
    """The opponent's half-turn right after user turn n, as a replay script with the opponent active
    (turnreplay.replay_opp_turn, docs/017 §2.2): its land plays, casts (its hand is hidden, so every
    recorded cast is taken from hand, or flashback from its graveyard), activated abilities and
    attacks (spec aliases B:..., new:<name> for a creature that entered that turn), plus the user's
    blocks and instant-speed plays in it (`_offturn`)."""
    ids = ids or Ids.load(g.meta.get("expansion") or "FDN")
    ana = analyze(g, ids)
    u, q = g.user_slot(n), g.next_slot(n)
    if u is None or q is None or q.side != "oppo" or not q.played:
        raise ValueError(f"row {g.row_index} has no opponent turn after user turn {n}")
    end_u = ana.states[u.seq]
    b_start, a_start = end_u.bf["oppo"], end_u.bf["user"]
    lab: dict = {"user_turn": n, "opp_turn": q.n, "global_turn": q.global_turn, "terminal": q.terminal}
    lab["lands"] = sorted((_key_entry(ids.name(c), *ids.play_key(ids.name(c))) for c in q.L("lands_played")),
                          key=lambda d: d["name"])
    once = recorded_once(g, q, ids)
    hand = Counter(ids.name(c) for c in q.L("creatures_cast") + q.L("non_creatures_cast")
                   + q.L("oppo_instants_sorceries_cast"))
    lab["casts"] = _casts(q, "oppo", hand, Counter(end_u.gy["oppo"]), ids)
    board = {i.name for i in b_start} | {ids.name(c) for c in _eot_bf(q, "oppo")}
    lab["activations"] = _activations(once.get("oppo_abilities", q.L("oppo_abilities")), board, ids)
    # the opponent's attacks, as turn_label does the user's
    pacified = _pacified(a_start, ids)
    att_list = once.get("creatures_attacked", q.L("creatures_attacked"))
    attacked_grps = Counter(att_list)
    left_early = Counter(q.L("oppo_creatures_killed_non_combat"))
    eligible, notes = [], []
    for i in b_start:
        if i.kind != "crea" or "Defender" in ids.info(i.name).keywords:
            continue
        if i.iid in pacified:
            notes.append(f"B:{alias(i)} enchanted by a hostile Aura: excluded")
            continue
        if "doesnt_untap" in ids.info(i.name).features and not attacked_grps.get(i.grp):
            notes.append(f"B:{alias(i)} may not have untapped: excluded")
            continue
        eligible.append(i)
    attacked, new = _match(attacked_grps, eligible)
    att_ids = {i.iid for i in attacked}
    lab["attacks"] = {}
    for i in sorted(eligible, key=lambda i: i.iid):
        if i.iid not in att_ids and left_early.get(i.grp):
            notes.append(f"B:{alias(i)} died outside combat this turn: excluded")
            continue
        lab["attacks"][f"B:{alias(i)}"] = i.iid in att_ids
    for g_, k in new.items():
        for j in range(k):
            lab["attacks"][f"new:{ids.name(g_)}" + (f"#{j + 1}" if k > 1 else "")] = True
    dup = [g_ for g_, k in Counter(i.grp for i in eligible).items() if k > 1 and 0 < attacked_grps.get(g_, 0) < k]
    if dup:
        notes.append("which copy attacked is a guess for: " + ", ".join(sorted(ids.name(x) for x in dup)))
    lab["attack_notes"] = notes
    lab.update(_offturn(g, u, q, ana, ids))
    return lab


def after_turn_label(g: Game, n: int, ids: Ids | None = None) -> dict:
    """The labels of `reconstruct.state_after_user_turn(g, n)`: what the user did during the
    opponent's half-turn right after user turn n (blocks, block_pairing, offturn_instants,
    offturn_flash, offturn_activations, offturn_deduped, as in `turn_label`), plus `attacked`, the
    opponent's attackers of that turn by name (the spec at DECLARE_ATTACKERS declares them)."""
    ids = ids or Ids.load(g.meta.get("expansion") or "FDN")
    ana = analyze(g, ids)
    u, q = g.user_slot(n), g.next_slot(n)
    if u is None or q is None:
        raise ValueError(f"row {g.row_index} has no opponent turn after user turn {n}")
    att = recorded_once(g, q, ids).get("creatures_attacked", q.L("creatures_attacked"))
    lab: dict = {"user_turn": n, "opp_turn": q.n, "global_turn": q.global_turn, "terminal": q.terminal,
                 "attacked": sorted(ids.name(c) for c in att)}
    lab.update(_offturn(g, u, q, ana, ids))
    return lab


def _eot_bf(t: TurnRecord, w: str) -> list[int]:
    return t.L(f"eot_{w}_lands_in_play") + t.L(f"eot_{w}_creatures_in_play") + t.L(f"eot_{w}_non_creatures_in_play")


def _pacified(auras: list[Inst], ids: Ids) -> set[int]:
    """Hosts of the Auras among `auras` whose creature can neither attack nor block (Pacifism) or
    stays tapped (Starlight Snare). Witness Protection and Eaten by Piranhas leave a plain 1/1 that
    can do both, and Imprisoned in the Moon makes a land (no longer in the creature list)."""
    return {i.host for i in auras if i.host is not None
            and ("host_cant_attack_block" in ids.info(i.name).features or ids.info(i.name).attach.startswith("aura:freeze"))}


def _cant_block(i: Inst, flyers: bool, ids: Ids) -> bool:
    """A creature that can't block (Vampire Soulcaller, the Rat token that can't block), or blocks
    only flyers (Brazen Borrower) when no attacker flies: the engine does not ask about it."""
    if i.token:
        return "CantBlock" in (i.token_class or "")
    f = ids.info(i.name).features
    return "cant_block" in f or ("blocks_only_flyers" in f and not flyers)


def _blocks(rec: dict, potential: list[Inst], b_insts: list[Inst], a_after: list[Inst], ids: Ids,
            left: Counter | None = None, may_fly: bool = False):
    """rec: the opponent turn's combat lists, each once (`recorded_once`); may_fly: the user had a
    flying grant (`flying_granted`)."""
    blocked = rec["creatures_blocked"]
    blockers = rec["creatures_blocking"]
    status, good = block_assignments(ids, blocked, blockers, rec["oppo_creatures_killed_combat"],
                                     rec["user_creatures_killed_combat"], blockers_may_fly=may_fly)
    if not blockers:
        status = "unique"                  # nobody blocked: every potential blocker answered "none"
    # blocker ids -> instances (flash blockers that entered during q come from the later state)
    seen_b, bl_inst = set(), []
    for grp in blockers:
        pool = [i for i in potential + a_after if i.grp == grp and i.iid not in seen_b]
        inst = pool[0] if pool else None
        if inst is not None:
            seen_b.add(inst.iid)
        bl_inst.append(inst)
    att_key = []
    used_a = set()
    uniq_b = {}
    for i in b_insts:
        uniq_b.setdefault(i.iid, i)
    for grp in blocked:
        pool = sorted([i for i in uniq_b.values() if i.grp == grp and i.iid not in used_a], key=lambda i: (i.entered, i.iid))
        if pool:
            used_a.add(pool[0].iid)
            att_key.append(f"B:{alias(pool[0])}")
        else:
            att_key.append(f"new:{ids.name(grp)}")
    rows = []
    for k, inst in enumerate(bl_inst):
        key = f"A:{alias(inst)}" if inst is not None else f"new:{ids.name(blockers[k])}"
        if good:
            # an inconsistent pairing's one assignment is a legal guess, not the record
            choices = {a[k] for a in good}
            rows.append([key, att_key[good[0][k]], status != "inconsistent" and len(choices) == 1])
        else:
            rows.append([key, att_key[0] if att_key else None, False])
    for i in potential:
        if i.iid not in seen_b:
            # "did not block" is certain unless it left the battlefield outside combat that turn
            rows.append([f"A:{alias(i)}", None, not (left or {}).get(i.grp)])
    return status, sorted(rows, key=lambda r: r[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m draftzero.gameplay.labels", description="Print one turn label.")
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--turn", type=int, required=True)
    ap.add_argument("--after", action="store_true", help="only the opponent turn after it (after_turn_label)")
    ap.add_argument("--path", default=None)
    a = ap.parse_args(argv)
    from draftzero.gameplay.replay import read_games
    g = read_games([a.row], a.path)[a.row]
    print(json.dumps((after_turn_label if a.after else turn_label)(g, a.turn), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
