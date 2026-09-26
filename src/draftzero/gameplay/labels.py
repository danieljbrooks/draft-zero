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
                       turn ("new:<name>" for a creature that entered this turn and attacked)
  blocks               [[blocker, attacker or None, exact]] for every potential blocker in the next
                       opponent turn; `block_pairing` is unique | ambiguous | inconsistent |
                       too_many | no_attack | none (research pairing by printed P/T and deaths)
  offturn_instants[]   instants cast on the following opponent turn (auxiliary target)
  offturn_flash[]      flash permanents inferred on that turn (hand -> battlefield, no cast record)
  mulligan, bottomed   user turn 1 only: {"kept_after": k, "hands": [[names], ...]}, bottomed names

Aliases ("A:GleamingBarrier_4") are the permanent ids of the StateSpec that
`reconstruct.state_at_user_turn(game, n)` builds, so a bridge can apply the labels to that spec.
Labels of a game's last slot (`terminal`) can be incomplete: the game ended during it.

  python -m draftzero.gameplay.labels --row 0 --turn 3
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter

from draftzero.gameplay.ids import Ids
from draftzero.gameplay.reconstruct import Inst, _tapped_by_abilities, alias, analyze
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


def _activations(t: TurnRecord, who: str, board: set, ids: Ids) -> list[dict]:
    out = []
    for aid in t.L(f"{who}_abilities"):
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


# --- blocker -> attacker pairing (port of research semantics2.py) -----------------------------------

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
                      dead_blk: list[int], cap: int = 50000) -> tuple[str, list[tuple[int, ...]]]:
    """(status, consistent assignments): each assignment gives, per blocker index, the index of the
    blocked attacker it blocked. status: unique | ambiguous | inconsistent | too_many | none."""
    nb, nk = len(blocked), len(blockers)
    if nb == 0 or nk == 0:
        return "none", []
    if nk < nb:
        return "inconsistent", []
    if nb == 1:
        return "unique", [tuple([0] * nk)]
    if nb ** nk > cap:
        return "too_many", []
    dA, dB = Counter(dead_att), Counter(dead_blk)
    seen, good = set(), []
    for assign in itertools.product(range(nb), repeat=nk):
        if len(set(assign)) < nb:
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
    if not good:
        return "inconsistent", []
    return ("unique" if len(good) == 1 else "ambiguous"), good


# --- the label ---------------------------------------------------------------------------------------

def turn_label(g: Game, n: int, ids: Ids | None = None) -> dict:
    ids = ids or Ids.load(g.meta.get("expansion") or "FDN")
    ana = analyze(g, ids)
    u = g.user_slot(n)
    if u is None:
        raise ValueError(f"row {g.row_index} has no user turn {n}")
    p = g.prev_slot(n)
    q = g.next_slot(n)
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
    board = {i.name for i in a_start} | {ids.name(c) for c in _eot_bf(u, "user")}
    lab["activations"] = _activations(u, "user", board, ids)
    # --- attacks -----------------------------------------------------------------------------------
    b_start = start.bf["oppo"] if start else []
    pacified = {i.host for i in b_start if i.host is not None
                and ids.info(i.name).attach.startswith(("aura:hostile", "aura:freeze"))}
    eligible, notes = [], []
    left_early = Counter(u.L("user_creatures_killed_non_combat"))
    attacked_grps = Counter(u.L("creatures_attacked"))
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
    attacked, new = _match(Counter(u.L("creatures_attacked")), eligible)
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
           if k > 1 and 0 < Counter(u.L("creatures_attacked")).get(g_, 0) < k]
    if dup:
        notes.append("which copy attacked is a guess for: " + ", ".join(sorted(ids.name(x) for x in dup)))
    lab["attack_notes"] = notes
    # --- the next opponent turn: blocks and instant-speed plays ------------------------------------
    lab["blocks"], lab["block_pairing"] = [], "none"
    lab["offturn_instants"], lab["offturn_flash"], lab["offturn_activations"] = [], [], []
    if q is not None and q.side == "oppo":
        end_u = ana.states[u.seq]
        tapped = _tapped_after(u, end_u.bf["user"], ids)
        tapped |= _tapped_by_abilities(u, "user", end_u.bf["user"], ids, tapped)[0]   # paid {T}: can't block
        tapped |= {i.host for i in end_u.bf["oppo"] if i.host is not None
                   and ids.info(i.name).attach.startswith("aura:freeze")}       # stays tapped: can't block
        pot = [i for i in end_u.bf["user"] if i.kind == "crea" and i.iid not in tapped]
        att = q.L("creatures_attacked")
        if att:
            lab["block_pairing"], lab["blocks"] = _blocks(q, pot, end_u.bf["oppo"] + ana.states[q.seq].bf["oppo"],
                                                          ana.states[q.seq].bf["user"], ids)
        else:
            lab["block_pairing"] = "no_attack"
        hand_q = Counter({ids.name(c): k for c, k in Counter(u.L("eot_user_cards_in_hand")).items()})
        lab["offturn_instants"] = _casts(q, "user", hand_q, Counter(end_u.gy["user"]), ids)
        flash = [c for z, sign, cat, c in ana.checks[q.seq].cats if sign == "+" and z in ("user_crea", "user_nonc")
                 and cat in ("from_hand_without_cast_record", "flash_creature_on_opp_turn_uncast")]
        lab["offturn_flash"] = sorted((_key_entry(ids.name(c), *ids.cast_key(ids.name(c))[:3]) for c in flash),
                                      key=lambda d: d["name"])
        lab["offturn_activations"] = _activations(q, "user", {i.name for i in end_u.bf["user"]}, ids)
    # --- mulligan ----------------------------------------------------------------------------------
    if n == 1:
        lab["mulligan"] = {"kept_after": int(g.meta.get("num_mulligans") or 0),
                           "hands": [sorted(ids.name(c) for c in h) for h in g.candidate_hands]}
        lab["bottomed"] = sorted(ids.name(c) for c in g.bottomed)
        lab["bottomed_exact"] = g.bottomed_exact
    return lab


def _eot_bf(t: TurnRecord, w: str) -> list[int]:
    return t.L(f"eot_{w}_lands_in_play") + t.L(f"eot_{w}_creatures_in_play") + t.L(f"eot_{w}_non_creatures_in_play")


def _tapped_after(u: TurnRecord, insts: list[Inst], ids: Ids) -> set[int]:
    """The user's attackers (no vigilance) stay tapped through the opponent's next turn."""
    out = set()
    for grp, k in Counter(u.L("creatures_attacked")).items():
        cand = sorted([i for i in insts if i.grp == grp], key=lambda i: (i.entered, i.iid))
        if cand and "Vigilance" not in ids.info(cand[0].name).keywords:
            out.update(i.iid for i in cand[:k])
    return out


def _blocks(q: TurnRecord, potential: list[Inst], b_insts: list[Inst], a_after: list[Inst], ids: Ids):
    blocked = q.L("creatures_blocked")
    blockers = q.L("creatures_blocking")
    status, good = block_assignments(ids, blocked, blockers, q.L("oppo_creatures_killed_combat"),
                                     q.L("user_creatures_killed_combat"))
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
            choices = {a[k] for a in good}
            rows.append([key, att_key[good[0][k]], len(choices) == 1])
        else:
            rows.append([key, att_key[0] if att_key else None, False])
    for i in potential:
        if i.iid not in seen_b:
            rows.append([f"A:{alias(i)}", None, True])
    return status, sorted(rows, key=lambda r: r[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m draftzero.gameplay.labels", description="Print one turn label.")
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--turn", type=int, required=True)
    ap.add_argument("--path", default=None)
    a = ap.parse_args(argv)
    from draftzero.gameplay.replay import read_games
    g = read_games([a.row], a.path)[a.row]
    print(json.dumps(turn_label(g, a.turn), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
