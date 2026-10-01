"""
Replay one 17lands user turn through XMage and check that it reaches the recorded end of turn.

This is the fidelity test past the turn start: `reconstruct` builds the state at the start of user
turn n, `labels` says what the user did in it, and the bridge's `replay_turn` op (java/mzbridge,
TurnReplay.java) plays that turn with both seats as puppets:

  A (the user)   plays its recorded lands, spells and activated abilities (17lands has no order:
                 lands first, spells by mana value, before or after combat; a spell that is not
                 legal yet is retried at every later window) and declares exactly its recorded
                 attackers;
  B (opponent)   blocks as recorded (`creatures_blocking`, paired with the blocked attackers by
                 `labels.block_assignments`; every consistent pairing is an alternative) and casts
                 its recorded instants, flash permanents and abilities in a window read off the card
                 (a counterspell in response, a trick after blocks, removal after attackers, the
                 rest in the end step);
  targets        a TargetResolver prefers the candidate whose fate explains the end snapshot (it
                 left the battlefield, the hand, the library ...), else a heuristic; guessed
                 targets are flagged.

B's recorded instant-speed cards go into its (otherwise hidden) hand and the lands it needs are
left untapped (`_add_to_hand`, `_untap_for`). The op tries a bounded set of orderings and windows
(java/mzbridge/README.md, replay_turn) and reports the first one whose end of turn has the
recorded life totals, A's hand (as cleanup began: 17lands' snapshot precedes the discard to hand
size) and both battlefields (name multisets per controller), and in which every recorded play of
A's happened and every recorded attack and block between spec permanents was declared (a
matching end state without them is a coincidence: "undone:A", "attack:A:unrealised",
"block:B:unrealised" ... in diffKeys). On the way it records A's decisions
{type, text, legal, chosen, label_kind} with label_kind exact | imputed_order | guessed_target:
per-decision imitation rows (rows from turns that did not reproduce should not be trained on).

    from draftzero.gameplay.turnreplay import replay_turn
    r = replay_turn(bridge_or_pool, game, n)        # ok (= reproduced), diff, policy, decisions, ...

    python -m draftzero.gameplay.turnreplay --row 4 --turn 3 [--path fixture.csv.gz] [--encode]
    python -m draftzero.gameplay.turnreplay measure --turns 2000 --workers 2
        # stratified sample over the whole FDN file: reproduction rate by tier and turn, time per
        # turn, attempts, failure categories (row/turn refs only)

17lands public data is CC BY 4.0 (https://www.17lands.com/public_datasets).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from draftzero.gameplay.ids import DATA_DIR, Ids
from draftzero.gameplay.labels import block_assignments, turn_label
from draftzero.gameplay.reconstruct import alias, analyze, state_at_user_turn
from draftzero.gameplay.replay import Game, TurnRecord, iter_games, read_games
from draftzero.gameplay.statespec import StateSpec

OUT_DIR = DATA_DIR.parent / "gameplay" / "turnreplay"
MAX_PAIRINGS = 5
# primary failure category: the first diff key of the best attempt in this order
CATEGORY_ORDER = ("bridge_error", "engine_error", "end_not_reached", "game_over", "unscripted:A", "undone:A",
                  "attack:A:unrealised", "attack:A:extra", "block:B:unrealised", "hand:A:extra",
                  "hand:A:missing", "hand:A:short", "bf:A:missing", "bf:A:extra", "bf:B:missing", "bf:B:extra",
                  "life:A", "life:B", "undone:B", "deaths:A", "deaths:B")
TIERS = ("T0", "T1", "T2", "T3")


# --- the request -------------------------------------------------------------------------------------

def _eot_bf(t: TurnRecord, w: str) -> list[int]:
    return t.L(f"eot_{w}_lands_in_play") + t.L(f"eot_{w}_creatures_in_play") + t.L(f"eot_{w}_non_creatures_in_play")


def perm_key(grp: int, ids: Ids) -> str:
    """The name the bridge compares permanents by: a card's name, a token's XMage name without
    " Token" (17lands names tokens by type, e.g. "Cat"; XMage's class may say "Cat Token")."""
    if ids.is_token(grp):
        x = (ids.token_rows.get(grp) or {}).get("xmage_name") or ids.name(grp)
        return x[:-6] if x.endswith(" Token") else x
    return ids.name(grp)


def expected_snapshot(g: Game, n: int, ids: Ids, slot: TurnRecord | None = None) -> dict:
    """The recorded end of user turn n (or of `slot`, e.g. the opponent's turn after it): life, A's
    hand (unknown ids by count), both battlefields (perm_key multisets) and the non-token creatures
    that died (combat / non-combat)."""
    u = slot if slot is not None else g.user_slot(n)
    hand = u.L("eot_user_cards_in_hand")
    known = [ids.name(c) for c in hand if c != -1 and ids.cards17.get(c)]
    return {
        "life": {"A": int(u.num("eot_user_life", 20.0)), "B": int(u.num("eot_oppo_life", 20.0))},
        "hand": {"A": sorted(known), "unknownA": len(hand) - len(known)},
        "battlefield": {s: sorted(perm_key(c, ids) for c in _eot_bf(u, w) if c != -1)
                        for s, w in (("A", "user"), ("B", "oppo"))},
        "deaths": {s: {"combat": sorted(ids.name(c) for c in u.L(f"{w}_creatures_killed_combat")),
                       "noncombat": sorted(ids.name(c) for c in u.L(f"{w}_creatures_killed_non_combat"))}
                   for s, w in (("A", "user"), ("B", "oppo"))},
    }


def _cost_mv(key: str) -> int:
    """Mana value of the cost in an XMage key such as "Flashback {2}{U}"."""
    mv = 0
    for sym in re.findall(r"\{([^}]*)\}", key):
        mv += int(sym) if sym.isdigit() else (0 if sym in ("X", "T", "Q") else 1)
    return mv


def opponent_plays(g: Game, n: int, ids: Ids, b_gy: Counter) -> tuple[list[dict], list[str], list[str]]:
    """B's recorded plays during user turn n: (script items, card names that must be in its hand,
    notes). Instants/sorceries from `oppo_instants_sorceries_cast` (a flashback cast when its
    graveyard holds a copy and the card has flashback), flash permanents inferred from the zone
    changes (no cast record for the non-active player), activated non-mana abilities."""
    u = g.user_slot(n)
    ana = analyze(g, ids)
    items, hand, notes = [], [], []
    gy = Counter(b_gy)
    for c in u.L("oppo_instants_sorceries_cast"):
        name = ids.name(c)
        key, idx, exact, zone = ids.cast_key(name, Counter(), gy)
        mv = ids.mv(c)
        if zone == "graveyard":
            gy[name] -= 1
            mv = _cost_mv(key)                  # the flashback cost, not the printed one
        else:
            key = "Cast " + name
            hand.append(name)
        items.append({"kind": "cast", "key": key, "name": name, "family": "instant_sorcery", "mv": mv})
    flash_cats = {"oppo_creature_appeared_on_user_turn(flash)", "oppo_permanent_appeared_on_user_turn(flash)",
                  "killed_listed_but_never_on_bf(flash/uncast creature died in slot)"}
    for z, sign, cat, c in ana.checks[u.seq].cats:
        if z.startswith("oppo_") and sign == "+" and cat in flash_cats and not ids.is_token(c):
            name = ids.name(c)
            if "Flash" not in ids.info(name).keywords and "flash" not in ids.info(name).features:
                notes.append(f"opponent permanent {name} appeared on the user's turn without flash")
            items.append({"kind": "cast", "key": "Cast " + name, "name": name, "family": "flash", "mv": ids.mv(c)})
            hand.append(name)
    board = {i.name for i in ana.states[g.prev_slot(n).seq].bf["oppo"]} if g.prev_slot(n) else set()
    for aid in u.L("oppo_abilities"):
        a = ids.ability(aid)
        if not a.is_action:
            continue
        key, idx, exact, unique = ids.ability_key(aid, board)
        if key is None:
            notes.append(f"opponent ability {aid} has no XMage key: not scripted")
            continue
        src = next((s for s, k, _, _ in a.keys_by_source if s in board), a.source_cards[0] if a.source_cards else None)
        items.append({"kind": "activation", "key": key, "name": src})
    return items, hand, notes


def opponent_blocks(g: Game, n: int, ids: Ids, attacks: dict) -> tuple[list, str, list[str]]:
    """B's blocks in user turn n as alternatives of complete pairings [[blocker ref, attacker ref]],
    refs being spec aliases ("B:Bear_4", "A:Drake_6") or "new:<name>" (entered this turn). The
    pairing status comes from labels.block_assignments (printed P/T and the deaths)."""
    u, p = g.user_slot(n), g.prev_slot(n)
    blocked, blockers = u.L("creatures_blocked"), u.L("creatures_blocking")
    if not blockers:
        return [], "none", []
    notes = []
    status, good = block_assignments(ids, blocked, blockers, u.L("user_creatures_killed_combat"),
                                     u.L("oppo_creatures_killed_combat"))
    ana = analyze(g, ids)
    start = ana.states[p.seq] if p is not None else None
    b_pool = sorted([i for i in (start.bf["oppo"] if start else []) if i.kind == "crea"], key=lambda i: (i.entered, i.iid))
    a_pool = sorted([i for i in (start.bf["user"] if start else []) if i.kind == "crea"], key=lambda i: (i.entered, i.iid))
    used, b_ref = set(), []
    for grp in blockers:
        inst = next((i for i in b_pool if i.grp == grp and i.iid not in used), None)
        if inst is not None:
            used.add(inst.iid)
            b_ref.append(f"B:{alias(inst)}")
        else:
            b_ref.append(f"new:{perm_key(grp, ids)}")
    attacked = {k for k, v in attacks.items() if v}
    used, a_ref = set(), []
    for grp in blocked:
        inst = next((i for i in a_pool if i.grp == grp and i.iid not in used and f"A:{alias(i)}" in attacked), None)
        if inst is None:
            inst = next((i for i in a_pool if i.grp == grp and i.iid not in used), None)
        if inst is not None:
            used.add(inst.iid)
            a_ref.append(f"A:{alias(inst)}")
        else:
            a_ref.append(f"new:{perm_key(grp, ids)}")
    if not good:
        # inconsistent with printed P/T (pumps, counters) or too many to enumerate: spread the
        # blockers over the blocked attackers so every blocked attacker is blocked
        notes.append(f"block pairing {status}: blockers spread over the blocked attackers")
        good = [tuple(k % max(1, len(blocked)) for k in range(len(blockers)))] if blocked else []
    alts = []
    for assign in good[:MAX_PAIRINGS]:
        alts.append([[b_ref[k], a_ref[assign[k]] if blocked else None] for k in range(len(blockers))])
    return alts, status, notes


def _add_to_hand(spec: StateSpec, names: list[str], flags: list[str]) -> None:
    """Put B's recorded instant-speed cards into its (otherwise hidden) hand, out of handUnknown, and
    make sure its decklist has enough copies (the placeholder deck only has what was seen so far)."""
    B = spec.players["B"]
    if not names:
        return
    B.hand = sorted(B.hand + names)
    if B.handUnknown >= len(names):
        B.handUnknown -= len(names)
    else:
        flags.append("opp_hand_count_short")
        B.handUnknown = 0
    owned = [x.name for s, P in spec.players.items() for x in P.battlefield
             if x.name and not (x.token or x.tokenClass) and (x.owner or s) == "B" for _ in range(x.count)]
    need = Counter(B.hand + B.graveyard + B.exile + B.libraryTop + owned)
    have = Counter(B.decklist)
    extra = need - have
    if extra:
        B.decklist = sorted(B.decklist + list(extra.elements()))
        flags.append("opp_decklist_extended_for_script")


def _untap_for(spec: StateSpec, items: list[dict], ids: Ids, flags: list[str], seat: str = "B") -> None:
    """B pays for its recorded instant-speed plays (spells and activated abilities) with the lands it
    left untapped at the end of its turn. Which lands it tapped there is inferred from mana spent
    (reconstruct.choose_tapped_lands), so the inference can leave the wrong colours or too few
    lands open: first swap a tapped land of
    a colour B's plays need for an untapped one they do not need (the count, i.e. the mana spent,
    stays), then untap more lands if the count is short."""
    from draftzero.gameplay.statespec import Perm
    B = spec.players[seat]
    pips = Counter()
    need_total = 0
    for it in items:
        if it["kind"] == "cast":
            cost = it["key"].split(" ", 1)[1] if it["key"].startswith("Flashback ") else ids.info(it["name"]).mana_cost
            need_total += it.get("mv", 0)
        else:
            # an activated ability pays the mana symbols before its colon ("{2}{W}: Look at ...";
            # "{1}{U}, {T}: Draw ..."); {T}, {this} and non-mana costs need no land
            cost = it["key"].split(":", 1)[0] if ":" in it["key"] else ""
            need_total += sum(int(x) if x.isdigit() else 1 for x in re.findall(r"\{([0-9]+|[WUBRGC])\}", cost))
        pips.update(x for x in re.findall(r"\{([WUBRG])\}", cost or ""))
    if not need_total:
        return

    def lands():
        return [p for p in B.battlefield if p.name and not (p.token or p.tokenClass) and ids.info(p.name).land_mana]

    def split(p: Perm, tapped: bool) -> Perm:
        """One copy of a land group with the other tapped state (a new single entry)."""
        if p.count == 1 and not p.attachTo:
            p.tapped = tapped
            return p
        p.count -= 1
        q = Perm(name=p.name, count=1, tapped=tapped)
        B.battlefield.append(q)
        return q
    changed = False
    for color, k in pips.items():
        have = sum(p.count for p in lands() if not p.tapped and color in ids.info(p.name).land_mana)
        while have < k:
            src = next((p for p in lands() if p.tapped and color in ids.info(p.name).land_mana and not p.attachTo), None)
            if src is None:
                break
            split(src, False)
            have += 1
            changed = True
            # keep the tapped count: tap an open land whose colours B's plays do not need
            spare = next((p for p in lands() if not p.tapped and not p.attachTo
                          and not set(ids.info(p.name).land_mana) & set(pips)), None)
            if spare is not None:
                split(spare, True)
    short = need_total - sum(p.count for p in lands() if not p.tapped)
    for p in lands():
        if short <= 0:
            break
        if p.tapped and not p.attachTo:
            split(p, False)
            short -= 1
            changed = True
    if changed:
        flags.append("opp_lands_untapped_for_script" if seat == "B" else "user_lands_untapped_for_script")


def replay_request(g: Game, n: int, ids: Ids | None = None, partner: Game | None = None) -> tuple[dict, dict, dict]:
    """(spec dict, request options, meta) for replaying user turn n of g. The spec is the
    eot_rollover state before the turn with its later draws on A's library (future_draws);
    options carry the script and the expected end of turn; meta has the ref, tier and notes."""
    ids = ids or Ids.load(g.meta.get("expansion") or "FDN")
    u = g.user_slot(n)
    if u is None:
        raise ValueError(f"row {g.row_index} has no user turn {n}")
    if partner is not None:
        from draftzero.gameplay.reconstruct import exact_spec
        spec = exact_spec(g, partner, n, "eot_rollover", ids=ids, future_draws=True)
    else:
        spec = state_at_user_turn(g, n, "eot_rollover", ids=ids, future_draws=True)
    lab = turn_label(g, n, ids)
    flags: list[str] = []
    notes: list[str] = list(lab.get("attack_notes", []))
    unscripted = 0
    casts = []
    for c in lab["casts"]:
        casts.append({"kind": "cast", "key": c["key"], "name": c["name"], "family": c["family"]})
    acts = []
    for a in lab["activations"]:
        if not a["key"]:
            notes.append(f"user ability {a['id']} has no XMage key: not scripted")
            flags.append("activation_unkeyed")
            unscripted += 1          # the op then never labels a Pass exact, nor reproduces the turn
            continue
        acts.append({"kind": "activation", "key": a["key"], "name": a["source"]})
    opp_items, opp_hand, opp_notes = opponent_plays(g, n, ids, Counter(spec.players["B"].graveyard))
    notes += opp_notes
    _add_to_hand(spec, opp_hand, flags)
    _untap_for(spec, opp_items, ids, flags)
    blocks, pairing, bnotes = opponent_blocks(g, n, ids, lab["attacks"])
    notes += bnotes
    guess = [x.split(": ", 1)[1] for x in lab.get("attack_notes", []) if x.startswith("which copy attacked is a guess for: ")]
    script = {
        "seat": "A", "turn": u.global_turn,
        "lands": [{"kind": "land", "key": x["key"], "name": x["name"]} for x in lab["lands"]],
        "casts": casts, "activations": acts, "opp": opp_items,
        "attacks": lab["attacks"], "attackGuess": [n_ for x in guess for n_ in x.split(", ")],
        "blocks": blocks, "blockPairing": pairing, "unscripted": unscripted,
    }
    d = spec.to_dict()
    tier = spec.provenance.tier if spec.provenance else None
    meta = {"row": g.row_index, "turn": n, "global_turn": u.global_turn, "tier": tier,
            "spec_flags": list(spec.provenance.flags) if spec.provenance else [], "flags": flags, "notes": notes,
            "n_casts": len(casts), "n_opp": len(opp_items), "n_blocks": len(u.L("creatures_blocking")),
            "block_pairing": pairing, "on_play": g.on_play}
    return d, {"script": script, "expected": expected_snapshot(g, n, ids)}, meta


def replay_turn(bridge, g: Game, n: int, ids: Ids | None = None, partner: Game | None = None, *, seed: int = 0,
                encode: bool = False, perfect_info: bool = False, max_attempts: int = 12, timeout: float | None = None,
                **options) -> dict:
    """Run the replay_turn op (a Bridge or a BridgePool) on user turn n of g. Returns the op's
    response plus "meta" (ref, tier, request notes), with "ok" set to the verdict "reproduced"; a
    request the bridge rejects comes back with ok False and "bridge_error" instead of raising.
    partner: the mirrored row of the same game (pairs.py), for B's real deck and hand."""
    from draftzero.gameplay.bridge import BridgeError
    spec, opt, meta = replay_request(g, n, ids, partner)
    try:
        kw = {"timeout": timeout} if timeout and not hasattr(bridge, "workers") else {}
        r = bridge.request("replay_turn", spec, seed=seed, encode=encode or None, perfectInfo=perfect_info or None,
                           maxAttempts=max_attempts, **opt, **options, **kw)
    except BridgeError as e:
        err = e.error or {}
        r = {"reproduced": False, "bridge_error": f"{err.get('type')}: {err.get('message')}"[:500],
             "diffKeys": ["bridge_error"]}
    # the worker's envelope "ok" only says the request ran (a failed one raised above): from here on
    # "ok" is the verdict, the end of turn matched the snapshot
    r["ok"] = bool(r.get("reproduced"))
    r["meta"] = meta
    return r


def replay_request_opp(g: Game, n: int, ids: Ids | None = None) -> tuple[dict, dict, dict]:
    """(spec dict, request options, meta) for replaying the opponent's turn right after user turn n,
    the opponent scripted and active, the user's decisions recorded (recordSeat A; docs/017 §2.2).
    The spec is the end of user turn n (reconstruct.state_after_user_turn, eot_rollover). The
    opponent's hand is hidden, so its recorded plays are put in it (hindsight, flagged); the user's
    off-turn instants, flash and abilities are its script items with their windows read off the
    card, and its blocks are the recorded pairing."""
    from draftzero.gameplay.labels import opponent_turn_label
    from draftzero.gameplay.reconstruct import state_after_user_turn
    ids = ids or Ids.load(g.meta.get("expansion") or "FDN")
    spec = state_after_user_turn(g, n, "eot_rollover", ids=ids)
    lab = opponent_turn_label(g, n, ids)
    q = g.next_slot(n)
    flags: list[str] = ["opp_turn_hindsight_hand"]
    notes: list[str] = list(lab.get("attack_notes", []))
    unscripted = 0
    b_hand = [x["name"] for x in lab["lands"]] + [c["name"] for c in lab["casts"] if c.get("zone") != "graveyard"]
    _add_to_hand(spec, b_hand, flags)
    casts = [{"kind": "cast", "key": c["key"], "name": c["name"], "family": c["family"]} for c in lab["casts"] if c["key"]]
    acts = []
    for a in lab["activations"]:
        if not a["key"]:
            notes.append(f"opponent ability {a['id']} has no XMage key: not scripted")
            unscripted += 1
            continue
        acts.append({"kind": "activation", "key": a["key"], "name": a["source"]})
    mine = []          # the user's plays in the opponent's turn: the other seat's items
    for x in lab["offturn_instants"]:
        if x.get("key"):
            mine.append({"kind": "cast", "key": x["key"], "name": x["name"], "family": "instant_sorcery",
                         "mv": _cost_mv(x["key"] if x["key"].startswith("Flashback ") else ids.info(x["name"]).mana_cost)})
    for x in lab["offturn_flash"]:
        if x.get("key"):
            mine.append({"kind": "cast", "key": x["key"], "name": x["name"], "family": "flash",
                         "mv": _cost_mv(ids.info(x["name"]).mana_cost)})
    for a in lab["offturn_activations"]:
        if a.get("key"):
            mine.append({"kind": "activation", "key": a["key"], "name": a["source"]})
        else:
            notes.append(f"user ability {a['id']} has no XMage key: not scripted")
            flags.append("user_activation_unkeyed")
    _untap_for(spec, mine, ids, flags, seat="A")
    pairs = [[b, a] for b, a, _ in lab["blocks"] if a is not None]
    guess = [x.split(": ", 1)[1] for x in notes if x.startswith("which copy attacked is a guess for: ")]
    script = {
        "seat": "B", "turn": lab["global_turn"],
        "lands": [{"kind": "land", "key": x["key"], "name": x["name"]} for x in lab["lands"] if x.get("key")],
        "casts": casts, "activations": acts, "opp": mine,
        "attacks": lab["attacks"], "attackGuess": [n_ for x in guess for n_ in x.split(", ")],
        "blocks": [pairs] if pairs else [], "blockPairing": lab["block_pairing"], "unscripted": unscripted,
    }
    tier = spec.provenance.tier if spec.provenance else None
    meta = {"row": g.row_index, "turn": n, "opp_turn": True, "global_turn": lab["global_turn"], "tier": tier,
            "spec_flags": list(spec.provenance.flags) if spec.provenance else [], "flags": flags, "notes": notes,
            "n_casts": len(casts), "n_user": len(mine), "block_pairing": lab["block_pairing"], "on_play": g.on_play}
    return spec.to_dict(), {"script": script, "expected": expected_snapshot(g, n, ids, slot=q), "recordSeat": "A"}, meta


def replay_opp_turn(bridge, g: Game, n: int, ids: Ids | None = None, *, seed: int = 0, encode: bool = False,
                    perfect_info: bool = False, max_attempts: int = 12, timeout: float | None = None, **options) -> dict:
    """replay_turn for the opponent's turn right after user turn n, recording the user's decisions
    in it (replay_request_opp). Same response as replay_turn."""
    from draftzero.gameplay.bridge import BridgeError
    spec, opt, meta = replay_request_opp(g, n, ids)
    try:
        kw = {"timeout": timeout} if timeout and not hasattr(bridge, "workers") else {}
        r = bridge.request("replay_turn", spec, seed=seed, encode=encode or None, perfectInfo=perfect_info or None,
                           maxAttempts=max_attempts, **opt, **options, **kw)
    except BridgeError as e:
        err = e.error or {}
        r = {"reproduced": False, "bridge_error": f"{err.get('type')}: {err.get('message')}"[:500],
             "diffKeys": ["bridge_error"]}
    r["ok"] = bool(r.get("reproduced"))
    r["meta"] = meta
    return r


# --- measurement ---------------------------------------------------------------------------------------

def category(r: dict) -> str:
    keys = r.get("diffKeys") or []
    if r.get("ok"):
        return "ok"
    for c in CATEGORY_ORDER:
        if c in keys:
            return c
    return keys[0] if keys else "unknown"


def sample_turns(n_turns: int, path=None, seed: int = 1, per_tier: bool = True, every: int | None = None,
                 total_rows: int = 791_159, ids: Ids | None = None, log=None) -> tuple[list, Counter]:
    """A sample of decision turns spread over the whole file. Pass 1 visits every `every`-th game
    (about 3 x n_turns games, from the first row to the last), draws one decision turn per game
    (seeded) and grades its turn-start spec; then n_turns/4 turns per tier T0..T3 (per_tier; else
    n_turns overall) are taken evenly spaced by row, and pass 2 reads those games again.
    Returns ([(game, n, tier)], tiers of every visited turn: the natural weights)."""
    ids = ids or Ids.load()
    rng = random.Random(seed)
    every = every or max(1, total_rows // (n_turns * 3))
    refs: dict[str, list[tuple[int, int]]] = defaultdict(list)
    seen = Counter()
    t0 = time.time()
    for g in iter_games(path, every=every, start=seed % every):
        turns = g.decision_turns()
        if not turns:
            continue
        n = rng.choice(turns)
        try:
            tier = state_at_user_turn(g, n, "eot_rollover", ids=ids).provenance.tier
        except ValueError:
            continue
        seen[tier] += 1
        refs[tier if per_tier else "all"].append((g.row_index, n))

    def spread(xs: list, k: int) -> list:
        if len(xs) <= k:
            return list(xs)
        return [xs[int(i * len(xs) / k)] for i in range(k)]
    if per_tier:
        chosen = {t: spread(refs[t], (n_turns + 3) // 4) for t in TIERS}
    else:
        chosen = {"all": spread(refs["all"], n_turns)}
    want = {row: (n, t) for t, xs in chosen.items() for row, n in xs}
    t1 = time.time()
    games = read_games(sorted(want), path)
    out = []
    for row in sorted(want):
        n, t = want[row]
        g = games[row]
        tier = t if per_tier else state_at_user_turn(g, n, "eot_rollover", ids=ids).provenance.tier
        out.append((g, n, tier))
    if log:
        log(f"sampled {len(out)} turns from {sum(seen.values())} games (every {every}th row, pass 1 {t1 - t0:.0f} s, "
            f"pass 2 {time.time() - t1:.0f} s); tiers seen {dict(sorted(seen.items()))}")
    return out, seen


def summarize(rows: list[dict], tiers_seen: Counter | None = None) -> dict:
    """Reproduction rates overall / by tier / by turn, time per turn, attempts, failure categories."""
    def rate(xs):
        return {"n": len(xs), "ok": sum(1 for x in xs if x.get("ok")),
                "rate": round(sum(1 for x in xs if x.get("ok")) / len(xs), 4) if xs else None}
    by_tier = defaultdict(list)
    by_turn = defaultdict(list)
    for r in rows:
        by_tier[r["meta"]["tier"]].append(r)
        t = r["meta"]["turn"]
        by_turn[str(t) if t < 10 else "10+"].append(r)
    out = {"overall": rate(rows), "by_tier": {t: rate(by_tier[t]) for t in TIERS if by_tier[t]},
           "by_turn": {k: rate(by_turn[k]) for k in sorted(by_turn, key=lambda k: (len(k), k))}}
    if tiers_seen:
        tot = sum(tiers_seen.values())
        w = {t: tiers_seen[t] / tot for t in TIERS if tiers_seen[t]}
        est = sum(w[t] * out["by_tier"][t]["rate"] for t in w if t in out["by_tier"])
        out["overall_natural_weights"] = {"rate": round(est, 4), "weights": {t: round(x, 4) for t, x in w.items()}}
    times = sorted((r.get("timing_ms") or {}).get("total", 0) for r in rows if r.get("timing_ms"))
    if times:
        out["time_ms"] = {"median": times[len(times) // 2], "p90": times[int(len(times) * 0.9)], "max": times[-1],
                          "mean": round(sum(times) / len(times), 1)}
    ok_times = sorted((r.get("timing_ms") or {}).get("total", 0) for r in rows if r.get("ok") and r.get("timing_ms"))
    if ok_times:
        out["time_ms_ok"] = {"median": ok_times[len(ok_times) // 2], "p90": ok_times[int(len(ok_times) * 0.9)]}
    out["attempts_needed"] = dict(sorted(Counter(r.get("attempt") for r in rows if r.get("ok")).items()))
    pol = Counter(json.dumps(r.get("policy"), sort_keys=True) for r in rows if r.get("ok"))
    out["winning_policies"] = [{"policy": json.loads(k), "n": v} for k, v in pol.most_common(8)]
    fails = [r for r in rows if not r.get("ok")]
    cats = Counter(category(r) for r in fails)
    ex = defaultdict(list)
    for r in fails:
        c = category(r)
        if len(ex[c]) < 5:
            ex[c].append(f"row={r['meta']['row']} turn={r['meta']['turn']} ({r['meta']['tier']})")
    out["fail_categories"] = [{"category": c, "n": k, "share_of_failures": round(k / len(fails), 3), "examples": ex[c]}
                              for c, k in cats.most_common()]
    out["diff_keys_in_failures"] = dict(Counter(k for r in fails for k in r.get("diffKeys") or []).most_common())
    miss, extra = Counter(), Counter()
    for r in fails:
        d = r.get("diff") or {}
        for s, v in (d.get("battlefield") or {}).items():
            miss.update(f"{s}:{x}" for x in v.get("missing", []))
            extra.update(f"{s}:{x}" for x in v.get("extra", []))
        h = d.get("hand") or {}
        miss.update(f"hand:{x}" for x in h.get("missing", []))
        extra.update(f"hand:{x}" for x in h.get("extra", []))
    out["top_missing"] = miss.most_common(20)
    out["top_extra"] = extra.most_common(20)
    fl_ok = Counter(f for r in rows if r.get("ok") for f in r.get("flags") or [])
    fl_bad = Counter(f for r in fails for f in r.get("flags") or [])
    out["flags"] = {f: {"ok": fl_ok[f], "failed": fl_bad[f]} for f in sorted(set(fl_ok) | set(fl_bad), key=lambda f: -(fl_ok[f] + fl_bad[f]))}
    undone = Counter(u for r in fails for u in (r.get("diff") or {}).get("undone", []))
    out["top_undone"] = undone.most_common(20)
    errs = Counter((r.get("error") or r.get("bridge_error") or "")[:120] for r in rows if r.get("error") or r.get("bridge_error"))
    out["errors"] = errs.most_common(10)
    kinds = Counter()
    for r in rows:
        if r.get("ok"):
            kinds.update(r.get("decisionKinds") or Counter(f"{d['type']}:{d['label_kind']}" for d in r.get("decisions") or []))
    out["decisions_in_ok_turns"] = dict(sorted(kinds.items()))
    out["decisions_per_ok_turn"] = round(sum(kinds.values()) / max(1, out["overall"]["ok"]), 2)
    return out


def measure(n_turns: int = 2000, workers: int = 2, path=None, seed: int = 1, heap: str = "2500m",
            runtime_root: Path | None = None, out_dir: Path | None = None, max_attempts: int = 12, log=print,
            sample: tuple | None = None, bridge_kw: dict | None = None) -> dict:
    """Sample (or take `sample` = sample_turns(...)'s result), replay through a BridgePool of
    `workers` JVMs, write rows + summary under data/gameplay/turnreplay/ (row/turn refs only)."""
    from concurrent.futures import ThreadPoolExecutor
    from draftzero.gameplay.bridge import BridgePool
    ids = Ids.load()
    sample, seen = sample or sample_turns(n_turns, path, seed, ids=ids, log=log)
    out_dir = Path(out_dir or OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"rows_seed{seed}_n{len(sample)}.jsonl"
    kw = dict(bridge_kw or {})
    if runtime_root:
        kw["runtime_root"] = runtime_root
    t0 = time.time()
    rows = []
    with BridgePool(workers, prefix="turnreplay", heap=heap, **kw) as pool, open(rows_path, "w") as f:
        def one(item):
            g, n, tier = item
            try:
                r = replay_turn(pool, g, n, ids, seed=seed, max_attempts=max_attempts)
            except Exception as e:     # a request that cannot even be built: counted, not fatal
                r = {"ok": False, "bridge_error": f"{type(e).__name__}: {e}"[:500], "diffKeys": ["bridge_error"],
                     "meta": {"row": g.row_index, "turn": n, "tier": tier}}
            # the rows file keeps a per-turn count of the decision rows, not the rows themselves
            ds = r.pop("decisions", None) or []
            r["decisionKinds"] = dict(Counter(f"{d['type']}:{d['label_kind']}" for d in ds))
            return r
        with ThreadPoolExecutor(workers) as ex:
            for i, r in enumerate(ex.map(one, sample)):
                rows.append(r)
                f.write(json.dumps(r) + "\n")
                if log and (i + 1) % 200 == 0:
                    ok = sum(1 for x in rows if x.get("ok"))
                    log(f"  {i + 1}/{len(sample)} turns, ok {ok / len(rows):.3f}, {time.time() - t0:.0f} s")
    wall = time.time() - t0
    s = summarize(rows, seen)
    s["wall_s"] = round(wall, 1)
    s["workers"] = workers
    s["turns_per_s"] = round(len(rows) / wall, 2)
    s["rows_file"] = str(rows_path)
    (out_dir / f"summary_seed{seed}_n{len(sample)}.json").write_text(json.dumps(s, indent=1))
    return s


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "measure":
        ap = argparse.ArgumentParser(prog="python -m draftzero.gameplay.turnreplay measure")
        ap.add_argument("--turns", type=int, default=2000)
        ap.add_argument("--workers", type=int, default=2)
        ap.add_argument("--seed", type=int, default=1)
        ap.add_argument("--path", default=None)
        ap.add_argument("--heap", default="2500m")
        ap.add_argument("--max-attempts", type=int, default=12)
        ap.add_argument("--runtime-root", default=None)
        ap.add_argument("--out-dir", default=None)
        a = ap.parse_args(argv[1:])
        s = measure(a.turns, a.workers, a.path, a.seed, a.heap, Path(a.runtime_root) if a.runtime_root else None,
                    Path(a.out_dir) if a.out_dir else None, a.max_attempts)
        print(json.dumps({k: s[k] for k in ("overall", "overall_natural_weights", "by_tier", "time_ms", "wall_s")
                          if k in s}, indent=1))
        return 0
    ap = argparse.ArgumentParser(prog="python -m draftzero.gameplay.turnreplay",
                                 description="Replay one 17lands user turn through XMage (or: measure ...).")
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--turn", type=int, required=True)
    ap.add_argument("--path", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--request", action="store_true", help="print the request only (no bridge)")
    ap.add_argument("--name", default="turnreplay-cli")
    a = ap.parse_args(argv)
    g = read_games([a.row], a.path)[a.row]
    if a.request:
        spec, opt, meta = replay_request(g, a.turn)
        print(json.dumps({"meta": meta, "options": opt, "spec": spec}, indent=1))
        return 0
    from draftzero.gameplay.bridge import Bridge
    with Bridge(a.name, heap="2500m") as b:
        r = replay_turn(b, g, a.turn, seed=a.seed, encode=a.encode)
    if not a.encode:
        for d in r.get("decisions", []):
            d.pop("features", None)
    print(json.dumps(r, indent=1))
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
