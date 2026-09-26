"""17lands replay -> ids, games, StateSpecs and labels (draftzero.gameplay WP1).

The golden tests run on tests/fixtures/gameplay/fdn_premier_rows.csv.gz: FDN Premier Draft rows
0, 1, 4, 8, 26, 42, 76, 130 and 198 of the 17lands public replay file (CC BY 4.0, see the fixture
README). Rows 0 and 4 are the ones the research phase inspected by hand. They need only committed
files. The regression test streams a seeded 5,000-game sample of the real file and skips without
it; the engine test builds fixture specs in XMage through java/mzbridge and skips without Java or
the XMage build.
"""
from collections import Counter
from pathlib import Path

import pytest

from draftzero.gameplay import labels as lb
from draftzero.gameplay import reconstruct as rc
from draftzero.gameplay import replay
from draftzero.gameplay.ids import Ids
from draftzero.gameplay.statespec import TIERS, StateSpec

FIXTURE = Path(__file__).parent / "fixtures" / "gameplay" / "fdn_premier_rows.csv.gz"
REPLAY = replay.replay_path("FDN", "PremierDraft")


@pytest.fixture(scope="module")
def ids():
    return Ids.load()


@pytest.fixture(scope="module")
def games():
    return {g.row_index: g for g in replay.iter_games(FIXTURE)}


def names(ids, grps):
    return sorted(ids.name(c) for c in grps)


def bf(spec, seat):
    return spec.players[seat].battlefield


def perm(spec, seat, name=None, token=None):
    hits = [p for p in bf(spec, seat) if (name and p.name == name) or (token and p.tokenClass == token)]
    assert hits, f"{name or token} not on {seat}'s battlefield"
    return hits


# --- ids -----------------------------------------------------------------------------------------

def test_tokens_map_by_id_not_name(ids):
    assert ids.card(94156).token_class == "CatToken3"      # 1/1 Cat
    assert ids.card(94157).token_class == "CatToken"       # 2/2 Cat, same name
    assert ids.card(94171).token_class == "DragonToken"    # 4/4
    assert ids.card(94172).token_class == "DragonToken2"   # 5/5
    assert ids.token_rows[94156]["power"] == "1" and ids.token_rows[94157]["power"] == "2"
    observed = [g for g, t in ids.token_rows.items() if int(t["games"]) > 0]
    assert len(observed) == 23 and all(ids.token_rows[g]["token_class"] for g in observed)


def test_cosmetic_basics_classified_by_type_line(ids):
    for gid in (95196, 66499):                              # FDN Swamp, XLN Plains
        c = ids.card(gid)
        assert c.kind == "basic" and c.xmage_set == "FDN" and c.play_key.startswith("Play ") and c.vocab_exact


def test_card_keys_and_vocab(ids):
    kaito = ids.card(93757)
    assert (kaito.kind, kaito.cast_key, kaito.vocab_idx, kaito.vocab_exact) == (
        "card", "Cast Kaito, Cunning Infiltrator", 279, True)
    from draftzero.gameplay.ids import check
    assert check(ids) == []                                 # every FDN booster card maps and keys exactly


def test_abilities_sources_and_keys(ids):
    # one Arena id, two vocab slots: the Equip key differs in reminder text between equipment
    eq = ids.ability(1268)
    assert eq.category == "activated" and eq.confidence == "multi"
    assert ids.ability_key(1268, {"Goldvein Pick"})[:3] == (
        "Equip {1} <i>({1}: Attach to target creature you control. Equip only as a sorcery.)</i>", 507, True)
    assert ids.ability_key(1268, {"Swiftfoot Boots"})[:3] == ("Equip {1}", 506, True)
    assert ids.ability_key(1268, set())[3] is False                     # board does not settle it
    # one of the 26 ids 17lands' abilities.csv lacks: source card and XMage key only
    ghoul = ids.ability(175817)
    assert ghoul.source_cards == ("Hungry Ghoul",) and ghoul.method == "cooccurrence"
    assert (ghoul.xmage_key, ghoul.vocab_idx, ghoul.self_p1p1) == (
        "{1}, Sacrifice another creature: Put a +1/+1 counter on {this}.", 569, 1)
    assert ids.ability(175795).loyalty == 1 and ids.ability(175796).loyalty == -2    # Kaito +1 / -2
    assert ids.ability(90050).category == "triggered" and ids.ability(90050).xmage_key is None
    assert ids.ability(1003).category == "mana" and not ids.ability(1003).is_action


def test_flashback_cast_key(ids):
    from draftzero.gameplay.ids import is_flashback_cast
    key, idx, exact, zone = ids.cast_key("Think Twice", Counter(), Counter({"Think Twice": 1}))
    assert (key, idx, exact, zone) == ("Flashback {2}{U}", 515, True, "graveyard")
    assert ids.cast_key("Think Twice", Counter({"Think Twice": 1}), Counter({"Think Twice": 1}))[0] == "Cast Think Twice"
    assert ids.cast_key("Stab", Counter(), Counter({"Stab": 1}))[3] == "unknown"   # no flashback
    assert is_flashback_cast("Think Twice", Counter(), Counter({"Think Twice": 1}), ids)


def test_card_facts_for_tapped_state_and_counters(ids):
    from draftzero.gameplay.ids import cost_taps_source
    # the committed table must be what build_tables writes: it once lagged the code (Starlight
    # Snare was "hostile", so the freeze logic never ran)
    assert ids.info("Starlight Snare").attach == "aura:freeze:creature"
    assert "etb_tapped" in ids.info("Thornwood Falls").features and "etb_tapped" in ids.info("Azorius Guildgate").features
    assert "etb_tapped" not in ids.info("Forest").features and "etb_tapped" not in ids.info("Evolving Wilds").features
    assert "doesnt_untap" in ids.info("Slumbering Cerberus").features
    assert "upkeep_draw" in ids.info("Phyrexian Arena").features and "opp_draw_trigger" in ids.info("Scrawling Crawler").features
    assert (ids.info("Mossborn Hydra").etb_counters, ids.info("Mossborn Hydra").counter_mode) == ("1", "self_tracked")
    assert ids.info("Wildwood Scourge").etb_counters == "X" and ids.info("Heroes' Bane").etb_counters == "4"
    assert ids.info("Goblin Boarders").etb_counters == "" and ids.info("Gnarlid Colony").etb_counters == ""  # conditional
    assert ids.ability(175873).self_double and not ids.ability(92970).self_double
    # {T} in the cost: Strix Lookout and Rune-Sealed Wall tap; Equip and loyalty abilities do not
    assert cost_taps_source(ids.ability(8760)) and cost_taps_source(ids.ability(141700))
    assert not cost_taps_source(ids.ability(1268)) and not cost_taps_source(ids.ability(175795))


def test_choose_tapped_lands(ids):
    ch = rc.choose_tapped_lands
    assert ch(["Island", "Mountain"], 0, [], ids) == (Counter(), False)
    assert ch(["Island", "Mountain"], 2, [], ids) == (Counter({"Island": 1, "Mountain": 1}), False)
    # the red pip can only come from a Mountain; a blue spell leaves the Plains untapped
    assert ch(["Island", "Mountain", "Mountain"], 1, ["Burst Lightning"], ids) == (Counter({"Mountain": 1}), False)
    assert ch(["Plains", "Island", "Island"], 1, ["Think Twice"], ids) == (Counter({"Island": 1}), False)
    # a tapped land played this turn is tapped on top of the lands that paid (row 26, turn 4)
    tapped, amb = ch(["Thornwood Falls", "Mountain", "Forest", "Island", "Forest"], 2, ["Firebrand Archer"], ids,
                     entered_tapped=["Thornwood Falls"])
    assert tapped["Thornwood Falls"] == 1 and tapped["Mountain"] == 1 and sum(tapped.values()) == 3 and amb
    assert ch(["Scoured Barrens", "Plains"], 0, [], ids, entered_tapped=["Scoured Barrens"]) == (
        Counter({"Scoured Barrens": 1}), False)


def test_etb_x_counters_estimate(ids):
    # Wildwood Scourge ({X}{G}, mana value 1) cast with 5 mana beside Hare Apparent (2): X = 5 - 2 - 1
    t = replay.TurnRecord("oppo", 3, seq=4, f={"creatures_cast": [93949, 93728], "oppo_mana_spent": 5.0})
    assert (ids.mv(93949), ids.mv(93728)) == (1, 2) and rc._etb_x(t, "oppo", 93949, ids) == 2
    t = replay.TurnRecord("oppo", 3, seq=4, f={"creatures_cast": [], "oppo_mana_spent": 0.0})
    assert rc._etb_x(t, "oppo", 93949, ids) == 1          # not cast (reanimated, copied): at least 1


def test_ids_work_without_the_17lands_download(tmp_path):
    ids = Ids.load(data_dir=tmp_path)            # empty dir: committed tables only
    assert ids.card(93757).name == "Kaito, Cunning Infiltrator" and ids.card(93757).xmage_set == "FDN"
    assert ids.ability(92970).text_flags == frozenset({"counter_on"})


# --- replay parsing --------------------------------------------------------------------------------

def test_parse_order_and_global_turns(games, ids):
    g0, g4 = games[0], games[4]
    assert g0.on_play and g0.won and g0.num_turns == 8 and sum(g0.deck.values()) == 40
    assert [(t.side, t.n) for t in g0.turns[:4]] == [("user", 1), ("oppo", 1), ("user", 2), ("oppo", 2)]
    assert [t.global_turn for t in g0.turns] == list(range(1, 17))
    assert g0.user_slot(8).global_turn == 15 and g0.turns[-1].terminal
    assert [t.terminal for t in g0.turns].count(True) == 1
    assert g0.decision_turns() == list(range(1, 9))
    assert not g4.on_play and (g4.turns[0].side, g4.turns[0].global_turn) == ("oppo", 1)
    assert g4.user_slot(1).global_turn == 2 and sum(g4.deck.values()) == 41
    # padding past num_turns is gone; the opening hand is the pre-bottom 7
    assert len(g0.turns) == 16 and len(g0.opening_hand) == 7


def test_bottomed_inferred(games, ids):
    g = games[8]
    assert g.meta["num_mulligans"] == 1 and len(g.candidate_hands) == 2
    assert names(ids, g.bottomed) == ["Needletooth Pack"] and g.bottomed_exact


# --- specs -----------------------------------------------------------------------------------------

def test_golden_row0_turn3_main1(games, ids):
    s = rc.state_at_user_turn(games[0], 3, "main1", ids=ids)
    A, B = s.players["A"], s.players["B"]
    assert (s.turn, s.activePlayer, s.phase, s.step, s.enterMode) == (5, "A", "PRECOMBAT_MAIN", "PRECOMBAT_MAIN",
                                                                      "PRIORITY_FRESH")
    assert A.hand == sorted(["Island", "Dreadwing Scavenger", "Archmage of Runes", "Seeker's Folly",
                             "Kaito, Cunning Infiltrator", "Eaten Alive"])
    assert (A.life, B.life, B.handUnknown, A.decklistSource, len(A.decklist)) == (20, 20, 6, "exact", 40)
    assert [(p.name, p.count, p.tapped) for p in bf(s, "A") if p.name == "Island"] == [("Island", 2, False)]
    pride = perm(s, "B", "Ajani's Pridemate")[0]
    assert pride.sick and not pride.tapped
    # the opponent spent 2 mana on its turn with 2 lands: both stay tapped through our turn
    assert sorted((p.name, p.tapped) for p in bf(s, "B") if p.name in ("Plains", "Island")) == [
        ("Island", True), ("Plains", True)]
    assert s.provenance.tier == "T0" and s.provenance.ref == "FDN_PremierDraft:row=0:user_turn=3"
    assert B.decklistSource == "placeholder" and len(B.decklist) == 40


def test_golden_row0_turn3_eot_rollover(games, ids):
    s = rc.state_at_user_turn(games[0], 3, ids=ids)
    A, B = s.players["A"], s.players["B"]
    assert (s.turn, s.activePlayer, s.phase, s.step) == (4, "B", "END", "END_TURN")
    assert A.libraryTop == ["Eaten Alive"] and "Eaten Alive" not in A.hand
    # the user's lands tapped on its own turn 2 are still tapped at the end of the opponent's turn
    assert [(p.name, p.count, p.tapped) for p in bf(s, "A") if p.name == "Island"] == [("Island", 2, True)]
    assert perm(s, "A", "Gleaming Barrier")[0].sick           # entered on the user's last turn
    assert B.landsPlayed == 1


def test_golden_row0_turn8_tokens_loyalty(games, ids):
    s = rc.state_at_user_turn(games[0], 8, ids=ids)
    assert perm(s, "A", "Kaito, Cunning Infiltrator")[0].counters == {"LOYALTY": 4}   # 3 -2 +1 +1 +1
    assert perm(s, "A", token="NinjaToken2")
    assert s.players["A"].libraryTop == ["Tolarian Terror"]    # cards_drawn[0]; two cards drawn
    assert "natural_draw_multi" in s.provenance.flags and s.provenance.tier == "T3"
    assert sorted(s.players["B"].graveyard) == ["Ajani's Pridemate", "Burnished Hart", "Clinquant Skymage",
                                                "Stroke of Midnight", "Time Stop"]
    assert s.players["B"].exile == ["Think Twice"]             # cast twice: the second was its flashback


def test_golden_row4_draw_and_attachments(games, ids):
    s1 = rc.state_at_user_turn(games[4], 1, ids=ids)          # on the draw: rollover from global turn 1
    assert (s1.turn, s1.activePlayer, s1.startingPlayer) == (1, "B", "B")
    assert s1.players["A"].libraryTop == ["Banishing Light"] and len(s1.players["A"].hand) == 7
    s8 = rc.state_at_user_turn(games[4], 8, "main1", ids=ids)
    pup = perm(s8, "B", "Mischievous Pup")[0]
    assert perm(s8, "A", "Imprisoned in the Moon")[0].attachTo == f"B:{pup.id}"
    assert perm(s8, "B", "Goldvein Pick")[0].attachTo is None  # falls off: the Pup is a land now
    s9 = rc.state_at_user_turn(games[4], 9, "main1", ids=ids)
    ice = perm(s9, "B", "Icewind Elemental")[0]
    assert perm(s9, "B", "Goldvein Pick")[0].attachTo == f"B:{ice.id}"
    assert "Banishing Light" in s9.players["A"].exile          # exiled by the opponent's Banishing Light


def test_golden_counters_and_mulligan(games, ids):
    s = rc.state_at_user_turn(games[1], 9, "main1", ids=ids)
    assert perm(s, "A", "Clinquant Skymage")[0].counters == {"P1P1": 1}
    s = rc.state_at_user_turn(games[8], 1, ids=ids)           # user turn 1 on the play: main1
    assert (s.turn, s.activePlayer, s.phase) == (1, "A", "PRECOMBAT_MAIN")
    assert "Needletooth Pack" not in s.players["A"].hand and len(s.players["A"].hand) == 6
    assert "entry_main1_t1_on_play" in s.provenance.flags and s.players["B"].handUnknown == 7


def test_opponent_deck_and_hand_from_a_pair(games, ids):
    # a mirrored row of the same game would supply the opponent's real 40 and its hand
    deck = ["Island"] * 17 + ["Plains"] * 16 + ["Ajani's Pridemate", "Burnished Hart", "Clinquant Skymage",
                                                 "Stroke of Midnight", "Think Twice", "Time Stop", "Dazzling Angel"]
    s = rc.state_at_user_turn(games[0], 5, opp_decklist=deck, opp_decklist_source="exact",
                              opp_hand=["Clinquant Skymage", "Stroke of Midnight", "Island", "Plains", "Think Twice"],
                              ids=ids)
    B = s.players["B"]
    assert (B.decklistSource, B.handUnknown, len(B.hand)) == ("exact", 0, 5)
    assert not s.is_partial() and s.validate() == []
    assert "opp_decklist_placeholder" not in s.provenance.flags


def test_golden_opponent_tapped_state(games, ids):
    # row 26, user turn 4: the opponent played Thornwood Falls (enters tapped) and spent 2 of its
    # other 4 lands on Firebrand Archer {1}{R}: three lands stay tapped through our turn
    s = rc.state_at_user_turn(games[26], 4, "main1", ids=ids)
    B = bf(s, "B")
    lands = {"Thornwood Falls", "Mountain", "Forest", "Island"}
    assert sorted((p.name, p.tapped) for p in B if p.name in ("Thornwood Falls", "Mountain")) == [
        ("Mountain", True), ("Thornwood Falls", True)]                      # the Mountain: the only red source
    assert sum(p.count for p in B if p.name in lands and p.tapped) == 3
    assert sum(p.count for p in B if p.name in lands) == 5
    # row 130: Rune-Sealed Wall taps to surveil on each of the opponent's turns, Strix Lookout
    # from turn 6; both stay tapped (they could not block), a fresh Lookout is sick
    s4, s5, s6 = (rc.state_at_user_turn(games[130], n, "main1", ids=ids) for n in (4, 5, 6))
    assert perm(s4, "B", "Rune-Sealed Wall")[0].tapped and perm(s5, "B", "Rune-Sealed Wall")[0].tapped
    look5, look6 = perm(s5, "B", "Strix Lookout")[0], perm(s6, "B", "Strix Lookout")[0]
    assert (look5.tapped, look5.sick, look6.tapped, look6.sick) == (False, True, True, False)
    assert s6.provenance.tier == "T1"


def test_golden_aura_host_from_type_change(games, ids):
    # row 26: the opponent's Imprisoned in the Moon arrives as the user's Lathril moves from the
    # creature to the land list; the host is Lathril (not the higher-power Hungry Ghoul) and certain
    s = rc.state_at_user_turn(games[26], 5, "main1", ids=ids)
    lathril = perm(s, "A", "Lathril, Blade of the Elves")[0]
    assert perm(s, "B", "Imprisoned in the Moon")[0].attachTo == f"A:{lathril.id}"
    assert "attach_heuristic" not in s.provenance.flags and s.provenance.tier == "T0"


def test_golden_counters_entered_with_and_doubled(games, ids):
    # row 76: Mossborn Hydra enters with a +1/+1 counter and its landfall doubled it the same turn
    # (a 0/0 without counters would be buried by XMage's state-based actions)
    assert perm(rc.state_at_user_turn(games[76], 5, ids=ids), "A", "Mossborn Hydra")[0].counters == {"P1P1": 2}
    assert perm(rc.state_at_user_turn(games[76], 6, ids=ids), "A", "Mossborn Hydra")[0].counters == {"P1P1": 4}


def test_golden_upkeep_draw_comes_first(games, ids):
    # row 198, user turn 5: Phyrexian Arena draws at upkeep, before the draw step: both cards are
    # drawn before the first main phase
    e = rc.state_at_user_turn(games[198], 5, ids=ids)
    m = rc.state_at_user_turn(games[198], 5, "main1", ids=ids)
    assert e.players["A"].libraryTop == ["Sanguine Syphoner", "Island"]
    assert {"Sanguine Syphoner", "Island"} <= set(m.players["A"].hand)
    assert "main1_turn_start_triggers_skipped" in m.provenance.flags
    assert "main1_turn_start_triggers_skipped" not in e.provenance.flags


def test_unrecorded_draw_is_unknown(games, ids):
    # row 0, user turn 7 lists no drawn card (0.014% of mid-game user turns): main1 holds an
    # unknown card and the tier says a visible identity is uncertain
    m = rc.state_at_user_turn(games[0], 7, "main1", ids=ids)
    assert m.players["A"].handUnknown == 1 and m.is_partial() and m.provenance.tier == "T3"
    assert "draw_unknown_card" in m.provenance.flags


def all_specs(games, ids, entry):
    for g in games.values():
        for n in g.user_turn_numbers():
            yield g, n, rc.state_at_user_turn(g, n, entry, ids=ids, labels=True)


@pytest.mark.parametrize("entry", rc.ENTRIES)
def test_every_fixture_spec_is_valid(games, ids, entry):
    k = 0
    for g, n, s in all_specs(games, ids, entry):
        assert s.validate() == [], (g.row_index, n, s.validate())
        assert s.provenance.tier in TIERS and s.provenance.source == "17lands"
        assert s.provenance.ref == f"FDN_PremierDraft:row={g.row_index}:user_turn={n}"
        assert StateSpec.from_json(s.to_json()).to_dict() == s.to_dict()
        assert "draft_id" not in s.to_json()
        assert s.labels["user_turn"] == n
        k += 1
    assert k == sum(len(g.user_turn_numbers()) for g in games.values())


def test_eot_rollover_and_main1_agree(games, ids):
    for g in games.values():
        for n in g.user_turn_numbers():
            if g.prev_slot(n) is None:
                continue
            e = rc.state_at_user_turn(g, n, "eot_rollover", ids=ids)
            m = rc.state_at_user_turn(g, n, "main1", ids=ids)
            eA, mA, eB, mB = e.players["A"], m.players["A"], e.players["B"], m.players["B"]
            assert m.turn == e.turn + 1 and (e.activePlayer, m.activePlayer) == ("B", "A")
            # the engine draws libraryTop (the draw step's card, after any upkeep draws) itself;
            # an unrecorded draw is a blind draw there and an unknown card in main1
            assert mA.hand == sorted(eA.hand + eA.libraryTop)
            assert mA.handUnknown == eA.handUnknown + (not g.user_slot(n).L("cards_drawn"))
            assert eB.handUnknown == mB.handUnknown
            for x, y in ((eA, mA), (eB, mB)):
                assert (x.life, x.graveyard, x.exile, x.decklist) == (y.life, y.graveyard, y.exile, y.decklist)
                cx = Counter((p.name, p.tokenClass, p.id, p.attachTo, tuple(sorted(p.counters.items())))
                             for p in x.battlefield for _ in range(p.count))
                cy = Counter((p.name, p.tokenClass, p.id, p.attachTo, tuple(sorted(p.counters.items())))
                             for p in y.battlefield for _ in range(p.count))
                assert cx == cy, (g.row_index, n)
            # the opponent's permanents stay as they were; the user's are untapped after the untap step
            assert sorted((p.id or p.name, p.tapped, p.count) for p in eB.battlefield) == \
                   sorted((p.id or p.name, p.tapped, p.count) for p in mB.battlefield)
            assert not any(p.tapped or p.sick for p in mA.battlefield)
            # main1 can only be worse: it skips the upkeep / draw-step triggers the engine plays
            skipped = "main1_turn_start_triggers_skipped" in m.provenance.flags
            assert e.provenance.tier == m.provenance.tier or (skipped and m.provenance.tier > e.provenance.tier)


# --- labels ----------------------------------------------------------------------------------------

def test_labels_row0(games, ids):
    lab = lb.turn_label(games[0], 3, ids)
    assert [(x["key"], x["idx"]) for x in lab["lands"]] == [("Play Island", 533)]
    assert [(x["key"], x["idx"], x["zone"]) for x in lab["casts"]] == [("Cast Kaito, Cunning Infiltrator", 279, "hand")]
    assert [(x["key"], x["idx"]) for x in lab["activations"]] == [("-2: Create a 2/1 blue Ninja creature token.", 13)]
    assert lab["attacks"] == {}                                # Gleaming Barrier has defender
    lab6 = lb.turn_label(games[0], 6, ids)
    spec6 = rc.state_at_user_turn(games[0], 6, "main1", ids=ids)
    creatures = {f"A:{p.id}": p for p in bf(spec6, "A") if p.id}
    assert set(lab6["attacks"]) <= set(creatures)             # keys are the spec's aliases
    assert sorted((creatures[k].name or creatures[k].tokenClass, v) for k, v in lab6["attacks"].items()) == [
        ("Aegis Turtle", False), ("HumanToken", True), ("NinjaToken2", True)]


def test_labels_flashback_block_mulligan(games, ids):
    assert [x["key"] for x in lb.turn_label(games[1], 3, ids)["casts"]] == ["Cast Think Twice"]
    fb = lb.turn_label(games[1], 4, ids)["casts"]
    assert [(x["key"], x["idx"], x["zone"]) for x in fb] == [("Flashback {2}{U}", 515, "graveyard")]
    # opponent turn 4 after user turn 4: the Faerie token blocked the Cat token
    lab = lb.turn_label(games[1], 4, ids)
    assert lab["block_pairing"] == "unique"
    assert [(b[0].split("_")[0], b[1].split("_")[0], b[2]) for b in lab["blocks"]] == [("A:Faerie", "B:Cat", True)]
    m = lb.turn_label(games[8], 1, ids)
    assert m["mulligan"]["kept_after"] == 1 and len(m["mulligan"]["hands"]) == 2
    assert m["bottomed"] == ["Needletooth Pack"]
    amb = lb.turn_label(games[42], 9, ids)
    assert amb["block_pairing"] == "ambiguous" and any(not b[2] for b in amb["blocks"])


def test_block_pairing_trivial_cases(ids):
    assert lb.block_assignments(ids, [], [93727], [], [])[0] == "none"
    assert lb.block_assignments(ids, [93727], [93727, 93672], [], []) == ("unique", [(0, 0)])
    assert lb.block_assignments(ids, [93727, 93672], [93727], [], [])[0] == "inconsistent"


# --- through the engine ----------------------------------------------------------------------------

# fixture turns whose engine turn start legitimately differs from the main1 spec (checked by hand):
# a blind draw where the row lists none; Clinquant Skymage counters the engine adds (the row logs
# its trigger with a single draw); Dazzling Angel gaining life from the bridge's injected Faerie
# tokens (their ETB triggers fire, a bridge issue); Giada's extra counters on Angels, which the
# bridge applies at injection and the row does not record
ENGINE_KNOWN_DIFFS = {(0, 7), (1, 8), (1, 9), (4, 7), (198, 4), (198, 5)}


def _bf_sig(p: dict) -> Counter:
    return Counter((x.get("name") or "class:" + x.get("tokenClass", "").rsplit(".", 1)[-1], bool(x.get("tapped")),
                    tuple(sorted((x.get("counters") or {}).items()))) for x in p.get("battlefield", [])
                   for _ in range(x.get("count", 1)))


def test_specs_build_and_roll_over_in_xmage(games, ids):
    """Every fixture decision state builds in XMage in both entries, and the engine's own cleanup,
    untap, upkeep and draw from the eot_rollover spec reach the main1 spec's hand, battlefields
    (identity, tapped, counters) and life."""
    from draftzero.gameplay import bridge
    problems = bridge.environment_problems()
    if problems:
        pytest.skip("mzbridge unavailable: " + "; ".join(problems))
    differ, n = set(), 0
    with bridge.Bridge("pytest17lands", log_level="error") as br:
        for g in games.values():
            for t in g.decision_turns():
                if g.prev_slot(t) is None:
                    continue
                m = rc.state_at_user_turn(g, t, "main1", ids=ids)
                e = rc.state_at_user_turn(g, t, ids=ids)
                built = br.build(m, seed=1, advance=False)["dump"]            # raises on a rejected spec
                r = br.build(e, seed=1, decisionPlayer="A", dumpDecisionState=True,
                             decideFrom={"turn": g.user_slot(t).global_turn, "step": "PRECOMBAT_MAIN"})
                ds, where = r.get("decisionState"), (r.get("decision") or {}).get("where") or {}
                n += 1
                if bridge.diff_dump(m, built) or not ds or where.get("step") != "PRECOMBAT_MAIN":
                    differ.add((g.row_index, t))
                    continue
                md = m.to_dict()["players"]
                if Counter(md["A"].get("hand", [])) != Counter(ds["players"]["A"]["hand"]) or any(
                        md[s].get("life", 20) != ds["players"][s]["life"] or _bf_sig(md[s]) != _bf_sig(ds["players"][s])
                        for s in "AB"):
                    differ.add((g.row_index, t))
    assert n >= 60 and differ <= ENGINE_KNOWN_DIFFS, sorted(differ - ENGINE_KNOWN_DIFFS)


# --- regression on the real file ---------------------------------------------------------------------

# replay_empirics.md §4.2 (conservation, % of slots) and §8 (cumulative ladder, % of user turns)
RE_ZONE_STRICT = {"user_hand": 92.7, "user_lands": 98.8, "user_crea": 87.8, "user_nonc": 96.7,
                  "oppo_hand_n": 93.3, "oppo_lands": 98.8, "oppo_crea": 88.0, "oppo_nonc": 96.8}
RE_SLOT = {"strict:user_zones": 80.8, "strict:all_zones": 67.5, "benign:user_zones": 96.9,
           "benign:all_zones": 92.8, "nounexp:user_zones": 99.3, "nounexp:all_zones": 99.0}
RE_LADDER = {"T0": 88.7, "T1": 86.8, "T1b": 59.5, "T2": 43.3, "T3": 36.7}


@pytest.mark.skipif(not REPLAY.exists(), reason="17lands FDN replay file not downloaded")
def test_regression_5000_games(ids):
    from draftzero.gameplay.seventeenlands import collect
    # seeded sample spread over the whole file: every 158th row from row 0 (the research used
    # every 16th; the first rows alone are not representative: early-format games differ ~1 pt)
    st = collect(replay.iter_games(REPLAY, limit=5000, every=158, start=0), ids, build_specs=True)
    r = st.rates()
    assert r["counts"]["games"] == 5000
    for z, v in RE_ZONE_STRICT.items():
        assert abs(r["conservation_zone"][f"strict:{z}"] - v) <= 0.5, (z, r["conservation_zone"][f"strict:{z}"])
    for k, v in RE_SLOT.items():
        assert abs(r["conservation_slot"][k] - v) <= 0.5, (k, r["conservation_slot"][k])
    for k, v in RE_LADDER.items():
        assert abs(r["re_ladder"][k] - v) <= 0.5, (k, r["re_ladder"][k])
    assert r["coverage"]["event_card_ids_mapped"] == 100.0
    assert r["coverage"]["ability_ids_in_table"] == 100.0
    assert r["counts"].get("invalid_specs", 0) == 0, r["spec_invalid"]
