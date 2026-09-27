"""17lands replay -> ids, games, StateSpecs and labels (draftzero.gameplay WP1).

The golden tests run on tests/fixtures/gameplay/fdn_premier_rows.csv.gz: FDN Premier Draft rows
0, 1, 4, 8, 26, 42, 76, 130 and 198 of the 17lands public replay file (CC BY 4.0, see the fixture
README), plus rows 64435 and 89383 in fdn_premier_rows_extra.csv.gz. Rows 0 and 4 are the ones the
research phase inspected by hand. They need only committed files. Mirrored pairs are tested on a
synthetic partner row built from a fixture row (no pair is committed) and, when the data is there,
on real pairs from data/gameplay/pairs_FDN_PremierDraft.jsonl. The regression test streams a seeded
5,000-game sample of the real file and the doubled-list test reads five of its rows; both skip
without it. The engine test builds fixture specs in XMage through java/mzbridge and skips without
Java or the XMage build.
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
EXTRA = Path(__file__).parent / "fixtures" / "gameplay" / "fdn_premier_rows_extra.csv.gz"
REPLAY = replay.replay_path("FDN", "PremierDraft")
PAIRS = Path(__file__).resolve().parents[1] / "data" / "gameplay" / "pairs_FDN_PremierDraft.jsonl"


@pytest.fixture(scope="module")
def ids():
    return Ids.load()


@pytest.fixture(scope="module")
def games():
    return {g.row_index: g for g in replay.iter_games(FIXTURE)}


@pytest.fixture(scope="module")
def extra():
    return {g.row_index: g for g in replay.iter_games(EXTRA)}


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
    # a known hand that does not add up to the row's count is flagged and graded (was ungraded)
    s = rc.state_at_user_turn(games[0], 5, opp_hand=["Island"], ids=ids)
    assert "opp_hand_count_mismatch" in s.provenance.flags and rc.spec_tier({}, {"opp_hand_count_mismatch"}) == "T1"


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


# --- bridge options, mirrored pairs, the state after a turn, unknown destinations, owners -------

def test_every_spec_carries_its_bridge_options(games, ids):
    """labels['bridge'] reaches A's decision of the turn the spec describes, with labels or not:
    the main phase of user turn n (the eot_rollover spec is B's END_TURN, one turn earlier, where
    the bridge would otherwise decide for B), or the opponent's following turn for the specs after
    it (its upkeep for instants and flash, DECLARE_BLOCKERS for blocks)."""
    from draftzero.gameplay.bridge import turn_start_options
    for g in games.values():
        for n in g.user_turn_numbers():
            u = g.user_slot(n)
            want = {"decisionPlayer": "A", "decideFrom": {"turn": u.global_turn, "step": "PRECOMBAT_MAIN"}}
            for entry in rc.ENTRIES:
                for with_labels in (False, True):
                    s = rc.state_at_user_turn(g, n, entry, ids=ids, labels=with_labels)
                    assert s.labels["bridge"] == want, (g.row_index, n, entry)
                    if s.step == "END_TURN":
                        assert s.labels["bridge"] == turn_start_options(s)
            q = g.next_slot(n)
            if not (u.played and q is not None and q.played):
                continue
            s = rc.state_after_user_turn(g, n, ids=ids)
            assert s.labels["bridge"] == {"decisionPlayer": "A", "decideFrom": {"turn": q.global_turn, "step": "UPKEEP"}}
            if q.L("creatures_attacked"):
                try:
                    s = rc.state_after_user_turn(g, n, "declare_attackers", ids=ids)
                except ValueError:
                    continue
                assert s.labels["bridge"] == {"decisionPlayer": "A",
                                              "decideFrom": {"turn": q.global_turn, "step": "DECLARE_BLOCKERS"}}


def _grp(ids, name):
    return min(g for g, r in ids.cards17.items() if r["name"] == name)


def mirror(g, ids, hand_at, draws_at=lambda seq: [], board=lambda t, f: f):
    """A synthetic partner row for g: the same game recorded by g's opponent (sides swapped, boards
    and life mirrored). Its own hand at the end of slot k is hand_at(k) (grp ids), its draws in its
    own turn k draws_at(k); its deck is every card g's opponent revealed plus 20 Plains and 20 Islands."""
    turns = []
    for t in g.turns:
        f = {"eot_user_life": t.num("eot_oppo_life"), "eot_oppo_life": t.num("eot_user_life"),
             "eot_user_cards_in_hand": hand_at(t.seq), "cards_drawn": draws_at(t.seq) if t.side == "oppo" else []}
        for k in ("lands", "creatures", "non_creatures"):
            f[f"eot_user_{k}_in_play"] = t.L(f"eot_oppo_{k}_in_play")
            f[f"eot_oppo_{k}_in_play"] = t.L(f"eot_user_{k}_in_play")
        turns.append(replay.TurnRecord(t.other, t.n, t.seq, t.global_turn, board(t, f), t.played, t.terminal))
    deck = Counter(rc.analyze(g, ids).states[-1].revealed) + Counter({"Plains": 20, "Island": 20})
    k = int(g.meta.get("opp_num_mulligans") or 0)
    plains = _grp(ids, "Plains")
    return replay.Game(-1, {"on_play": not g.on_play, "draft_id": "synthetic-partner"}, deck, Counter(), [],
                       [plains] * 7, [plains] * k, True, turns)


def test_exact_spec_from_a_mirrored_pair(games, ids):
    """exact_spec takes B's deck from the partner row and B's hand from the partner's own hand at
    the end of the half-turn the spec starts from (not the one before or after it); the state after
    a user turn takes B's hand at the end of that turn and B's next draw from the partner's turn."""
    island, plains, swamp = _grp(ids, "Island"), _grp(ids, "Plains"), _grp(ids, "Swamp")
    for row in (0, 4):                                  # on the play (turn 1 from the opening hand), on the draw
        g = games[row]

        def hand_at(seq, g=g):
            k = int(g.turns[seq].num("eot_oppo_cards_in_hand"))
            j = seq % (k + 1)                           # a different mix at every slot: catches off-by-ones
            return [island] * j + [plains] * (k - j)
        partner = mirror(g, ids, hand_at, draws_at=lambda seq: [swamp])
        for n in g.decision_turns():
            s = rc.exact_spec(g, partner, n, ids=ids, labels=True)
            p = g.prev_slot(n)
            want = hand_at(p.seq) if p is not None else [plains] * (7 - int(g.meta.get("opp_num_mulligans") or 0))
            B = s.players["B"]
            assert B.hand == names(ids, want), (row, n)
            assert (B.decklistSource, sorted(B.decklist)) == ("exact", sorted(partner.deck.elements()))
            assert not s.is_partial() and s.validate() == [], (row, n, s.validate())
            assert "pair_view_mismatch" not in s.provenance.flags and "opp_decklist_placeholder" not in s.provenance.flags
            assert s.labels["bridge"]["decideFrom"]["turn"] == g.user_slot(n).global_turn
            q = g.next_slot(n)
            if q is None or not q.played:
                continue
            a = rc.exact_spec(g, partner, n, ids=ids, after=True)
            assert a.players["B"].hand == names(ids, hand_at(g.user_slot(n).seq)), (row, n)
            assert a.players["B"].libraryTop == ["Swamp"] and not a.is_partial() and a.validate() == []
            if q.L("creatures_attacked"):
                b = rc.exact_spec(g, partner, n, "declare_attackers", ids=ids, after=True)
                assert not b.is_partial() and b.validate() == []
                # its hand at the end of our turn + the draw - what it played before combat
                played = Counter(p.name for p in b.players["B"].battlefield if p.name for _ in range(p.count)) \
                    - Counter(p.name for p in a.players["B"].battlefield if p.name for _ in range(p.count))
                assert Counter(b.players["B"].hand) == Counter(a.players["B"].hand) + Counter(["Swamp"]) - played
    # rows that disagree about the board are flagged; two rows on the same side are no pair
    g = games[0]
    odd = mirror(g, ids, lambda seq: [plains] * int(g.turns[seq].num("eot_oppo_cards_in_hand")),
                 board=lambda t, f: {**f, "eot_user_lands_in_play": f["eot_user_lands_in_play"][1:]})
    s = rc.exact_spec(g, odd, 5, ids=ids)
    assert "pair_view_mismatch" in s.provenance.flags and s.provenance.tier == "T3"
    same = replay.Game(-2, {"on_play": g.on_play}, Counter(), Counter(), [], [], [], True, [])
    with pytest.raises(ValueError, match="not a mirrored pair"):
        rc.exact_spec(g, same, 5, ids=ids)


def test_holdout_drafts_names_both_rows_drafts(games, ids):
    g = games[0]
    partner = mirror(g, ids, lambda seq: [])
    real = replay.Game(g.row_index, {**g.meta, "draft_id": "user-draft"}, g.deck, g.sideboard, g.candidate_hands,
                       g.opening_hand, g.bottomed, g.bottomed_exact, g.turns)
    assert rc.holdout_drafts(real, partner) == ["synthetic-partner", "user-draft"]
    assert rc.holdout_drafts(real) == ["user-draft"]
    assert rc.holdout_drafts(g) == []                     # the fixture blanks draft_id


@pytest.mark.skipif(not (REPLAY.exists() and PAIRS.exists()), reason="17lands replay file or pairs file missing")
def test_exact_specs_on_real_pairs(ids):
    """On real mirrored pairs (both rows below 6000): every exact spec is valid and not partial; B's
    battlefield is the partner's own view of its battlefield, B's hand the partner's own hand of that
    half-turn, B's deck the partner's deck; the states after a turn hold B's hand and next draw."""
    from draftzero.gameplay.pairs import load_pairs
    pairs = [q for q in load_pairs(PAIRS) if max(q.row_a, q.row_b) < 6000][:40]
    assert len(pairs) >= 20
    got = replay.read_games({r for q in pairs for r in (q.row_a, q.row_b)})
    k = 0
    for q in pairs:
        for a, b in ((q.row_a, q.row_b), (q.row_b, q.row_a)):
            g, o = got[a], got[b]
            drafts = rc.holdout_drafts(g, o)
            assert len(drafts) == 2 and set(drafts) == {g.meta["draft_id"], o.meta["draft_id"]}
            for n in g.decision_turns():
                s = rc.exact_spec(g, o, n, ids=ids)
                assert s.validate() == [] and not s.is_partial(), (a, n)
                B = s.players["B"]
                # token copies are recorded under the copied card's id: a copy beyond the deck's
                # count extends the list (flagged, T2); nothing else may differ from the real deck
                assert B.decklistSource == "exact" and Counter(o.deck) <= Counter(B.decklist)
                assert Counter(B.decklist) == Counter(o.deck) or {"B_decklist_extended", "token_copy_possible"} <= set(
                    s.provenance.flags), (a, n)
                p = g.prev_slot(n)
                if p is None or not p.played:
                    continue
                m = rc.mirror_slot(g, o, p)
                assert (m.side, m.n) == ("user", p.n)
                theirs = Counter(ids.card(c).token_class if ids.is_token(c) else ids.name(c)
                                 for f in ("lands", "creatures", "non_creatures") for c in m.L(f"eot_user_{f}_in_play"))
                mine = Counter(x.tokenClass or x.name for x in B.battlefield for _ in range(x.count))
                assert mine == theirs and B.hand == names(ids, m.L("eot_user_cards_in_hand")), (a, n)
                assert "pair_view_mismatch" not in s.provenance.flags
                qs = g.next_slot(n)
                if qs is not None and qs.played and qs.side == "oppo":
                    after = rc.exact_spec(g, o, n, ids=ids, after=True)
                    mu, mq = rc.mirror_slot(g, o, g.user_slot(n)), rc.mirror_slot(g, o, qs)
                    assert after.players["B"].hand == names(ids, mu.L("eot_user_cards_in_hand"))
                    assert after.players["B"].libraryTop == [ids.name(c) for c in mq.L("cards_drawn")][:1]
                    assert after.validate() == [] and not after.is_partial()
                k += 1
    assert k >= 200


def test_state_after_user_turn(games, ids):
    # row 1, user turn 4 (global turn 7): the opponent then attacks with Prideful Parent (vigilance)
    # and a Cat token, and the user's Faerie token blocks the Cat
    g = games[1]
    e = rc.state_after_user_turn(g, 4, ids=ids, labels=True)
    assert (e.turn, e.activePlayer, e.phase, e.step, e.enterMode, e.priorityPlayer) == (
        7, "A", "END", "END_TURN", "PRIORITY_FRESH", "A")
    assert e.provenance.ref == "FDN_PremierDraft:row=1:user_turn=4:after"
    assert e.labels["blocks"] == [["A:Faerie_6", "B:Cat_10", True]] and e.labels["attacked"] == ["Cat", "Prideful Parent"]
    # the user's lands tapped on its own turn stay tapped through the opponent's
    assert all(p.tapped for p in bf(e, "A") if p.name in ("Plains", "Island"))
    b = rc.state_after_user_turn(g, 4, "declare_attackers", ids=ids, labels=True)
    assert (b.turn, b.activePlayer, b.phase, b.step, b.enterMode, b.priorityPlayer) == (
        8, "B", "COMBAT", "DECLARE_ATTACKERS", "PRIORITY_HELD", "B")
    assert sorted(x.attacker for x in b.attackers) == ["B:Cat_10", "B:PridefulParent_8"]
    assert {x.defender for x in b.attackers} == {"player:A"}
    B = b.players["B"]
    # its untap step untapped everything; its land drop was played before combat
    assert not any(p.tapped for p in B.battlefield if p.name in ("Plains", "Mountain"))
    assert B.landsPlayed == 1 and sum(p.count for p in B.battlefield if p.name == "Plains") == 3
    assert perm(b, "B", "Prideful Parent")[0].tapped is False            # vigilance
    assert B.handUnknown == e.players["B"].handUnknown + 1 - 1           # + its draw - the land
    assert not perm(b, "A", token="FaerieToken")[0].tapped
    assert "opp_turn_timing_assumed" in b.provenance.flags and b.provenance.tier == "T1"
    assert b.labels["block_pairing"] == "unique" and b.validate() == [] and e.validate() == []
    # the labels' attacker keys are the spec's attackers
    for blocker, attacker, _ in b.labels["blocks"]:
        assert attacker is None or attacker in {x.attacker for x in b.attackers}
    with pytest.raises(ValueError, match="did not attack"):
        rc.state_after_user_turn(g, 1, "declare_attackers", ids=ids)
    with pytest.raises(ValueError, match="no opponent turn"):
        rc.state_after_user_turn(g, g.user_turn_numbers()[-1], ids=ids)


@pytest.mark.parametrize("entry", rc.AFTER_ENTRIES)
def test_every_fixture_spec_after_a_turn_is_valid(games, extra, ids, entry):
    k = 0
    for g in list(games.values()) + list(extra.values()):
        for n in g.decision_turns():
            try:
                s = rc.state_after_user_turn(g, n, entry, ids=ids, labels=True)
            except ValueError:
                continue
            assert s.validate() == [], (g.row_index, n, s.validate())
            assert s.provenance.tier in TIERS and StateSpec.from_json(s.to_json()).to_dict() == s.to_dict()
            assert s.labels["user_turn"] == n and s.labels["opp_turn"] == g.next_slot(n).n
            k += 1
    assert k >= (60 if entry == "eot_rollover" else 25)


def test_bounced_opponent_card_has_no_zone(games, ids):
    """Row 4: the opponent's Mischievous Pup returns its own Island to its hand on its turn 5. The
    departure has no record, so the Island is in none of B's zones (not in exile, where it used to
    go): the library as far as the spec is concerned, and a belief may put it in the hand. It is
    flagged until the opponent plays an Island again (its turn 6)."""
    g = games[4]
    for n in (5, 6):
        s = rc.state_at_user_turn(g, n, ids=ids)
        assert rc.opp_location_unknown(g, n, ids) == Counter({"Island": 1})
        assert "Island" not in s.players["B"].exile + s.players["B"].graveyard
        assert "opp_dest_unknown" in s.provenance.flags
        assert "Island" in s.players["B"].decklist                     # a revealed card of the placeholder
    s = rc.state_at_user_turn(g, 7, ids=ids)
    assert not rc.opp_location_unknown(g, 7, ids) and "opp_dest_unknown" not in s.provenance.flags
    assert rc.opp_location_unknown(g, 4, ids, after=True) == Counter()
    assert rc.opp_location_unknown(g, 5, ids, after=True) == Counter({"Island": 1})
    # a known opponent hand that holds it accounts for it
    s = rc.state_at_user_turn(g, 5, ids=ids, opp_hand=["Island"])
    assert "opp_dest_unknown" not in s.provenance.flags
    assert rc.spec_tier({}, {"opp_dest_unknown"}) == "T1"
    # the Pup's slot shows one more card in the opponent's hand than its events explain: likely in
    # hand; row 42's Vampire Nighthawk left with no such rise: hand, library or exile
    assert rc.opp_location_unknown(g, 5, ids, split=True) == (Counter({"Island": 1}), Counter())
    assert rc.opp_location_unknown(games[42], 6, ids, split=True) == (Counter(), Counter({"Vampire Nighthawk": 1}))
    # the user's own unrecorded departures still go to exile: its hand is known, so not there
    # (row 1: its Lightshell Duo leaves on the opponent's turn 5 with no record)
    assert "Lightshell Duo" in rc.state_at_user_turn(games[1], 6, ids=ids).players["A"].exile
    # an opponent's permanent that leaves while the user exiled something goes to exile (row 26)
    assert rc.state_at_user_turn(games[26], 8, ids=ids).players["B"].exile == ["Skyship Buccaneer"]


def test_control_change_names_the_owner(extra, ids):
    """Row 64435: at the end of the opponent's turn 5 it controls the user's Helpful Hunter. The
    permanent is listed under its controller with owner A, so its card comes out of A's decklist
    (not B's placeholder); the spec flags that the change may be until end of turn."""
    g = extra[64435]
    s = rc.state_at_user_turn(g, 5, ids=ids)
    hunter = perm(s, "B", "Helpful Hunter")[0]
    assert hunter.owner == "A" and not any(p.owner for p in bf(s, "A"))
    assert "A_decklist_extended" not in s.provenance.flags and "control_changed" in s.provenance.flags
    assert s.provenance.tier == "T2" and s.validate() == []
    assert Counter(s.players["A"].decklist) == Counter(g.deck)
    t4, t6 = (rc.state_at_user_turn(g, n, ids=ids) for n in (4, 6))
    assert not any(p.owner for x in (t4, t6) for seat in "AB" for p in bf(x, seat))


def test_imprisoned_in_the_moon_without_a_creature_turning_into_a_land(extra, ids):
    """Row 89383: the opponent's Imprisoned in the Moon arrives and no creature moves to the land
    list, so it enchants a land (on a creature, XMage would turn that creature into a land and it
    could no longer block)."""
    g = extra[89383]
    s = rc.state_at_user_turn(g, 8, ids=ids)
    moon = perm(s, "B", "Imprisoned in the Moon")[0]
    host = next(p for p in bf(s, "A") if f"A:{p.id}" == moon.attachTo)
    assert host.name == "Mountain" and "attach_heuristic" in s.provenance.flags
    b = rc.state_after_user_turn(g, 8, "declare_attackers", ids=ids, labels=True)
    # the opponent's Meteor Golem destroyed the Scourge that turn, before or after combat: that it
    # "did not block" is not certain (it may have been dead by then)
    assert b.labels["blocks"] == [["A:WildwoodScourge_26", None, False]] and len(b.attackers) == 3
    assert "offturn_timing_unknown" in b.provenance.flags


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


def test_who_can_attack_and_block(games, ids):
    """Eligibility follows what the engine allows. Row 4, user turns 6-7: its Drake Hatcher under the
    opponent's Witness Protection (a plain 1/1, which can attack) attacked; it used to be excluded as
    under a "hostile" Aura and its attack keyed "new:Drake Hatcher", a name no spec permanent has.
    A Pacifism host, Vampire Soulcaller and the Rat token that can't block are no potential
    blockers (the engine never asks about them); Brazen Borrower is one only against flyers."""
    for n in (6, 7):
        att = lb.turn_label(games[4], n, ids)["attacks"]
        assert att["A:DrakeHatcher_13"] is True and not any(k.startswith("new:") for k in att), att
    assert "host_cant_attack_block" in ids.info("Pacifism").features
    assert "host_cant_attack_block" not in ids.info("Witness Protection").features

    def inst(iid, name, host=None, token_class=None):
        return rc.Inst(iid, 0, name, "user", 0, token=token_class is not None, token_class=token_class, host=host)
    assert lb._pacified([inst(1, "Pacifism", 7), inst(2, "Witness Protection", 8), inst(3, "Starlight Snare", 9)],
                        ids) == {7, 9}
    assert lb._cant_block(inst(4, "Vampire Soulcaller"), True, ids)
    assert lb._cant_block(inst(5, "Rat", token_class="RatCantBlockToken"), False, ids)
    assert lb._cant_block(inst(6, "Brazen Borrower"), False, ids) and not lb._cant_block(inst(6, "Brazen Borrower"), True, ids)
    assert not lb._cant_block(inst(7, "Gleaming Barrier"), False, ids)       # defender: it blocks


def test_block_spec_flags_what_it_cannot_show(games, extra, ids):
    """The declare-attackers spec is the state before the user's own plays of that turn and after
    none of the declaration's triggers: both are flagged, not silently dropped."""
    # row 64435, user turn 2: the user flashed in Resolute Reinforcements (and got its Soldier)
    # during the opponent's turn and blocked with both; neither is in the state (was unflagged)
    s = rc.state_after_user_turn(extra[64435], 2, "declare_attackers", ids=ids, labels=True)
    assert "offturn_timing_unknown" in s.provenance.flags and s.labels["offturn_flash"]
    aliases = {f"A:{p.id}" for p in bf(s, "A") if p.id}
    assert [b for b, a, _ in s.labels["blocks"] if a] and all(b not in aliases for b, a, _ in s.labels["blocks"] if a)
    # a "whenever ... attacks" trigger (Frenzied Goblin's: a creature can't block) resolved before
    # blocks; the PRIORITY_HELD entry skips it. Row 1, user turn 4 with that trigger added:
    g = games[1]
    q = g.next_slot(4)
    assert "attack_trigger" in ids.ability(96640).text_flags          # committed table: no download needed
    assert "attack_triggers_skipped" not in rc.state_after_user_turn(g, 4, "declare_attackers", ids=ids).provenance.flags
    turns = [replay.TurnRecord(t.side, t.n, t.seq, t.global_turn,
                               {**t.f, "oppo_abilities": t.L("oppo_abilities") + [96640]} if t is q else t.f,
                               t.played, t.terminal) for t in g.turns]
    g2 = replay.Game(g.row_index, g.meta, g.deck, g.sideboard, g.candidate_hands, g.opening_hand, g.bottomed,
                     g.bottomed_exact, turns)
    s = rc.state_after_user_turn(g2, 4, "declare_attackers", ids=ids)
    assert "attack_triggers_skipped" in s.provenance.flags and rc.spec_tier({}, {"attack_triggers_skipped"}) == "T1"


def test_block_pairing_trivial_cases(ids):
    assert lb.block_assignments(ids, [], [93727], [], [])[0] == "none"
    assert lb.block_assignments(ids, [93727], [93727, 93672], [], []) == ("unique", [(0, 0)])
    assert lb.block_assignments(ids, [93727, 93672], [93727], [], [])[0] == "inconsistent"


HAWK, PANDA, VESSEL, SAGE, LIONS, SERRA, HUNTER, BARRIER = 93855, 93833, 93776, 93777, 93859, 93860, 93729, 93965
SNARESPINNER, FEASTER, SQUIRE, COURAGEOUS_GOBLIN = 93827, 93772, 93736, 93795
INSECT, FAERIE, CAT = 94176, 94164, 94156          # token ids: a flying Insect and Faerie, a 1/1 Cat


def test_block_pairing_respects_evasion(ids):
    """A pairing must be legal: a flyer (Healer's Hawk, a Faerie or Insect token) is blocked by
    Flying or Reach only, and a Menace attacker (Crypt Feaster) by two or more. Conditional evasion
    (Skyknight Squire flies with three counters, Courageous Goblin's menace) is unknown, as is a
    token of an unlisted class: they restrict nothing."""
    assert lb._can_block(ids, SNARESPINNER, HAWK) and lb._can_block(ids, SERRA, FAERIE)
    assert not lb._can_block(ids, HUNTER, HAWK) and not lb._can_block(ids, CAT, INSECT)
    assert lb._can_block(ids, HUNTER, SQUIRE) and lb._evasion(ids, COURAGEOUS_GOBLIN) is None
    # ambiguous by P/T alone (Helpful Hunter trading with either attacker), unique by evasion
    assert lb.block_assignments(ids, [HAWK, LIONS], [SERRA, HUNTER], [HAWK, LIONS], [HUNTER]) == ("unique", [(0, 1)])
    assert lb.block_assignments(ids, [FEASTER, LIONS], [HUNTER, BARRIER, SERRA], [LIONS], []) == ("unique", [(0, 0, 1)])
    assert lb.block_assignments(ids, [COURAGEOUS_GOBLIN, LIONS], [HUNTER, SERRA], [LIONS], [])[0] == "unique"
    # 17lands row 458570, user turn 6 (the opponent's blocks): the old "consistent" pairings had
    # Infernal Vessel blocking Healer's Hawk, which XMage refused. No legal one explains the deaths
    # by printed P/T, so it is inconsistent with the one legal guess: the Insect token on the Hawk
    st, good = lb.block_assignments(ids, [HAWK, PANDA], [VESSEL, INSECT, SAGE], [HAWK], [VESSEL])
    assert (st, good) == ("inconsistent", [(1, 0, 1)])
    # nothing legal (a trick gave a blocker flying or reach): no guess
    assert lb.block_assignments(ids, [HAWK, PANDA], [LIONS, HUNTER], [], []) == ("inconsistent", [])


def test_block_pairing_allows_a_granted_flyer(ids):
    """A defender with a flying grant (Fleeting Flight cast that turn, Celestial Armor, Angelic
    Destiny, an anthem) may block a flyer with a printed ground creature: the evasion rule is lifted
    for its blockers (not for Ajani's sorcery-speed -3 or a Vehicle's own Flying). 17lands row
    238760, the opponent's turn 11: two Soldier tokens, one under Fleeting Flight, blocked
    Courageous Goblin and Flamewake Phoenix (the rule alone left no legal pairing)."""
    GOBLIN, PHOENIX, SOLDIER = COURAGEOUS_GOBLIN, 93911, 94161
    assert lb.flying_granted(ids, ["Plains", "Fleeting Flight"]) and lb.flying_granted(ids, {"Celestial Armor"})
    assert lb.flying_granted(ids, ["Angelic Destiny"]) and lb.flying_granted(ids, ["Dropkick Bomber"])
    assert not lb.flying_granted(ids, ["Ajani, Caller of the Pride", "Skysovereign, Consul Flagship", "Healer's Hawk",
                                       "Sure Strike", "Kitesail Corsair"])
    args = (ids, [GOBLIN, PHOENIX], [SOLDIER, SOLDIER], [GOBLIN, PHOENIX], [])
    assert lb.block_assignments(*args) == ("inconsistent", [])
    assert lb.block_assignments(*args, blockers_may_fly=True) == ("unique", [(0, 1)])
    assert not lb._can_block(ids, HUNTER, HAWK) and lb._can_block(ids, HUNTER, HAWK, blockers_may_fly=True)


def _with(g, slot, **fields):
    """Game g with some lists of one slot replaced (the same row, a 17lands quirk added)."""
    turns = [replay.TurnRecord(t.side, t.n, t.seq, t.global_turn, {**t.f, **fields} if t is slot else t.f,
                               t.played, t.terminal) for t in g.turns]
    return replay.Game(g.row_index, g.meta, g.deck, g.sideboard, g.candidate_hands, g.opening_hand, g.bottomed,
                       g.bottomed_exact, turns)


def test_a_list_recorded_twice_is_counted_once(games, ids):
    """17lands lists a turn's attackers (and often its blocks, activations) twice over in ~3% of
    attacking turns. A doubled list is halved when the board cannot explain the second copy, and
    labels say so; one the board explains (two copies that both attacked) is kept."""
    def doubled(g, slot, *fields):
        return _with(g, slot, **{f: slot.L(f) * 2 for f in fields})
    # row 4, user turn 6: Drake Hatcher and two Faerie tokens attacked; one Hatcher on the battlefield
    g = games[4]
    lab = lb.turn_label(g, 6, ids)
    dbl = lb.turn_label(doubled(g, g.user_slot(6), "creatures_attacked"), 6, ids)
    assert dbl["deduped"] == ["creatures_attacked"] and dbl["attacks"] == lab["attacks"]
    assert not any(k.startswith("new:") for k in dbl["attacks"]) and "twice over" in dbl["attack_notes"][0]
    # row 8, user turn 9: a Brazen Scourge attacked with a second one cast that turn (haste);
    # [Scourge, Scourge] alone is two copies the board explains, not a double
    g = games[8]
    u = g.user_slot(9)
    two = [c for c in u.L("creatures_attacked") if ids.name(c) == "Brazen Scourge"]
    lab = lb.turn_label(_with(g, u, creatures_attacked=two), 9, ids)
    assert lab["deduped"] == [] and {k: v for k, v in lab["attacks"].items() if "Scourge" in k} == {
        "A:BrazenScourge_21": True, "new:Brazen Scourge": True}
    # the opponent's turn after row 1's user turn 8: Cat and Brazen Scourge attacked, Firebrand
    # Archer blocked the Cat; all three lists doubled (the Cat token is no evidence on its own)
    g = games[1]
    q = g.next_slot(8)
    lab = lb.turn_label(g, 8, ids)
    dbl = lb.turn_label(doubled(g, q, "creatures_attacked", "creatures_blocked", "creatures_blocking"), 8, ids)
    assert dbl["offturn_deduped"] == ["creatures_attacked", "creatures_blocked", "creatures_blocking"]
    assert (dbl["blocks"], dbl["block_pairing"]) == (lab["blocks"], lab["block_pairing"]) == (
        [["A:ClinquantSkymage_21", None, True], ["A:FirebrandArcher_22", "B:Cat_18", True]], "unique")
    assert lb.after_turn_label(doubled(g, q, "creatures_attacked"), 8, ids)["attacked"] == ["Brazen Scourge", "Cat"]
    # activations: row 0, user turn 3, Kaito's -2 twice (a loyalty ability, one Kaito); row 198,
    # user turn 3, Equip {1} twice with 3 mana spent on Goldvein Pick (2) and one Equip
    g = games[0]
    dbl = lb.turn_label(doubled(g, g.user_slot(3), "user_abilities"), 3, ids)
    assert dbl["deduped"] == ["user_abilities"] and dbl["activations"] == lb.turn_label(g, 3, ids)["activations"]
    g = games[198]
    u = g.user_slot(3)
    once = lb.turn_label(g, 3, ids)["activations"]
    assert u.num("user_mana_spent") == 3 and [a["key"][:9] for a in once] == ["Equip {1}"]
    dbl = lb.turn_label(doubled(g, u, "user_abilities"), 3, ids)
    assert dbl["deduped"] == ["user_abilities"] and len(dbl["activations"]) == 1
    paid = lb.turn_label(_with(g, u, user_abilities=u.L("user_abilities") * 2, user_mana_spent=4.0), 3, ids)
    assert paid["deduped"] == [] and len(paid["activations"]) == 2          # 4 mana pays for both Equips


@pytest.mark.skipif(not REPLAY.exists(), reason="17lands FDN replay file not downloaded")
def test_doubled_lists_and_evasion_on_real_rows(ids):
    """The turns the replay review found, and two the cleanup review found (reads the file up to
    row 458570, ~4 s)."""
    from draftzero.gameplay.turnreplay import opponent_blocks
    g = replay.read_games([80, 600, 116964, 123680, 238760, 449959, 458570], REPLAY)
    # [Skyknight Squire, Cat Collector] x2 with one of each: the second copy was "new:" attackers
    lab = lb.turn_label(g[116964], 5, ids)
    assert lab["deduped"] == ["creatures_attacked"]
    assert lab["attacks"] == {"A:SkyknightSquire_8": True, "A:CatCollector_14": True}
    # Sower of Chaos' {2}{R} twice, with 5 mana spent on Firebrand Archer (2) and one activation
    lab = lb.turn_label(g[449959], 6, ids)
    assert lab["deduped"] == ["user_abilities"] and [a["id"] for a in lab["activations"]] == [175859]
    # [Inspiring Paladin] x2 with two Paladins on the battlefield: both attacked (one blocked and
    # killed, one unblocked), not a double
    lab = lb.turn_label(g[80], 5, ids)
    assert lab["deduped"] == [] and lab["attacks"] == {"A:InspiringPaladin_7": True, "A:InspiringPaladin_11": True}
    # the opponent's turn after user turn 5: its combat recorded twice over (one Vampire Gourmand
    # blocked one Strongbox Raider); it was an "inconsistent" pairing with a "new:" blocker
    lab = lb.turn_label(g[600], 5, ids)
    assert lab["offturn_deduped"] == ["creatures_attacked", "creatures_blocked", "creatures_blocking"]
    assert (lab["block_pairing"], lab["blocks"]) == ("unique", [["A:VampireGourmand_13", "B:StrongboxRaider_11", True]])
    # user turn 6: the replay's pairing had Infernal Vessel blocking Healer's Hawk (flying), which
    # XMage refused; its one alternative is now the legal guess, the Insect token on the Hawk
    alts, status, _ = opponent_blocks(g[458570], 6, ids, lb.turn_label(g[458570], 6, ids)["attacks"])
    assert status == "inconsistent" and alts == [[["B:InfernalVessel_8", "A:FiendishPanda_16"],
                                                  ["B:Insect_21", "A:HealersHawk_3"],
                                                  ["B:InfestationSage_22", "A:FiendishPanda_16"]]]
    # the opponent's turn after user turn 16: Scavenging Ooze's {G} twice, 6 mana spent on Claws Out
    # (mana value 5, {1} less with the user's Cat, Helpful Hunter) and two activations: not a double
    lab = lb.turn_label(g[123680], 16, ids)
    assert g[123680].next_slot(16).num("user_mana_spent") == 6
    assert lab["offturn_deduped"] == [] and [a["id"] for a in lab["offturn_activations"]] == [90106, 90106]
    # row 238760, the opponent's turn after user turn 10: the user cast Fleeting Flight, and its Soldier tokens
    # blocked Courageous Goblin and Flamewake Phoenix; without the grant no pairing was legal
    lab = lb.turn_label(g[238760], 10, ids)
    assert "Fleeting Flight" in [ids.name(c) for c in g[238760].next_slot(10).L("user_instants_sorceries_cast")]
    assert (lab["block_pairing"], [r for r in lab["blocks"] if r[1]]) == (
        "unique", [["A:Soldier_22", "B:CourageousGoblin_14", True], ["A:Soldier_23", "B:FlamewakePhoenix_25", True]])


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


def test_specs_build_and_roll_over_in_xmage(games, extra, ids, tmp_path):
    """Every fixture decision state builds in XMage in both entries, and the engine's own cleanup,
    untap, upkeep and draw from the eot_rollover spec reach the main1 spec's hand, battlefields
    (identity, tapped, counters) and life. Each spec's labels['bridge'] reaches A's precombat-main
    decision of the user turn it describes. The states after a turn build too; the block specs'
    decision is A's block question at DECLARE_BLOCKERS of the opponent's turn (or, with no creature
    able to block, a later one), and the END_TURN ones decide in the opponent's turn or not at all."""
    from draftzero.gameplay import bridge
    problems = bridge.environment_problems()
    if problems:
        pytest.skip("mzbridge unavailable: " + "; ".join(problems))
    differ, differ_b, n, blocks, block_questions = set(), set(), 0, 0, 0
    with bridge.Bridge("pytest17lands", log_level="error", heap="2g", runtime_root=tmp_path) as br:
        for g in list(games.values()) + list(extra.values()):
            for t in g.decision_turns():
                u, q = g.user_slot(t), g.next_slot(t)
                if g.prev_slot(t) is not None:
                    m = rc.state_at_user_turn(g, t, "main1", ids=ids)
                    e = rc.state_at_user_turn(g, t, ids=ids)
                    built = br.build(m, seed=1, advance=False)["dump"]            # raises on a rejected spec
                    r = br.build(e, seed=1, dumpDecisionState=True, **e.labels["bridge"])
                    rm = br.build(m, seed=1, **m.labels["bridge"])
                    for x in (r, rm):
                        d = x.get("decision")
                        assert d is None or (d["player"], d["where"]["turn"], d["where"]["step"]) == (
                            "A", u.global_turn, "PRECOMBAT_MAIN"), (g.row_index, t, d and d["where"])
                    ds, where = r.get("decisionState"), (r.get("decision") or {}).get("where") or {}
                    n += 1
                    if bridge.diff_dump(m, built) or not ds or where.get("step") != "PRECOMBAT_MAIN":
                        differ.add((g.row_index, t))
                    else:
                        md = m.to_dict()["players"]
                        if Counter(md["A"].get("hand", [])) != Counter(ds["players"]["A"]["hand"]) or any(
                                md[s].get("life", 20) != ds["players"][s]["life"] or _bf_sig(md[s]) != _bf_sig(ds["players"][s])
                                for s in "AB"):
                            differ.add((g.row_index, t))
                if q is None or not q.played or q.side != "oppo":
                    continue
                a = rc.state_after_user_turn(g, t, ids=ids)
                d = br.build(a, seed=1, **a.labels["bridge"]).get("decision")
                assert d is None or (d["player"], d["where"]["turn"]) == ("A", q.global_turn), (g.row_index, t)
                if q.L("creatures_attacked"):
                    b = rc.state_after_user_turn(g, t, "declare_attackers", ids=ids, labels=True)
                    r = br.build(b, seed=1, dumpDecisionState=True, **b.labels["bridge"])
                    if bridge.diff_dump(b, r["dump"]):
                        differ_b.add((g.row_index, t))      # Giada's counters (row 198), as above
                    d, blocks = r.get("decision"), blocks + 1
                    if d is not None and d["type"] == "CHOOSE_TARGET":
                        assert (d["player"], d["where"]["turn"], d["where"]["step"]) == ("A", q.global_turn, "DECLARE_BLOCKERS")
                        block_questions += 1
                        # the labels' potential blockers are the creatures the engine lets block;
                        # a blocker flashed in during that turn is not in the state (flagged)
                        can = {f"A:{p['id']}" for p in r["decisionState"]["players"]["A"]["battlefield"]
                               if p.get("id") and (p.get("x") or {}).get("canBlock")}
                        pot = {k for k, _, _ in b.labels["blocks"] if k.startswith("A:")}
                        here = {f"A:{p.id}" for p in b.players["A"].battlefield if p.id}
                        assert can == pot & here, (g.row_index, t, can, pot)
                        assert pot <= here or "offturn_timing_unknown" in b.provenance.flags, (g.row_index, t)
                    else:                                   # no block question: no potential blocker either
                        assert not any(k.startswith("A:") for k, _, _ in b.labels["blocks"]), (g.row_index, t)
    assert n >= 60 and differ <= ENGINE_KNOWN_DIFFS, sorted(differ - ENGINE_KNOWN_DIFFS)
    assert differ_b <= ENGINE_KNOWN_DIFFS, sorted(differ_b - ENGINE_KNOWN_DIFFS)
    assert blocks >= 25 and block_questions >= blocks // 2


# --- regression on the real file ---------------------------------------------------------------------

# replay_empirics.md §4.2 (conservation, % of slots) and §8 (cumulative ladder, % of user turns)
RE_ZONE_STRICT = {"user_hand": 92.7, "user_lands": 98.8, "user_crea": 87.8, "user_nonc": 96.7,
                  "oppo_hand_n": 93.3, "oppo_lands": 98.8, "oppo_crea": 88.0, "oppo_nonc": 96.8}
RE_SLOT = {"strict:user_zones": 80.8, "strict:all_zones": 67.5, "benign:user_zones": 96.9,
           "benign:all_zones": 92.8, "nounexp:user_zones": 99.3, "nounexp:all_zones": 99.0}
RE_LADDER = {"T0": 88.7, "T1": 86.8, "T1b": 59.5, "T2": 43.3, "T3": 36.7}
TOL = 0.75


@pytest.mark.skipif(not REPLAY.exists(), reason="17lands FDN replay file not downloaded")
def test_regression_5000_games(ids):
    from draftzero.gameplay.seventeenlands import collect
    # seeded sample spread over the whole file: every 158th row from row 0 (the research used
    # every 16th; the first rows alone are not representative: early-format games differ ~1 pt)
    st = collect(replay.iter_games(REPLAY, limit=5000, every=158, start=0), ids, build_specs=True)
    r = st.rates()
    assert r["counts"]["games"] == 5000
    # ±0.75 pt: a 5,000-game sample's own noise is about ±0.5 pt (start=157 gives T2 -0.53)
    for z, v in RE_ZONE_STRICT.items():
        assert abs(r["conservation_zone"][f"strict:{z}"] - v) <= TOL, (z, r["conservation_zone"][f"strict:{z}"])
    for k, v in RE_SLOT.items():
        assert abs(r["conservation_slot"][k] - v) <= TOL, (k, r["conservation_slot"][k])
    for k, v in RE_LADDER.items():
        assert abs(r["re_ladder"][k] - v) <= TOL, (k, r["re_ladder"][k])
    assert r["coverage"]["event_card_ids_mapped"] == 100.0
    assert r["coverage"]["ability_ids_in_table"] == 100.0
    assert r["counts"].get("invalid_specs", 0) == 0, r["spec_invalid"]
