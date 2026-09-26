"""Behavioural fingerprints (draftzero.gameplay.fingerprints, experiment E0).

Synthetic tests build 17lands `Game` objects, MageZero JVM-log excerpts and ProbePlayer JSONL by
hand and check every record field and aggregate against values worked out on paper. The fixture
tests run on tests/fixtures/gameplay/fdn_premier_rows.csv.gz (nine 17lands FDN rows, CC BY 4.0,
see the fixture README) and need only committed files; the sample test streams the first rows
of the real replay file and skips without it.
"""
import json
from collections import Counter
from pathlib import Path

import pytest

from draftzero.gameplay import fingerprints as fp
from draftzero.gameplay import labels as lb
from draftzero.gameplay import replay
from draftzero.gameplay.ids import Ids
from draftzero.gameplay.replay import Game, TurnRecord

FIXTURE = Path(__file__).parent / "fixtures" / "gameplay" / "fdn_premier_rows.csv.gz"
REPLAY = replay.replay_path("FDN", "PremierDraft")


@pytest.fixture(scope="module")
def ids():
    return Ids.load()


@pytest.fixture(scope="module")
def G(ids):
    """card name -> FDN grpId"""
    out = {}
    for g, r in ids.cards17.items():
        if r.get("expansion") == "FDN" and r["name"] not in out:
            out[r["name"]] = g
    return out


# --- building blocks ------------------------------------------------------------------------------

def test_payable_is_halls_condition():
    W, U, B = 1, 2, 4
    assert fp.payable((W,), 1, ((W, 1),))
    assert not fp.payable((U,), 1, ((W, 1),))
    assert not fp.payable((W, W), 3, ())                       # not enough lands
    assert fp.payable((W, U), 2, ((W, 1), (U, 1)))
    assert not fp.payable((W | U, W | U), 3, ((W, 1), (U, 1)))  # three mana from two lands
    assert fp.payable((W | U, W | U, 0), 3, ((W, 1), (U, 1)))   # a colourless land pays the generic
    # Hero's Downfall {1}{B}{B}: one dual counts once even though it makes either colour
    assert not fp.payable((B | U, U, U), 3, ((B, 2),))
    assert fp.payable((B | U, B, U), 3, ((B, 2),))


def test_castable_skips_taplands_and_fetches(ids):
    F = fp._facts(ids)
    falls, wilds, forest = F.name("Thornwood Falls"), F.name("Evolving Wilds"), F.name("Forest")
    elves = F.name("Llanowar Elves")                            # {G}
    assert falls.etb_tapped and falls.mana_land and not wilds.mana_land
    assert fp._castable([elves], [falls]) == (True, True)
    assert fp._castable([elves], [falls], played=[falls]) == (False, False)   # entered tapped this turn
    assert fp._castable([elves], [wilds]) == (False, False)                   # makes no mana
    assert fp._castable([elves], [forest, falls], played=[falls]) == (True, True)
    assert fp._castable([F.name("Hero's Downfall")], [forest] * 3) == (True, False)


def test_skill_groups():
    g = lambda b, n: fp.skill_groups({"user_game_win_rate_bucket": b, "user_n_games_bucket": n})
    assert g(0.48, 100) == ["all", "lt50", "lt50_n100"]
    assert g(0.48, 50) == ["all", "lt50"]
    assert g(0.5, 10) == ["all", "50to58"]
    assert g(0.58, 10) == ["all", "ge58"]                        # float buckets: 0.58 is ge58
    assert g(0.6, 100) == ["all", "ge58", "top"]
    assert g(0.6, 50) == ["all", "ge58"]                         # low-n users: the bucket leaks
    assert g(0.7000000000000001, 1000) == ["all", "ge58", "top"]
    assert fp.skill_groups({"user_game_win_rate_bucket": None}) == ["all"]


# --- a synthetic 17lands game --------------------------------------------------------------------------

def _slot(side, n, seq, **f):
    return TurnRecord(side, n, seq=seq, global_turn=seq + 1, f=f, played=True)


def synthetic_game(G) -> Game:
    """User on the play, four turns each; the opponent's last turn ends the game.

    U1 Plains, holds Savannah Lions (castable, idle) | O1 -
    U2 Island, Savannah Lions                        | O2 user Think Twice; Cavalry attacks, Lions blocks
    U3 Plains, Drake Hatcher; Lions attacks          | O3 two Cavalry attack, Hatcher blocks (Lions is
                                                     |    tapped); the opponent's Pacifism enters
    U4 no land (Forest and Island in hand), Gleaming Barrier; Hatcher removed before combat and
       Lions pacified: nothing could attack           | O4 (terminal)

    Pacifism is not in FDN boosters, so the committed id table (Ids without data/17lands/cards.csv)
    lacks it; the Lions are then removed before combat too, which gives the same records.
    """
    P, I, F_ = G["Plains"], G["Island"], G["Forest"]
    lions, tt, hd, gb, dh = (G["Savannah Lions"], G["Think Twice"], G["Hero's Downfall"],
                             G["Gleaming Barrier"], G["Drake Hatcher"])
    cav, pac = G["Axgard Cavalry"], G.get("Pacifism")
    u4_gone = [dh] if pac else [dh, lions]
    turns = [
        _slot("user", 1, 0, lands_played=[P], eot_user_lands_in_play=[P],
              eot_user_cards_in_hand=[P, I, lions, tt, hd, gb], user_mana_spent=0.0),
        _slot("oppo", 1, 1, eot_user_lands_in_play=[P], eot_user_cards_in_hand=[P, I, lions, tt, hd, gb]),
        _slot("user", 2, 2, cards_drawn=[dh], lands_played=[I], creatures_cast=[lions], user_mana_spent=1.0,
              eot_user_lands_in_play=[P, I], eot_user_creatures_in_play=[lions],
              eot_user_cards_in_hand=[P, tt, hd, gb, dh]),
        _slot("oppo", 2, 3, user_instants_sorceries_cast=[tt], creatures_attacked=[cav], creatures_blocking=[lions],
              eot_user_lands_in_play=[P, I], eot_user_creatures_in_play=[lions],
              eot_user_cards_in_hand=[P, hd, gb, dh]),
        _slot("user", 3, 4, cards_drawn=[F_], lands_played=[P], creatures_cast=[dh], creatures_attacked=[lions],
              user_mana_spent=2.0, eot_user_lands_in_play=[P, I, P], eot_user_creatures_in_play=[lions, dh],
              eot_user_cards_in_hand=[hd, gb, F_]),
        _slot("oppo", 3, 5, creatures_attacked=[cav, cav], creatures_blocking=[dh],
              eot_user_lands_in_play=[P, I, P], eot_user_creatures_in_play=[lions, dh],
              eot_oppo_non_creatures_in_play=[pac] if pac else [], eot_user_cards_in_hand=[hd, gb, F_]),
        _slot("user", 4, 6, cards_drawn=[I], creatures_cast=[gb], user_creatures_killed_non_combat=u4_gone,
              user_mana_spent=2.0, eot_user_lands_in_play=[P, I, P],
              eot_user_creatures_in_play=[lions, gb] if pac else [gb], eot_user_cards_in_hand=[hd, F_, I]),
        _slot("oppo", 4, 7, creatures_attacked=[cav], eot_user_lands_in_play=[P, I, P]),
    ]
    turns[-1].terminal = True
    meta = {"on_play": True, "won": False, "num_mulligans": 0, "user_game_win_rate_bucket": 0.62,
            "user_n_games_bucket": 100}
    return Game(7, meta, Counter(), Counter(), [], [P, P, I, lions, tt, hd, gb], [], True, turns)


def test_records_from_synthetic_game(ids, G):
    recs = fp.records_from_game(synthetic_game(G), ids)
    assert [(r["side"], r["turn"]) for r in recs] == [("me", 1), ("opp", 1), ("me", 2), ("opp", 2),
                                                      ("me", 3), ("opp", 3), ("me", 4), ("opp", 4)]
    u1, o1, u2, o2, u3, o3, u4, o4 = recs
    assert u1["mulligans"] == 0 and u1["won"] is False and o4["terminal"]
    assert (u1["lands_played"], u1["creatures_cast"], u1["castable"], u1["castable_mv"]) == (1, 0, True, True)
    assert (u2["creatures_cast"], u2["cast_mv"], u2["mana_spent"], u2["lands_in_play"]) == (1, 1, 1, 2)
    assert (o1["opp_attackers"], o1["potential_blockers"]) == (0, 0)
    assert (o2["opp_attackers"], o2["blockers"], o2["potential_blockers"], o2["instants_cast"]) == (1, 1, 1, 1)
    assert (u3["attack_eligible"], u3["attackers"], u3["attackers_new"]) == (1, 1, 0)
    # the Lions attacked on U3 and stay tapped through O3: only the Hatcher (vigilance anyway) can block
    assert (o3["opp_attackers"], o3["blockers"], o3["potential_blockers"]) == (2, 1, 1)
    # U4: Barrier has Defender, Hatcher died before combat, Pacifism holds the Lions
    assert (u4["attack_eligible"], u4["attackers"]) == (0, 0)
    assert (u4["lands_played"], u4["land_in_hand"]) == (0, True)
    # U3's hand held Hero's Downfall ({1}{B}{B}: no black source) and Gleaming Barrier ({2}): castable
    assert u3["castable"] and u3["castable_mv"]


def test_fingerprint_of_synthetic_game(ids, G):
    s = fp.fingerprint_from_records([fp.records_from_game(synthetic_game(G), ids)])
    a, bt = s["all"], s["all"]["by_turn"]
    assert s["play"]["games"] == 1 and s["draw"]["games"] == 0 and a["games"] == 1
    assert (a["turns_global_mean"], a["own_turns_mean"], a["spells_per_game"]) == (8, 4, 3)
    assert a["win_rate"] == 0 and a["mulligan_rate"] == 0
    assert a["first_spell_turn"] == {"median": 2, "mean": 2.0, "never": 0.0, "hist": {"2": 1}}
    assert bt["land_drop"] == {"1": 1.0, "2": 1.0, "3": 1.0, "4": 0.0, "all": 0.75}
    assert bt["land_miss"]["all"] == 0.25
    assert bt["spells"]["all"] == 0.75 and bt["spells"]["1"] == 0.0
    assert bt["idle"] == {"1": 1.0, "2": 0.0, "3": 0.0, "4": 0.0, "all": 0.25}
    assert bt["attack_rate"] == {"3": 1.0, "all": 1.0}              # U4 had nothing eligible
    assert bt["opp_blockers_per_attacker"]["all"] == round(2 / 3, 4)   # O4 is terminal: skipped
    assert bt["opp_block_any"]["all"] == 1.0 and bt["opp_block_rate"]["all"] == 1.0
    assert bt["opp_instants"]["all"] == round(1 / 3, 4)
    assert a["offturn_instants_per_game"] == 1
    assert a["cast_mix"]["creature"] == 0.75 and a["cast_mix"]["offturn_instant"] == 0.25
    assert bt["mana_eff"]["2"] == 0.5 and bt["cast_mv_eff"]["3"] == round(2 / 3, 4)
    assert s["all"]["n"]["spells"]["all"] == 4


def test_fingerprint_from_flat_records_and_idle_off_turn():
    """A flat record stream, split into games by `game`; only an instant on the FOLLOWING
    opponent turn keeps an own turn from being idle."""
    held = {"lands_played": 1, "creatures_cast": 0, "castable": True}
    recs = [
        {"game": 0, "side": "me", "turn": 1, "on_play": True, **held},
        {"game": 0, "side": "opp", "turn": 1, "on_play": True, "instants_cast": 1},
        {"game": 0, "side": "me", "turn": 2, "on_play": True, "terminal": True},
        {"game": 1, "side": "opp", "turn": 1, "on_play": False, "instants_cast": 1},
        {"game": 1, "side": "me", "turn": 1, "on_play": False, **held},
        {"game": 1, "side": "opp", "turn": 2, "on_play": False, "instants_cast": 0, "terminal": True},
    ]
    s = fp.fingerprint_from_records(recs)
    assert s["all"]["games"] == 2 and s["play"]["games"] == 1 and s["draw"]["games"] == 1
    assert s["play"]["by_turn"]["idle"]["1"] == 0.0          # held the card for the opponent's turn
    assert s["draw"]["by_turn"]["idle"]["1"] == 1.0          # its instant came BEFORE its own turn
    assert s["play"]["first_spell_turn"]["never"] == 1.0
    assert s["play"]["turns_global_mean"] == 3 and s["draw"]["turns_global_mean"] == 3
    assert s["all"]["offturn_instants_per_game"] == 1.0


def test_unknown_fields_are_skipped():
    recs = [{"game": 1, "side": "me", "turn": 1, "on_play": True},
            {"game": 1, "side": "opp", "turn": 1, "on_play": True, "terminal": True}]
    a = fp.fingerprint_from_records(recs)["all"]
    assert a["games"] == 1 and a["turns_global_mean"] == 2
    assert a["by_turn"] == {} and a["spells_per_game"] is None and a["mulligan_rate"] is None


def flash_game(G) -> Game:
    """User on the draw; 17lands lists no cast for a flash permanent on the opponent's turn.

    O1 -                            | U1 Island; holds Spectral Sailor ({U}, flash): castable
    O2 the user flashes in a Sailor | U2 Forest; the Sailor attacks
    O3 Cavalry attacks; a second Sailor flashes in, blocks and dies | U3 (terminal)
    """
    I, P, F_ = G["Island"], G["Plains"], G["Forest"]
    ss, aw, cav = G["Spectral Sailor"], G["Ambush Wolf"], G["Axgard Cavalry"]
    turns = [
        _slot("oppo", 1, 0, eot_user_cards_in_hand=[I, I, F_, ss, ss, aw, P]),
        _slot("user", 1, 1, cards_drawn=[P], lands_played=[I], eot_user_lands_in_play=[I],
              eot_user_cards_in_hand=[I, F_, ss, ss, aw, P, P]),
        _slot("oppo", 2, 2, user_mana_spent=1.0, eot_user_lands_in_play=[I], eot_user_creatures_in_play=[ss],
              eot_user_cards_in_hand=[I, F_, ss, aw, P, P]),
        _slot("user", 2, 3, cards_drawn=[F_], lands_played=[F_], creatures_attacked=[ss],
              eot_user_lands_in_play=[I, F_], eot_user_creatures_in_play=[ss],
              eot_user_cards_in_hand=[I, ss, aw, P, P, F_]),
        _slot("oppo", 3, 4, creatures_attacked=[cav], creatures_blocking=[ss], user_creatures_killed_combat=[ss],
              user_mana_spent=1.0, eot_user_lands_in_play=[I, F_], eot_user_creatures_in_play=[ss],
              eot_user_cards_in_hand=[I, aw, P, P, F_]),
        _slot("user", 3, 5, cards_drawn=[I], lands_played=[I], eot_user_lands_in_play=[I, F_, I],
              eot_user_creatures_in_play=[ss], eot_user_cards_in_hand=[aw, P, P, F_, I]),
    ]
    turns[-1].terminal = True
    meta = {"on_play": False, "won": True, "num_mulligans": 0}
    return Game(3, meta, Counter(), Counter(), [], [I, I, F_, ss, ss, aw, P], [], True, turns)


def test_flash_casts_on_opponent_turns(ids, G):
    """Regression: the human flash casts come from the snapshots, so holding a flash creature for
    the opponent's turn is not "idle" (MageZero logs count flash casts the same way)."""
    recs = fp.records_from_game(flash_game(G), ids)
    assert [(r["side"], r["turn"]) for r in recs] == [("opp", 1), ("me", 1), ("opp", 2), ("me", 2),
                                                      ("opp", 3), ("me", 3)]
    o1, u1, o2, u2, o3, u3 = recs
    assert o1["flash_cast"] == 0 and u1["castable"] and u1["creatures_cast"] == 0
    assert (o2["flash_cast"], o2["instants_cast"]) == (1, 0)
    assert (u2["attack_eligible"], u2["attackers"]) == (1, 1)
    # the second Sailor never shows on an end-of-turn battlefield, only among the combat deaths;
    # the first attacked on U2 and is still tapped
    assert (o3["flash_cast"], o3["opp_attackers"], o3["blockers"], o3["potential_blockers"]) == (1, 1, 1, 0)
    s = fp.fingerprint_from_records([recs])["all"]
    assert s["by_turn"]["idle"] == {"1": 0.0, "2": 0.0, "all": 0.0}
    assert s["offturn_casts_per_game"] == 2 and s["offturn_instants_per_game"] == 0
    assert s["cast_mix"]["offturn_flash"] == 1.0 and s["turns_global_mean"] == 6


def test_mulligan_bottom_hole_slot_and_missed_land(ids, G):
    """On the play after one mulligan: turn 1's hand is the kept 7 minus the bottomed Savannah
    Lions, so nothing is castable off one Plains. O1 is an empty "hole" slot (0.12% of games), so
    U2 starts from U1's snapshot. U2 plays an Island the hand model missed: a land was held."""
    P, I = G["Plains"], G["Island"]
    lions, hd, gb, dh, tt = (G["Savannah Lions"], G["Hero's Downfall"], G["Gleaming Barrier"],
                             G["Drake Hatcher"], G["Think Twice"])
    turns = [
        _slot("user", 1, 0, lands_played=[P], eot_user_lands_in_play=[P], eot_user_cards_in_hand=[hd, gb, dh, tt, tt]),
        TurnRecord("oppo", 1, seq=1, global_turn=2, f={}, played=False),
        _slot("user", 2, 2, cards_drawn=[gb], lands_played=[I], creatures_cast=[dh], user_mana_spent=2.0,
              eot_user_lands_in_play=[P, I], eot_user_creatures_in_play=[dh], eot_user_cards_in_hand=[hd, gb, tt, tt, gb]),
        _slot("oppo", 2, 3, eot_user_lands_in_play=[P, I], eot_user_creatures_in_play=[dh],
              eot_user_cards_in_hand=[hd, gb, tt, tt, gb]),
    ]
    turns[-1].terminal = True
    meta = {"on_play": True, "won": False, "num_mulligans": 1}
    g = Game(5, meta, Counter(), Counter(), [], [P, lions, hd, gb, dh, tt, tt], [], True, turns)
    g.bottomed, g.bottomed_exact = replay.infer_bottomed(g)
    assert g.bottomed == [lions] and g.bottomed_exact
    recs = fp.records_from_game(g, ids)
    assert [(r["side"], r["turn"]) for r in recs] == [("me", 1), ("me", 2), ("opp", 2)]
    u1, u2, o2 = recs
    assert (u1["castable_mv"], u1["castable"], u1["mulligans"]) == (False, False, 1)
    assert u2["land_in_hand"] and u2["castable"] and o2["potential_blockers"] == 1
    s = fp.fingerprint_from_records([recs])["all"]
    assert s["turns_global_mean"] == 4 and s["own_turns_mean"] == 2 and s["mulligan_rate"] == 1
    assert s["by_turn"]["land_miss"] == {"1": 0.0, "2": 0.0, "all": 0.0}
    assert s["by_turn"]["idle"] == {"1": 0.0, "2": 0.0, "all": 0.0}


# --- MageZero logs --------------------------------------------------------------------------------------

def _log(lines, thread="pool-3-thread-1"):
    return "".join(f"INFO  2026-09-26 00:30:00,{i % 1000:03d} {m:<80} =>[{thread}] X.y \n" for i, m in enumerate(lines))


JVM_GAME = [
    "Player A won the die roll",
    "[1:Beginning:UPKEEP]PlayerA hand: : Plains,Plains,Savannah Lions,Balmor, Battlemage Captain,",
    "[1:Beginning:UPKEEP]PlayerB hand: : Island,Island,Think Twice,Brineborn Cutthroat,",
    "Player: PlayerA simulated 10 evaluations in 0.1 seconds - nodes in tree: 5",
    "PRECOMBAT_MAIN0pool= actions: [Pass score: 0.1 count: 3] [Play Plains score: 0.2 count: 7]",
    "playable abilities: [Play Plains, Pass]",
    "[1:Precombat Main:PRECOMBAT_MAIN]chose action:Play Plains success ratio: 0.2",
    "[PlayerA], life = 20",
    "Player: PlayerA simulated 10 evaluations in 0.1 seconds - nodes in tree: 5",
    "POSTCOMBAT_MAIN1 (top: stack ability (When {this} enters, draw a card.))pool= actions: "
    "[Pass score: 0.3 count: 9] [Cast Savannah Lions score: 0.1 count: 1]",
    "auto pass",
    "[2:Beginning:UPKEEP]PlayerB hand: : Island,Island,Think Twice,Brineborn Cutthroat,Island,",
    "[2:Beginning:UPKEEP]PlayerA hand: : Plains,Savannah Lions,Balmor, Battlemage Captain,",
    "[2:Precombat Main:PRECOMBAT_MAIN]chose action:Play Island success ratio: 0.2",
    "[PlayerB], life = 20",
    "[3:Beginning:UPKEEP]PlayerA hand: : Plains,Savannah Lions,Balmor, Battlemage Captain,Plains,",
    "[3:Precombat Main:PRECOMBAT_MAIN]chose action:Play Plains success ratio: 0.2",
    "[PlayerA], life = 20",
    "[3:Precombat Main:PRECOMBAT_MAIN]chose action:Cast Savannah Lions success ratio: 0.2",
    "[PlayerA], life = 20",
    "[4:Beginning:UPKEEP]PlayerB hand: : Island,Think Twice,Brineborn Cutthroat,Island,",
    "[4:Precombat Main:PRECOMBAT_MAIN]chose action:Play Island success ratio: 0.2",
    "[PlayerB], life = 20",
    "[4:Precombat Main:PRECOMBAT_MAIN]chose action:Cast Brineborn Cutthroat success ratio: 0.2",
    "[PlayerB], life = 20",
    "[4:End Turn:END_TURN]chose action:Cast Savannah Lions success ratio: 0.2",   # (a flash stand-in)
    "[PlayerA], life = 20",
    "[5:Beginning:UPKEEP]PlayerA hand: : Plains,Balmor, Battlemage Captain,",
    "base choose use attack with: Savannah Lions?",
    "Player: PlayerA simulated 2 evaluations in 0.1 seconds - nodes in tree: 5",
    "DECLARE_ATTACKERS0pool= actions: [false score: 0.5 count: 7] [true score: 0.4 count: 3]",
    "use attack with: Savannah Lions?: false",
    "[6:Beginning:UPKEEP]PlayerB hand: : Think Twice,Island,",
    "base choose use attack with: Brineborn Cutthroat?",
    "Player: PlayerB simulated 2 evaluations in 0.1 seconds - nodes in tree: 5",
    "use attack with: Brineborn Cutthroat?: true",
    "base choose target choose which creature to block for Savannah Lions",
    "possible targets: 1",
    "Player: PlayerA simulated 1 evaluations in 0.1 seconds - nodes in tree: 5",
    "DECLARE_BLOCKERS0pool= actions: [Stop Choosing score: -0.5 count: 2] [Brineborn Cutthroat score: -0.6 count: 1]",
    "Targeting Brineborn Cutthroat",
    'GAME_SUMMARY {"game":1,"seed":1,"first":"A","winner":"B","turns":6}',
]


@pytest.fixture
def jvm_log(tmp_path):
    p = tmp_path / "jvm_0.log"
    failed = ["Player B won the die roll", "[1:Precombat Main:PRECOMBAT_MAIN]chose action:Play Island success ratio: 1",
              "[PlayerB], life = 20", "A game simulation failed"]
    text = _log(JVM_GAME[:10]) + _log(failed, "pool-3-thread-2") + _log(JVM_GAME[10:])
    p.write_text("WARNING: a JVM banner line\n" + text)
    return p


def test_records_from_jvm_log(jvm_log, ids):
    assert fp.detect_format(jvm_log) == "jvm"
    games = list(fp.records_from_jvm_log(jvm_log, ids))
    assert len(games) == 2                                    # one finished game, two seats; the failed one dropped
    a, b = games
    assert [(r["side"], r["turn"]) for r in a] == [("me", 1), ("opp", 1), ("me", 2), ("opp", 2), ("me", 3), ("opp", 3)]
    assert a[0]["on_play"] and not b[0]["on_play"] and a[0]["won"] is False and b[0]["won"] is True
    a1, a2 = a[0], a[2]
    assert (a1["lands_played"], a1["creatures_cast"], a1["castable_engine"]) == (1, 0, True)
    # the upkeep hand splits "Balmor, Battlemage Captain" correctly and Lions was castable off one Plains
    assert a1["castable"] and a1["land_in_hand"]
    assert (a2["lands_played"], a2["creatures_cast"], a2["cast_mv"], a2["lands_in_play"]) == (2 - 1, 1, 1, 2)
    assert a[3]["flash_cast"] == 1 and a[3]["instants_cast"] == 0      # a creature on B's turn
    assert (a[4]["attack_eligible"], a[4]["attackers"]) == (1, 0)
    assert (a[5]["opp_attackers"], a[5]["blockers"], a[5]["potential_blockers"], a[5]["terminal"]) == (1, 1, 1, True)
    assert (b[3]["creatures_cast"], b[3]["lands_played"]) == (1, 1)
    assert (b[5]["attack_eligible"], b[5]["attackers"]) == (1, 1)
    s = fp.fingerprint_from_records(games)["all"]
    assert s["games"] == 2 and s["mulligan_rate"] is None and s["turns_global_mean"] == 6
    assert s["by_turn"]["idle_engine"]["1"] == 0.5              # A passed with Lions castable; B had nothing


def test_records_from_probe(tmp_path, ids):
    rows = [
        {"tag": "g0", "decision_no": 1, "turn": 1, "step": "PRECOMBAT_MAIN", "active": True, "decision": "PRIORITY",
         "hand": ["Plains", "Savannah Lions"], "children": [{"action": "Pass"}, {"action": "Play Plains"}],
         "played": "Play Plains"},
        {"tag": "g0", "decision_no": 2, "turn": 1, "step": "PRECOMBAT_MAIN", "active": True, "decision": "PRIORITY",
         "hand": ["Savannah Lions"], "children": [{"action": "Pass"}, {"action": "Cast Savannah Lions"}],
         "played": "Cast Savannah Lions"},
        {"tag": "g0", "decision_no": 3, "turn": 2, "step": "DECLARE_BLOCKERS", "active": False,
         "decision": "CHOOSE_TARGET", "message": "choose which creature to block for Savannah Lions :: attacking creature",
         "hand": [], "children": [{"action": "Stop Choosing"}, {"action": "Goblin Token"}, {"action": "Axgard Cavalry"}],
         "played": "Stop Choosing"},
        {"tag": "g0", "game_over": True, "winner": "A", "turns": 4},
    ]
    p = tmp_path / "probe.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert fp.detect_format(p) == "probe"
    (recs,) = list(fp.records_from_probe(p, ids))
    assert [(r["side"], r["turn"]) for r in recs] == [("me", 1), ("opp", 1), ("me", 2), ("opp", 2)]
    assert recs[0]["won"] is True and recs[0]["creatures_cast"] == 1 and recs[0]["castable"]
    assert (recs[1]["opp_attackers"], recs[1]["blockers"], recs[1]["potential_blockers"]) == (2, 0, 1)
    # own turn 2: no logged search, so XMage offered nothing; the opponent's turn 2 is unknown
    assert recs[2]["castable"] is False and recs[2]["castable_mv"] is None and recs[2]["lands_played"] == 0
    assert recs[3]["opp_attackers"] is None


def test_probe_rejects_interleaved_games(tmp_path, ids):
    """Regression: research/out/selfplay.jsonl interleaves concurrent JVMs whose tags repeat; reading it
    as one game per game_over line silently mixed games (wrong first player, empty games)."""
    dec = lambda no, turn, played: {"tag": "selfplay_g0", "decision_no": no, "turn": turn, "step": "PRECOMBAT_MAIN",
                                    "active": True, "decision": "PRIORITY", "hand": ["Plains"],
                                    "children": [{"action": "Pass"}, {"action": "Play Plains"}], "played": played}
    rows = [dec(1, 1, "Play Plains"), dec(1, 2, "Play Plains"), dec(2, 3, "Pass"),
            {"tag": "selfplay_g0", "game_over": True, "winner": "A", "turns": 3}]
    p = tmp_path / "interleaved.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="interleaves"):
        list(fp.records_from_probe(p, ids))


def test_records_from_game_summaries(tmp_path):
    p = tmp_path / "games.jsonl"
    p.write_text(json.dumps({"kind": "selfplay", "first": "B", "winner": "A", "turns": 5}) + "\n"
                 + 'INFO  x GAME_SUMMARY {"game":2,"first":"A","winner":"draw","turns":4} =>[pool-1] P.q \n')
    games = list(fp.records_from_game_summaries(p, seats="AB"))
    assert len(games) == 4
    a1 = games[0]
    assert [(r["side"], r["turn"]) for r in a1] == [("opp", 1), ("me", 1), ("opp", 2), ("me", 2), ("opp", 3)]
    assert a1[0]["won"] is True and a1[-1]["terminal"] and games[2][0]["won"] is None
    s = fp.fingerprint_from_records(games)["all"]
    assert s["games"] == 4 and s["turns_global_mean"] == 4.5 and s["by_turn"] == {}


# --- 17lands fixture and sample ----------------------------------------------------------------------------

def test_fixture_attack_eligibility_matches_labels(ids):
    """The fast eligibility agrees with labels.turn_label (full reconstruction) on every fixture
    turn. Labels used to exclude every creature under a "hostile" Aura, and disagreed on row 4's
    Drake Hatcher under Witness Protection (a plain 1/1 that attacked); they now exclude only
    Pacifism-like and freezing Auras' hosts."""
    agree = total = 0
    for g in replay.iter_games(FIXTURE):
        for r in fp.records_from_game(g, ids):
            if r["side"] != "me" or r["terminal"]:
                continue
            att = {k: v for k, v in lb.turn_label(g, r["turn"], ids)["attacks"].items() if k.startswith("A:")}
            total += 1
            agree += (len(att), sum(att.values())) == (r["attack_eligible"], r["attackers"])
    assert total == 71 and agree == 71


def test_fixture_fingerprint(ids):
    games = list(replay.iter_games(FIXTURE))
    s = fp.fingerprint_from_records(fp.records_from_game(g, ids) for g in games)
    a = s["all"]
    assert a["games"] == 9 and s["play"]["games"] + s["draw"]["games"] == 9
    # totals equal the raw columns: every user cast on an own turn is counted once
    own = sum(len(t.L("creatures_cast")) + len(t.L("non_creatures_cast")) + len(t.L("user_instants_sorceries_cast"))
              for g in games for t in g.turns if t.side == "user")
    assert round(a["spells_per_game"] * 9) == own
    assert a["mulligan_rate"] == round(sum(g.meta["num_mulligans"] > 0 for g in games) / 9, 4)
    assert a["by_turn"]["land_drop"]["1"] == 1.0
    assert 0 < a["by_turn"]["attack_rate"]["all"] <= 1


@pytest.mark.skipif(not REPLAY.exists(), reason="17lands FDN replay file not downloaded")
def test_real_sample_is_plausible(ids):
    out = fp.human_fingerprints(limit=400, ids=ids)
    assert out["meta"]["games"] == 400
    a = out["groups"]["all"]["all"]
    bt = a["by_turn"]
    assert 6.0 < a["spells_per_game"] < 8.5                 # RE §2: 7.31 own-turn casts per game
    assert 0.08 < a["mulligan_rate"] < 0.17                 # RE §2: 12.0%
    assert bt["land_drop"]["1"] > 0.9 and bt["spells"]["1"] < bt["spells"]["3"]
    assert 0.3 < bt["attack_rate"]["all"] < 0.7 and a["first_spell_turn"]["median"] in (2, 3)
    # RE §2: 0.90 instants and 0.28 inferred flash permanents per game on opponent turns
    assert 0.6 < a["offturn_instants_per_game"] < 1.2
    assert 0.15 < a["offturn_casts_per_game"] - a["offturn_instants_per_game"] < 0.5
    assert sum(out["groups"][g]["all"]["games"] for g in ("lt50", "50to58", "ge58")) <= 400


def test_magezero_fingerprint_counts_games_per_file(tmp_path, jvm_log, ids, capsys):
    """A log that gives no finished game (a WARN-level CoachProbe log, a run killed before its
    first GAME_SUMMARY) is reported, not silently counted as zero games."""
    warn = tmp_path / "selfplay_x.log"
    warn.write_text("WARN  MCTSNode FOUND DUPLICATE ACTION Play Swamp\n")
    out = fp.magezero_fingerprint([str(jvm_log), str(warn)], ids)
    assert [(f["format"], f["games"]) for f in out["files"]] == [("jvm", 2), ("jvm", 0)]
    assert "gave no finished games" in capsys.readouterr().err and out["all"]["games"] == 2


def test_main_writes_json_and_tables(tmp_path, jvm_log, capsys):
    out, mz = tmp_path / "fp.json", tmp_path / "mz.json"
    assert fp.main(["--path", str(FIXTURE), "--out", str(out)]) == 0
    human = json.loads(out.read_text())
    assert human["meta"]["games"] == 9 and set(human["groups"]) == set(fp.GROUPS)
    assert "| spells per own turn |" in capsys.readouterr().out
    assert fp.main(["--human", str(out), "--mz", f"pilot={jvm_log}", "--mz-out", str(mz)]) == 0
    table = capsys.readouterr().out
    assert "MZ pilot" in table and "human all" in table
    assert json.loads(mz.read_text())["magezero"]["pilot"]["all"]["games"] == 2
