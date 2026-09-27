"""Arena Player.log parser: log reading, state accumulation, request/response pairing, the
StateSpec it writes, and the privacy scrub.

Most tests run on tests/fixtures/gameplay/arena_synthetic.log, a hand-built log with fake names and
ids that mimics the real structure (see tests/fixtures/gameplay/README.md). Tests that need the
17lands card table, Arena's card database or a real Player.log skip when those are absent.
"""
import collections
import json
import re
from pathlib import Path

import pytest

from draftzero.gameplay import arena, bridge
from draftzero.gameplay.statespec import PHASE_STEPS, StateSpec

FIXTURE = Path(__file__).parent / "fixtures" / "gameplay" / "arena_synthetic.log"
REAL_LOG = Path.home() / "Library/Logs/Wizards Of The Coast/MTGA/Player-prev.log"

# the fixture's cards (17lands card ids = Arena grpIds; CC BY 4.0 17lands data)
CARDS = {95192: ("Plains", "FDN"), 95199: ("Forest", "FDN"), 93859: ("Savannah Lions", "FDN"),
         93650: ("Pacifism", "FDN"), 93725: ("Felidar Savior", "FDN"), 93741: ("Vanguard Seraph", "FDN"),
         93734: ("Prideful Parent", "FDN"), 93781: ("Sanguine Syphoner", "FDN"), 93784: ("Stab", "FDN"),
         95195: ("Swamp", "FDN")}
ABILITIES = {1001: "{oT}: Add {oW}.",
             175773: "When CARDNAME enters, create a 1/1 white Cat creature token."}


def names(token_classes=None) -> arena.CardNames:
    cards = {g: arena.CardInfo(g, n, s) for g, (n, s) in CARDS.items()}
    cards[94156] = arena.CardInfo(94156, "Cat", "FDN", token=True, front=False)
    return arena.CardNames(cards, ABILITIES, token_classes=token_classes,
                           vocab=arena.read_vocab(arena.VOCAB_TSV))


@pytest.fixture(scope="module")
def parsed():
    return arena.parse_logs([FIXTURE], names())


def decision(parser, game, index):
    return parser.games[game - 1].decisions[index - 1].spec


def perm(spec, seat, alias):
    return next(p for p in spec.players[seat].battlefield if p.id == alias)


# --- log reading ---------------------------------------------------------------------------------
def test_reads_both_directions_and_the_summarized_event():
    msgs = list(arena.iter_messages(FIXTURE))
    kinds = collections.Counter(m.kind for m in msgs if m.obj is not None)
    assert kinds["GreToClientEvent"] == 26 and kinds["ClientToGremessage"] == 17   # + 1 summarized event
    assert kinds["AuthenticateResponse"] == 1 and kinds["ClientToGreuimessage"] == 1
    # client->GRE JSON is pretty-printed over many lines and still parses whole
    client = [m.obj for m in msgs if m.kind == "ClientToGremessage"]
    assert all("payload" in c for c in client)
    summaries = [m for m in msgs if m.summary is not None]
    assert len(summaries) == 1 and "ActionsAvailableReq" in summaries[0].summary


@pytest.mark.parametrize("stamp", ["9/1/2026 1:02:03 PM: ", "2026-09-01 13:02:03: ", "01.09.2026 13:02:03: ",
                                   "1/9/2026 13:02:03: ", ""])
def test_header_accepts_locale_timestamps(stamp):
    m = arena.HEADER.match(f"[UnityCrossThreadLogger]{stamp}Match to ABC123DEF: GreToClientEvent")
    assert m and m.group("kind") == "GreToClientEvent" and m.group("dir").startswith("Match to")
    m = arena.HEADER.match(f"[UnityCrossThreadLogger]{stamp}ABC123DEF to Match: ClientToGremessage")
    assert m and m.group("kind") == "ClientToGremessage"


def test_log_info(tmp_path):
    info = arena.log_info(FIXTURE)
    assert info == {"detailed_logs": True, "client_version": "2026.99.10", "grp_build": 999}
    off = tmp_path / "Player.log"
    off.write_text("boot\nDETAILED LOGS: DISABLED\n")
    assert arena.log_info(off)["detailed_logs"] is False


def test_timestamps_in_both_units():
    assert arena.gre_ms("1790000000000") == 1790000000000                 # GRE: epoch ms
    assert arena.gre_ms(str(1790000000000 * 10000 + 621355968000000000)) == 1790000000000  # client: ticks
    assert arena.gre_ms(None) is None


# --- state accumulation ----------------------------------------------------------------------------
def _gsm(gsid, prev=None, full=False, **kw):
    return dict(kw, type="GameStateType_Full" if full else "GameStateType_Diff", gameStateId=gsid,
                prevGameStateId=prev if prev is not None else gsid - 1)


def test_full_then_diffs_replace_whole_objects_and_delete():
    st = arena.GameState()
    st.apply(_gsm(1, full=True, zones=[{"zoneId": 28, "type": "ZoneType_Battlefield", "objectInstanceIds": [5, 6]}],
                  gameObjects=[{"instanceId": 5, "grpId": 1, "isTapped": True}, {"instanceId": 6, "grpId": 2}],
                  players=[{"systemSeatNumber": 1, "lifeTotal": 20}, {"systemSeatNumber": 2, "lifeTotal": 20}],
                  turnInfo={"turnNumber": 3, "phase": "Phase_Combat", "step": "Step_DeclareAttack"},
                  persistentAnnotations=[{"id": 9, "type": ["AnnotationType_Counter"], "affectedIds": [6],
                                          "details": [{"key": "count", "valueInt32": [2]},
                                                      {"key": "counter_type", "valueInt32": [1]}]}]))
    assert st.objects[5]["isTapped"] and st.counters() == {6: {1: 2}}
    st.apply(_gsm(2, gameObjects=[{"instanceId": 5, "grpId": 1}],            # re-sent without isTapped
                  diffDeletedInstanceIds=[6], diffDeletedPersistentAnnotationIds=[9],
                  players=[{"systemSeatNumber": 2, "lifeTotal": 17}],
                  turnInfo={"turnNumber": 3, "phase": "Phase_Main2"}))    # no step in a main phase
    assert not st.objects[5].get("isTapped") and 6 not in st.objects
    assert st.counters() == {} and st.players[1]["lifeTotal"] == 20 and st.players[2]["lifeTotal"] == 17
    assert "step" not in st.turn                                          # replaced, not merged
    assert st.gaps == []
    assert not st.apply(_gsm(2)) and st.n_skipped == 1                     # a re-send is skipped


def test_lineage_through_zone_changes_and_shuffles():
    st = arena.GameState()
    st.apply(_gsm(1, full=True))
    st.apply(_gsm(2, annotations=[
        {"id": 1, "type": ["AnnotationType_ObjectIdChanged"],
         "details": [{"key": "orig_id", "valueInt32": [10]}, {"key": "new_id", "valueInt32": [20]}]},
        {"id": 2, "type": ["AnnotationType_Shuffle"],
         "details": [{"key": "OldIds", "valueInt32": [7, 8]}, {"key": "NewIds", "valueInt32": [30, 31]}]}]))
    st.apply(_gsm(3, annotations=[{"id": 3, "type": ["AnnotationType_ObjectIdChanged"],
                                   "details": [{"key": "orig_id", "valueInt32": [20]},
                                               {"key": "new_id", "valueInt32": [40]}]}]))
    assert st.root(40) == 10 and st.root(31) == 8 and st.root(99) == 99


def test_chain_gap_marks_state_stale_until_resent():
    st = arena.GameState()
    st.apply(_gsm(1, full=True, zones=[{"zoneId": 28, "type": "ZoneType_Battlefield", "objectInstanceIds": [5]}],
                  gameObjects=[{"instanceId": 5, "grpId": 1}]))
    st.apply(_gsm(3, prev=2))                                              # state 2 was lost
    assert st.gaps == [{"expected_prev": 1, "got_prev": 2, "gameStateId": 3}]
    assert st.stale_objects == {5} and 28 in st.stale_zones
    st.apply(_gsm(4, zones=[{"zoneId": 28, "type": "ZoneType_Battlefield", "objectInstanceIds": [5]}],
                  gameObjects=[{"instanceId": 5, "grpId": 1}]))
    assert not st.stale_objects and 28 not in st.stale_zones


def test_attachment_direction_and_target_order():
    st = arena.GameState()
    st.apply(_gsm(1, full=True, persistentAnnotations=[
        {"id": 1, "type": ["AnnotationType_Attachment"], "affectorId": 161, "affectedIds": [252]},
        {"id": 2, "type": ["AnnotationType_TargetSpec"], "affectorId": 70, "affectedIds": [9],
         "details": [{"key": "index", "valueInt32": [2]}]},
        {"id": 3, "type": ["AnnotationType_TargetSpec"], "affectorId": 70, "affectedIds": [8],
         "details": [{"key": "index", "valueInt32": [1]}]}]))
    assert st.attachments() == {161: 252}            # the Aura is the affector, the host affected
    assert st.targets() == {70: [8, 9]}


# --- games, pairing and logical decisions ----------------------------------------------------------
def test_games_and_pairing(parsed):
    g1, g2 = parsed.games
    assert [len(g.decisions) for g in parsed.games] == [11, 1]
    assert sum(g1.requests_by_type.values()) == 14 and g1.answered_by_type == g1.requests_by_type
    assert g1.counts["unpaired_client_messages"] == 1          # the answer to the lost request
    assert g1.counts["requests_lost_to_summary"] == 1
    assert len(g1.state.gaps) == 1 and g1.state.n_full == 1
    assert g1.local_seat == 1 and g1.starting_seat == 1 and g1.result() == "win"
    assert g2.state.info["gameNumber"] == 2 and g2.decisions[0].kind == "MulliganReq"
    kinds = [d.kind for d in g1.decisions]
    assert kinds == ["ChooseStartingPlayerReq", "MulliganReq", "ActionsAvailableReq", "ActionsAvailableReq",
                     "ActionsAvailableReq", "ActionsAvailableReq", "SelectTargetsReq", "ActionsAvailableReq",
                     "DeclareAttackersReq", "DeclareBlockersReq", "ActionsAvailableReq"]


def test_priority_decision_offers_arena_legal_set(parsed):
    s = decision(parsed, 1, 3)
    L = s.labels
    assert [o["mz_key"] for o in L["options"]] == ["Play Plains"] * 3 + ["Pass"]
    assert [o["mz_key"] for o in L["inactive"]] == ["Cast Savannah Lions", "Cast Pacifism"]
    assert L["chosen"][0]["instanceId"] == 101 and L["chosen_index"] == 0 and L["auto_pass"] == "Yes"
    assert L["options"][0]["mz_idx"] == 539 and L["options"][3]["mz_idx"] == 0
    assert (s.phase, s.step, s.enterMode, s.priorityPlayer) == ("PRECOMBAT_MAIN", "PRECOMBAT_MAIN",
                                                                 "PRIORITY_FRESH", "A")
    mana = decision(parsed, 1, 4).labels["options"][1]
    assert mana["arena_action_type"] == "Activate_Mana" and mana["mz_key"] == "{T}: Add {W}." and mana["mz_idx"] == 5


def test_target_chain_is_one_decision_linked_to_its_cast(parsed):
    t = decision(parsed, 1, 7)
    assert t.labels["decision_type"] == "SelectTargetsReq" and t.labels["round_trips"] == 1
    assert t.labels["options"][0]["candidates"] == ["B:252", "B:254", "A:152"]
    assert t.labels["chosen"] == ["B:252"] and t.labels["answered_with"] == "SubmitTargetsReq"
    assert t.labels["part_of"] == 6 and "mid_action" in t.provenance.flags
    cast = decision(parsed, 1, 6)
    assert cast.labels["chosen"][0]["mz_key"] == "Cast Pacifism" and cast.labels["chosen"][0]["targets"] == ["B:252"]
    held = decision(parsed, 1, 8)                    # A holds priority over its own Pacifism
    assert [(x.controller, x.card, x.targets) for x in held.stack] == [("A", "Pacifism", ["B:252"])]
    assert (held.enterMode, held.priorityPlayer, held.passedPlayers) == ("PRIORITY_HELD", "A", [])


def test_attack_chain(parsed):
    s = decision(parsed, 1, 9)
    assert s.labels["round_trips"] == 1 and s.labels["answered_with"] == "SubmitAttackersReq"
    assert s.labels["chosen"] == [{"attacker": "A:152", "name": "Savannah Lions", "defender": "player:B"}]
    assert s.labels["attack"] == {"A:152": True} and s.labels["state_check"] is True
    # entered one step early so XMage's declare-attackers step asks for the attack
    assert (s.phase, s.step, s.enterMode) == ("COMBAT", "BEGIN_COMBAT", "PRIORITY_FRESH")
    assert s.labels["arena"]["step"] == "DeclareAttack" and s.attackers == []
    # A may hold an instant at begin combat: the bridge must open the window at the declaration
    assert s.labels["bridge"] == {"decisionPlayer": "A", "decideFrom": {"turn": s.turn, "step": "DECLARE_ATTACKERS"}}


def test_block_chain(parsed):
    s = decision(parsed, 1, 10)
    assert s.labels["block"] == {"A:152": ["B:254"]} and s.labels["state_check"] is True
    assert [(a.attacker, a.defender) for a in s.attackers] == [("B:254", "player:A")]
    assert (s.step, s.enterMode, s.priorityPlayer, s.activePlayer) == ("DECLARE_ATTACKERS", "PRIORITY_HELD", "B", "B")
    # the bridge would decide for the priority player (B) unless told to decide for A
    assert s.labels["bridge"] == {"decisionPlayer": "A", "decideFrom": {"turn": s.turn, "step": "DECLARE_BLOCKERS"}}


def test_opponent_turn_priority_with_a_trigger_on_the_stack(parsed):
    s = decision(parsed, 1, 5)
    assert (s.activePlayer, s.enterMode, s.priorityPlayer, s.passedPlayers) == ("B", "PRIORITY_HELD", "A", ["B"])
    assert s.stack == [] and s.labels["stack_abilities"][0]["source"] == "Prideful Parent"
    assert s.labels["stack_abilities"][0]["ability"] == "When {this} enters, create a 1/1 white Cat creature token."
    assert s.provenance.tier == "T2" and "stack_ability_omitted" in s.provenance.flags


def test_board_details(parsed):
    before, s = decision(parsed, 1, 5), decision(parsed, 1, 9)
    assert perm(before, "A", "150").tapped                         # tapped for the Lions on turn 1
    assert not perm(decision(parsed, 1, 6), "A", "150").tapped     # re-sent without isTapped at untap
    assert perm(s, "A", "150").tapped                              # tapped again for Pacifism
    lions = perm(s, "A", "152")
    assert lions.counters == {"P1P1": 1} and not lions.sick
    assert perm(before, "A", "152").sick                           # entered this turn / sick flag
    assert perm(s, "A", "161").attachTo == "B:252"
    cat = perm(s, "B", "254")
    assert (cat.token, cat.set, cat.tokenClass) == ("Cat", "FDN", None)
    assert "tokens_by_name:1" in s.provenance.flags
    assert s.players["B"].life == 20 and decision(parsed, 1, 10).players["B"].life == 17
    assert s.players["A"].landsPlayed == 0 and decision(parsed, 1, 4).players["A"].landsPlayed == 1
    assert s.startingPlayer == "A"


def test_token_class_from_the_ids_table(tmp_path):
    table = tmp_path / "XYZ_tokens.tsv"
    table.write_text("# comment line\ngrpId\tname\ttoken_class\n94156\tCat\tCatToken3\n94179\tCopy\t\n")
    classes = arena.read_token_table(table)
    assert classes == {94156: "CatToken3"}                     # "Copy" has no class
    p = arena.parse_logs([FIXTURE], names(token_classes=classes))
    s = decision(p, 1, 9)
    cat = perm(s, "B", "254")
    assert (cat.tokenClass, cat.token) == ("CatToken3", None)
    assert not any(f.startswith("tokens_by_name") for f in s.provenance.flags) and s.validate() == []


@pytest.mark.skipif(not arena.TOKEN_TABLES, reason="assets/gameplay/*_tokens.tsv not present")
def test_committed_token_tables_load():
    n = arena.CardNames.load(cards_csv=None, abilities_csv=None)
    assert n.token_class(94156) == "CatToken3" and n.token_class(94157) == "CatToken"


def test_after_the_gap(parsed):
    s = decision(parsed, 1, 11)
    assert s.provenance.tier == "T1" and "stale_after_gap" in s.provenance.flags
    assert "254" not in {p.id for p in s.players["B"].battlefield}          # the Cat died (deleted)
    assert perm(s, "A", "152").damage == 1 and s.turn == 5


def test_hidden_information_and_decklists(parsed):
    s = decision(parsed, 1, 6)
    A, B = s.players["A"], s.players["B"]
    assert A.decklistSource == "exact" and len(A.decklist) == 40 and A.handUnknown == 0
    assert A.librarySize == 32 and A.hand.count("Plains") == 3
    assert B.decklistSource == "placeholder" and len(B.decklist) == 40 and B.hand == [] and B.handUnknown == 6
    c = collections.Counter(B.decklist)
    assert c["Prideful Parent"] == 1 and c["Plains"] == 39      # seen cards + basics of B's colour (W)
    g2 = decision(parsed, 2, 1)
    assert collections.Counter(g2.players["A"].decklist)["Plains"] == 18    # the sideboarded deck
    assert g2.provenance.ref == "game2:decision1" and g2.labels["chosen"] == ["Mulligan"]


def test_every_spec_validates_and_round_trips(parsed):
    specs = parsed.specs()
    assert len(specs) == 12
    for s in specs:
        assert s.validate() == [], s.provenance.ref
        assert s.provenance.source == "arena" and s.provenance.ref.startswith("game")
        assert s.phase in PHASE_STEPS and s.step in PHASE_STEPS[s.phase]
        again = StateSpec.from_json(s.to_json())
        assert again.to_dict() == s.to_dict()
    assert all(s.is_partial() for s in specs if s.players["B"].handUnknown)
    assert decision(parsed, 1, 1).labels["chosen"] == ["A"]                  # chose to play first


def bridge_rule_errors(s: StateSpec) -> list[str]:
    """The extra checks java/mzbridge Spec.validate() makes on top of StateSpec.validate()."""
    errs = []
    if s.enterMode == "BEGIN_STEP" and s.step == "UNTAP":
        errs.append("BEGIN_STEP at UNTAP")
    if s.step in ("UNTAP", "CLEANUP") and s.enterMode != "BEGIN_STEP":
        errs.append("no priority window")
    errs += [f"attacker {a.attacker} not the active player's" for a in s.attackers
             if not a.attacker.startswith(s.activePlayer + ":")]
    return errs


def test_specs_follow_the_bridge_rules(parsed):
    for s in parsed.specs():
        assert bridge_rule_errors(s) == [], s.provenance.ref
    pre = decision(parsed, 1, 2)                                  # the mulligan, before turn 1
    assert (pre.turn, pre.phase, pre.step, pre.enterMode, pre.activePlayer) == (1, "BEGINNING", "UPKEEP",
                                                                               "BEGIN_STEP", "A")
    assert "pregame" in pre.provenance.flags


# --- privacy -----------------------------------------------------------------------------------------
def test_identifiers_are_collected_from_the_log():
    ids = arena.Identifiers.from_logs([FIXTURE])
    counts = ids.counts()
    for cat in ("account_id", "screenName", "playerName", "userId", "matchId", "uuid", "os_user", "clientId"):
        assert counts.get(cat), cat
    assert "Testy McFakeface" in ids.words["playerName_bare"]


def test_scrub_passes_clean_output_and_fails_loudly_on_a_leak(parsed):
    ids = arena.Identifiers.from_logs([FIXTURE])
    lines = [arena.spec_json(s) for s in parsed.specs()]
    arena.check_privacy(lines, ids)                                          # nothing leaks
    for leak in ("FAKEACCT0000042", "Testy McFakeface#12345", "00000000-1111-2222-3333-444444444444",
                 "FAKEUSERIDOPPON000000002", "fakeuser", "Testy McFakeface"):
        with pytest.raises(arena.PrivacyError) as e:
            arena.check_privacy(lines + [f'{{"comment": "{leak}"}}'], ids)
        assert leak not in str(e.value)                                      # the error names categories only
    # a bare screen name that is also a word in card text we emit is not a leak
    ids.words["playerName_bare"].add("Savannah")
    arena.check_privacy(['"Savannah Lions"'], ids, allowed_words={"Savannah", "Lions"})


def test_main_writes_decisions_and_summary(tmp_path, capsys):
    out = tmp_path / "arena"
    assert arena.main([str(FIXTURE), "--out", str(out), "--no-card-db"]) == 0
    rows = [json.loads(line) for line in (out / "decisions.jsonl").read_text().splitlines()]
    summary = json.loads((out / "summary.json").read_text())
    assert len(rows) == 12 and summary["games"] == 2 and summary["fraction_requests_paired"] == 1.0
    assert summary["fraction_specs_valid"] == 1.0 and summary["unpaired_client_messages"] == 1
    assert summary["attack_block_state_check"] == {"checked": 2, "consistent": 2}
    text = (out / "decisions.jsonl").read_text() + (out / "summary.json").read_text() + capsys.readouterr().out
    for secret in ("FAKEACCT", "Fakey", "Testy", "Fakename", "FAKEUSERID", "00000000-1111", "aaaaaaaa-",
                   "bbbbbbbb-", "fakeuser", "1790000"):
        assert secret not in text


def test_main_writes_nothing_when_a_leak_is_found(tmp_path, monkeypatch):
    real = arena.Parser.specs

    def leaky(self):
        specs = real(self)
        specs[0].comment = "played by Opponent Fakename#67890"
        return specs

    monkeypatch.setattr(arena.Parser, "specs", leaky)
    out = tmp_path / "arena"
    with pytest.raises(arena.PrivacyError):
        arena.main([str(FIXTURE), "--out", str(out), "--no-card-db"])
    assert not out.exists()


def test_list_games(capsys):
    assert arena.main([str(FIXTURE), "--list-games", "--no-card-db"]) == 0
    text = capsys.readouterr().out
    assert "game 1: Limited game 1 of its match, 5 turns, on the play, win, 11 decisions" in text
    assert "game 2: Limited game 2" in text


def test_local_seat_two(tmp_path):
    """The local player is seat 2: A must still be the local seat."""
    hdr_in = "[UnityCrossThreadLogger]9/1/2026 1:00:00 PM: Match to FAKE2: GreToClientEvent"
    hdr_out = "[UnityCrossThreadLogger]9/1/2026 1:00:01 PM: FAKE2 to Match: ClientToGremessage"
    zones = [{"zoneId": 28, "type": "ZoneType_Battlefield", "objectInstanceIds": [300]},
             {"zoneId": 31, "type": "ZoneType_Hand", "ownerSeatId": 1, "objectInstanceIds": [101, 102]},
             {"zoneId": 35, "type": "ZoneType_Hand", "ownerSeatId": 2, "objectInstanceIds": [201]}]
    gsm = {"type": "GameStateType_Full", "gameStateId": 1, "zones": zones,
           "players": [{"systemSeatNumber": 1, "lifeTotal": 18}, {"systemSeatNumber": 2, "lifeTotal": 20}],
           "turnInfo": {"turnNumber": 2, "phase": "Phase_Main1", "activePlayer": 2, "priorityPlayer": 2},
           "gameObjects": [{"instanceId": 201, "grpId": 93859, "type": "GameObjectType_Card", "zoneId": 35,
                            "ownerSeatId": 2, "controllerSeatId": 2},
                           {"instanceId": 300, "grpId": 95192, "type": "GameObjectType_Card", "zoneId": 28,
                            "ownerSeatId": 1, "controllerSeatId": 1}]}
    ev = {"timestamp": "1790000000000", "greToClientEvent": {"greToClientMessages": [
        {"type": "GREMessageType_ConnectResp", "systemSeatIds": [2],
         "connectResp": {"deckMessage": {"deckCards": [93859] * 4 + [95192] * 36}}},
        {"type": "GREMessageType_GameStateMessage", "systemSeatIds": [2], "gameStateMessage": gsm},
        {"type": "GREMessageType_ActionsAvailableReq", "systemSeatIds": [2], "msgId": 5, "gameStateId": 1,
         "actionsAvailableReq": {"actions": [{"actionType": "ActionType_Cast", "grpId": 93859, "instanceId": 201},
                                             {"actionType": "ActionType_Pass"}]}}]}}
    resp = {"timestamp": str(1790000002000 * 10000 + 621355968000000000),
            "payload": {"type": "ClientMessageType_PerformActionResp", "respId": 5, "gameStateId": 1,
                        "performActionResp": {"actions": [{"actionType": "ActionType_Pass"}]}}}
    log = tmp_path / "Player.log"
    log.write_text("\n".join([hdr_in, json.dumps(ev), "", hdr_out] + json.dumps(resp, indent=2).split("\n")) + "\n")
    p = arena.parse_logs([log], names())
    s = p.specs()[0]
    assert p.games[0].local_seat == 2
    assert s.players["A"].hand == ["Savannah Lions"] and s.players["A"].life == 20
    assert s.players["B"].handUnknown == 2 and s.players["B"].life == 18
    assert [x.name for x in s.players["B"].battlefield] == ["Plains"]
    assert (s.activePlayer, s.enterMode) == ("A", "PRIORITY_FRESH")
    assert s.labels["chosen"][0]["mz_key"] == "Pass" and s.labels["latency_s"] == 2.0
    assert s.validate() == []


# --- regression tests on small inline logs (local seat 1, opponent seat 2) -------------------------
HDR_IN = "[UnityCrossThreadLogger]9/1/2026 1:00:00 PM: Match to FAKE3: GreToClientEvent"
HDR_OUT = "[UnityCrossThreadLogger]9/1/2026 1:00:01 PM: FAKE3 to Match: ClientToGremessage"
TRACE = "UnityEngine.DebugLogHandler:Internal_Log(LogType, LogOption, String, Object)"
PLAINS, LIONS, PACIFISM, SERAPH = 95192, 93859, 93650, 93741
DECK = [LIONS] * 4 + [PACIFISM] * 4 + [SERAPH] * 4 + [PLAINS] * 28
ZONES = {27: ("Stack", None), 28: ("Battlefield", None), 29: ("Exile", None), 31: ("Hand", 1), 32: ("Library", 1),
         33: ("Graveyard", 1), 35: ("Hand", 2), 36: ("Library", 2), 37: ("Graveyard", 2)}
HIDDEN = {32: list(range(1000, 1020)), 35: [2000, 2001, 2002], 36: list(range(2100, 2125))}


def card(iid, grp, zone, owner=1, controller=None, **kw):
    return dict(kw, instanceId=iid, grpId=grp, type="GameObjectType_Card", zoneId=zone, ownerSeatId=owner,
                controllerSeatId=controller or owner)


def game_state(gsid=1, objects=(), order=None, turn=5, phase="Phase_Main1", step=None, active=1, prio=1,
               full=True, **kw):
    """A GameStateMessage with Arena's zone ids; `order` overrides a zone's id list (Arena's order)."""
    by_zone = collections.defaultdict(list, {k: list(v) for k, v in HIDDEN.items()})
    for o in objects:
        by_zone[o["zoneId"]].append(o["instanceId"])
    by_zone.update(order or {})
    zones = [dict({"zoneId": z, "type": "ZoneType_" + t, "objectInstanceIds": by_zone[z]},
                  **({"ownerSeatId": owner} if owner else {})) for z, (t, owner) in ZONES.items()]
    ti = dict({"turnNumber": turn, "phase": phase, "activePlayer": active, "priorityPlayer": prio,
               "decisionPlayer": prio}, **({"step": step} if step else {}))
    gsm = dict(kw, type="GameStateType_Full" if full else "GameStateType_Diff", gameStateId=gsid,
               zones=zones, gameObjects=list(objects), turnInfo=ti,
               players=[{"systemSeatNumber": 1, "lifeTotal": 20}, {"systemSeatNumber": 2, "lifeTotal": 20}])
    if full:
        gsm["gameInfo"] = {"matchID": "fake-match", "gameNumber": 1}
    return {"type": "GREMessageType_GameStateMessage", "systemSeatIds": [1], "gameStateMessage": gsm}


def connect(deck=DECK):
    return {"type": "GREMessageType_ConnectResp", "systemSeatIds": [1],
            "connectResp": {"deckMessage": {"deckCards": deck}}}


def req(kind, msg_id, gsid=1, **body):
    return dict(body, type="GREMessageType_" + kind, systemSeatIds=[1], msgId=msg_id, gameStateId=gsid)


def aar(msg_id, gsid=1, actions=None):
    return req("ActionsAvailableReq", msg_id, gsid,
               actionsAvailableReq={"actions": actions or [{"actionType": "ActionType_Pass"}]})


def attack_req(msg_id, iid):
    atk = {"attackerInstanceId": iid,
           "legalDamageRecipients": [{"type": "DamageRecType_Player", "playerSystemSeatId": 2}]}
    return req("DeclareAttackersReq", msg_id, declareAttackersReq={"attackers": [atk], "qualifiedAttackers": [atk]})


def combat(objects, step="Step_DeclareAttack"):
    return game_state(objects=objects, phase="Phase_Combat", step=step)


def write_log(path, *items):
    """Each item is one GRE event (a list of messages) or one client payload (a dict)."""
    lines, t = [], 1790000000000
    for item in items:
        t += 1000
        if isinstance(item, list):
            lines += [HDR_IN, json.dumps({"timestamp": str(t), "greToClientEvent": {"greToClientMessages": item}})]
        else:
            ticks = str(t * 10000 + 621355968000000000)
            lines += [HDR_OUT, *json.dumps({"timestamp": ticks, "payload": item}, indent=2).split("\n")]
        lines += [TRACE, ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse(tmp_path, *items):
    return arena.parse_logs([write_log(tmp_path / "Player.log", *items)], names())


def test_stack_and_graveyard_are_written_bottom_to_top(tmp_path):
    """Arena lists the stack and graveyards newest first; the spec (and the bridge) want bottom first."""
    objs = [card(400, LIONS, 33), card(401, SERAPH, 33),               # Lions died first, then Seraph
            card(501, LIONS, 27), card(502, PACIFISM, 27, owner=2)]    # A's Lions, then B's Pacifism on top
    p = parse(tmp_path, [connect(), game_state(objects=objs, order={33: [401, 400], 27: [502, 501]},
                                               active=2, prio=1), aar(5)])
    s = p.specs()[0]
    assert s.players["A"].graveyard == ["Savannah Lions", "Vanguard Seraph"]
    assert [(x.controller, x.card) for x in s.stack] == [("A", "Savannah Lions"), ("B", "Pacifism")]
    assert (s.enterMode, s.priorityPlayer, s.passedPlayers) == ("PRIORITY_HELD", "A", ["B"])   # B cast last
    assert s.validate() == [] and bridge_rule_errors(s) == []


def test_written_labels_keep_empty_and_missing_answers(tmp_path):
    """An attack with no attackers is chosen: [] and an unanswered request chosen: null in the file;
    StateSpec.to_json() would drop both keys."""
    p = parse(tmp_path, [connect(), combat([card(150, LIONS, 28)]), attack_req(7, 150)],
              {"type": "ClientMessageType_SubmitAttackersReq", "respId": 7, "gameStateId": 1},
              [aar(8)])
    rows = [json.loads(arena.spec_json(s)) for s in p.specs()]
    assert rows[0]["labels"]["chosen"] == [] and rows[0]["labels"]["attack"] == {"A:150": False}
    assert rows[1]["labels"]["chosen"] is None and rows[1]["labels"]["unanswered"] is True
    assert "chosen" not in p.specs()[0].to_dict()["labels"]          # what StateSpec.to_json() would write
    assert StateSpec.from_json(arena.spec_json(p.specs()[0])).labels["chosen"] == []


def test_priority_inside_the_declare_steps_is_entered_mid_round(tmp_path):
    """PRIORITY_FRESH at DECLARE_ATTACKERS/BLOCKERS makes XMage replay the declaration
    (DeclareAttackersStep.resumeBeginStep): attack triggers fire again. PRIORITY_HELD does not."""
    lions = card(150, LIONS, 28, attackState="AttackState_Attacking", attackInfo={"targetId": 2})
    p = parse(tmp_path, [connect(), combat([lions]), aar(5)])
    s = p.specs()[0]
    assert (s.step, s.enterMode, s.priorityPlayer, s.passedPlayers) == ("DECLARE_ATTACKERS", "PRIORITY_HELD", "A", [])
    assert [(a.attacker, a.defender) for a in s.attackers] == [("A:150", "player:B")]
    seraph = card(250, SERAPH, 28, owner=2, blockState="BlockState_Blocking", blockInfo={"attackerIds": [150]})
    p = parse(tmp_path, [connect(), combat([lions, seraph], step="Step_DeclareBlock"), aar(5)])
    s = p.specs()[0]
    assert (s.step, s.enterMode, s.priorityPlayer) == ("DECLARE_BLOCKERS", "PRIORITY_HELD", "A")
    assert [(b.blocker, b.attacker) for b in s.blockers] == [("B:250", "A:150")]
    assert s.validate() == [] and bridge_rule_errors(s) == []


def test_attack_decision_drops_a_preselected_attacker(tmp_path):
    """The attack spec is rewound to BEGIN_COMBAT, where no attack can be declared yet."""
    lions = card(150, LIONS, 28, attackState="AttackState_Declared", attackInfo={"targetId": 2})
    p = parse(tmp_path, [connect(), combat([lions]), attack_req(7, 150)])
    s = p.specs()[0]
    assert (s.step, s.attackers) == ("BEGIN_COMBAT", []) and "preselected_attackers_dropped" in s.provenance.flags


def unaccounted(s: StateSpec) -> list[str]:
    return [f for f in s.provenance.flags if f.startswith(("A.cards_unaccounted", "A.card_not_in_decklist"))]


def to_battlefield(iid, src_zone, category="Return"):
    return {"id": 70, "type": ["AnnotationType_ZoneTransfer"], "affectedIds": [iid],
            "details": [{"key": "zone_src", "valueInt32": [src_zone]}, {"key": "zone_dest", "valueInt32": [28]},
                        {"key": "category", "valueString": [category]}]}


def test_reanimated_creature_is_listed_under_its_controller(tmp_path):
    """A creature card of B's that A returned to the battlefield under A's control is A's permanent
    with owner B: it attacks for A, its card comes out of B's decklist (not A's), the labels name it
    by the spec's alias, and since it has been A's since it entered nothing about it is missing."""
    back = card(251, SERAPH, 28, owner=2, controller=1)
    attacking = dict(back, attackState="AttackState_Attacking", attackInfo={"targetId": 2})
    p = parse(tmp_path, [connect(), game_state(1, objects=[card(250, SERAPH, 37, owner=2)]), aar(5, 1)],
              [game_state(2, objects=[back], full=False, annotations=[to_battlefield(251, 37)]), aar(6, 2)],
              [game_state(3, objects=[back], full=False, phase="Phase_Combat", step="Step_DeclareAttack"),
               attack_req(7, 251)],
              {"type": "ClientMessageType_SubmitAttackersReq", "respId": 7, "gameStateId": 3},
              [game_state(4, objects=[attacking], full=False, phase="Phase_Combat", step="Step_DeclareAttack"),
               aar(8, 4)])
    before, s, atk, after = p.specs()
    for x in (s, atk, after):
        assert [(b.id, b.name, b.owner) for b in x.players["A"].battlefield] == [("251", "Vanguard Seraph", "B")]
        assert x.players["B"].battlefield == [] and "Vanguard Seraph" in x.players["B"].decklist
        assert "owner_differs:1" in x.provenance.flags and "control_changed" not in x.provenance.flags
        assert x.provenance.tier == "T0" and x.validate() == [] and bridge_rule_errors(x) == []
        assert unaccounted(x) == unaccounted(before)          # A's library is not one card short
    assert [(a.attacker, a.defender) for a in after.attackers] == [("A:251", "player:B")]
    assert [o["attacker"] for o in atk.labels["options"]] == ["A:251"] and "A:251" in atk.aliases()


def test_control_taken_by_an_effect_is_flagged(tmp_path):
    """A creature that changed controller while on the battlefield (Threaten, Mind Control) is listed
    under its controller too, attacks included, but the effect may end, and the bridge makes control
    permanent: T2. An Animate Dead-style Aura of the controller explains a creature whose entry the
    log does not show (a log that starts mid-game); B's creature stolen from A is A's card."""
    seraph = card(250, SERAPH, 28, owner=2)
    stolen = dict(seraph, controllerSeatId=1, attackState="AttackState_Attacking", attackInfo={"targetId": 2})
    p = parse(tmp_path, [connect(), game_state(1, objects=[seraph]), aar(5, 1)],
              [game_state(2, objects=[stolen], full=False, phase="Phase_Combat", step="Step_DeclareAttack"),
               aar(6, 2)])
    s = p.specs()[1]
    assert [(b.id, b.owner) for b in s.players["A"].battlefield] == [("250", "B")]
    assert [(a.attacker, a.defender) for a in s.attackers] == [("A:250", "player:B")]
    assert s.provenance.tier == "T2" and "control_changed" in s.provenance.flags and s.validate() == []
    # the same creature under an Animate Dead of A's, first seen in a Full state
    dead = arena.CardNames({**names().cards, 99001: arena.CardInfo(99001, "Animate Dead", "EMA")},
                           ABILITIES, vocab={})
    aura = card(300, 99001, 28)
    hold = {"id": 91, "type": ["AnnotationType_Attachment"], "affectorId": 300, "affectedIds": [250]}
    for objs, tier in (([dict(seraph, controllerSeatId=1), aura], "T0"), ([dict(seraph, controllerSeatId=1)], "T2")):
        log = write_log(tmp_path / "Player.log", [connect(DECK[:-1] + [99001]),
                                                  game_state(objects=objs, persistentAnnotations=[hold]), aar(5)])
        s = arena.parse_logs([log], dead).specs()[0]
        assert s.provenance.tier == tier and s.validate() == [], objs
        assert ("owner_differs:1" in s.provenance.flags) == (tier == "T0")
    assert perm(s, "A", "250").owner == "B"
    # B took A's Lions: listed under B with owner A; still A's card for A's library count
    lions = card(150, LIONS, 28)
    p = parse(tmp_path, [connect(), game_state(1, objects=[lions]), aar(5, 1)],
              [game_state(2, objects=[dict(lions, controllerSeatId=2)], full=False), aar(6, 2)])
    mine, taken = p.specs()
    assert [(b.id, b.owner) for b in taken.players["B"].battlefield] == [("150", "A")]
    assert taken.players["A"].battlefield == [] and "Savannah Lions" not in taken.players["B"].decklist
    assert unaccounted(taken) == unaccounted(mine) and taken.validate() == []
    # a stolen token keeps no owner (StateSpec tokens have none)
    cat = dict(card(260, 94156, 28, owner=2, controller=1), type="GameObjectType_Token")
    s = parse(tmp_path, [connect(), game_state(objects=[cat]), aar(5)]).specs()[0]
    assert [(b.id, b.token, b.owner) for b in s.players["A"].battlefield] == [("260", "Cat", None)]
    assert {"token_owner_dropped", "control_changed"} <= set(s.provenance.flags) and s.validate() == []


def test_permanent_back_under_its_owner_is_flagged(tmp_path):
    """B's creature that A reanimated (it entered under A) and that B controls again (an effect that
    ends, such as until end of turn) is listed under B with no owner, as it looks. But the bridge
    would make B its permanent controller, so control never goes back to A: T2."""
    back = card(251, SERAPH, 28, owner=2, controller=1)
    p = parse(tmp_path, [connect(), game_state(1, objects=[card(250, SERAPH, 37, owner=2)]), aar(5, 1)],
              [game_state(2, objects=[back], full=False, annotations=[to_battlefield(251, 37)]), aar(6, 2)],
              [game_state(3, objects=[dict(back, controllerSeatId=2)], full=False), aar(7, 3)])
    _, under_a, taken = p.specs()
    assert under_a.provenance.tier == "T0" and "control_changed" not in under_a.provenance.flags
    assert [(b.id, b.owner) for b in taken.players["B"].battlefield] == [("251", None)]
    assert "control_changed" in taken.provenance.flags and taken.provenance.tier == "T2" and taken.validate() == []
    # a creature B cast, still B's: nothing changed hands
    own = card(252, SERAPH, 28, owner=2)
    s = parse(tmp_path, [connect(), game_state(1, objects=[card(250, SERAPH, 27, owner=2)]), aar(5, 1)],
              [game_state(2, objects=[own], full=False, annotations=[to_battlefield(252, 27, "Resolve")]),
               aar(6, 2)]).specs()[1]
    assert "control_changed" not in s.provenance.flags and s.provenance.tier == "T0"


def test_stack_targets_keep_their_slots(tmp_path):
    """The bridge fills target slot k with targets[k]: one target per slot, nothing after a slot it
    cannot fill."""
    def spec_(slot_targets):
        objs = [card(150, LIONS, 28), card(151, LIONS, 28), card(401, SERAPH, 33), card(501, PACIFISM, 27)]
        pann = [{"id": 90 + k, "type": ["AnnotationType_TargetSpec"], "affectorId": 501, "affectedIds": ts,
                 "details": [{"key": "index", "valueInt32": [k]}]} for k, ts in slot_targets.items()]
        return parse(tmp_path, [connect(), game_state(objects=objs, persistentAnnotations=pann), aar(5)]).specs()[0]
    s = spec_({1: [150], 2: [2]})
    assert s.stack[0].targets == ["A:150", "player:B"] and s.provenance.tier == "T0"
    s = spec_({1: [150, 151]})                                     # "any number of target creatures"
    assert s.stack[0].targets == ["A:150"] and "stack_multi_target_slot" in s.provenance.flags
    s = spec_({1: [401], 2: [150]})                                # a card in a graveyard, then a creature
    assert s.stack[0].targets == [] and "stack_target_unrepresentable" in s.provenance.flags
    s = spec_({2: [150]})                                          # optional slot 1 left empty
    assert s.stack[0].targets == [] and s.provenance.tier == "T2"


def test_gap_leaves_annotations_unverified_until_a_full_state(tmp_path):
    """A lost message's counters/attachments are not re-sent with their objects, so every spec after
    a gap is T1 even when all objects came again; the next Full state clears it."""
    lions = card(150, LIONS, 28)
    counter = {"id": 9, "type": ["AnnotationType_Counter"], "affectedIds": [150],
               "details": [{"key": "count", "valueInt32": [1]}, {"key": "counter_type", "valueInt32": [1]}]}
    after_gap = game_state(3, objects=[lions], full=False, prevGameStateId=2)   # state 2 was lost
    p = parse(tmp_path, [connect(), game_state(1, objects=[lions], persistentAnnotations=[counter]), aar(5, 1)],
              [after_gap, aar(6, 3)], [game_state(4, objects=[lions]), aar(7, 4)])
    before, gap, resync = p.specs()
    assert before.provenance.tier == "T0"
    assert gap.provenance.tier == "T1" and "annotations_unverified_after_gap" in gap.provenance.flags
    assert "stale_after_gap" not in gap.provenance.flags                   # every object was re-sent
    assert resync.provenance.tier == "T0" and p.games[0].resyncs == 1


def test_starting_player_of_a_log_that_begins_mid_game(tmp_path):
    """Global turn 6 with seat 2 active: seat 1 (A) was on the play."""
    s = parse(tmp_path, [connect(), game_state(turn=6, active=2, prio=1), aar(5)]).specs()[0]
    assert (s.turn, s.activePlayer, s.startingPlayer) == (6, "B", "A")
    assert "starting_player_from_turn_parity" in s.provenance.flags
    s = parse(tmp_path, [connect(), game_state(turn=1, active=2, prio=2), aar(5)]).specs()[0]
    assert s.startingPlayer == "B" and "starting_player_from_turn_parity" not in s.provenance.flags


def test_select_n_response_keeps_ids_that_are_not_instances(tmp_path):
    p = parse(tmp_path, [connect(), game_state(objects=[card(150, LIONS, 28)]),
                         req("SelectNReq", 5, selectNReq={"ids": [1, 2, 3], "idType": "IdType_PromptParameterIndex",
                                                          "minSel": 1, "maxSel": 1})],
              {"type": "ClientMessageType_SelectNResp", "respId": 5, "selectNResp": {"ids": [2]}})
    L = p.specs()[0].labels
    assert L["options"] == [1, 2, 3] and L["chosen"] == [2]


def test_untap_step_is_entered_as_upkeep(tmp_path):
    p = parse(tmp_path, [connect(), game_state(phase="Phase_Beginning", step="Step_Untap"),
                         req("OptionalActionMessage", 5, optionalActionMessage={})])
    s = p.specs()[0]
    assert (s.phase, s.step, s.enterMode) == ("BEGINNING", "UPKEEP", "BEGIN_STEP")
    assert bridge_rule_errors(s) == [] and "entry_untap_as_upkeep" in s.provenance.flags


def test_alternative_cost_tells_two_casts_of_one_card_apart(tmp_path):
    cast = {"actionType": "ActionType_Cast", "grpId": LIONS, "instanceId": 101}
    alt = dict(cast, alternativeGrpId=4242)
    p = parse(tmp_path, [connect(), game_state(objects=[card(101, LIONS, 31)]),
                         aar(5, actions=[cast, alt, {"actionType": "ActionType_Pass"}])],
              {"type": "ClientMessageType_PerformActionResp", "respId": 5, "performActionResp": {"actions": [alt]}})
    L = p.specs()[0].labels
    assert L["options"][1]["alt_grpId"] == 4242 and L["chosen_index"] == 1


def test_client_body_is_decoded_once_and_a_truncated_one_ends_at_the_next_header(tmp_path, monkeypatch):
    calls = collections.Counter()

    class Counting(json.JSONDecoder):
        def raw_decode(self, s, idx=0):
            calls["n"] += 1
            return super().raw_decode(s, idx)

    monkeypatch.setattr(arena.json, "JSONDecoder", Counting)
    big = {"type": "ClientMessageType_PerformActionResp", "respId": 5,
           "performActionResp": {"actions": [{"actionType": "ActionType_Pass", "k": list(range(300))}]}}
    log = write_log(tmp_path / "Player.log", big)
    assert [m.obj["payload"]["respId"] for m in arena.iter_messages(log)] == [5]
    assert calls["n"] <= 2                        # not once per line of the 300-line body
    # a body cut short (Arena crashed mid-write) must not swallow the next message
    text = log.read_text().split("\n")
    cut = "\n".join([HDR_OUT, "{", '  "payload": {', TRACE, "", HDR_IN,
                     json.dumps({"greToClientEvent": {"greToClientMessages": [connect()]}}), ""])
    log.write_text(cut + "\n" + "\n".join(text))
    msgs = list(arena.iter_messages(log))
    assert [(m.direction, m.obj is not None) for m in msgs] == [("out", False), ("in", True), ("out", True)]


def test_empty_summary_line_is_ignored():
    arena.Parser(names()).feed(arena.LogMessage(1, "in", "GreToClientEvent", summary=["", "GameStateMessage"]))


def test_untagged_and_short_screen_names_are_searched_as_words():
    """Arena dropped the #tag in 2024: names come bare, often as ordinary words, sometimes 3 letters."""
    ids = arena.Identifiers()
    ids.add_line('{"reservedPlayers": [{"playerName": "Bob"}, {"playerName": "Goblin"}], "screenName": "Zed Q"}')
    assert ids.scan('{"comment": "Bob"}') == {"playerName": 1}
    assert ids.scan('{"card": "Bobcat"}') == {}                             # not a whole word
    assert ids.scan('{"card": "Goblin Guide"}', {"Goblin", "Guide"}) == {}  # card text we emit
    assert ids.scan('{"x": "Goblin"}') == {"playerName": 1}                 # not card text here
    assert ids.scan('{"x": "Zed Q"}') == {"screenName": 1}


@pytest.mark.parametrize("in_log", ['{"playerName": "Zo\\u00eb Fakename"}',     # the log may escape it
                                    '{"playerName": "Zoë Fakename"}'])      # or not
def test_non_ascii_names_are_found_in_what_is_written(tmp_path, in_log):
    ids = arena.Identifiers()
    ids.add_line(in_log)
    p = parse(tmp_path, [connect(), game_state(), aar(5)])
    s = p.specs()[0]
    s.comment = "Zoë Fakename"
    with pytest.raises(arena.PrivacyError):
        arena.check_privacy([arena.spec_json(s)], ids)
    assert "\\u00eb" in s.to_json()                                           # escaped: a scan would miss it


# --- the specs in XMage (java/mzbridge; skipped without java and the XMage build) -------------------
BRIDGE_PROBLEMS = bridge.environment_problems()


@pytest.fixture(scope="module")
def xmage(tmp_path_factory):
    b = bridge.Bridge("arena-pytest", heap="2g", runtime_root=tmp_path_factory.mktemp("mzbridge"))
    yield b
    b.close()


@pytest.mark.skipif(bool(BRIDGE_PROBLEMS), reason="mzbridge unavailable: " + "; ".join(BRIDGE_PROBLEMS))
def test_specs_reach_the_same_decision_in_xmage(xmage, tmp_path):
    """Attack and block specs, built with their labels.bridge options, stop at the attack / block
    decision; a priority spec inside DECLARE_ATTACKERS does not re-fire the attack trigger."""
    p = arena.parse_logs([FIXTURE], names(token_classes={94156: "CatToken3"}))
    atk, blk = decision(p, 1, 9), decision(p, 1, 10)
    d = xmage.build(atk, seed=1, **atk.labels["bridge"])["decision"]
    assert (d["player"], d["type"], d["where"]["step"]) == ("A", "CHOOSE_USE", "DECLARE_ATTACKERS")
    assert d["text"] == "attack with: Savannah Lions?"
    d = xmage.build(blk, seed=1, **blk.labels["bridge"])["decision"]
    assert (d["player"], d["type"], d["where"]["step"]) == ("A", "CHOOSE_TARGET", "DECLARE_BLOCKERS")
    # Sanguine Syphoner's "whenever this creature attacks" already triggered in the Arena game
    deck = [93781] * 4 + [93784] * 4 + [95195] * 32
    objs = [card(150, 93781, 28, attackState="AttackState_Attacking", attackInfo={"targetId": 2}),
            card(151, 95195, 28), card(101, 93784, 31), card(250, LIONS, 28, owner=2)]
    s = parse(tmp_path, [connect(deck), combat(objs), aar(5)]).specs()[0]
    d = xmage.build(s, seed=1, **s.labels["bridge"])["decision"]
    assert (d["type"], d["where"]["step"], d["where"]["stack"]) == ("PRIORITY", "DECLARE_ATTACKERS", 0)


@pytest.mark.skipif(bool(BRIDGE_PROBLEMS), reason="mzbridge unavailable: " + "; ".join(BRIDGE_PROBLEMS))
def test_reanimated_creature_attacks_for_its_controller_in_xmage(xmage, tmp_path):
    """B's creature that A returned under A's control: XMage builds it as A's, owned by B, with no
    control warning, and asks A whether it attacks (the question Arena asked)."""
    back = card(251, SERAPH, 28, owner=2, controller=1)
    p = parse(tmp_path, [connect(), game_state(1, objects=[card(250, SERAPH, 37, owner=2)]), aar(5, 1)],
              [game_state(2, objects=[back], full=False, annotations=[to_battlefield(251, 37)]), aar(6, 2)],
              [game_state(3, objects=[back], full=False, phase="Phase_Combat", step="Step_DeclareAttack"),
               attack_req(7, 251)],
              {"type": "ClientMessageType_SubmitAttackersReq", "respId": 7, "gameStateId": 3})
    atk = p.specs()[2]
    assert atk.labels["decision_type"] == "DeclareAttackersReq" and atk.provenance.tier == "T0"
    r = xmage.build(atk, seed=1, **atk.labels["bridge"])
    d = r["decision"]
    assert (d["player"], d["type"], d["where"]["step"]) == ("A", "CHOOSE_USE", "DECLARE_ATTACKERS")
    assert d["text"] == "attack with: Vanguard Seraph?"
    assert not [w for w in r["warnings"] if re.search(r"control|owned by", w)], r["warnings"]
    assert [(x["name"], x.get("owner")) for x in r["dump"]["players"]["A"]["battlefield"]] == [("Vanguard Seraph", "B")]
    assert r["dump"]["players"]["B"]["battlefield"] == []


# --- mapping tables and helpers ------------------------------------------------------------------
def test_arena_text_to_xmage_keys():
    x = arena.xmage_text
    assert x("{oT}, Sacrifice CARDNAME: Add one mana of any color.") == "{T}, Sacrifice {this}: Add one mana of any color."
    assert x("{o2oG}: Put a +1/+1 counter on target creature you control.") == \
        "{2}{G}: Put a +1/+1 counter on target creature you control."
    assert x("{o(B/G)}, {oT}: Draw a card.") == "{B/G}, {T}: Draw a card."
    assert x("<i>Landfall</i> — Whenever a land enters, Kellan gets +1/+1.", "Kellan, the Fae-Blooded") == \
        "Landfall — Whenever a land enters, {this} gets +1/+1."
    assert x("{o4}: This land becomes a 3/3 creature.") == "{4}: {this} becomes a 3/3 creature."
    assert arena.mz_key("Pass", None) == "Pass"
    assert arena.mz_key("CastAdventure", "Bonecrusher Giant") == "Cast Bonecrusher Giant"
    assert arena.mz_key("PlayMdfc", "Plains") == "Play Plains"
    assert arena.mz_key("Activate", "Goldvein Pick", "Equip {1}") == "Equip {1}"
    assert arena.mz_key("Cast", "Think Twice", alt_text="Flashback {2}{U}") == "Flashback {2}{U}"
    assert arena.mz_key("FloatMana", None) is None
    assert arena.mz_key("Activate", "Boseiju, Who Endures", "Channel — {1}{G}, Discard this card: Destroy it.") == \
        "{1}{G}, Discard this card: Destroy it."
    assert arena.same_action("Swampcycling {1}", "Swampcycling {1} <i>({1}, Discard this card: Search.)</i>")
    assert not arena.same_action("Cast Stab", "Cast Stab Wound") and not arena.same_action(None, "Pass")
    assert arena.mana_str([{"color": ["ManaColor_Generic"], "count": 2},
                           {"color": ["ManaColor_Black"], "count": 2}]) == "{2}{B}{B}"


def test_card_names_use_xmage_ascii_spelling():
    assert arena.xmage_name("Troll of Khazad-dûm") == "Troll of Khazad-dum"
    assert arena.xmage_name("Æther Vial") == "Aether Vial" and arena.xmage_name("Stab") == "Stab"


def test_vocab_matches_reminder_text_variants():
    n = arena.CardNames(vocab={"Equip {1} <i>(Attach to target creature you control.)</i>": 507, "Pass": 0})
    assert n.vocab_index("Equip {1}") == 507 and n.vocab_index("Pass") == 0 and n.vocab_index("Equip {2}") is None


def test_phase_and_counter_tables():
    for ph, step in arena.PHASES.values():
        assert step in PHASE_STEPS[ph]
    assert arena.xmage_phase("Phase_Main2", None) == ("POSTCOMBAT_MAIN", "POSTCOMBAT_MAIN")
    assert arena.xmage_phase("Phase_Combat", "Step_FirstStrikeDamage") == ("COMBAT", "FIRST_COMBAT_DAMAGE")
    assert arena.xmage_phase(None, None) is None
    assert arena.COUNTER_TYPES[1] == "P1P1" and arena.COUNTER_TYPES[7] == "LOYALTY"
    assert arena.COUNTER_TYPES[108] == "LORE" and 111 not in arena.COUNTER_TYPES
    assert all(v.isupper() for v in arena.COUNTER_TYPES.values())


def test_placeholder_basics():
    assert arena.basics_for(collections.Counter({"W": 3, "G": 1}), 8) == ["Plains"] * 6 + ["Forest"] * 2
    five = arena.basics_for(collections.Counter(), 7)
    assert len(five) == 7 and set(five) == set(arena.BASICS.values())


# --- optional: real data on this machine ------------------------------------------------------------
@pytest.mark.skipif(not arena.CARDS_CSV.exists(), reason="data/17lands/cards.csv not present")
def test_cards_csv_names():
    n = arena.CardNames.load(card_db=None)
    assert n.name(93859) == "Savannah Lions"
    cat = n.card(94156)
    assert (cat.name, cat.set, cat.token) == ("Cat", "FDN", True)


@pytest.mark.skipif(not arena.find_card_dbs(), reason="no Arena card database on this machine")
def test_arena_card_db():
    db = arena.ArenaCardDB(arena.pick_card_db(None))
    assert db.card(93859).name == "Savannah Lions"
    assert arena.xmage_text(db.ability_text(1001)) == "{T}: Add {W}."


@pytest.mark.skipif(not REAL_LOG.exists(), reason="no local Arena log")
def test_real_log_smoke():
    """Aggregate checks only; nothing from the log is printed."""
    card_db = arena.pick_card_db(arena.log_info(REAL_LOG)["grp_build"])
    n = arena.CardNames.load(card_db=card_db)
    p = arena.parse_logs([REAL_LOG], n)
    specs = p.specs()
    if not specs:
        pytest.skip("the local log holds no games")
    assert sum(1 for s in specs if not s.validate()) / len(specs) >= 0.9
    arena.check_privacy([arena.spec_json(s) for s in specs], arena.Identifiers.from_logs([REAL_LOG]),
                        {w for s in n.emitted for w in re.findall(r"\w+", s)})
