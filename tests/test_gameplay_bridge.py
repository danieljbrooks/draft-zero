"""Tests for the Java XMage bridge (java/mzbridge) and its Python client.

The worker tests need java and the XMage build (xmage/lib, xmage/db) and skip cleanly without
them. They start two workers once (BridgePool(2): the machine rule is at most two JVMs) and
keep every search small, so the file runs in well under a minute.
"""
import copy
import json

import pytest

from draftzero.gameplay import bridge
from draftzero.gameplay.bridge import BridgeError, diff_dump, golden_specs
from draftzero.gameplay.statespec import StateSpec

PROBLEMS = bridge.environment_problems()
needs_worker = pytest.mark.skipif(bool(PROBLEMS), reason="mzbridge unavailable: " + "; ".join(PROBLEMS))


def spec(name: str) -> dict:
    return json.loads(golden_specs()[name].read_text())


def decider(s: dict) -> str | None:
    return s.get("labels", {}).get("decisionPlayer")


def perm(dump: dict, seat: str, name: str) -> dict:
    for p in dump["players"][seat]["battlefield"]:
        if p["x"]["name"] == name:
            return p
    raise AssertionError(f"{name} not on {seat}'s battlefield")


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    # a private runtime root: two test sessions on this shared machine must not collide on the
    # worker names (run.sh refuses a name that is already running)
    p = bridge.BridgePool(2, prefix="pytest", heap="2g", runtime_root=tmp_path_factory.mktemp("mzbridge"))
    yield p
    p.close()


@pytest.fixture(scope="module")
def worker(pool):
    return pool.workers[0]


@pytest.fixture(scope="module")
def builds(pool):
    """Every golden spec built once (seed 1), in parallel over the pool, with the decision state."""
    names = list(golden_specs())
    results = pool.map("build", [spec(n) for n in names], seed=1, dumpDecisionState=True,
                       options_for=lambda s: {"decisionPlayer": decider(s)})
    return dict(zip(names, results))


# ------------------------------------------------------------------------------ no JVM needed

def test_golden_specs_are_valid_statespecs():
    names = set(golden_specs())
    assert {"main_phase", "mana_library", "attack", "block", "block_beginstep", "respond", "eot_rollover"} <= names
    for n in names:
        assert StateSpec.from_dict(spec(n)).validate() == [], n


def test_diff_dump_flags_differences():
    s = spec("main_phase")
    fake = copy.deepcopy(s)
    fake["players"]["A"]["hand"] = ["Stab", "Plains"]
    fake["players"]["B"]["battlefield"][2]["damage"] = 0      # the Prowler had 1 damage
    fake["stack"] = []
    for seat in "AB":                                           # a dump lists one entry per permanent
        bf = []
        for p in fake["players"][seat]["battlefield"]:
            bf += [dict(p, count=1) for _ in range(p.get("count", 1))]
        fake["players"][seat]["battlefield"] = bf
    diffs = diff_dump(s, fake)
    assert any("A.hand" in d for d in diffs)
    assert any("B.battlefield" in d for d in diffs)
    clean = copy.deepcopy(s)
    clean["startingPlayer"] = "A"                               # turn 7, A active: A is on the play
    for seat in "AB":
        bf = []
        for p in clean["players"][seat]["battlefield"]:
            bf += [dict(p, count=1) for _ in range(p.get("count", 1))]
        clean["players"][seat]["battlefield"] = bf
    assert diff_dump(s, clean) == []
    assert any("startingPlayer" in d for d in diff_dump(s, dict(clean, startingPlayer="B")))


def test_diff_dump_compares_owner():
    """A control-changed permanent is listed under its controller with owner = the other seat; the
    dump says the same, and a dump that lost the owner (or a spec that states owner = its own seat,
    which is the default) is handled."""
    s = spec("main_phase")
    s["players"]["A"]["battlefield"].append({"name": "Courageous Goblin", "owner": "B", "id": "z"})
    dump = copy.deepcopy(s)
    for seat in "AB":
        bf = []
        for p in dump["players"][seat]["battlefield"]:
            bf += [dict(p, count=1) for _ in range(p.get("count", 1))]
        dump["players"][seat]["battlefield"] = bf
    dump["startingPlayer"] = "A"
    assert diff_dump(s, dump) == []
    lost = copy.deepcopy(dump)
    lost["players"]["A"]["battlefield"][-1].pop("owner")
    assert any("A.battlefield" in d for d in diff_dump(s, lost))
    same = copy.deepcopy(s)
    same["players"]["A"]["battlefield"][0]["owner"] = "A"            # owner = listing seat: the default
    assert diff_dump(same, dump) == []


def test_turn_start_options():
    eot = spec("eot_rollover")                                  # B's end step of turn 6
    assert bridge.turn_start_options(eot) == {"decisionPlayer": "A",
                                              "decideFrom": {"turn": 7, "step": "PRECOMBAT_MAIN"}}
    assert bridge.turn_start_options(spec("main_phase")) == {"decisionPlayer": "A"}


def test_spec_dict_accepts_statespec_dict_and_path():
    path = golden_specs()["respond"]
    as_path = bridge.spec_dict(path)
    assert bridge.spec_dict(as_path) is as_path
    assert bridge.spec_dict(StateSpec.from_dict(as_path))["players"]["A"]["hand"] == as_path["players"]["A"]["hand"]


# ------------------------------------------------------------------------------ worker

@needs_worker
def test_worker_ready_and_ping(worker):
    assert worker.ready["ok"] and worker.ready["deterministicIds"]
    assert worker.ready["actionDim"] == 1024
    assert worker.ping()["ok"]


@needs_worker
def test_golden_specs_round_trip(builds):
    for name, r in builds.items():
        assert r["warnings"] == [], name
        assert diff_dump(spec(name), r["dump"]) == [], name


@needs_worker
def test_main_phase_rules_and_legal_actions(builds):
    r = builds["main_phase"]
    dump = r["dump"]
    goblin = perm(dump, "B", "Courageous Goblin")          # enchanted with A's Pacifism
    assert goblin["x"]["canBlock"] is False and goblin["x"]["attachments"] == ["Pacifism"]
    seraph = perm(dump, "A", "Vanguard Seraph")            # came under A's control this turn
    assert seraph["sick"] and seraph["x"]["canAttack"] is False and seraph["x"]["canBlock"] is True
    prowler = perm(dump, "B", "Cackling Prowler")          # 4/3 with a +1/+1 counter and 1 damage
    assert (prowler["x"]["power"], prowler["x"]["toughness"], prowler["damage"]) == (5, 4, 1)
    beast = perm(dump, "B", "Beast Token")
    assert beast["tokenClass"] == "BeastToken" and (beast["x"]["power"], beast["x"]["toughness"]) == (3, 3)
    d = r["decision"]
    assert (d["player"], d["type"], d["text"]) == ("A", "PRIORITY", "priority")
    assert {(o["label"], o["idx"]) for o in d["legal"]} == {
        ("Cast Stab", 427), ("Cast Burglar Rat", 85), ("Cast Hero's Downfall", 249), ("Play Plains", 539), ("Pass", 0)}


@needs_worker
def test_mana_pool_library_and_exile(builds):
    a = builds["mana_library"]["dump"]["players"]["A"]
    assert a["manaPool"] == "BB" and a["librarySize"] == 10 and a["libraryTop"] == ["Squad Rallier"]
    assert builds["mana_library"]["dump"]["players"]["B"]["exile"] == ["Bushwhack"]


@needs_worker
def test_attack_decision_skips_pacified_creature(builds):
    r = builds["attack"]
    assert perm(r["dump"], "B", "Courageous Goblin")["x"]["canAttack"] is False
    d = r["decision"]
    assert d["player"] == "B" and d["type"] == "CHOOSE_USE"
    assert d["text"] in {f"attack with: {n}?" for n in ("Cackling Prowler", "Needletooth Pack", "Beast Token")}
    assert [(o["label"], o["idx"]) for o in d["legal"]] == [("no", 0), ("yes", 1)]


@needs_worker
def test_block_decision(builds):
    d = builds["block"]["decision"]
    assert d["player"] == "A" and d["type"] == "CHOOSE_TARGET" and d["where"]["step"] == "DECLARE_BLOCKERS"
    assert d["text"].startswith("choose which creature to block for ")
    assert {o["label"] for o in d["legal"]} == {"Cackling Prowler", "Needletooth Pack", "Beast Token", "Stop Choosing"}
    stop = [o for o in d["legal"] if o["label"] == "Stop Choosing"][0]
    assert stop["idx"] == 0


@needs_worker
def test_respond_to_stack(builds):
    r = builds["respond"]
    assert r["dump"]["stack"] == [{"controller": "B", "card": "Burst Lightning", "targets": ["A:rat"]}]
    d = r["decision"]
    assert d["player"] == "A" and d["type"] == "PRIORITY" and d["where"]["stack"] == 1
    assert {o["label"] for o in d["legal"]} == {"Cast Stab", "Cast Fleeting Flight", "Pass"}


@needs_worker
def test_eot_rollover_cleans_up_untaps_and_draws_known_top_card(builds):
    r = builds["eot_rollover"]
    d = r["decision"]
    assert (d["player"], d["where"]["turn"], d["where"]["step"]) == ("A", 7, "DRAW")
    assert {o["label"] for o in d["legal"]} == {"Cast Hero's Downfall", "Pass"}
    after = r["decisionState"]
    assert "Hero's Downfall" in after["players"]["A"]["hand"]                      # the known top card
    assert after["players"]["A"]["librarySize"] == r["dump"]["players"]["A"]["librarySize"] - 1
    assert perm(after, "B", "Cackling Prowler")["damage"] == 0                    # cleanup
    assert perm(after, "A", "Felidar Savior")["tapped"] is False                   # untap
    assert perm(after, "A", "Vanguard Seraph")["x"]["canAttack"] is True           # sickness wore off


@needs_worker
def test_decide_from_skips_earlier_windows(worker):
    """An end-of-turn snapshot's first decision is the draw-step instant; decideFrom moves the
    window to the main phase (the upkeep/draw windows are passed and recorded)."""
    r = worker.request("build", spec("eot_rollover"), seed=1, decisionPlayer="A",
                       decideFrom={"turn": 7, "step": "PRECOMBAT_MAIN"})
    d = r["decision"]
    assert (d["where"]["turn"], d["where"]["step"]) == (7, "PRECOMBAT_MAIN")
    assert {"Cast Hero's Downfall", "Cast Burglar Rat", "Play Plains", "Pass"} <= {o["label"] for o in d["legal"]}
    c = worker.request("coach", spec("eot_rollover"), seed=1, decisionPlayer="A", determinizations=1, budget=40,
                       decideFrom={"turn": 7, "step": "PRECOMBAT_MAIN"})
    assert c["decision"]["where"]["step"] == "PRECOMBAT_MAIN" and c["determinizations"][0]["rootVisits"] >= 40


@needs_worker
def test_begin_step_block_entry_searches_the_right_state(worker):
    """The research's anchor negative case (3b): entering DECLARE_BLOCKERS with BEGIN_STEP used to
    search from a stale or wrongly-resumed anchor (an NPE, or 'target id is null' and no blocks).
    Now it must reach the same block decision, with the same search, as the PRIORITY_HELD entry."""
    opts = dict(seed=3, decisionPlayer="A", determinizations=1, budget=60)
    held = worker.request("coach", spec("block"), **opts)
    begin = worker.request("coach", spec("block_beginstep"), **opts)
    assert begin["decision"]["type"] == "CHOOSE_TARGET"
    assert begin["decision"]["text"] == held["decision"]["text"]
    kids = begin["determinizations"][0]["children"]
    assert {c["label"] for c in kids} == {"Cackling Prowler", "Needletooth Pack", "Beast Token", "Stop Choosing"}
    assert sum(c["N"] for c in kids) > 0
    assert kids == held["determinizations"][0]["children"]


@needs_worker
def test_coach_is_deterministic_offline(worker):
    s = spec("main_phase")
    opts = dict(seed=11, determinizations=2, budget=80, humanAction="Play Plains")
    r1 = worker.request("coach", s, **opts)
    r2 = worker.request("coach", s, **opts)
    strip = lambda r: [[(c["label"], c["N"], c["Q"]) for c in d["children"]] for d in r["determinizations"]]
    assert strip(r1) == strip(r2)
    assert r1["aggregate"] == r2["aggregate"]
    assert all(d["rootVisits"] >= 80 for d in r1["determinizations"])     # fresh tree: the budget is new visits
    assert r1["consistent"] and r1["settings"]["resample"] == ["B"]
    labels = {a["label"] for a in r1["aggregate"]}
    assert labels == {"Cast Stab", "Cast Burglar Rat", "Cast Hero's Downfall", "Play Plains", "Pass"}
    h = r1["human"]
    assert h["found"] and h["label"] == "Play Plains" and h["regret"] >= 0 and 1 <= h["rank"] <= 5
    # the resampled opponent hands differ between determinizations but not between runs
    hands = [tuple(d["sampledHands"]["B"]) for d in r1["determinizations"]]
    assert hands == [tuple(d["sampledHands"]["B"]) for d in r2["determinizations"]]


@needs_worker
def test_coach_is_identical_across_workers(pool):
    """Same spec and seed on two JVMs with different histories: the same question and statistics."""
    s = spec("attack")
    opts = dict(seed=5, decisionPlayer="B", determinizations=2, budget=60, humanAction="attack")
    a, b = (w.request("coach", s, **opts) for w in pool.workers)
    assert a["decision"]["text"] == b["decision"]["text"] and a["consistent"]
    assert [d["children"] for d in a["determinizations"]] == [d["children"] for d in b["determinizations"]]
    assert a["human"]["found"] and a["human"]["label"] == "yes"   # yes/no aliases of a CHOOSE_USE answer


@needs_worker
def test_encode_features(worker):
    s = spec("main_phase")
    hidden = worker.request("encode", s, seed=1)
    full = worker.request("encode", s, seed=1, perfectInfo=True)
    assert hidden["decision"]["type"] == "PRIORITY" and hidden["nFeatures"] > 500
    assert hidden["features"] == sorted(hidden["features"])
    assert full["nFeatures"] > hidden["nFeatures"]                     # the opponent's hand adds features
    assert worker.request("encode", s, seed=1)["features"] == hidden["features"]
    blk = worker.request("encode", spec("block"), seed=1, decisionPlayer="A")
    assert blk["decision"]["type"] == "CHOOSE_TARGET" and blk["nFeatures"] > 500


@needs_worker
def test_hidden_hand_is_seed_deterministic(worker):
    s = spec("main_phase")
    b = s["players"]["B"]
    b["handUnknown"] = len(b.pop("hand"))
    hands = {seed: worker.request("build", s, seed=seed, advance=False)["dump"]["players"]["B"]["hand"]
             for seed in (1, 2, 3)}
    again = worker.request("build", s, seed=1, advance=False)["dump"]
    assert again["players"]["B"]["hand"] == hands[1] and len(hands[1]) == 3
    assert len({tuple(h) for h in hands.values()}) > 1
    assert diff_dump(s, again) == []


@needs_worker
def test_bad_specs_fail_with_clear_errors(worker):
    s = spec("main_phase")
    s["players"]["A"]["hand"].append("Llanowar Elves")
    with pytest.raises(BridgeError) as e:
        worker.request("build", s)
    assert any("missing from decklist" in p for p in e.value.problems)
    s = spec("main_phase")
    s["players"]["B"]["battlefield"][-1] = {"token": "Beast", "set": "FDN"}   # FDN has a 3/3 and a 4/4 Beast
    with pytest.raises(BridgeError, match="tokenClass"):
        worker.request("build", s)
    with pytest.raises(BridgeError, match="unknown op"):
        worker.request("fly")
    assert worker.ping()["ok"]                                       # the worker survives bad requests


@needs_worker
def test_encode_matches_magezero_root_encoding(worker):
    """`encode` must give exactly the features MageZero's own search computes for the same
    decision (the root's MCTSNode.stateVector), for each decision type and both perfectInfo
    settings; otherwise human rows and self-play rows would disagree on the input."""
    for name in ("main_phase", "attack", "block", "respond"):
        s = spec(name)
        for pi in (False, True):
            e = worker.request("encode", s, seed=2, decisionPlayer=decider(s), perfectInfo=pi)
            c = worker.request("coach", s, seed=2, decisionPlayer=decider(s), perfectInfo=pi, determinizations=1,
                               budget=5, resample=[], rootFeatures=True)
            assert e["decision"]["type"] == c["decision"]["type"], name
            assert e["features"] == c["determinizations"][0]["rootFeatures"], (name, pi)


@needs_worker
def test_leyline_in_opening_seven_is_not_put_onto_the_battlefield(worker):
    """init() draws the first seven decklist cards and offers opening-hand actions. A Leyline
    among them used to be put onto the battlefield: an extra permanent (or 'no Leyline Axe left in
    the library' when the spec had it in play). 14 of the first 770 17lands specs hit this."""
    s = spec("main_phase")
    deck = s["players"]["B"]["decklist"]
    deck.remove("Forest")
    deck.insert(0, "Leyline Axe")                               # among the seven cards init() draws
    r = worker.request("build", s, seed=1, advance=False)
    assert diff_dump(s, r["dump"]) == [] and r["warnings"] == []
    s["players"]["B"]["battlefield"].append({"name": "Leyline Axe"})
    assert diff_dump(s, worker.request("build", s, seed=1, advance=False)["dump"]) == []


@needs_worker
def test_starting_player_skips_the_turn_1_draw(worker):
    """The TurnMod that makes the player on the play skip its first draw is cleared by the
    injector (G2) and must come back when the spec is entered before turn 1's draw (Arena
    mulligan / starting-player decisions); the player used to draw an 8th card."""
    s = spec("main_phase")
    s.update(turn=1, phase="BEGINNING", step="UPKEEP", enterMode="BEGIN_STEP")
    for seat in "AB":
        p = s["players"][seat]
        p.update(hand=[], graveyard=[], exile=[], battlefield=[], libraryTop=[], handUnknown=7, life=20)
    s["players"]["B"]["exile"] = []
    for mode, step in (("BEGIN_STEP", "UPKEEP"), ("PRIORITY_FRESH", "UPKEEP"), ("BEGIN_STEP", "DRAW")):
        r = worker.request("build", dict(s, enterMode=mode, step=step), seed=1, decisionPlayer="A",
                           dumpDecisionState=True, decideFrom={"turn": 1, "step": "PRECOMBAT_MAIN"})
        assert r["decision"]["where"]["step"] == "PRECOMBAT_MAIN"
        assert len(r["decisionState"]["players"]["A"]["hand"]) == 7, (mode, step)
    bad = dict(s, startingPlayer="B")                           # turn 1 is the starting player's
    with pytest.raises(BridgeError, match="turn 1 belongs to the starting player"):
        worker.request("build", bad)


@needs_worker
def test_decision_player_default_and_turn_start_options(worker):
    """An end-of-turn snapshot of B's turn defaults to deciding for B (the active player), who
    has nothing to do in A's turn; turn_start_options asks for A's main phase instead, and
    `passedBefore` tells that earlier windows were passed."""
    s = spec("eot_rollover")
    r = worker.request("build", s, seed=1)
    assert r["decision"] is None and "decisionPlayer" in r["noDecision"]
    d = worker.request("build", s, seed=1, **bridge.turn_start_options(s))["decision"]
    assert (d["player"], d["where"]["turn"], d["where"]["step"]) == ("A", 7, "PRECOMBAT_MAIN")
    assert d["where"]["passedBefore"] >= 2                      # B's end step, A's draw step at least
    assert worker.request("build", spec("main_phase"), seed=1)["decision"]["where"]["passedBefore"] == 0
    odd = dict(spec("main_phase"), startingPlayer="B")          # B on the play, yet A active on turn 7
    assert any("startingPlayer" in w for w in worker.request("build", odd, advance=False)["warnings"])
    begin = dict(spec("mana_library"), enterMode="BEGIN_STEP")  # floating mana cannot survive into a new step
    assert any("manaPool" in w for w in worker.request("build", begin, advance=False)["warnings"])


@needs_worker
def test_pool_restarts_a_dead_worker(pool):
    w = pool.workers[1]
    w.proc.kill()
    w.proc.wait()
    rs = [pool.request("ping") for _ in range(3)]              # one of these lands on the dead worker
    assert all(r["ok"] for r in rs) and w.proc.poll() is None


@needs_worker
def test_planeswalker_defender_is_not_a_decision(worker):
    """Critique C5, documented: when the defending player controls a planeswalker, MageZero still
    asks only "attack with X?" (yes/no) and sends the attacker at whichever defender its HashSet
    yields first, so the choice of player vs planeswalker is not a decision the bridge can
    capture or coach. A spec can still state an attack on the planeswalker."""
    s = spec("attack")                                         # B attacks; A gets a planeswalker
    deck = s["players"]["A"]["decklist"]
    deck.remove("Plains")
    deck.append("Ajani, Caller of the Pride")
    s["players"]["A"]["battlefield"].append({"name": "Ajani, Caller of the Pride", "id": "ajani",
                                             "counters": {"LOYALTY": 3}})
    r = worker.request("build", s, seed=1, decisionPlayer="B")
    assert diff_dump(s, r["dump"]) == []
    d = r["decision"]
    assert d["type"] == "CHOOSE_USE" and [o["label"] for o in d["legal"]] == ["no", "yes"]
    held = dict(s, step="DECLARE_ATTACKERS", enterMode="PRIORITY_HELD", priorityPlayer="B",
                attackers=[{"attacker": "B:prowler", "defender": "A:ajani"}])
    r = worker.request("build", held, seed=1, decisionPlayer="A", advance=False)
    assert r["dump"]["attackers"] == [{"attacker": "B:prowler", "defender": "A:ajani"}]


# ------------------------------------------------------------------------------ bridge fixes (phase 2)

def with_cards(s: dict, seat: str, add: list[str], drop: list[str]) -> dict:
    """Swap decklist cards (the decklist must stay >= 40 and account for every named card)."""
    deck = s["players"][seat]["decklist"]
    for new, old in zip(add, drop):
        deck[deck.index(old)] = new
    return s


def angel_and_tokens() -> dict:
    """main_phase with 'whenever a creature enters' permanents next to injected tokens on both sides."""
    s = with_cards(spec("main_phase"), "A", ["Dazzling Angel"], ["Vanguard Seraph"])
    with_cards(s, "B", ["Authority of the Consuls"], ["Forest"])
    a, b = s["players"]["A"], s["players"]["B"]
    a["battlefield"] = [x for x in a["battlefield"] if x.get("name") != "Vanguard Seraph"]
    a["battlefield"] += [{"name": "Dazzling Angel"}, {"tokenClass": "FaerieToken", "count": 2}]
    b["battlefield"] += [{"name": "Authority of the Consuls"}, {"tokenClass": "GoblinToken", "count": 3}]
    return s


@needs_worker
def test_injected_tokens_fire_no_enter_triggers(worker):
    """Tokens used to enter through Token.putOntoBattlefield, whose ENTERS_THE_BATTLEFIELD events
    wait in GameState's simultaneous-event queue until the game resumes, i.e. after the injector
    cleared the pending triggers: Dazzling Angel / Authority of the Consuls triggered once per
    injected token (17lands row 560303 u9: 6 triggers on the stack, +6 life after rollover)."""
    s = angel_and_tokens()
    r = worker.request("build", s, seed=1, decisionPlayer="A", dumpDecisionState=True)
    assert diff_dump(s, r["dump"]) == [] and r["warnings"] == []
    assert r["decision"]["where"]["stack"] == 0 and r["decision"]["where"]["passedBefore"] == 0
    assert {x: r["decisionState"]["players"][x]["life"] for x in "AB"} == {"A": 13, "B": 9}
    # through a whole rollover (B's end step -> A's next main phase) nothing drifts either
    eot = dict(copy.deepcopy(s), turn=6, activePlayer="B", phase="END", step="END_TURN")
    r = worker.request("build", eot, seed=1, dumpDecisionState=True, **bridge.turn_start_options(eot))
    assert r["decision"]["where"]["step"] == "PRECOMBAT_MAIN"
    assert {x: r["decisionState"]["players"][x]["life"] for x in "AB"} == {"A": 13, "B": 9}


@needs_worker
def test_later_enter_triggers_still_fire(worker):
    """Only injection is event-free: a creature that enters after the resume triggers as usual.
    A's Burglar Rat resolves (B passed, A has only a land): Dazzling Angel gains exactly 1 life
    (not 1 + 2 for the injected Faeries) and the Rat's own trigger makes B discard."""
    s = angel_and_tokens()
    s["players"]["A"]["hand"] = ["Plains"]
    s.update(enterMode="PRIORITY_HELD", priorityPlayer="A", passedPlayers=["B"],
             stack=[{"controller": "A", "card": "Burglar Rat"}])
    r = worker.request("build", s, seed=1, decisionPlayer="A", dumpDecisionState=True)
    ds = r["decisionState"]
    assert (r["decision"]["where"]["turn"], r["decision"]["where"]["step"]) == (7, "PRECOMBAT_MAIN")
    assert ds["players"]["A"]["life"] == 14 and len(ds["players"]["B"]["hand"]) == 2
    assert "Burglar Rat" in {p["x"]["name"] for p in ds["players"]["A"]["battlefield"]}


@needs_worker
def test_fixture_row_with_tokens_rolls_over_without_bogus_triggers(worker):
    """The 17lands evidence: fixture row 4, user turn 7 (Dazzling Angel + 2 Faerie tokens). The
    main1 decision used to have 2 Angel triggers on the stack and the rollover gained 2 life."""
    from pathlib import Path
    from draftzero.gameplay import reconstruct as rc, replay
    from draftzero.gameplay.ids import Ids
    fixture = Path(__file__).parent / "fixtures" / "gameplay" / "fdn_premier_rows.csv.gz"
    if not fixture.exists():
        pytest.skip("17lands fixture missing")
    ids = Ids.load()
    g = next(g for g in replay.iter_games(fixture) if g.row_index == 4)
    m = rc.state_at_user_turn(g, 7, "main1", ids=ids)
    e = rc.state_at_user_turn(g, 7, ids=ids)
    assert any(p.tokenClass == "FaerieToken" for p in m.players["A"].battlefield)
    r = worker.request("build", m, seed=1, decisionPlayer="A")
    assert diff_dump(m, r["dump"]) == [] and r["decision"]["where"]["stack"] == 0
    r = worker.request("build", e, seed=1, decisionPlayer="A", dumpDecisionState=True,
                       decideFrom={"turn": g.user_slot(7).global_turn, "step": "PRECOMBAT_MAIN"})
    assert {x: r["decisionState"]["players"][x]["life"] for x in "AB"} == {x: m.players[x].life for x in "AB"}


def animate_dead_spec() -> dict:
    """A reanimated B's Courageous Goblin with Animate Dead: A controls it, B owns it (Perm.owner)."""
    s = with_cards(spec("main_phase"), "A", ["Animate Dead"], ["Plains"])
    s["players"]["A"]["battlefield"] += [{"name": "Courageous Goblin", "owner": "B", "id": "zombie"},
                                         {"name": "Animate Dead", "id": "ad", "attachTo": "A:zombie"}]
    return s


@needs_worker
def test_control_changed_permanent_with_animate_dead(worker):
    """Perm.owner: the card comes out of B's library and enters under A's control (originalControllerId
    = A, as a reanimation does), so control does not revert when effects are reapplied, across a
    whole turn. Animate Dead enchants 'creature card in a graveyard' until its ETB trigger (not
    fired by injection) swaps it: the bridge rebuilds that swap and the sacrifice link. This was
    46 of the 105 real Arena specs ('cannot attach: A: Animate Dead -> ... was refused by XMage')."""
    s = animate_dead_spec()
    assert StateSpec.from_dict(s).validate() == []
    r = worker.request("build", s, seed=1, decisionPlayer="A", dumpDecisionState=True)
    assert diff_dump(s, r["dump"]) == [] and r["warnings"] == []
    for state in (r["dump"], r["decisionState"]):
        zombie = perm(state, "A", "Courageous Goblin")
        assert zombie["owner"] == "B" and zombie["x"]["attachments"] == ["Animate Dead"]
        assert zombie["x"]["power"] == 1 and zombie["x"]["canAttack"]        # 2/2, Animate Dead gives -1/-0
        assert perm(state, "A", "Animate Dead")["attachTo"] == "A:zombie"
    assert not any(p.get("owner") for p in r["dump"]["players"]["B"]["battlefield"])
    # a turn later (B's end step -> A's main phase) A still controls it
    eot = dict(copy.deepcopy(s), turn=6, activePlayer="B", phase="END", step="END_TURN")
    r = worker.request("build", eot, seed=1, dumpDecisionState=True, **bridge.turn_start_options(eot))
    assert r["decision"]["where"]["turn"] == 7
    assert perm(r["decisionState"], "A", "Courageous Goblin")["owner"] == "B"
    # destroying Animate Dead sacrifices the creature, which goes to its owner's graveyard
    d = with_cards(copy.deepcopy(s), "B", ["Disenchant"], ["Forest"])
    d["players"]["A"]["hand"] = ["Plains"]
    d.update(turn=8, activePlayer="B", enterMode="PRIORITY_HELD", priorityPlayer="A", passedPlayers=["B"],
             stack=[{"controller": "B", "card": "Disenchant", "targets": ["A:ad"]}])
    ds = worker.request("build", d, seed=1, decisionPlayer="A", dumpDecisionState=True)["decisionState"]
    assert "Courageous Goblin" not in {p["x"]["name"] for p in ds["players"]["A"]["battlefield"]}
    assert "Courageous Goblin" in ds["players"]["B"]["graveyard"] and "Animate Dead" in ds["players"]["A"]["graveyard"]


@needs_worker
def test_owner_validation_matches_statespec(worker):
    """Spec.validate (Java) and StateSpec.validate (Python) accept and reject the same specs, the
    owner's decklist supplying a control-changed card."""
    base = animate_dead_spec()

    def case(fn):
        s = copy.deepcopy(base)
        fn(s)
        return s

    zombie = lambda s: s["players"]["A"]["battlefield"][-2]                      # noqa: E731
    cases = {
        "ok": case(lambda s: None),
        "owner is the listing seat": case(lambda s: zombie(s).update(owner="A", name="Vampire Soulcaller")),
        "not in the owner's decklist": case(lambda s: zombie(s).update(name="Vampire Soulcaller")),
        "owner's copies used up": case(lambda s: s["players"]["B"]["hand"].extend(["Courageous Goblin"] * 2)),
        "owner on a token": case(lambda s: s["players"]["A"]["battlefield"].append({"tokenClass": "BeastToken",
                                                                                     "owner": "B"})),
        "owner's library too small": case(lambda s: s["players"]["B"].update(handUnknown=40)),
    }
    for name, s in cases.items():
        py = StateSpec.from_dict(s).validate()
        try:
            worker.request("build", s, seed=1, advance=False)
            java = []
        except BridgeError as e:
            java = e.problems
        assert bool(py) == bool(java), (name, py, java)
        assert bool(py) == (name not in ("ok", "owner is the listing seat")), (name, py)


@needs_worker
def test_counters_set_the_count(worker):
    """Spec counters SET the count of their type on top of what the card entered with (a saga's
    first lore counter, a planeswalker's printed loyalty) instead of adding to it, at the build
    and at the decision; an unlisted type keeps the entered-with count (the arena review had
    seen loyalty 4 built as 7)."""
    s = with_cards(spec("main_phase"), "A", ["Ajani, Caller of the Pride", "Fable of the Mirror-Breaker"],
                   ["Plains", "Plains"])
    s["players"]["A"]["battlefield"] += [{"name": "Ajani, Caller of the Pride", "counters": {"LOYALTY": 7}},
                                         {"name": "Fable of the Mirror-Breaker", "counters": {"LORE": 2}}]
    r = worker.request("build", s, seed=1, decisionPlayer="A", dumpDecisionState=True)
    assert diff_dump(s, r["dump"]) == [] and r["decision"]["where"]["stack"] == 0
    for state in (r["dump"], r["decisionState"]):
        assert perm(state, "A", "Ajani, Caller of the Pride")["counters"] == {"LOYALTY": 7}
        assert perm(state, "A", "Fable of the Mirror-Breaker")["counters"] == {"LORE": 2}
    for p in s["players"]["A"]["battlefield"][-2:]:
        p.pop("counters")
    dump = worker.request("build", s, seed=1, advance=False)["dump"]
    assert perm(dump, "A", "Ajani, Caller of the Pride")["counters"] == {"LOYALTY": 4}      # printed
    assert perm(dump, "A", "Fable of the Mirror-Breaker")["counters"] == {"LORE": 1}


@needs_worker
def test_no_land_play_in_the_opponents_turn(worker):
    """Active-player bookkeeping: A holding priority in B's main phase (PRIORITY_HELD, with or
    without a spell on the stack) is offered instants only, never its land. The Arena review's
    'land plays during B's main phase' were decisions of A's NEXT turn (where.turn 6, active A):
    its only instant had been substituted by a Plains, so the bridge passed on to A's turn."""
    for stack in ([{"controller": "B", "card": "Burst Lightning", "targets": ["A:rat"]}], []):
        s = dict(spec("respond"), stack=stack)
        d = worker.request("build", s, seed=1, decisionPlayer="A")["decision"]
        assert (d["where"]["turn"], d["where"]["activePlayer"], d["where"]["passedBefore"]) == (8, "B", 0)
        assert {o["label"] for o in d["legal"]} == {"Cast Stab", "Cast Fleeting Flight", "Pass"}


@needs_worker
def test_land_on_the_stack_is_a_clear_error(worker):
    """Card.cast on a land threw 'NullPointerException ... SpellAbility.getSpellAbilityToResolve'
    (the arena review's 3 stack NPEs were substituted Plains on the stack)."""
    s = with_cards(spec("respond"), "B", ["Forest"], ["Burst Lightning"])
    s["stack"] = [{"controller": "B", "card": "Forest"}]
    with pytest.raises(BridgeError) as e:
        worker.request("build", s, seed=1)
    assert e.value.error["type"] == "SpecException" and "is not a spell" in e.value.problems[0]


@needs_worker
def test_substitute_missing_cards(worker):
    """Default strict: every card XMage 1.4.58 lacks is listed in one error. options.substitute
    builds the spec with stand-ins and says what it replaced or dropped."""
    missing = "Wan Shi Tong, Librarian"                         # 2026 card in the Arena log, not in 1.4.58
    s = with_cards(spec("respond"), "A", [missing, "Moonshadow"], ["Pilfer", "Stab"])
    s["players"]["A"]["hand"] = [missing, "Fleeting Flight", "Plains"]
    s["players"]["A"]["battlefield"].append({"name": "Moonshadow", "id": "moon", "counters": {"P1P1": 1}})
    with_cards(s, "B", [missing], ["Burst Lightning"])
    s["stack"] = [{"controller": "B", "card": missing, "targets": ["A:rat"]}]
    with pytest.raises(BridgeError) as e:
        worker.request("build", s, seed=1)
    assert "2 card(s) not in the XMage card database" in e.value.problems[0] and missing in e.value.problems[0]
    r = worker.request("build", s, seed=1, decisionPlayer="A", substitute={"missing": "Plains"})
    assert r["substitutions"] == {missing: "Plains", "Moonshadow": "Plains"}
    w = "\n".join(r["warnings"])
    assert f"'{missing}' -> 'Plains' (not in the XMage card database)" in w and "dropped B's" in w
    assert r["dump"]["stack"] == [] and r["dump"]["players"]["A"]["hand"].count("Plains") == 2
    assert {o["label"] for o in r["decision"]["legal"]} == {"Cast Fleeting Flight", "Pass"}
    # an explicit map (applied even to known cards), and tokens XMage cannot resolve
    t = spec("main_phase")
    t["players"]["B"]["battlefield"].append({"token": "Boo", "set": "HBG", "id": "boo", "counters": {"P1P1": 2}})
    with pytest.raises(BridgeError, match="no token 'Boo'"):
        worker.request("build", t)
    r = worker.request("build", t, seed=1, advance=False,
                       substitute={"missingToken": "GoblinToken", "Hero's Downfall": "Stab"})
    assert r["substitutions"] == {"Hero's Downfall": "Stab", "token Boo/HBG": "GoblinToken"}
    assert r["dump"]["players"]["A"]["hand"].count("Stab") == 2
    assert perm(r["dump"], "B", "Goblin Token")["counters"] == {"P1P1": 2}
    with pytest.raises(BridgeError, match="not in the XMage card database either"):
        worker.request("build", t, substitute={"missing": "Not A Real Card"})


@needs_worker
def test_coach_over_given_determinizations(worker):
    """options.specs: K concrete determinizations of one decision (the belief model's samples;
    here B's hand and one unseen decklist card differ). One fresh search per spec, one shared
    idSeed, the same question; humanActions reports a set-valued human action."""
    base = spec("main_phase")
    hands = [["Burst Lightning", "Forest", "Bite Down"], ["Snakeskin Veil", "Mountain", "Felling Blow"],
             ["Forest", "Forest", "Bushwhack"]]
    specs = []
    for i, h in enumerate(hands):
        s = copy.deepcopy(base)
        s["players"]["B"]["hand"] = h
        s["players"]["B"]["exile"] = [] if "Bushwhack" in h else ["Bushwhack"]
        if i == 2:
            with_cards(s, "B", ["Mountain"], ["Sower of Chaos"])
        specs.append(s)
    opts = dict(seed=3, budget=40, decisionPlayer="A", humanAction="Play Plains",
                humanActions=["Play Plains", "Cast Burglar Rat", "Cast Llanowar Elves"])
    r = worker.coach_specs(specs, **opts)
    assert r["consistent"] and r["settings"]["specs"] == 3 and r["settings"]["resample"] == []
    assert [d["hands"]["B"] for d in r["determinizations"]] == hands
    assert {a["nDet"] for a in r["aggregate"]} == {3}
    assert r["human"]["found"] and r["human"]["label"] == "Play Plains"            # the single form still works
    hs = r["humanSet"]
    assert [m["found"] for m in hs["members"]] == [True, True, False] and hs["nLegal"] == 2
    q = {a["label"]: a["meanQ"] for a in r["aggregate"]}
    best = r["aggregate"][0]["meanQ"]
    members = [q["Play Plains"], q["Cast Burglar Rat"]]
    assert hs["regret_best_of_set"] == pytest.approx(best - max(members))
    assert hs["regret_mean_of_set"] == pytest.approx(best - sum(members) / 2)
    assert 0 <= hs["regret_best_of_set"] <= hs["regret_mean_of_set"]
    again = worker.coach_specs(specs, **opts)
    assert [d["children"] for d in again["determinizations"]] == [d["children"] for d in r["determinizations"]]
    with pytest.raises(BridgeError, match="do not also pass resample"):
        worker.coach_specs(specs, resample=["B"], budget=5)


@needs_worker
def test_card_ids_do_not_depend_on_the_rest_of_the_decklist(worker):
    """Determinizations with different belief decklists must ask the same question. Attack and
    block questions come in UUID order, and card UUIDs used to be one sequential stream over both
    decklists, so changing one of A's library cards renumbered every B card."""
    s = spec("attack")
    other = with_cards(copy.deepcopy(s), "A", ["Vampire Soulcaller"], ["Swamp"])   # more abilities: more ids
    dumps = [worker.request("build", x, seed=5, advance=False)["dump"] for x in (s, other)]
    ids = [{p["x"]["name"]: p["x"]["uuid"] for p in d["players"]["B"]["battlefield"]} for d in dumps]
    assert ids[0] == ids[1]
    texts = [worker.request("build", x, seed=5, decisionPlayer="B")["decision"]["text"] for x in (s, other)]
    assert texts[0] == texts[1]


@needs_worker
def test_pregame_spec_is_deterministic(pool):
    """The Arena ChooseStartingPlayerReq spec (turn 1 upkeep, empty hands) has no decision for A
    before the safety stop: A skips its first draw, B's turn 2 gives A nothing to do. Its one-off
    PRIORITY decision in the arena review came from a jar before the turn-1 draw fix (A drew an
    8th card: here, the first card). Same answer on both workers and on repeats."""
    s = spec("main_phase")
    s.update(turn=1, phase="BEGINNING", step="UPKEEP", enterMode="BEGIN_STEP", startingPlayer="A")
    for seat in "AB":
        s["players"][seat].update(hand=[], graveyard=[], exile=[], battlefield=[], libraryTop=[], handUnknown=0,
                                  life=20)
    out = [w.request("build", s, seed=1, decisionPlayer="A") for w in pool.workers for _ in range(2)]
    assert all(o["decision"] is None for o in out)
    assert len({o["noDecision"] for o in out}) == 1 and "end of turn 2" in out[0]["noDecision"]


@needs_worker
def test_injection_answers_no_to_enter_questions(worker):
    """Injected permanents ask 'as this enters' questions: the puppet used to pay 2 life for a
    shock land (39 of 105 Arena specs), whose queued life-loss event then triggered A's
    Bloodthirsty Conqueror ('whenever an opponent loses life, you gain that much') on resume.
    Injection now declines every question and drops whatever events it queued."""
    s = with_cards(spec("main_phase"), "A", ["Bloodthirsty Conqueror"], ["Vanguard Seraph"])
    a = s["players"]["A"]
    a["battlefield"] = [x for x in a["battlefield"] if x.get("name") != "Vanguard Seraph"]
    a["battlefield"].append({"name": "Bloodthirsty Conqueror"})
    with_cards(s, "B", ["Stomping Ground"], ["Forest"])
    s["players"]["B"]["battlefield"].append({"name": "Stomping Ground"})
    r = worker.request("build", s, seed=1, decisionPlayer="A", dumpDecisionState=True)
    assert diff_dump(s, r["dump"]) == [] and r["warnings"] == []
    assert r["decision"]["where"]["stack"] == 0
    assert {x: r["decisionState"]["players"][x]["life"] for x in "AB"} == {"A": 13, "B": 9}
    assert perm(r["dump"], "B", "Stomping Ground")["tapped"] is False           # the spec's state wins
