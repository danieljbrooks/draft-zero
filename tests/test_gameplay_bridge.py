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
