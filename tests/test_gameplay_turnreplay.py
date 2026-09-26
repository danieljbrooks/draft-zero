"""Tests for the replay_turn op (java/mzbridge TurnReplay) and its driver (gameplay.turnreplay).

The request tests need only the committed 17lands fixture. The replay tests need java and the
XMage build and skip cleanly without them; they run one bridge worker under a pytest temporary
runtime root (one JVM, as the shared machine allows) and take a few seconds.
"""
import copy
from collections import Counter
from pathlib import Path

import pytest

from draftzero.gameplay import bridge
from draftzero.gameplay.ids import Ids
from draftzero.gameplay.replay import read_games
from draftzero.gameplay.statespec import StateSpec
from draftzero.gameplay.turnreplay import category, expected_snapshot, replay_request, replay_turn, summarize

FIX = Path(__file__).parent / "fixtures" / "gameplay"
MAIN = FIX / "fdn_premier_rows.csv.gz"
EXTRA = FIX / "fdn_premier_rows_extra.csv.gz"

PROBLEMS = bridge.environment_problems()
needs_worker = pytest.mark.skipif(bool(PROBLEMS), reason="mzbridge unavailable: " + "; ".join(PROBLEMS))

# (file, row, user turn): turns that must reproduce, each exercising a different part of the op
GOLDEN = [
    (MAIN, 4, 3),      # B counters A's Dazzling Angel in response (Refute): the respond window
    (MAIN, 4, 7),      # B's Aetherize bounces A's attackers; A recasts Drake Hatcher after combat (main2 attempt)
    (MAIN, 42, 4),     # Witness Protection on the blocker the snapshot says dies; a forced ("attacks each combat") hasty attacker
    (MAIN, 0, 6),      # a modal sorcery (-1/-1 to B's creatures), a planeswalker +1 with a discard, one block
    (MAIN, 8, 6),      # two recorded blocks and B's flashback Think Twice
    (EXTRA, 89383, 8),  # two B blockers on one attacker, B's instant in the turn
]


@pytest.fixture(scope="module")
def ids():
    return Ids.load()


@pytest.fixture(scope="module")
def games():
    out = {}
    for path, rows in ((MAIN, [0, 4, 8, 42]), (EXTRA, [89383])):
        out.update(read_games(rows, path))
    return out


@pytest.fixture(scope="module")
def worker(tmp_path_factory):
    w = bridge.Bridge("pytest-turnreplay", heap="2500m", runtime_root=tmp_path_factory.mktemp("mzbridge"))
    yield w
    w.close()


# ------------------------------------------------------------------------------ no JVM needed

def test_request_holds_script_expected_and_a_valid_spec(games, ids):
    spec, opt, meta = replay_request(games[4], 3, ids)
    assert StateSpec.from_dict(spec).validate() == []
    s = opt["script"]
    assert (s["seat"], s["turn"]) == ("A", 6)
    assert [x["key"] for x in s["lands"]] == ["Play Plains"]
    assert [x["key"] for x in s["casts"]] == ["Cast Dazzling Angel"]
    assert s["attacks"] == {"A:DrakeHatcher_6": True}
    assert [x["key"] for x in s["opp"]] == ["Cast Refute"]
    # B's instant must be in its hand: out of the hidden cards, and in its (placeholder) decklist
    assert "Refute" in spec["players"]["B"]["hand"] and "Refute" in spec["players"]["B"]["decklist"]
    # the turn's later draws sit on A's library in order (future_draws), after the natural draw
    assert spec["players"]["A"]["libraryTop"] == ["Banishing Light"]
    e = opt["expected"]
    assert e["life"] == {"A": 20, "B": 19}
    assert Counter(e["battlefield"]["A"]) == Counter({"Plains": 2, "Island": 1, "Drake Hatcher": 1})
    assert e["hand"]["unknownA"] == 0 and "Dazzling Angel" not in e["hand"]["A"]
    assert meta["tier"] == spec["provenance"]["tier"] and meta["row"] == 4


def test_opponent_blocks_become_pairings_of_spec_aliases(games, ids):
    _, opt, _ = replay_request(games[8], 6, ids)
    s = opt["script"]
    assert s["blockPairing"] == "unique"
    assert s["blocks"] == [[["B:CampusGuide_6", "A:SlumberingCerberus_4"], ["B:AegisTurtle_9", "A:SpitfireLagac_14"]]]
    _, opt, _ = replay_request(games[89383], 8, ids)
    assert sorted(b for b, a in opt["script"]["blocks"][0]) == ["B:Cat_16", "B:Soldier_8"]
    assert {a for b, a in opt["script"]["blocks"][0]} == {"A:Soldier_24"}


def test_expected_snapshot_names_tokens_like_the_engine(games, ids):
    # 17lands names the token "Faerie"; XMage's permanent is "Faerie Token": both become "Faerie"
    e = expected_snapshot(games[4], 6, ids)
    assert e["battlefield"]["A"].count("Faerie") == 2


def test_summarize_rates_categories_and_weights():
    def row(ok, tier, turn, keys=(), t=10):
        return {"ok": ok, "attempt": 1 if ok else 3, "diffKeys": list(keys), "timing_ms": {"total": t},
                "meta": {"row": 1, "turn": turn, "tier": tier}, "policy": {"order": "mv_asc"}}
    rows = [row(True, "T0", 3), row(True, "T0", 4), row(False, "T1", 5, ["life:B", "undone:B"]),
            row(False, "T1", 11, ["undone:A", "hand:A:extra"])]
    s = summarize(rows, Counter({"T0": 3, "T1": 1}))
    assert s["overall"] == {"n": 4, "ok": 2, "rate": 0.5}
    assert s["by_tier"]["T0"]["rate"] == 1.0 and s["by_tier"]["T1"]["rate"] == 0.0
    assert s["overall_natural_weights"]["rate"] == 0.75
    assert {c["category"] for c in s["fail_categories"]} == {"life:B", "undone:A"}
    assert category(rows[3]) == "undone:A" and category(rows[0]) == "ok"
    assert set(s["by_turn"]) == {"3", "4", "5", "10+"}


# ------------------------------------------------------------------------------ the op

@needs_worker
@pytest.mark.parametrize("path,row,turn", GOLDEN)
def test_golden_turns_reproduce(worker, ids, path, row, turn):
    g = read_games([row], path)[row]
    r = replay_turn(worker, g, turn, ids)
    assert r["ok"] and r["reproduced"], (r.get("diff"), r.get("attemptLog"))
    # every recorded play of A's happened (B's may not show in the snapshot: its hand is not compared)
    assert all(it["done"] for it in r["items"] if it["seat"] == "A"), r["items"]
    kinds = {d["label_kind"] for d in r["decisions"]}
    assert kinds <= {"exact", "imputed_order", "guessed_target"}
    for d in r["decisions"]:
        assert d["chosen"] in {o["label"] for o in d["legal"]}, d
        assert len(d["legal"]) >= 2
    attacks = [d for d in r["decisions"] if d["text"].startswith("attack with: ")]
    assert attacks and all(d["label_kind"] in ("exact", "imputed_order") for d in attacks)


@needs_worker
def test_counterspell_and_decision_labels(worker, games, ids):
    r = replay_turn(worker, games[4], 3, ids)
    assert r["ok"]
    b = next(it for it in r["items"] if it["seat"] == "B")
    assert (b["key"], b["due"], b["done"]) == ("Cast Refute", "respond", True)
    d = r["decisions"]
    assert (d[0]["type"], d[0]["chosen"], d[0]["label_kind"]) == ("PRIORITY", "Play Plains", "imputed_order")
    assert set(d[0]["set"]) == {"Play Plains"} | ({"Cast Dazzling Angel"} & {o["label"] for o in d[0]["legal"]})
    att = [x for x in d if x["type"] == "CHOOSE_USE"]
    assert [(x["text"], x["chosen"], x["label_kind"]) for x in att] == [("attack with: Drake Hatcher?", "yes", "exact")]
    assert r["end"]["battlefield"]["A"].count("Dazzling Angel") == 0


@needs_worker
def test_wrong_labels_fail_with_a_clear_diff(worker, games, ids):
    spec, opt, _ = replay_request(games[4], 3, ids)
    # the Drake Hatcher did not attack: B keeps its life, whatever the attempts try
    wrong = copy.deepcopy(opt)
    wrong["script"]["attacks"] = {"A:DrakeHatcher_6": False}
    r = worker.request("replay_turn", spec, seed=0, **wrong)
    assert r["reproduced"] is False
    assert r["diff"]["life"] == {"B": [19, 20]} and "life:B" in r["diffKeys"]
    assert r["attempts"] == len(r["attemptLog"]) >= 2 and not any(a["ok"] for a in r["attemptLog"])
    # no land drop: the Plains stays in hand and is missing from the battlefield
    wrong = copy.deepcopy(opt)
    wrong["script"]["lands"] = []
    r = worker.request("replay_turn", spec, seed=0, **wrong)
    assert r["reproduced"] is False
    assert "Plains" in r["diff"]["hand"]["extra"] and "Plains" in r["diff"]["battlefield"]["A"]["missing"]
    assert {"hand:A:extra", "bf:A:missing"} <= set(r["diffKeys"])
    # a recorded cast that cannot happen: the end state still matches, but a turn whose recorded
    # play never happened is not reproduced (its decisions would miss the play)
    wrong = copy.deepcopy(opt)
    wrong["script"]["casts"].append({"kind": "cast", "key": "Cast Serra Angel", "name": "Serra Angel"})
    r = worker.request("replay_turn", spec, seed=0, maxAttempts=2, **wrong)
    assert r["reproduced"] is False
    assert "A: Cast Serra Angel" in r["diff"]["undone"] and r["diffKeys"] == ["undone:A"]
    assert not {"life", "hand", "battlefield"} & set(r["diff"])
    # a recorded action the request could not script (an ability with no XMage key): the turn is
    # not reproduced, and no Pass is labelled exact (the human did something at some point). A
    # Stab in hand that the human kept makes Pass a real choice once the recorded plays are done
    s3, o3 = copy.deepcopy(spec), copy.deepcopy(opt)
    _add(s3, "A", "battlefield", _perm("Swamp"), _perm("Swamp"), _perm("Swamp"))  # one survives the Angel's auto-tap
    _add(s3, "A", "hand", "Stab")
    o3["expected"]["battlefield"]["A"] += ["Swamp"] * 3
    o3["expected"]["hand"]["A"] = sorted(o3["expected"]["hand"]["A"] + ["Stab"])
    r = worker.request("replay_turn", s3, seed=0, maxAttempts=1, **o3)
    passes = [d for d in r["decisions"] if d["type"] == "PRIORITY" and d["chosen"] == "Pass"]
    assert r["reproduced"] and passes and {d["label_kind"] for d in passes} == {"exact"}
    o3["script"]["unscripted"] = 1
    r = worker.request("replay_turn", s3, seed=0, maxAttempts=1, **o3)
    assert r["reproduced"] is False and r["diffKeys"] == ["unscripted:A"]
    passes = [d for d in r["decisions"] if d["type"] == "PRIORITY" and d["chosen"] == "Pass"]
    assert passes and {d["label_kind"] for d in passes} == {"imputed_order"}


def _perm(name, alias=None, tapped=False):
    p = {"name": name, "count": 1, "tapped": tapped, "sick": False, "damage": 0, "faceDown": False}
    if alias:
        p["id"] = alias
    return p


def _add(spec, seat, zone, *cards):
    """Add cards to a seat's zone and its decklist (the accounting stays exact)."""
    P = spec["players"][seat]
    for c in cards:
        if zone == "hand":
            P["hand"] = sorted(P["hand"] + [c])
        else:
            P.setdefault("battlefield", []).append(c)
        P["decklist"] = sorted(P["decklist"] + [c if zone == "hand" else c["name"]])


@needs_worker
def test_recorded_attack_or_block_that_did_not_happen_fails_the_verdict(worker, games, ids):
    # the end state matches in both cases; before the review both came back reproduced
    spec, opt, _ = replay_request(games[4], 3, ids)
    # a recorded attacker that cannot attack (Defender): never asked, so the engine's declarations decide
    s1, o1 = copy.deepcopy(spec), copy.deepcopy(opt)
    _add(s1, "A", "battlefield", _perm("Gleaming Barrier", "GleamingBarrier_92"))
    o1["script"]["attacks"]["A:GleamingBarrier_92"] = True
    o1["expected"]["battlefield"]["A"].append("Gleaming Barrier")
    r = worker.request("replay_turn", s1, seed=0, **o1)
    assert r["reproduced"] is False
    assert r["diff"]["divergence"] == {"attack:A:unrealised": ["A:GleamingBarrier_92"]}
    assert "attack:A:unrealised" in r["diffKeys"] and not {"life", "hand", "battlefield"} & set(r["diff"])
    # a recorded block by a creature that is tapped
    s2, o2 = copy.deepcopy(spec), copy.deepcopy(opt)
    _add(s2, "B", "battlefield", _perm("Savannah Lions", "SavannahLions_93", tapped=True))
    o2["script"]["blocks"] = [[["B:SavannahLions_93", "A:DrakeHatcher_6"]]]
    o2["expected"]["battlefield"]["B"].append("Savannah Lions")
    r = worker.request("replay_turn", s2, seed=0, maxAttempts=3, **o2)
    assert r["reproduced"] is False
    assert r["diff"]["divergence"] == {"block:B:unrealised": ["B:SavannahLions_93 -> A:DrakeHatcher_6"]}
    assert not {"life", "hand", "battlefield"} & set(r["diff"])


@needs_worker
def test_harmful_targets_prefer_the_other_seats_doomed_creature(worker, games, ids):
    # A's Stab and B's Stab; one creature of each seat dies, both outside combat: each Stab kills
    # the other seat's creature. Before the review A's Stab killed A's own (bigger) Eager
    # Trufflesnout, B's Stab then killed B's own Frogmite, and the end state still matched
    spec, opt, _ = replay_request(games[4], 3, ids)
    _add(spec, "A", "battlefield", _perm("Swamp"), _perm("Eager Trufflesnout", "EagerTrufflesnout_90"))
    _add(spec, "A", "hand", "Stab")
    _add(spec, "B", "battlefield", _perm("Swamp"), _perm("Frogmite", "Frogmite_91"))
    _add(spec, "B", "hand", "Stab")
    s = opt["script"]
    s["casts"].append({"kind": "cast", "key": "Cast Stab", "name": "Stab", "family": "instant_sorcery"})
    s["opp"].append({"kind": "cast", "key": "Cast Stab", "name": "Stab", "family": "instant_sorcery", "mv": 1})
    # no attack (Drake Hatcher's prowess would make the Stab's timing matter)
    s["attacks"] = {"A:DrakeHatcher_6": False, "A:EagerTrufflesnout_90": False}
    e = opt["expected"]
    e["life"]["B"] = 20
    e["battlefield"]["A"].append("Swamp")
    e["battlefield"]["B"].append("Swamp")
    e["deaths"] = {"A": {"combat": [], "noncombat": ["Eager Trufflesnout"]}, "B": {"combat": [], "noncombat": ["Frogmite"]}}
    r = worker.request("replay_turn", spec, seed=0, **opt)
    assert r["reproduced"] and r["attempt"] == 1, (r["diff"], r["targets"])
    stabs = {(t["seat"], t["target"], t["evidence"]) for t in r["targets"] if t["source"] == "Stab"}
    assert stabs == {("A", "Frogmite", "fate"), ("B", "Eager Trufflesnout", "fate")}


@needs_worker
def test_a_guessed_attacking_copy_is_matched_by_count(worker, games, ids):
    # two Slumbering Cerberus, one attacked; labels could not tell which (attackGuess) and picked
    # the copy that stayed tapped (it does not untap). The other copy attacks instead, labelled a
    # guess; before the review neither attacked
    spec, opt, _ = replay_request(games[4], 3, ids)
    _add(spec, "A", "battlefield", _perm("Slumbering Cerberus", "SlumberingCerberus_80", tapped=True),
         _perm("Slumbering Cerberus", "SlumberingCerberus_81"))
    s = opt["script"]
    s["attacks"].update({"A:SlumberingCerberus_80": True, "A:SlumberingCerberus_81": False})
    s["attackGuess"] = ["Slumbering Cerberus"]
    e = opt["expected"]
    e["life"]["B"] = 15                             # Drake Hatcher 1 + Slumbering Cerberus 4
    e["battlefield"]["A"] += ["Slumbering Cerberus", "Slumbering Cerberus"]
    r = worker.request("replay_turn", spec, seed=0, maxAttempts=2, **opt)
    assert r["reproduced"], (r["diff"], r["flags"])
    asked = [d for d in r["decisions"] if d["text"] == "attack with: Slumbering Cerberus?"]
    assert [(d["chosen"], d["label_kind"], d["evidence"]) for d in asked] == [("yes", "imputed_order", "copy_guess")]
    # a recorded count the engine cannot meet is a divergence, named by the card
    s["attacks"]["A:SlumberingCerberus_81"] = True
    e["life"]["B"] = 11
    r = worker.request("replay_turn", spec, seed=0, maxAttempts=2, **opt)
    assert r["reproduced"] is False and r["diff"]["divergence"] == {"attack:A:unrealised": ["Slumbering Cerberus x1"]}


@needs_worker
def test_a_flashback_recorded_as_a_cast_is_replayed(worker, games, ids):
    # 17lands logs flashback as a cast; labels only sees it when no copy was in hand at the turn
    # start. Think Twice cast from hand, then flashed back in the same turn, comes as two "Cast
    # Think Twice": the second is replayed as the flashback (it used to stay undone)
    spec, opt, _ = replay_request(games[4], 3, ids)
    _add(spec, "A", "battlefield", _perm("Island"), _perm("Island"))
    s = opt["script"]
    s["casts"] = [{"kind": "cast", "key": "Cast Think Twice", "name": "Think Twice", "family": "instant_sorcery"}] * 2
    s["opp"] = []                                   # no Dazzling Angel to counter
    e = opt["expected"]
    e["battlefield"]["A"] += ["Island", "Island"]
    e["hand"]["A"] = sorted([c for c in e["hand"]["A"] if c != "Think Twice"] + ["Dazzling Angel"])
    e["hand"]["unknownA"] = 2                       # the two cards Think Twice draws
    r = worker.request("replay_turn", spec, seed=0, **opt)
    assert r["reproduced"], (r["diff"], r["items"])
    assert all(it["done"] for it in r["items"] if it["seat"] == "A") and "cast_as_flashback" in r["flags"]
    chosen = [d["chosen"] for d in r["decisions"] if d["type"] == "PRIORITY"]
    assert "Cast Think Twice" in chosen and "Flashback {2}{U}" in chosen
    fb = next(d for d in r["decisions"] if d.get("chosen") == "Flashback {2}{U}")
    assert fb["label_kind"] == "imputed_order" and "Flashback {2}{U}" in fb["set"]


@needs_worker
def test_replay_is_deterministic_across_calls_and_jvms(worker, games, ids):
    def key(r):
        return {k: r[k] for k in ("reproduced", "attempt", "policy", "diff", "items", "targets", "end", "decisions")}
    a = replay_turn(worker, games[4], 7, ids, encode=True)
    b = replay_turn(worker, games[4], 7, ids, encode=True)
    assert key(a) == key(b)
    assert all(len(d["features"]) > 500 for d in a["decisions"])
    worker.close()
    worker.start()            # a fresh JVM: no class loading or id history from the calls above
    c = replay_turn(worker, games[4], 7, ids, encode=True)
    assert key(a) == key(c)
