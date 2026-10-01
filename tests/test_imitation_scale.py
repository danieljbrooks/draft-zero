"""Experiment #4's tools (docs/017): the exp4 split, the game runner's pairing and scoring
(tools/imitation_scale/play.py), and the bridge's play op (BenchPlayer: IS-MCTS for both seats).
The play-op test needs the bridge and skips cleanly without it."""
import importlib.util
from pathlib import Path

import pytest

from draftzero.gameplay import bridge

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "imitation_scale"
DECKS = Path(__file__).resolve().parents[1] / "assets" / "sample" / "decks"
PROBLEMS = bridge.environment_problems()
needs_worker = pytest.mark.skipif(bool(PROBLEMS), reason="mzbridge unavailable: " + "; ".join(PROBLEMS))


def tool(name: str):
    spec = importlib.util.spec_from_file_location(f"imitation_scale_{name}", TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_exp4_split_keeps_test_inside_2b_test_range():
    build = tool("build")
    assert build.exp4_split(0.10) == 0 and build.exp4_split(0.95) == 0     # train
    assert build.exp4_split(0.82) == 1 and build.exp4_split(0.869) == 1    # val: #2b's val range
    assert build.exp4_split(0.87) == 2 and build.exp4_split(0.919) == 2    # test: inside #2b's test range
    assert build.exp4_split(0.92) == 0
    h = build.component_hash("some-draft")
    assert 0.0 <= h < 1.0 and h == build.component_hash("some-draft")


def test_play_pairs_and_scores_by_role():
    play = tool("play")
    pairs = play.deck_pairs(["a", "b", "c", "d"], 5, seed=1)
    assert pairs == play.deck_pairs(["a", "b", "c", "d"], 5, seed=1) and all(x != y for x, y in pairs)
    # a mirror match is still scored by role: bot1 won one of the pair's two games
    games = [{"pair": 0, "swap": False, "winner_role": "bot1", "turns": 10, "seconds": 1.0},
             {"pair": 0, "swap": True, "winner_role": "bot2", "turns": 12, "seconds": 1.0},
             {"pair": 1, "swap": False, "winner_role": None, "turns": 50, "seconds": 1.0},
             {"pair": 1, "swap": True, "error": "x", "seconds": 1.0}]
    s = play.summarize(games, "heuristic", "heuristic")
    assert s["games"] == 3 and s["errors"] == 1
    assert s["bot1_score"] == pytest.approx(1.5 / 3) and s["no_winner"] == 1 and s["pairs_split"] == 1
    lo, hi = s["ci95"]
    assert 0 < lo < 0.5 < hi < 1
    # a pair's two games share their shuffles, except in self-play, where they would be one game twice
    t = play.game_tasks(pairs[:2], 7, mirror=False)
    assert [x["game_seed"] for x in t] == [7000, 7000, 7001, 7001]
    t = play.game_tasks(pairs[:2], 7, mirror=True)
    assert len({x["game_seed"] for x in t}) == 4
    # a network bot's seat asks for experiment #4's defaults
    o = play.seat_options("il_bc", 1000, 50052, 300)
    assert o["priors"] and o["opponentPriors"] == "uniform" and o["isPolicyPerWorld"] and o["leaf"] == "net"
    assert play.seat_options("heuristic", 1000, None, 300)["evaluator"] == {"type": "offline"}
    assert play.seat_options("heuristic", 1000, None, 300, 50070, "FDN_top_00001_WB")["belief"] == \
        {"port": 50070, "exclude": "FDN_top_00001_WB", "worlds": 8}


@needs_worker
def test_play_op_runs_a_game_with_is_mcts_for_both_seats(tmp_path):
    decks = sorted(DECKS.glob("*.dck"))[:2]
    b = bridge.Bridge("pytest_play", heap="2g", runtime_root=tmp_path)
    try:
        seat = {"budget": 8, "evaluator": {"type": "offline"}}
        r = b.request("play", None, deckA=str(decks[0]), deckB=str(decks[1]), seatA=seat, seatB=seat,
                      seed=3, maxTurns=6, record=True, timeout=900)
        again = b.request("play", None, deckA=str(decks[0]), deckB=str(decks[1]), seatA=seat, seatB=seat,
                          seed=3, maxTurns=6, timeout=900)
    finally:
        b.close()
    assert r["winner"] in ("A", "B", None) and 1 <= r["turns"] <= 7
    for s in ("A", "B"):
        st = r["seats"][s]
        assert st["fallbacks"] == 0 and st["sims"] == 8 * st["decisions"]
        recs = r["records"][s]
        assert len(recs) == st["decisions"]
        for x in recs:
            assert x["features"] and x["type"] and x["turn"] >= 1 and -1 <= x["q"] <= 1
            assert len(x["legal"]) == len(x["visits"]) >= 2 and sum(x["visits"]) == 8
            assert -1 <= x["heuristic"] <= 1
        assert st["activationFailures"] >= 0
    # a seeded game replays the same way
    assert (again["winner"], again["turns"]) == (r["winner"], r["turns"])
    assert again["seats"]["A"]["decisions"] == r["seats"]["A"]["decisions"]


@needs_worker
def test_play_op_with_closed_decklists(tmp_path):
    """The opponent's hidden cards come from the belief service: 8 belief worlds every decision."""
    bs = tool("belief_server")
    if not bs.DECKS_JSONL.exists():
        pytest.skip("no decks.jsonl (tools/extract_decks.py) to map decks to drafts")
    beliefs = bs.Beliefs(bs.DECKS_JSONL)
    srv = bs.serve(0, beliefs)
    port = srv.server_address[1]
    decks = sorted(DECKS.glob("*.dck"))[:2]
    b = bridge.Bridge("pytest_play_belief", heap="2g", runtime_root=tmp_path)
    try:
        def seat(opp):
            return {"budget": 8, "evaluator": {"type": "offline"}, "belief": {"port": port, "exclude": opp.stem}}
        r = b.request("play", None, deckA=str(decks[0]), deckB=str(decks[1]), seatA=seat(decks[1]),
                      seatB=seat(decks[0]), seed=3, maxTurns=6, timeout=900)
    finally:
        b.close()
        srv.shutdown()
    for s in ("A", "B"):
        st = r["seats"][s]
        assert st["closedDecklists"] and st["beliefCalls"] == st["decisions"] > 0
        assert st["worldsBuilt"] == 8 * st["beliefCalls"] - st["worldsFailed"] and st["openFallbacks"] == 0


def _record_line(pair, swap, seat, result, recs):
    return {"pair": pair, "swap": swap, "seat": seat, "bot": "il_bc", "result": result, "records": recs}


def _rec(type_, legal, visits, q, turn, feats=(5, 9, 11)):
    return {"features": list(feats), "type": type_, "turn": turn, "legal": legal, "visits": visits, "q": q}


def test_selfplay_tables_soft_targets_td_values_and_game_splits(tmp_path):
    """play.py's records -> the soft tables: visit shares over the legal options, MageZero's TD(lambda)
    value targets, one head per decision type, whole deck pairs per split."""
    import gzip
    import json

    import h5py
    import numpy as np
    st = tool("selfplay_tables")
    q = np.array([0.2, -0.4, 0.6])
    t = st.td_targets(q, 1.0, 0.95)
    g2 = 0.95 * 1 + 0.05 * 0.6
    g1 = 0.95 * g2 + 0.05 * -0.4
    assert t == pytest.approx([0.95 * g1 + 0.05 * 0.2, g1, g2])
    recs = [_rec("PRIORITY", [0, 4, 7], [2, 6, 0], 0.2, 1), _rec("CHOOSE_TARGET", [3, 8], [1, 7], -0.4, 2),
            _rec("CHOOSE_USE", [0, 1], [5, 3], 0.6, 3), _rec("PRIORITY", [0], [8], 0.1, 3),        # one option: dropped
            _rec("CHOOSE_NUM", [0, 1, 2], [1, 1, 6], 0.0, 4)]                                       # no head: dropped
    d = tmp_path / "records"
    d.mkdir()
    pairs = range(12)
    for p in pairs:
        for swap in (False, True):
            lines = [_record_line(p, swap, "A", 1, recs), _record_line(p, swap, "B", None if p == 0 else -1, recs)]
            with gzip.open(d / f"p{p:05d}{'s' if swap else ''}.jsonl.gz", "wt") as f:
                f.write("".join(json.dumps(x) + "\n" for x in lines))
    s = st.build([d], tmp_path / "h5")
    assert s["seat_games"] == 48 and s["rows"] == 48 * 3 and s["decisions_by_type"]["CHOOSE_NUM"] == 48
    pair_of = {}
    for split in st.SPLITS:
        with h5py.File(tmp_path / "h5" / f"selfplay_{split}.h5", "r") as f:
            n = len(f["offsets"]) - 1
            assert n == s[split]["rows"]
            if not n:
                continue
            for pr in np.unique(f["meta/pair"][:]):
                assert pair_of.setdefault(int(pr), split) == split            # a pair never straddles splits
            sp, si, p = f["set_indptr"][:], f["set_idx"][:], f["set_p"][:]
            sums = np.add.reduceat(p, sp[:-1])
            assert sums == pytest.approx(np.ones(n), abs=1e-6)
            assert set(f["meta/atype"][:].tolist()) == {0, 3, 5}
            row0 = int(np.flatnonzero(f["meta/atype"][:] == 0)[0])
            assert si[sp[row0]:sp[row0 + 1]].tolist() == [0, 4]                  # the unvisited option leaves the set ...
            lp = f["legal_indptr"][:]
            assert f["legal_idx"][lp[row0]:lp[row0 + 1]].tolist() == [0, 4, 7]   # ... not the legal options
            assert p[sp[row0]:sp[row0 + 1]] == pytest.approx([0.25, 0.75])
            z, ztd, seat, pair = f["z"][:], f["z_td"][:], f["meta/seat"][:], f["meta/pair"][:]
            won = (seat == 0)
            assert (z[won] == 1).all() and ztd[won] == pytest.approx(np.tile(t, won.sum() // 3))
            nowin = (seat == 1) & (pair == 0)
            assert np.isnan(z[nowin]).all() and np.isnan(ztd[nowin]).all()
            assert f["meta/turn"][:].max() == 2 and f["meta/game_turn"][:].max() == 3   # (game turn + 1) // 2
    assert sorted(pair_of) == list(pairs)
