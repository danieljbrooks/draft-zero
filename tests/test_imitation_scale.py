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


def _toy_table(rows: list[tuple[list[int], list[int], list[int]]], first_row: int) -> dict:
    """A table in imitation's layout from (features, legal, set) per row."""
    import numpy as np
    t = {"indices": np.asarray([x for f, _, _ in rows for x in f], np.int32),
         "offsets": np.r_[0, np.cumsum([len(f) for f, _, _ in rows])].astype(np.int64),
         "legal_idx": np.asarray([x for _, l, _ in rows for x in l], np.int32),
         "legal_indptr": np.r_[0, np.cumsum([len(l) for _, l, _ in rows])].astype(np.int64),
         "set_idx": np.asarray([x for _, _, s in rows for x in s], np.int32),
         "set_indptr": np.r_[0, np.cumsum([len(s) for _, _, s in rows])].astype(np.int64),
         "z": np.asarray([1.0 if (first_row + i) % 2 else -1.0 for i in range(len(rows))], np.float32),
         "meta/label_status": np.zeros(len(rows), np.int32),
         "meta/row": np.arange(first_row, first_row + len(rows), dtype=np.int32),
         "labels": [[f"opt{x}" for x in l] for _, l, _ in rows], "lab_idx": [l for _, l, _ in rows]}
    return t


def test_merge_h5_concatenates_tables_and_shifts_csr_pointers(tmp_path):
    import h5py
    import numpy as np
    from draftzero.gameplay import imitation as im
    build = tool("build")
    rows = [([1, 2, 3], [4, 5], [5]), ([6], [7, 8, 9], [7, 9]), ([10, 11], [12, 13], []), ([14, 15, 16, 17], [1, 2], [2])]
    parts = [rows[:1], rows[1:3], rows[3:]]
    paths = []
    for i, p in enumerate(parts):
        paths.append(tmp_path / f"p{i}.h5")
        im._save_table(_toy_table(p, sum(len(q) for q in parts[:i])), paths[-1])
    im._save_table(_toy_table(rows, 0), tmp_path / "whole.h5")
    assert build.merge_h5(paths, tmp_path / "merged.h5") == len(rows)
    with h5py.File(tmp_path / "whole.h5") as a, h5py.File(tmp_path / "merged.h5") as b:
        assert sorted(build._h5_datasets(a)) == sorted(build._h5_datasets(b))
        for k in build._h5_datasets(a):
            x = a[k].asstr()[:] if h5py.check_string_dtype(a[k].dtype) else a[k][:]
            y = b[k].asstr()[:] if h5py.check_string_dtype(b[k].dtype) else b[k][:]
            assert np.array_equal(x, y), k
        assert dict(a.attrs).keys() == dict(b.attrs).keys()


def test_build_part_writer_resumes_after_an_unfinished_part(tmp_path):
    build = tool("build")
    game = lambda r: {"row": r, "ts": [{"row": r, "status": "ok"}], "rp": [], "bl": [], "op": []}   # noqa: E731
    w = build.PartWriter(tmp_path, part_games=3)
    for r in range(8):
        w.add(game(r))
    for f in w.files.values():              # a stop: part 2 is open with 2 games and no marker
        f.close()
    parts = build.completed_parts(tmp_path)
    assert [p["rows"] for p in parts] == [[0, 1, 2], [3, 4, 5]] and parts[0]["stats"]["ts_ok"] == 3
    w = build.PartWriter(tmp_path, part_games=3)
    assert w.done_rows == set(range(6)) and w.next_i == 2
    assert not list(tmp_path.glob("*.00002.pkl.gz"))
    for r in (6, 7):
        w.add(game(r))
    w.close()
    parts = build.completed_parts(tmp_path)
    assert [p["rows"] for p in parts][-1] == [6, 7]
    assert [r["row"] for p in parts for r in build.load_shard(p["paths"]["ts"])] == list(range(8))


def test_play_shards_split_games_by_deck_pair():
    play = tool("play")
    tasks = play.game_tasks(play.deck_pairs(["a", "b", "c", "d"], 7, seed=1), 1, mirror=False)
    parts = [play.shard_tasks(tasks, f"{i}/3") for i in range(3)]
    assert sorted((t["pair"], t["swap"]) for p in parts for t in p) == sorted((t["pair"], t["swap"]) for t in tasks)
    assert all(len({t["pair"] % 3 for t in p}) == 1 for p in parts) and play.shard_tasks(tasks, None) == tasks


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


def test_policy_bots_ask_for_policy_only_play():
    play = tool("play")
    o = play.seat_options("policy", 1000, 50052, 300, policy_fallback_budget=40)
    assert o["policyOnly"] and o["policyTemp"] == 1.0 and o["budget"] == 40 and o["priors"]
    assert play.seat_options("policy_greedy", 1000, 50052, 300)["policyTemp"] == 0.0
    assert "policyOnly" not in play.seat_options("il_bc", 1000, 50052, 300)
    # a bot's own budget: name@simulations
    assert play.seat_options("heuristic@100", 1000, None, 300)["budget"] == 100
    assert play.seat_options("il_bc@300", 1000, 50052, 300)["budget"] == 300
    assert play.seat_options("il_bc", 1000, 50052, 300)["budget"] == 1000
    assert play.parse_bot("policy") == ("policy", None)
    for bad in ("nope", "il_bc@0", "il_bc@x"):
        with pytest.raises(ValueError):
            play.parse_bot(bad)


def test_gih_counts_seats_with_the_card_in_hand_and_ranks_against_17lands():
    gih = tool("gih")
    g = lambda w, a, b: {"winner": w, "seats": {"A": {"inHand": a}, "B": {"inHand": b}}}  # noqa: E731
    games = [g("A", {"Stab": 2, "Plains": 3}, {"Refute": 1}), g("B", {"Stab": 1}, {"Refute": 1, "Stab": 1}),
             {"winner": None, "seats": {"A": {"inHand": {"Stab": 1}}, "B": {"inHand": {}}}}, {"error": "x"}]
    counts, seats = gih.gih_counts([x for x in games if not x.get("error")])
    assert seats == 4 and counts["Stab"] == [3, 2] and counts["Refute"] == [2, 1] and counts["Plains"] == [1, 1]
    ref = {"Stab": {"gih_wr": 0.55, "gih_games": 100}, "Refute": {"gih_wr": 0.60, "gih_games": 100},
           "Plains": {"gih_wr": 0.5, "gih_games": 100}}
    res = gih.compare(counts, ref, min_games=2)
    assert [r["card"] for r in res["rows"]] == ["Refute", "Stab"]        # basics out
    assert gih.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert gih.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert list(gih.ranks(__import__("numpy").array([5.0, 1.0, 5.0]))) == [2.5, 1.0, 2.5]


@needs_worker
def test_play_op_policy_only_plays_the_networks_policy(tmp_path):
    """policyOnly: one network call per decision with a head, no simulations there; inHand per seat."""
    from test_search_bench import FakeNet
    net = FakeNet()
    decks = sorted(DECKS.glob("*.dck"))[:2]
    b = bridge.Bridge("pytest_play_policy", heap="2g", runtime_root=tmp_path)
    try:
        seat = {"budget": 4, "policyOnly": True, "policyTemp": 1.0, "priors": True, "leaf": "net",
                "evaluator": {"type": "remote", "host": "127.0.0.1", "port": net.port}}
        greedy = {**seat, "policyTemp": 0.0}
        r = b.request("play", None, deckA=str(decks[0]), deckB=str(decks[1]), seatA=seat, seatB=greedy,
                      seed=5, maxTurns=8, timeout=900)
        again = b.request("play", None, deckA=str(decks[0]), deckB=str(decks[1]), seatA=seat, seatB=greedy,
                          seed=5, maxTurns=8, timeout=900)
    finally:
        b.close()
        net.close()
    assert r["winner"] in ("A", "B", None) and 1 <= r["turns"] <= 9
    for s in ("A", "B"):
        st = r["seats"][s]
        assert st["policyDecisions"] > 0 and st["decisions"] == st["policyDecisions"] + st["policySearched"]
        assert st["sims"] <= 4 * st["policySearched"]                    # only the headless decisions searched
        assert st["inHand"] and all(k > 0 for k in st["inHand"].values())
    # seeded sampling replays the same game
    assert (again["winner"], again["turns"]) == (r["winner"], r["turns"])
    assert again["seats"]["A"]["policyDecisions"] == r["seats"]["A"]["policyDecisions"]


def test_stage3_choice_rules():
    ch = tool("stage3_choice")
    d = {"replay_priority/set_nll": 0.240, "policy/set_nll": 0.258, "policy/top1_nonpass": 0.814,
         "value/auc": 0.767, "value/logloss": 0.558}
    a_close = {**d, "replay_priority/set_nll": 0.238}
    a_far = {**d, "replay_priority/set_nll": 0.235}
    v_good = {**d, "policy/set_nll": 0.253, "policy/top1_nonpass": 0.816, "value/auc": 0.764, "value/logloss": 0.562}
    v_value_cost = {**v_good, "value/auc": 0.760}
    assert ch.choose(d, a_close, d)["sets"] == []                                   # the config's defaults
    assert ch.choose(d, a_far, v_good)["sets"] == [
        "act_weights={opp_priority: 3.0, replay_priority: 3.0, turnstart: 3.0}", "value_weight=0.2"]
    assert ch.choose(d, a_close, v_value_cost)["sets"] == []                        # the value pays too much
    assert len(ch.choose(d, a_close, d)["why"]) == 2


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
