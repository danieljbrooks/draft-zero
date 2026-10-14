"""Paired-game scoring of dzgorge games.jsonl rows (draftzero.eval.paired) on synthetic records."""
from __future__ import annotations

import math

import pytest

from draftzero.eval.paired import game_points, policies, score_pairs, sprt, sprt_llr


def leg(pair, leg_, result, seed=None, a="az:sims=100", b="sbtactical"):
    """One games.jsonl row: leg 0 puts policy A in seat 0, leg 1 swaps the seats."""
    side = ["A", "B"] if leg_ == 0 else ["B", "A"]
    pols = [a, b] if leg_ == 0 else [b, a]
    winner = side.index(result) if result in ("A", "B") else -1
    return {"game": 2 * pair + leg_, "pair": pair, "leg": leg_, "seed": 1000 + pair if seed is None else seed,
            "decks": ["d1", "d2"], "side": side, "pols": pols, "winner": winner, "result": result, "turns": 12}


def old_score(games):
    """gorge/recipe/score.py (v0.1.0) on one file: anything but A/B counts half, pairs need both legs."""
    pairs = {}
    for g in games:
        r = g["result"]
        pairs.setdefault(g["pair"], []).append(1.0 if r == "A" else 0.0 if r == "B" else 0.5)
    full = [sum(v) / len(v) for v in pairs.values() if len(v) == 2]
    n = len(full)
    m = sum(full) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in full) / max(n - 1, 1))
    return m, sd / math.sqrt(n), n


GAMES = [leg(0, 0, "A"), leg(0, 1, "A"),          # A wins both: 1
         leg(1, 0, "A"), leg(1, 1, "B"),          # split: 0.5
         leg(2, 0, "draw"), leg(2, 1, "stall")]   # 0.5


def test_score_over_pairs_with_pair_level_se():
    r = score_pairs(GAMES)
    assert r["score"] == pytest.approx(2 / 3)
    assert r["se"] == pytest.approx(math.sqrt(1 / 12) / math.sqrt(3))
    assert (r["n_pairs"], r["n_games"], r["wins"], r["losses"], r["draws"], r["stalls"], r["errors"]) == (3, 6, 3, 1, 1, 1, 0)
    assert r["lo"] == pytest.approx(2 / 3 - 1.959964 * r["se"]) and r["hi"] == pytest.approx(2 / 3 + 1.959964 * r["se"])
    m, se, n = old_score(GAMES)
    assert (r["score"], r["se"], r["n_pairs"]) == pytest.approx((m, se, n))


def test_result_names_the_policy_not_the_seat():
    # leg 1: B sits in seat 0, A in seat 1 and wins (winner seat 1, result "A")
    g = leg(0, 1, "A")
    assert g["winner"] == 1 and g["side"][1] == "A"
    assert game_points(g) == 1.0 and game_points(g, "B") == 0.0
    assert game_points({"result": "draw"}) == 0.5 and game_points({"result": "stall"}, "B") == 0.5
    assert game_points({"result": "error"}) is None and game_points({}) is None


def test_hero_side_b_is_the_complement():
    a, b = score_pairs(GAMES), score_pairs(GAMES, hero_side="B")
    assert b["score"] == pytest.approx(1 - a["score"]) and b["se"] == pytest.approx(a["se"])
    assert (b["wins"], b["losses"]) == (a["losses"], a["wins"])
    with pytest.raises(ValueError):
        score_pairs(GAMES, hero_side="draw")


def test_errors_and_incomplete_pairs_are_dropped():
    games = GAMES + [leg(3, 0, "A"), leg(3, 1, "error"),   # an error drops its pair
                     leg(4, 0, "B")]                        # the other leg never finished
    r = score_pairs(games)
    assert r["n_pairs"] == 3 and r["score"] == pytest.approx(2 / 3)
    assert r["errors"] == 1 and r["dropped"] == 2 and r["n_games"] == 6


def test_duplicates_and_seeds():
    r = score_pairs(GAMES + [leg(0, 1, "B")])               # a replayed leg: the first row stands
    assert r["duplicates"] == 1 and r["score"] == pytest.approx(2 / 3)
    # the same pair index from runs with other seeds never makes a pair
    r = score_pairs([leg(0, 0, "A", seed=1), leg(0, 1, "A", seed=2)])
    assert r["n_pairs"] == 0 and r["score"] is None and r["se"] is None and r["dropped"] == 2


def test_single_leg_runs_and_edge_cases():
    single = [leg(i, 0, r) for i, r in enumerate("AABB")] + [leg(4, 0, "draw")]
    assert score_pairs(single)["n_pairs"] == 0
    r = score_pairs(single, paired=False)
    assert r["n_pairs"] == 5 and r["score"] == pytest.approx(0.5) and r["wins"] == 2
    assert score_pairs([])["score"] is None
    one = score_pairs(GAMES[:2])
    assert one["score"] == 1.0 and one["se"] == 0.0 and one["hi"] == 1.0   # score.py: one pair has SE 0


def test_policies():
    p = policies(GAMES)
    assert p == {"A": {"az:sims=100"}, "B": {"sbtactical"}}


def test_sprt_decisions():
    assert sprt(0, 0, 0) is None
    assert sprt(10, 2, 8) is None
    assert sprt(700, 100, 300) == "better"
    assert sprt(300, 100, 700) == "not better"
    s = sprt_llr(700, 100, 300)
    assert s["decision"] == "better" and s["llr"] >= s["upper"]
    assert s["lower"] == pytest.approx(math.log(0.05 / 0.95)) and s["upper"] == pytest.approx(math.log(0.95 / 0.05))
    # a test of a smaller edge needs more games
    assert sprt(560, 0, 440, p1=0.51) is None and sprt(560, 0, 440) == "better"
