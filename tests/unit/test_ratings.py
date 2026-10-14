"""Bradley-Terry ratings (draftzero.eval.ratings) on synthetic results."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pytest

from draftzero.eval.ratings import (ELO, fit_bt, game_triples, label, opponent_name, summary_triples, table,
                                    version_name)

TRUE = {"sbtactical": 0.0, "nonet": -60.0, "v1": 40.0, "v2": 150.0, "v3": 260.0}


def expected(a, b):
    return 1 / (1 + 10 ** ((TRUE[b] - TRUE[a]) / 400))


def old_fit(games, anchor):
    """v0.1.0 gorge/recipe/ratings.py's fit (ridge on the Hessian only), for parity."""
    players = sorted({p for g in games for p in g[:2]})
    ix = {p: i for i, p in enumerate(players)}
    A = np.array([ix[a] for a, _, _ in games])
    B = np.array([ix[b] for _, b, _ in games])
    S = np.array([s for _, _, s in games])
    n, r, pin = len(players), np.zeros(len(players)), ix[anchor]
    free = [i for i in range(n) if i != pin]
    for _ in range(100):
        p = 1 / (1 + np.exp(-(r[A] - r[B])))
        g = np.zeros(n)
        np.add.at(g, A, S - p)
        np.add.at(g, B, -(S - p))
        w = p * (1 - p)
        H = np.zeros((n, n))
        np.add.at(H, (A, A), -w)
        np.add.at(H, (B, B), -w)
        np.add.at(H, (A, B), w)
        np.add.at(H, (B, A), w)
        H -= 1e-6 * np.eye(n)
        step = np.linalg.solve(H[np.ix_(free, free)], g[free])
        r[free] -= step
        if np.abs(step).max() < 1e-9:
            break
    se = np.zeros(n)
    se[free] = np.sqrt(np.diag(np.linalg.inv(-H[np.ix_(free, free)])))
    return {p: (ELO * r[i], ELO * se[i]) for p, i in ix.items()}


def test_recovers_known_ratings_from_expected_scores():
    names = sorted(TRUE)
    res = [(a, b, expected(a, b), 400) for i, a in enumerate(names) for b in names[i + 1:]]
    fit = fit_bt(res)
    assert fit["sbtactical"] == (0.0, 0.0)
    for p, elo in TRUE.items():
        assert fit[p][0] == pytest.approx(elo, abs=0.05)
    assert all(fit[p][1] > 0 for p in TRUE if p != "sbtactical")
    # another anchor shifts every rating by the same amount
    fit2 = fit_bt(res, anchor="v1")
    for p in TRUE:
        assert fit2[p][0] == pytest.approx(fit[p][0] - fit["v1"][0], abs=0.05)


def test_weights_equal_repeated_games_and_se_scales():
    a = fit_bt([("x", "y", 0.6, 10)], anchor="y")
    b = fit_bt([("x", "y", 1.0)] * 6 + [("x", "y", 0.0)] * 4, anchor="y")
    c = fit_bt([("x", "y", 1.0)] * 5 + [("x", "y", 0.5)] * 2 + [("x", "y", 0.0)] * 3, anchor="y")
    assert a["x"] == pytest.approx(b["x"]) and a["x"] == pytest.approx(c["x"])
    assert a["x"][0] == pytest.approx(ELO * math.log(0.6 / 0.4), abs=1e-3)
    big = fit_bt([("x", "y", 0.6, 40)], anchor="y")
    assert big["x"][1] == pytest.approx(a["x"][1] / 2, rel=1e-3)
    # the reversed orientation is the same result
    assert fit_bt([("y", "x", 0.4, 10)], anchor="y")["x"] == pytest.approx(a["x"])


def test_matches_the_v010_fit_on_simulated_games():
    rng = np.random.default_rng(7)
    names = sorted(TRUE)
    games = []
    for _ in range(3000):
        a, b = rng.choice(names, 2, replace=False)
        u = rng.random()
        p = expected(a, b)
        games.append((a, b, 1.0 if u < p * 0.95 else 0.5 if u < p * 0.95 + 0.05 else 0.0))
    new, old = fit_bt(games), old_fit(games, "sbtactical")
    for p in names:
        assert new[p][0] == pytest.approx(old[p][0], abs=0.01)
        assert new[p][1] == pytest.approx(old[p][1], rel=1e-4)
        assert abs(new[p][0] - TRUE[p]) < 4 * new[p][1] + 1e-9


def test_edge_cases():
    assert fit_bt([]) == {}
    assert fit_bt([("x", "x", 1.0)]) == {}
    # a perfect record stays finite (the ridge)
    f = fit_bt([("x", "sbtactical", 1.0)] * 20)
    assert math.isfinite(f["x"][0]) and f["x"][0] > 1000 and math.isfinite(f["x"][1])
    # without the anchor, the alphabetically first player is pinned
    f = fit_bt([("b", "c", 0.75, 8), ("a", "b", 0.5, 8)])
    assert f["a"] == (0.0, 0.0) and f["b"][0] == pytest.approx(0, abs=1e-3) and f["c"][0] < -100
    with pytest.raises(ValueError):
        fit_bt([("x", "y", 1.5)])
    # min_games drops thin players with their results
    f = fit_bt([("x", "sbtactical", 0.5, 300), ("thin", "x", 1.0, 10)], min_games=200)
    assert set(f) == {"x", "sbtactical"}


def test_table_rows_best_first():
    rows = table([("v2", "sbtactical", 0.7, 100), ("v1", "sbtactical", 0.55, 100), ("v2", "v1", 0.6, 50)])
    assert [r["player"] for r in rows] == ["v2", "v1", "sbtactical"]
    assert rows[0]["games"] == 150 and rows[2]["games"] == 200 and rows[2]["elo"] == 0.0


def test_names_and_summary_triples():
    assert version_name(4) == version_name("v0004") == version_name("4") == "v4"
    assert version_name("sbtactical") == "sbtactical"
    assert opponent_name("prev", 5) == "v4" and opponent_name("v0003") == "v3"
    assert opponent_name("nonet", 2) == "nonet" and opponent_name("file:/x/v9.dzgw") == "file:/x/v9.dzgw"
    with pytest.raises(ValueError):
        opponent_name("prev")
    rows = [{"version": 5, "opponent": "prev", "score": 0.56, "n_games": 200, "n_pairs": 100},
            {"a": "v0005", "b": "sbtactical", "score": 0.7, "n_pairs": 50},
            {"version": 5, "opponent": "nonet", "score": None, "n_pairs": 0}]
    assert summary_triples(rows) == [("v5", "v4", 0.56, 200.0), ("v5", "sbtactical", 0.7, 100.0)]
    with pytest.raises(ValueError):
        summary_triples([{"score": 0.5, "n_games": 2}])


def test_game_triples_from_dzgorge_rows():
    net = "az:sims=100:autopay:oppnodes:oppfull:morekinds:topk=6:cpuct=0.25:gonet=/s/versions/player/v0004/model.dzgw"
    assert label(net) == "az:sims=100:cpuct=0.25:net=v0004"
    assert label("sbtactical") == "sbtactical"
    rows = [{"side": ["A", "B"], "pols": [net, "sbtactical"], "result": "A"},
            {"side": ["B", "A"], "pols": ["sbtactical", net], "result": "B"},
            {"side": ["A", "B"], "pols": [net, "sbtactical"], "result": "stall"},
            {"side": ["A", "B"], "pols": [net, "sbtactical"], "result": "error"},
            {"side": ["A", "B"], "pols": ["bot", "bot"], "result": "A"}]
    t = game_triples(rows)
    assert t == [("az:sims=100:cpuct=0.25:net=v0004", "sbtactical", 1.0),
                 ("az:sims=100:cpuct=0.25:net=v0004", "sbtactical", 0.0),
                 ("az:sims=100:cpuct=0.25:net=v0004", "sbtactical", 0.5)]
    assert game_triples(rows[:1], {net: "v4"}) == [("v4", "sbtactical", 1.0)]
    assert game_triples(rows[:1], lambda s: s[:2]) == [("az", "sb", 1.0)]
    counts = defaultdict(int)
    for a, b, _ in t:
        counts[a] += 1
    assert counts["az:sims=100:cpuct=0.25:net=v0004"] == 3
