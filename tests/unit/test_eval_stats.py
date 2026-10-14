"""Shared statistics (eval.stats), card statistics on dzgorge rows (eval.gih) and the analyze command line."""
from __future__ import annotations

import json

import pytest

from draftzero.eval import analyze, gih
from draftzero.eval.stats import ranks, spearman, wilson


def test_wilson():
    p, lo, hi = wilson(5, 10)
    assert p == 0.5 and lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    assert wilson(0, 0) == (None, None, None)
    p, lo, hi = wilson(10, 10)
    assert p == 1.0 and hi == 1.0 and 0.6 < lo < 0.8


def test_spearman_and_ranks():
    assert list(ranks([10, 30, 20, 20])) == [1.0, 4.0, 2.5, 2.5]
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0)        # rank-based: monotone is 1
    assert spearman([1, 2], [1, 2]) is None and spearman([1, 1, 1], [1, 2, 3]) is None
    with pytest.raises(ValueError):
        spearman([1, 2, 3], [1, 2])


def row(result, hands, decks=("d_wb", "d_ur"), side=("A", "B")):
    winner = list(side).index(result) if result in ("A", "B") else -1
    return {"pair": 0, "leg": 0, "decks": list(decks), "side": list(side), "pols": ["a", "b"],
            "winner": winner, "result": result, "in_hand": [list(h) for h in hands]}


GAMES = [row("A", (["Stab", "Stab", "Plains"], ["Shock"])),           # seat 0 (A) wins; Stab counted once
         row("B", (["Stab"], ["Shock"])),                             # seat 1 (B) wins
         row("A", (["Fog"], ["Stab"]), side=("B", "A")),              # A sits in seat 1 and wins
         row("draw", (["Stab"], ["Shock"])),                          # no winner: left out
         row("error", (["Stab"], ["Shock"]))]
REF = {"cards": {"Stab": {"gih_wr": 0.58}, "Shock": {"gih_wr": 0.55}, "Fog": {"gih_wr": 0.45},
                 "Plains": {"gih_wr": 0.5}},
       "colors": {"WB": {"wr": 0.57}, "UR": {"wr": 0.53}}}


def test_card_gih_on_dzgorge_rows():
    acc = gih.card_gih(GAMES)
    assert acc == {"Stab": [3, 2], "Plains": [1, 1], "Shock": [2, 1], "Fog": [1, 0]}
    assert gih.card_gih(GAMES, side="A") == {"Stab": [3, 2], "Plains": [1, 1]}
    rec = gih.color_records(GAMES, {"d_wb": "WB", "d_ur": "UR"})
    assert rec == {"WB": [3, 1], "UR": [3, 2]}
    assert gih.color_records(GAMES[:1], {}) == {"?": [2, 1]}


def test_gih_correlation_and_report():
    acc = gih.card_gih(GAMES)
    rho, n = gih.gih_correlation(acc, REF, min_games=1)
    assert n == 3 and rho == pytest.approx(1.0)                      # basics are excluded
    assert gih.gih_correlation(acc, REF, min_games=1, only={"Stab", "Fog"}) == (None, 2)
    assert gih.gih_correlation(acc, REF, min_games=3) == (None, 1)
    rep = gih.gih_report(GAMES, 1, reference=REF, rarity={"Stab": "common", "Shock": "common", "Fog": "common"},
                         colors={"d_wb": "WB", "d_ur": "UR"})
    assert rep["games"] == 5 and rep["player_games"] == 6
    assert rep["commons"]["cards"] == 3 and rep["commons"]["spearman"] == pytest.approx(1.0)
    assert [r[0] for r in rep["all"]["rows"]] == ["Stab", "Shock", "Fog"]
    assert rep["colors"]["rows"]["WB"] == {"games": 3, "wr": 1 / 3, "ref_wr": 0.57}
    assert rep["colors"]["spearman"] is None                         # two pairs: too few to rank
    json.dumps(rep)                                                   # JSON-safe (no NaN)
    empty = gih.gih_report([], 1, reference=REF, rarity={}, colors={})
    assert empty["all"]["cards"] == 0 and empty["all"]["spearman"] is None and empty["all"]["sd_sim"] is None


def test_assets_load():
    rar = gih.load_rarity()
    assert rar and "common" in set(rar.values())
    ref = gih.load_reference()
    assert len(ref["cards"]) > 200 and ref["colors"]
    assert gih.load_reference(None) == {"cards": {}, "colors": {}}
    assert gih.deck_colors("/nonexistent/pool.tsv") == {}


def paired_file(path, results, a="az:sims=100", b="sbtactical"):
    rows = []
    for p, (r0, r1) in enumerate(results):
        for leg, r in ((0, r0), (1, r1)):
            side = ["A", "B"] if leg == 0 else ["B", "A"]
            rows.append({"game": len(rows), "pair": p, "leg": leg, "seed": p, "decks": ["x", "y"], "side": side,
                         "pols": [a, b] if leg == 0 else [b, a], "result": r,
                         "winner": side.index(r) if r in ("A", "B") else -1, "turns": 10, "in_hand": [[], []]})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"partial": ')
    return path


def test_analyze_cli(tmp_path, capsys):
    f1 = paired_file(tmp_path / "net_v_bot.jsonl", [("A", "A"), ("A", "B"), ("B", "draw")])
    f2 = paired_file(tmp_path / "nonet_v_bot.jsonl", [("B", "B"), ("A", "B")], a="az:sims=100:nonet")
    assert len(analyze.load([f1])) == 6                               # the partial last line is skipped
    analyze.main(["wins", str(f1)])
    out = capsys.readouterr().out
    assert "az:sims=100" in out and "58.3%" in out and "3 pairs" in out
    js = tmp_path / "r.json"
    analyze.main(["ratings", str(tmp_path), "--min-games", "1", "--json", str(js)])
    rows = json.loads(js.read_text())
    assert {r["player"] for r in rows} == {"az:sims=100", "az:sims=100:nonet", "sbtactical"}
    assert next(r for r in rows if r["player"] == "sbtactical")["elo"] == 0.0
    analyze.main(["gih", str(f1), "--min-games", "1"])
    assert "player-games with a winner" in capsys.readouterr().out
