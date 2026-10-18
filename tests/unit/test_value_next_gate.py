"""tools/offline/value_next_gate.py's measurements on synthetic packs: the window classes, the castable instant in hand,
the clustered interval and the gate's rule."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from draftzero.nets import data as D
from draftzero.nets import packfmt as pf

TOOL = Path(__file__).resolve().parents[2] / "tools" / "offline" / "value_next_gate.py"


@pytest.fixture(scope="module")
def G():
    spec = importlib.util.spec_from_file_location("value_next_gate", TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_window_classes(G):
    """Whose turn (dense[19]) and the step one-hot (dense[2:14], gorge's stepNames order)."""
    cases = [(1, "upkeep", "own upkeep"), (1, "main1", "own main1"), (1, "declare-blockers", "own declare-blockers"),
             (1, "main2", "own main2"), (1, "end", "own end"), (1, "combat-damage", "own other"),
             (1, "cleanup", "own other"), (1, None, "own other"), (0, "upkeep", "opp upkeep"),
             (0, "draw", "opp draw"), (0, "main1", "opp main"), (0, "main2", "opp main"),
             (0, "begin-combat", "opp combat"), (0, "end-combat", "opp combat"), (0, "end", "opp end"),
             (0, "cleanup", "opp end"), (0, "untap", "opp other"), (0, None, "opp other")]
    dense = np.zeros((len(cases), pf.DENSE_W), np.float32)
    for i, (own, step, _) in enumerate(cases):
        dense[i, D.ACTIVE] = own
        if step:
            dense[i, 2 + G.STEPS.index(step)] = 1
    assert [G.CLASSES[c] for c in G.window_class(dense)] == [c for _, _, c in cases]
    assert set(G.GATE_WINDOWS) <= set(G.CLASSES)


def test_holding_a_castable_instant(G, tmp_path):
    """A card in hand with the castable and instant-speed bits; either bit alone, or the card elsewhere, is not."""
    pack, games = pf.synthetic(6, seed=40, max_cards=6)
    raw, grp, off = pack["card_raw"], pack["card_group"], pack["card_off"]
    raw[:, [G.CASTABLE, G.INSTANT]] = 0
    want = []
    for i in range(6):
        a, b = int(off[i]), int(off[i + 1])
        hand = [k for k in range(a, b) if grp[k] == G.HAND]
        other = [k for k in range(a, b) if grp[k] != G.HAND]
        if i == 0 and hand:
            raw[hand[0], [G.CASTABLE, G.INSTANT]] = 1              # held and castable
        elif i == 1 and hand:
            raw[hand[0], G.INSTANT] = 1                           # an instant it cannot cast
        elif i == 2 and other:
            raw[other[0], [G.CASTABLE, G.INSTANT]] = 1            # not in hand
        elif i == 3 and hand:
            raw[hand[0], G.CASTABLE] = 1                          # castable, not instant-speed
        want.append(i == 0 and bool(hand))
    pf.write_shard(tmp_path / "s", pack, games)
    assert G.holding(D.ShardSet(tmp_path / "s")).tolist() == want


def test_clustered_interval(G):
    x = np.array([1.0, 1.0, 3.0, 3.0, 2.0, 2.0])
    r = G.clustered(x, np.array([0, 0, 1, 1, 2, 2]))
    assert r["n"] == 6 and r["games"] == 3 and r["mean"] == pytest.approx(2.0)
    se = np.sqrt(3 / 2 * (2.0 ** 2 + 2.0 ** 2 + 0.0)) / 6                   # group sums of x - mean: -2, 2, 0
    assert r["hi"] - r["mean"] == pytest.approx(1.96 * se)
    one = G.clustered(np.array([1.0, 2.0]), np.array([5, 5]))               # one game: no interval
    assert one["mean"] == 1.5 and not np.isfinite(one["lo"])
    assert G.clustered(np.zeros(0), np.zeros(0))["mean"] is None


def test_gate_rule(G):
    def row(drift, ce, vl):
        return {"drift": {"gate windows | holding": {"n": 10, "games": 5, "mean": drift, "lo": drift - 1,
                                                       "hi": drift + 1}},
                "policy_ce": ce, "value_logloss": vl}
    co = row(2.0, 0.700, 0.480)
    assert G.gate({"control": co, "treatment": row(0.9, 0.704, 0.480)}, None)["pass"]
    assert G.gate({"control": co, "treatment": row(-1.0, 0.705, 0.470)}, None)["pass"]
    for bad in (row(1.2, 0.700, 0.470), row(0.0, 0.7051, 0.470), row(0.0, 0.700, 0.4801)):
        assert not G.gate({"control": co, "treatment": bad}, None)["pass"]
    assert not G.gate({"control": co}, None)["pass"]
