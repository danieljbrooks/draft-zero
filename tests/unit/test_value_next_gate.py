"""tools/offline/value_next_gate.py's measurements on synthetic packs: the window classes, the castable instant in hand,
the test set's references and the drift tables, the scoring against nets.train.evaluate, the clustered interval, the
paired differences, the gate's rule and the rerun check."""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from draftzero.nets import data as D
from draftzero.nets import packfmt as pf

TOOL = Path(__file__).resolve().parents[2] / "tools" / "offline" / "value_next_gate.py"
TINY_MLP = {"d": 32, "ff": 64, "layers": 2, "d_emb": 16, "card_d": 24, "head_hidden": 32}


@pytest.fixture(scope="module")
def G():
    spec = importlib.util.spec_from_file_location("value_next_gate", TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _set_cards(G, pack, i, cards):
    """Record i's cards: (group, castable, instant-speed) for its first cards in order, the rest in group 3 with
    neither bit (groups stay in order within the record, as the encoder writes them)."""
    a, b = int(pack["card_off"][i]), int(pack["card_off"][i + 1])
    assert b - a >= len(cards), "the synthetic record has too few cards for the case"
    pack["card_group"][a:b] = 3
    pack["card_raw"][a:b, [G.CASTABLE, G.INSTANT]] = 0
    for k, (grp, cast, inst) in enumerate(cards):
        pack["card_group"][a + k] = grp
        pack["card_raw"][a + k, G.CASTABLE] = cast
        pack["card_raw"][a + k, G.INSTANT] = inst


def _same(a: dict, b: dict) -> bool:
    """Equal tables, NaN intervals included (a one-game cell has none)."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


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
    """One card in hand with both the castable and the instant-speed bit; either bit alone, the two bits on two
    different cards, or the card outside the hand, is not."""
    H = G.HAND
    cases = [([(H, 1, 1)], True),                       # held, castable, instant-speed
             ([(H, 0, 1)], False),                      # an instant it cannot cast
             ([(0, 1, 1), (1, 1, 1)], False),           # castable instants, not in hand
             ([(H, 1, 0)], False),                      # castable, not instant-speed
             ([(H, 1, 0), (H, 0, 1)], False),           # one castable card and another instant-speed one
             ([(H, 0, 0), (H, 1, 1)], True),            # the second card in hand
             ([(H, 0, 0), (H, 0, 0), (3, 1, 1)], False)]
    pack, games = pf.synthetic(len(cases), seed=49, max_cards=8)
    for i, (cards, _) in enumerate(cases):
        _set_cards(G, pack, i, cards)
    pf.write_shard(tmp_path / "s", pack, games)
    assert G.holding(D.ShardSet(tmp_path / "s")).tolist() == [w for _, w in cases]


# game, seat, turn, own, step, root value, outcome, holding; then the value under test, the training target's
# next_rv (next own turn's root value, else the outcome), and the next own turn's root value (-1 none)
ROWS = [
    ("a", 0, 1, 1, "main2", 0.60, 1.0, True, 0.70, 0.50, 0.50),     # 0  -> record 4, the first of a/0's turn 3
    ("a", 1, 1, 0, "upkeep", 0.45, 0.0, False, 0.30, 0.40, 0.40),   # 1  -> record 2
    ("a", 1, 2, 1, "upkeep", 0.40, 0.0, True, 0.10, 0.00, -1.0),    # 2  ends first: its outcome
    ("b", 0, 1, 1, "end", 0.55, -1.0, True, 0.80, 0.65, 0.65),      # 3  -> record 5; the result unknown
    ("a", 0, 3, 1, "main1", 0.50, 1.0, False, 0.90, 1.00, -1.0),    # 4  ends first
    ("b", 0, 3, 1, "main1", 0.65, -1.0, False, 0.20, -1.0, -1.0),   # 5  ends first, the result unknown
    ("a", 0, 3, 1, "main2", 0.58, 1.0, True, 0.70, 1.00, -1.0),     # 6  ends first
    ("c", 0, 0, 1, None, 0.50, 0.5, False, 0.62, 0.52, 0.52),       # 7  the pregame (the starter active) -> 8
    ("c", 0, 1, 1, "draw", 0.52, 0.5, False, 0.40, 0.50, -1.0),     # 8  ends first, drawn
    ("c", 1, 1, 0, "declare-blockers", 0.48, 0.5, True, 0.55, 0.50, -1.0),   # 9  ends first
]


def _drift_pack(G, root):
    pack, _ = pf.synthetic(len(ROWS), seed=47, max_cards=8)
    games = list(dict.fromkeys(r[0] for r in ROWS))
    pack["game"] = np.array([games.index(r[0]) for r in ROWS], np.int32)
    pack["seat"] = np.array([r[1] for r in ROWS], np.uint8)
    pack["turn"] = np.array([r[2] for r in ROWS], np.int32)
    pack["dense"][:, 2:14] = 0
    for i, r in enumerate(ROWS):
        pack["dense"][i, D.ACTIVE] = r[3]
        if r[4]:
            pack["dense"][i, 2 + G.STEPS.index(r[4])] = 1
        _set_cards(G, pack, i, [(G.HAND, 1, 1)] if r[7] else [(G.HAND, 1, 0)])
    pack["root_value"] = np.array([r[5] for r in ROWS], np.float32)
    pack["outcome"] = np.array([r[6] for r in ROWS], np.float32)
    pf.write_shard(root, pack, games)
    return [str(root)]


def test_test_set_and_drift_tables(G, tmp_path):
    """The references by hand (the next own turn's own record, the outcome when the game ends first), the classes
    and holding; drift = 100 x (value - reference), positive when the value is above it, over records with a known
    result against next_rv and over records with a next own turn while the game continues."""
    t = G.test_set(_drift_pack(G, tmp_path / "p"))
    assert np.allclose(t["nrv"], [r[9] for r in ROWS]) and np.allclose(t["next"], [r[10] for r in ROWS])
    assert t["hold"].tolist() == [r[7] for r in ROWS]
    assert [G.CLASSES[c] for c in t["cls"]] == ["own main2", "opp upkeep", "own upkeep", "own end", "own main1",
                                                 "own main1", "own main2", "own other", "own draw", "opp combat"]
    value = np.array([r[8] for r in ROWS])
    d = G.drift_table(value, t)
    want = {0: 20, 1: -10, 2: 10, 4: -10, 6: -30, 7: 10, 8: -10, 9: 5}       # known results only (not 3 and 5)
    assert d["all"]["n"] == 8 and d["all"]["games"] == 2
    assert d["all"]["mean"] == pytest.approx(np.mean(list(want.values())))
    assert d["all"]["ends_first"] == pytest.approx(5 / 8)
    gw = d["gate windows | holding"]                                          # 0 and 6 (3: result unknown)
    assert gw["n"] == 2 and gw["mean"] == pytest.approx(-5) and gw["ends_first"] == pytest.approx(0.5)
    assert d["own upkeep | holding"]["mean"] == pytest.approx(10) and d["own upkeep | holding"]["ends_first"] == 1
    assert d["opp upkeep | not holding"]["mean"] == pytest.approx(-10)
    assert d["opp upkeep | holding"]["n"] == 0
    c = G.drift_table(value, t, "continues")                                  # 0, 1, 3 and 7: a next own turn
    assert c["all"]["n"] == 4 and c["all"]["mean"] == pytest.approx((20 - 10 + 15 + 10) / 4)
    assert "ends_first" not in c["all"]
    assert c["gate windows | holding"]["n"] == 2 and c["gate windows | holding"]["mean"] == pytest.approx(17.5)
    assert c["own upkeep | holding"]["n"] == 0
    s = G.score(None, t, 1)                                                   # the search: its own root value
    assert s["drift"]["all"]["mean"] == pytest.approx(np.mean([(ROWS[i][5] - ROWS[i][9]) * 100 for i in want]))
    assert _same(s["drift_continues"], G.drift_table(t["rv"], t, "continues"))


def test_score_matches_the_trainers_evaluation(G, tmp_path):
    """A checkpoint's policy CE and value metrics on the test set are nets.train.evaluate's; its per-record value is
    the network's and its drift tables are drift_table's of it; paired differences by hand."""
    import torch
    from draftzero.nets import models as M
    from draftzero.nets import train
    torch.manual_seed(0)
    model = M.build("mlp", TINY_MLP)
    with torch.no_grad():                                 # a fresh value head outputs 0 everywhere: perturb it
        for w in model.parameters():
            w.add_(0.2 * torch.randn_like(w))
    M.save(model, tmp_path / "m.pt")
    t = G.test_set(_drift_pack(G, tmp_path / "p"))
    r = G.score(tmp_path / "m.pt", t, 1, batch=4)
    model, _ = M.load(tmp_path / "m.pt")
    args = train.parse_args(["--train", "x", "--out", "y", "--workers", "1", "--device", "cpu"])
    ev = train.evaluate(model, t["ds"], np.arange(t["ds"].n), args, torch.device("cpu"))
    for k in ("policy_ce", "policy_ce_uniform", "value_logloss", "value_brier", "value_auc"):
        assert r[k] == pytest.approx(ev[k], rel=1e-5, abs=1e-6), k
    assert r["n_policy"] == ev["n_policy"] and r["n_value"] == ev["n_value"] == 8
    with torch.inference_mode():
        _, _, _, x = train.losses(model, t["ds"].collate(np.arange(t["ds"].n)), args, weighted=False)
    assert np.allclose(r["_vl"], x["vlogit"].numpy(), atol=1e-5) and r["_vl"].std() > 0.05
    v = 1 / (1 + np.exp(-r["_vl"]))
    assert _same(r["drift"], G.drift_table(v, t)) and _same(r["drift_continues"], G.drift_table(v, t, "continues"))
    b = {"_ce": r["_ce"] + 0.01, "_vl": r["_vl"] + 0.5}
    p = G.paired(r, b, t)
    assert p["policy_ce"]["mean"] == pytest.approx(0.01)
    y, l0, l1 = t["outcome"], r["_vl"], b["_vl"]
    k = y >= 0
    want_vl = np.mean((np.logaddexp(0, l1) - y * l1 - np.logaddexp(0, l0) + y * l0)[k])
    assert p["value_logloss"]["n"] == 8 and p["value_logloss"]["mean"] == pytest.approx(want_vl)
    g = [0, 6]                                                                # the gate windows, holding, known
    want_g = np.mean((1 / (1 + np.exp(-l1[g])) - 1 / (1 + np.exp(-l0[g]))) * 100)
    assert p["gate_drift"]["n"] == 2 and p["gate_drift"]["mean"] == pytest.approx(want_g)


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
    assert G.gate({"control": co, "treatment": row(0.9, 0.704, 0.480)})["treatment"]["pass"]
    assert G.gate({"control": co, "treatment": row(-1.0, 0.705, 0.470)})["treatment"]["pass"]
    for bad in (row(1.2, 0.700, 0.470), row(-1.2, 0.700, 0.470), row(0.0, 0.7051, 0.470), row(0.0, 0.700, 0.4801)):
        assert not G.gate({"control": co, "treatment": bad})["treatment"]["pass"]
    assert G.gate({"control": co}) == {} and G.gate({"treatment": co}) == {}
    g = G.gate({"control": co, "treatment": row(1.2, 0.700, 0.470), "next-only": row(0.4, 0.700, 0.470)})
    assert list(g) == ["treatment", "next-only"] and not g["treatment"]["pass"] and g["next-only"]["pass"]
    share = {k: [c for c in v["checks"] if not c.get("gating", True)] for k, v in g.items()}
    assert [c["pass"] for c in share["treatment"]] == [False] and [c["pass"] for c in share["next-only"]] == [True]
    g = G.gate({"control": row(1.5, 0.700, 0.480), "treatment": row(0.9, 0.700, 0.470)})["treatment"]
    assert g["pass"] and not g["checks"][-1]["pass"]                          # 0.9 is 0.6 of 1.5: not gating


def test_a_finished_arm_is_reused_only_with_the_same_settings(G, tmp_path, monkeypatch):
    """A rerun into the same OUT reads a finished arm back when its arguments, window and checkpoint pick match,
    and refuses (rather than scores a smoke run's arm as the full gate's) when they differ."""
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True)
        for f in ("best.pt", "last.pt"):
            (out / f).write_bytes(b"")
        return SimpleNamespace(returncode=0, stdout=json.dumps({"steps": 5, "best_step": 4, "best": {}}) + "\n")
    monkeypatch.setattr(subprocess, "run", fake_run)
    cfg = SimpleNamespace(learner=SimpleNamespace(batch=512, lr=1e-4, warmup=100, policy_target="visits"))

    def arm(window=("w/0", "w/1"), dry=False, **kw):
        a = SimpleNamespace(**{"eval_max": 100, "threads": 1, "steps": 5, "epochs": 2.0, "checkpoint": "best", **kw})
        return G.train_arm("treatment", 0.5, 0.5, Path("v36/model.pt"), list(window), cfg, 7, tmp_path, a, dry=dry)
    assert arm(dry=True) is None and not calls                                # nothing to reuse, nothing trained
    first = arm()
    assert len(calls) == 1 and first["pt"].endswith("best.pt") and first["steps"] == 5
    assert "--value-next" in calls[0] and first["argv"].count("<2 pack dirs>") == 1
    assert arm() == first and len(calls) == 1                                 # the same settings: read back
    for kw in ({"steps": 6}, {"steps": None}, {"window": ("w/0", "w/2")}, {"checkpoint": "last"}, {"eval_max": 50}):
        for dry in (True, False):
            with pytest.raises(SystemExit, match="other settings"):
                arm(dry=dry, **kw)
    assert arm(dry=True) == first and len(calls) == 1
