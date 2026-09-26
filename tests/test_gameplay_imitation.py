"""Tests for draftzero.gameplay.imitation: labels, splits, masking and the set loss, the HDF5
layout, and dataset building on the committed 17lands fixture (a fake bridge always; the real
bridge when java and the XMage build are present, else skipped)."""
from pathlib import Path

import numpy as np
import pytest

from draftzero.gameplay import bridge, imitation as im
from draftzero.gameplay.ids import Ids
from draftzero.gameplay.replay import read_games

FIX = Path(__file__).parent / "fixtures" / "gameplay" / "fdn_premier_rows.csv.gz"
PROBLEMS = bridge.environment_problems()
needs_worker = pytest.mark.skipif(bool(PROBLEMS), reason="mzbridge unavailable: " + "; ".join(PROBLEMS))


# ------------------------------------------------------------------------------ labels

def test_human_set_pass_set_unreachable():
    legal = ["Play Island", "Cast Stab", "Pass"]
    st, S, acts = im.human_set({"lands": [], "casts": [], "activations": []}, legal)
    assert (st, S, acts) == ("pass", ["Pass"], [])
    lab = {"lands": [{"key": "Play Island"}], "casts": [{"key": "Cast Stab"}, {"key": "Cast Kaito, Cunning Infiltrator"}]}
    st, S, acts = im.human_set(lab, legal)
    assert st == "set" and S == ["Play Island", "Cast Stab"]      # Kaito is not legal yet: not in S
    assert acts == ["Play Island", "Cast Stab", "Cast Kaito, Cunning Infiltrator"]
    st, S, _ = im.human_set({"casts": [{"key": "Cast Kaito, Cunning Infiltrator"}]}, legal)
    assert (st, S) == ("unreachable", [])


def test_human_set_matches_ability_text_without_cost_and_unkeyed_activation_is_not_pass():
    legal = ["-2: Create a 2/1 blue Ninja creature token.", "Pass"]
    st, S, _ = im.human_set({"activations": [{"key": "Create a 2/1 blue Ninja creature token."}]}, legal)
    assert st == "set" and S == ["-2: Create a 2/1 blue Ninja creature token."]
    # an activation 17lands logged but no XMage key exists for: the human acted, so not a Pass label
    st, S, _ = im.human_set({"activations": [{"id": 1, "key": None}]}, legal)
    assert st == "unreachable" and S == []


def test_heuristic_prefers_land_then_highest_mv_spell_then_pass():
    mv = {"Stab": 1, "Drake Hatcher": 2}.get
    sc = im.heuristic_scores(["Pass", "Cast Stab", "Cast Drake Hatcher", "Play Island"], mv)
    assert list(np.argsort(-sc)) == [3, 2, 1, 0]
    sc = im.heuristic_scores(["Pass", "Cast Stab", "Cast Drake Hatcher"], mv)
    assert int(np.argmax(sc)) == 2
    assert im.cost_mv("{2}{U}{U}") == 4 and im.cost_mv("{X}{R}") == 1 and im.cost_mv("{C}") == 1


def test_topk_in_set_expected_over_ties():
    legal = np.arange(4)
    assert im.topk_in_set(np.array([0, 0, 0, 0.0]), legal, np.array([2]), 1) == pytest.approx(0.25)
    assert im.topk_in_set(np.array([0, 0, 0, 0.0]), legal, np.array([2]), 3) == pytest.approx(0.75)
    assert im.topk_in_set(np.array([5, 1, 1, 0.0]), legal, np.array([0]), 1) == 1.0
    assert im.topk_in_set(np.array([5, 1, 1, 0.0]), legal, np.array([3]), 3) == 0.0
    assert im.topk_in_set(np.array([5, 1, 1, 0.0]), legal, np.array([2]), 2) == pytest.approx(0.5)


def test_wr_band_handles_float32_buckets():
    # 17lands win-rate buckets come back from HDF5 as float32: 0.58 -> 0.5799999833; as a Python
    # float that is below 0.58, so the band must not depend on the scalar type
    assert im.wr_band(np.float32(0.58)) == "0.58-1.01" == im.wr_band(float(np.float32(0.58)))
    assert im.wr_band(np.float32(0.54)) == "0.54-0.58"
    assert im.wr_band(np.float32(0.56)) == "0.54-0.58"
    assert im.wr_band(np.float32(0.52)) == "0.50-0.54"
    assert im.wr_band(np.float32(0.5)) == "0.50-0.54"
    assert im.wr_band(np.float32(0.48)) == "0.00-0.50"


def test_search_prior_temperature_and_non_pass_bonus():
    logits = np.array([2.0, 1.0, 0.0])                   # [Pass, cast, land]
    is_pass = np.array([True, False, False])
    p = im.search_prior(logits, is_pass, temp=1.0, bonus=0.0)
    np.testing.assert_allclose(p, np.exp(logits) / np.exp(logits).sum())
    q = im.search_prior(logits, is_pass, temp=1.5, bonus=0.1)
    assert q.sum() == pytest.approx(1.0) and q[1] > p[1] and q[0] < p[0]
    # Pass at 0.5 vs a cast at 0.45: the bonus flips the argmax, as it does in MCTSNode.setPriors
    r = im.search_prior(np.log([0.5, 0.45, 0.05]), is_pass, temp=1.0, bonus=0.1)
    assert int(np.argmax(r)) == 1


def test_auc_and_cluster_ci():
    y = np.array([0, 0, 1, 1, 0, 1])
    assert im.auc_score(np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7]), y) == 1.0
    assert im.auc_score(np.array([0.5] * 6), y) == 0.5
    assert np.isnan(im.auc_score(np.array([0.1, 0.2]), np.array([1, 1])))
    rng = np.random.default_rng(0)
    yy = rng.integers(0, 2, 400)
    p = yy * 0.3 + rng.random(400)
    c = im.cluster_auc_ci(p, yy, np.arange(400) // 2, n_boot=200)
    assert c["lo"] < c["mean"] < c["hi"] and c["games"] == 200


# ------------------------------------------------------------------------------ masking and the set loss

def test_masked_softmax_and_set_nll():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[1.0, 2.0, 3.0, 50.0]])
    legal = torch.tensor([[True, True, True, False]])     # the huge illegal logit must not count
    lp = im.masked_log_softmax(logits, legal)
    p = torch.softmax(torch.tensor([1.0, 2.0, 3.0]), -1)
    assert torch.allclose(lp[0, :3].exp(), p)
    assert lp[0, 3] == float("-inf")
    S = torch.tensor([[True, False, True, False]])
    nll = im.set_nll(lp, S & legal)
    assert float(nll) == pytest.approx(-float(torch.log(p[0] + p[2])), rel=1e-6)
    # a singleton set is the ordinary cross-entropy
    S1 = torch.tensor([[False, False, True, False]])
    assert float(im.set_nll(lp, S1)) == pytest.approx(-float(torch.log(p[2])), rel=1e-6)
    # gradients flow only into the legal logits
    x = logits.clone().requires_grad_(True)
    im.set_nll(im.masked_log_softmax(x, legal), S).sum().backward()
    assert float(x.grad[0, 3]) == 0.0 and float(x.grad[0, 1]) > 0


# ------------------------------------------------------------------------------ splits

def test_component_splits_keep_pairs_and_events_together():
    drafts = ["d1", "d1", "d2", "d3", "d3", "", "d4"]
    codes, stats = im.component_splits(drafts, [(1, 2), (4, 6)])       # d1-d2 and d3-d4 joined
    assert codes[0] == codes[1] == codes[2]
    assert codes[3] == codes[4] == codes[6]
    assert codes[0] == im.split_for_key("d1") and codes[3] == im.split_for_key("d3")   # smallest id keys
    assert stats["components_joined_by_pairs"] == 2
    # stable and roughly the configured fractions
    s = np.array([im.split_for_key(f"k{i}") for i in range(20000)])
    fr = np.bincount(s, minlength=3) / len(s)
    assert np.allclose(fr, im.SPLIT_FRACS, atol=0.015)


# ------------------------------------------------------------------------------ dataset building

class FakeBridge:
    """encode() answers with a canned main-phase decision (or a wrong one) and fixed features."""

    def __init__(self, legal, where_step="PRECOMBAT_MAIN", dtype="PRIORITY"):
        self.legal, self.step, self.type = legal, where_step, dtype
        self.calls = []

    def encode(self, spec, **opts):
        self.calls.append(opts)
        turn = (opts.get("decideFrom") or {}).get("turn")
        return {"decision": {"type": self.type, "text": "priority",
                             "where": {"turn": turn, "step": self.step, "passedBefore": 0},
                             "legal": [{"label": lab, "idx": i} for lab, i in self.legal]},
                "features": [3, 1, 2], "warnings": []}


@pytest.fixture(scope="module")
def ids():
    return Ids.load()


@pytest.fixture(scope="module")
def g0():
    return read_games([0], FIX)[0]


def test_turn_start_record_with_a_fake_bridge(g0, ids):
    # row 0 user turn 3: Play Island + Cast Kaito + Kaito's -2 (fixture golden)
    fb = FakeBridge([("Play Island", 533), ("Pass", 0)])
    r = im.turn_start_record(g0, 3, fb, ids, split=2)
    assert r["status"] == "ok" and r["label_status"] == "set" and r["S"] == ["Play Island"]
    assert fb.calls[0]["decisionPlayer"] == "A" and fb.calls[0]["perfectInfo"] is False
    assert fb.calls[0]["decideFrom"]["step"] == "PRECOMBAT_MAIN"
    assert r["features"].dtype == np.int32 and r["split"] == 2 and r["tier"] in im.TIERS
    # nothing the human did is legal here
    r = im.turn_start_record(g0, 3, FakeBridge([("Play Swamp", 540), ("Pass", 0)]), ids)
    assert r["label_status"] == "unreachable" and r["S"] == []
    # the bridge stopped at another decision (an attack question): not a turn-start row
    r = im.turn_start_record(g0, 3, FakeBridge([("no", 0), ("yes", 1)], "DECLARE_ATTACKERS", "CHOOSE_USE"), ids)
    assert r["status"] == "wrong_decision" and "features" not in r


def test_table_and_h5_layout(g0, ids, tmp_path):
    h5py = pytest.importorskip("h5py")
    recs = [im.turn_start_record(g0, n, FakeBridge([("Play Island", 533), ("Cast Kaito, Cunning Infiltrator", 279),
                                                    ("Pass", 0)]), ids) for n in (3, 4)]
    recs.append({**recs[0], "label_status": "unreachable", "S": []})
    t = im.ts_table(recs)
    t["lab_idx"] = [r["legal_idx"] for r in recs]
    path = tmp_path / "x.h5"
    im._save_table(t, path)
    with h5py.File(path, "r") as f:
        row = f["row"][:]
        assert row.shape == (3, im.A_DIM + 4)
        assert list(f["offsets"][:]) == [0, 3, 6, 9]
        np.testing.assert_allclose(row[:, :im.A_DIM].sum(1), [1, 1, 0])    # unreachable: no policy target
        s0 = f["set_idx"][f["set_indptr"][0]:f["set_indptr"][1]]
        np.testing.assert_allclose(row[0, s0], 1.0 / len(s0))
        assert set(row[:, im.A_DIM + 3]) == {0.0} and set(row[:, im.A_DIM + 2]) == {1.0}
        assert list(f["weight"][:]) == [1, 1, 0]
        assert set(np.abs(row[:, im.A_DIM])) == {1.0}                     # resultLabel = z = +-1
    back = im.load_table(path)
    assert back["labels"] == t["labels"] and list(back["meta/label_status"]) == list(t["meta/label_status"])
    sub = im.subset_table(back, np.array([2, 0]))
    assert list(sub["offsets"]) == [0, 3, 6] and sub["labels"][1] == back["labels"][0]


@needs_worker
def test_build_on_the_fixture_with_the_real_bridge(tmp_path):
    st = im.build(every=1, start=0, workers=1, limit=2, path=FIX, splits=np.zeros(10 ** 6, np.int8),
                  out_dir=tmp_path / "sh", runtime_root=tmp_path / "rt", replay_frac=1.0)
    assert st["games"] == 2 and st["ts"] > 0
    ts = im.load_shard(tmp_path / "sh" / "ts.pkl")
    ok = [r for r in ts if r["status"] == "ok"]
    assert len(ok) >= 0.9 * len(ts)
    for r in ok:
        assert set(r["S"]) <= set(r["legal"])
        assert len(r["features"]) > 100
        assert all(0 <= i < im.A_DIM for i in r["legal_idx"])
    rp = im.load_shard(tmp_path / "sh" / "rp.pkl")
    assert len(rp) == 2 and all("decisions" in r for r in rp)
    for t in rp:
        for d in t["decisions"]:
            assert d["chosen"] in d["legal"] or d["type"] != "PRIORITY"


def test_length_batches_budget_tokens_and_cover_every_row():
    assert im.batch_for(768, 32) == 32 and im.batch_for(1536, 32) == 8 and im.batch_for(3072, 32) == 2
    lens = np.array([100, 3000, 700, 2000, 800, 50, 1500] * 10)
    batches = im.length_batches(lens, 16)
    got = np.concatenate(batches)
    assert sorted(got.tolist()) == list(range(len(lens)))
    for b in batches:
        L = im.bucket_len(int(lens[b].max()))
        assert len(b) <= im.batch_for(L, 16)


def test_concat_tables_repacks_csr():
    a = {"indices": np.array([1, 2, 3]), "offsets": np.array([0, 1, 3]), "legal_idx": np.array([0, 5, 0, 7]),
         "legal_indptr": np.array([0, 2, 4]), "set_idx": np.array([5, 7]), "set_indptr": np.array([0, 1, 2]),
         "z": np.array([1.0, -1.0]), "labels": [["a"], ["b"]]}
    b = {"indices": np.array([9]), "offsets": np.array([0, 1]), "legal_idx": np.array([0, 9]),
         "legal_indptr": np.array([0, 2]), "set_idx": np.array([0]), "set_indptr": np.array([0, 1]),
         "z": np.array([1.0]), "labels": [["c"]]}
    c = im.concat_tables(a, b)
    assert list(c["offsets"]) == [0, 1, 3, 4] and list(c["indices"][3:]) == [9]
    assert list(c["set_indptr"]) == [0, 1, 2, 3] and list(c["set_idx"]) == [5, 7, 0]
    assert c["labels"] == [["a"], ["b"], ["c"]] and list(c["z"]) == [1.0, -1.0, 1.0]
