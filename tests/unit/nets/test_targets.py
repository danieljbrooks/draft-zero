import math

import numpy as np

from draftzero.nets import data as D
from draftzero.nets import packfmt as pf
from draftzero.nets import targets as T


def batch_of(states):
    """states: list of dicts with visits, prior, q, root_value, outcome."""
    recs = []
    base, _ = pf.synthetic(1, seed=0, min_opts=6, max_opts=6)
    r0 = pf.get_record(base, 0)
    for s in states:
        r = dict(r0)
        k = len(s["visits"])
        r["cands"] = [np.array([j % 6], np.int32) for j in range(k)]
        r["cand_visits"] = np.array(s["visits"], np.float32)
        r["cand_prior"] = np.array(s["prior"], np.float32)
        r["cand_q"] = np.array(s["q"], np.float32)
        r["root_value"] = s.get("root_value", 0.5)
        r["outcome"] = s.get("outcome", 1.0)
        recs.append(r)
    pack = pf.pack_records(recs)
    return D.collate(pack, np.arange(len(recs)))


def softmax(x):
    e = np.exp(np.array(x) - max(x))
    return e / e.sum()


def test_visits_target():
    b = batch_of([{"visits": [3, 1, 0], "prior": [0.3, 0.3, 0.4], "q": [0.6, 0.4, 0]},
                  {"visits": [0, 2], "prior": [0.5, 0.5], "q": [0, 0.7]}])
    t = T.visits_target(b).numpy()
    assert np.allclose(t, [0.75, 0.25, 0, 0, 1])
    t2 = T.visits_target(b, temp=0.5).numpy()
    assert np.allclose(t2, [0.9, 0.1, 0, 0, 1])


def test_cq_target_by_hand():
    s = {"visits": [2, 1, 0], "prior": [0.5, 0.3, 0.2], "q": [0.6, 0.4, 0.0], "root_value": 0.5}
    b = batch_of([s])
    # v_mix = (0.5 + 3 / 0.8 * (0.5 * 0.6 + 0.3 * 0.4)) / (1 + 3) = 0.51875
    v_mix = (0.5 + 3 / 0.8 * 0.42) / 4
    assert abs(v_mix - 0.51875) < 1e-12
    qn = [1.0, 0.0, (v_mix - 0.4) / 0.2]                       # completed q [0.6, 0.4, v_mix]
    sigma = [(50 + 2) * 0.1 * x for x in qn]
    exp = softmax([math.log(p + 1e-8) + g for p, g in zip(s["prior"], sigma)])
    got = T.cq_target(b).numpy()
    assert np.allclose(got, exp, atol=1e-6)
    got2 = T.cq_target(b, c_visit=10, c_scale=1.0).numpy()
    exp2 = softmax([math.log(p + 1e-8) + (10 + 2) * 1.0 * x for p, x in zip(s["prior"], qn)])
    assert np.allclose(got2, exp2, atol=1e-6)


def test_cq_constant_q_returns_prior():
    # every candidate visited with the same q: normalised q is 0.5 everywhere, the target is the prior
    b = batch_of([{"visits": [1, 1, 1], "prior": [0.2, 0.3, 0.5], "q": [0.4, 0.4, 0.4]},
                  # nothing visited: completed q = v_mix = root_value for all, again the prior
                  {"visits": [0, 0], "prior": [0.25, 0.75], "q": [0, 0], "root_value": 0.9}])
    assert np.allclose(T.cq_target(b).numpy(), [0.2, 0.3, 0.5, 0.25, 0.75], atol=1e-6)


def test_policy_and_value_masks():
    b = batch_of([{"visits": [3, 1], "prior": [0.5, 0.5], "q": [0.5, 0.5], "outcome": 1.0},
                  {"visits": [4], "prior": [1.0], "q": [0.5], "outcome": -1.0},           # one candidate
                  {"visits": [0, 0], "prior": [0.5, 0.5], "q": [0, 0], "outcome": 0.5},    # no visits
                  {"visits": [1, 1], "prior": [0.5, 0.5], "q": [0.5, 0.5], "outcome": 0.0, "root_value": 0.2}])
    assert T.policy_mask(b).tolist() == [True, False, False, True]
    t, m = T.value_target(b)
    assert m.tolist() == [True, False, True, True]
    assert np.allclose(t.numpy(), [1.0, 0.0, 0.5, 0.0])
    t, m = T.value_target(b, blend=0.25)
    assert np.allclose(t.numpy()[[0, 2, 3]], [0.75 + 0.25 * 0.5, 0.375 + 0.125, 0.25 * 0.2])


def _value_target_before_next(b, blend=0.0):
    """targets.value_target as it was before next_w, verbatim."""
    import torch
    z = b.outcome.float()
    mask = z >= 0
    t = z if blend == 0 else (1 - blend) * z + blend * b.root_value.float()
    return torch.where(mask, t, torch.zeros_like(t)), mask


def test_value_target_with_the_next_turns_share():
    """(1 - blend - next_w) outcome + blend root_value + next_w next_rv, the mask unchanged; next_w 0 is the target
    before next_rv, bit for bit (with or without a next_rv in the batch); next_w > 0 needs one."""
    import pytest
    import torch
    b = batch_of([{"visits": [1, 1], "prior": [0.5, 0.5], "q": [0.5, 0.5], "outcome": o, "root_value": r}
                  for o, r in ((1.0, 0.7), (-1.0, 0.4), (0.5, 0.55), (0.0, 0.2))])
    for blend in (0.0, 0.25, 0.5, 1.0):
        t0, m0 = _value_target_before_next(b, blend)
        t1, m1 = T.value_target(b, blend, 0.0)
        assert torch.equal(t0, t1) and torch.equal(m0, m1)
    with pytest.raises(ValueError, match="next_rv"):
        T.value_target(b, 0.5, 0.5)
    b.next_rv = torch.tensor([0.9, 0.3, 0.6, 0.1])
    for blend in (0.0, 0.5, 1.0):
        t0, _ = _value_target_before_next(b, blend)
        assert torch.equal(T.value_target(b, blend)[0], t0)              # a batch with next_rv, next_w 0: unchanged
    t, m = T.value_target(b, 0.5, 0.5)
    assert m.tolist() == [True, False, True, True]
    assert np.allclose(t.numpy(), [0.5 * 0.7 + 0.5 * 0.9, 0.0, 0.5 * 0.55 + 0.5 * 0.6, 0.5 * 0.2 + 0.5 * 0.1])
    t, _ = T.value_target(b, 0.25, 0.5)
    assert np.allclose(t.numpy()[[0, 2, 3]], [0.25 + 0.25 * 0.7 + 0.45, 0.125 + 0.25 * 0.55 + 0.3, 0.05 + 0.05])
    t, _ = T.value_target(b, 0.0, 1.0)
    assert np.allclose(t.numpy()[[0, 2, 3]], [0.9, 0.6, 0.1])


def test_policy_surprise_by_hand():
    b = batch_of([{"visits": [3, 1, 0], "prior": [0.5, 0.3, 0.2], "q": [0.6, 0.4, 0]},
                  {"visits": [0, 2], "prior": [2.0, 2.0], "q": [0, 0.7]},           # prior renormalised: 0.5, 0.5
                  {"visits": [1, 1, 1], "prior": [0, 0, 0], "q": [0.5, 0.5, 0.5]},  # no prior: uniform
                  {"visits": [2, 6], "prior": [0.25, 0.75], "q": [0.4, 0.6]},       # target = prior
                  {"visits": [0, 4], "prior": [1.0, 0.0], "q": [0, 0.6]}])          # visited, prior 0: floored
    kl = T.policy_surprise(b, T.visits_target(b)).numpy()
    want = [0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.3), math.log(2), 0.0, 0.0, -math.log(1e-6)]
    assert np.allclose(kl, want, atol=1e-5), kl
    assert (kl >= 0).all()


def test_surprise_weights():
    import torch
    s = torch.tensor([1.0, 3.0, 0.0, 100.0])
    mask = torch.tensor([True, True, True, False])
    w = T.surprise_weights(s, mask, 0.5)
    assert np.allclose(w[:3].numpy(), [0.5 + 0.5 * 0.75, 0.5 + 0.5 * 2.25, 0.5])    # mean surprise 4/3
    assert abs(float(w[mask].mean()) - 1) < 1e-6
    assert torch.equal(T.surprise_weights(s, mask, 0.0), torch.ones(4))
    assert torch.equal(T.surprise_weights(torch.zeros(4), mask, 0.5), torch.ones(4))   # no surprise: uniform
    assert np.allclose(T.surprise_weights(s, mask, 1.0)[:3].numpy(), [0.75, 2.25, 0.0])
