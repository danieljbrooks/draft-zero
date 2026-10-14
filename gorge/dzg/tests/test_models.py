import numpy as np
import pytest
import torch

from dzg import data as D
from dzg import models as M
from dzg import packfmt as pf

from .conftest import TINY, edge_records


def tiny(arch):
    torch.manual_seed(1)
    m = M.build(arch, TINY[arch]).eval()
    # the heads' last layers start at zero (uniform policy): give them weights so scores differ
    with torch.no_grad():
        for head in (m.policy, m.value):
            torch.nn.init.normal_(head[-1].weight, std=0.5)
            torch.nn.init.normal_(head[-1].bias, std=0.5)
    return m


def run(m, pack, sel=None):
    sel = np.arange(len(pack["dense"])) if sel is None else sel
    with torch.no_grad():
        return m(D.collate(pack, sel))


@pytest.mark.parametrize("arch", M.ARCHS)
def test_default_param_counts(arch):
    m = M.build(arch)
    c = m.parameter_counts()
    assert c["total"] == sum(p.numel() for p in m.parameters())
    assert c["table"] == pf.TABLE_ROWS * M.DEFAULTS[arch]["d_emb"]
    print(arch, c["total"])


@pytest.mark.parametrize("arch", M.ARCHS)
def test_wire_equals_collate(arch, syn):
    pack, _ = syn
    m = tiny(arch)
    sel = np.arange(0, 400, 9)
    s1, v1 = run(m, pack, sel)
    sub = {k: np.ascontiguousarray(v, dtype=pf.SPECS[k][0]) for k, v in D.take(pack, sel, targets=False).items()}
    wp, want = pf.decode_request(bytearray(pf.encode_request(sub)[4:]))
    with torch.no_grad():
        s2, v2 = m(D.batch_from_wire(wp, want))
    assert s1.dtype == torch.float32 and v1.dtype == torch.float32
    assert s1.shape == (int(sub["opt_off"][-1]),) and v1.shape == (len(sel),)
    assert torch.equal(s1, s2) and torch.equal(v1, v2)
    assert s1.std() > 0


def _with_options(rec, keep):
    r = dict(rec)
    keep = list(keep)
    for n in ("opt_dense", "opt_bot", "opt_ea", "opt_eb"):
        r[n] = rec[n][keep]
    for n in ("os_row", "os_val", "oh_row", "oh_val"):
        r[n] = [rec[n][k] for k in keep]
    r["cands"] = [np.array([j], np.int32) for j in range(len(keep))]
    for n in ("cand_visits", "cand_prior", "cand_q"):
        r[n] = np.ones(len(keep), np.float32)
    return r


@pytest.mark.parametrize("arch", M.ARCHS)
def test_option_independence(arch, syn):
    """An option's score depends only on its state and itself: not on the other options, their
    number, or the batch's padding (other states with more options or cards)."""
    pack, _ = syn
    m = tiny(arch)
    nopt = np.diff(pack["opt_off"])
    i = int(np.nonzero(nopt >= 6)[0][0])
    rec = pf.get_record(pack, i)
    full, vfull = run(m, pf.pack_records([_with_options(rec, range(len(rec["opt_bot"])))]))
    for keep in ([0, 2, 5], [5], [3, 1], [1, 0, 1]):   # subsets, reorders, a duplicate
        s, v = run(m, pf.pack_records([_with_options(rec, keep)]))
        assert torch.allclose(s, full[keep], atol=1e-5), keep
        assert torch.allclose(v, vfull, atol=1e-5)
    # padding: the same state inside a batch of other states with more options and cards
    big = int(np.argmax(nopt))
    others = [pf.get_record(pack, j) for j in (big, 0, 1)]
    batch = pf.pack_records([others[0], _with_options(rec, range(len(rec["opt_bot"]))), others[1], others[2]])
    s, v = run(m, batch)
    o0 = int(batch["opt_off"][1])
    assert torch.allclose(s[o0:o0 + len(full)], full, atol=1e-5)
    assert torch.allclose(v[1], vfull[0], atol=1e-5)


@pytest.mark.parametrize("arch", M.ARCHS)
def test_edge_states_forward_backward(arch):
    m = tiny(arch).train()
    pack = pf.pack_records(edge_records())
    for sel in (np.arange(6), np.array([3]), np.array([1]), np.array([0, 3]), np.array([1, 3]), np.arange(0)):
        b = D.collate(pack, sel)
        s, v = m(b)
        assert s.shape == (b.no,) and v.shape == (b.B,)
        assert torch.isfinite(s).all() and torch.isfinite(v).all()
        z, lp = M.candidate_logits(s, b)
        loss = v.sum() + s.sum() + (lp.exp().sum() if b.ncand else 0)
        m.zero_grad()
        loss.backward()
        for p in m.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()


def test_candidate_logits_sum_members(syn):
    pack, _ = syn
    b = D.collate(pack, np.arange(40))
    torch.manual_seed(0)
    s = torch.randn(b.no)
    z, lp = M.candidate_logits(s, b)
    co_off = np.asarray(D.take(pack, np.arange(40))["co_off"])
    co = b.co_opt.numpy()
    for k in range(b.ncand):
        members = co[co_off[k]:co_off[k + 1]]
        exp = float(s[members].sum()) if len(members) else 0.0
        assert abs(float(z[k]) - exp) < 1e-5
    # per-state softmax
    tot = torch.zeros(b.B).index_add(0, b.cand_state, lp.exp())
    has = torch.zeros(b.B).index_add(0, b.cand_state, torch.ones(b.ncand)) > 0
    assert torch.allclose(tot[has], torch.ones(int(has.sum())), atol=1e-5)
    # an empty candidate scores 0
    rec = pf.get_record(pack, 0)
    rec["cands"] = [np.zeros(0, np.int32)] + rec["cands"][1:]
    b2 = D.collate(pf.pack_records([rec]), np.array([0]))
    z2, _ = M.candidate_logits(torch.randn(b2.no), b2)
    assert float(z2[0]) == 0.0


@pytest.mark.parametrize("arch", M.ARCHS)
def test_save_load(arch, tmp_path, syn):
    pack, _ = syn
    m = tiny(arch)
    M.save(m, tmp_path / "m.pt", {"step": 3})
    m2, ck = M.load(tmp_path / "m.pt")
    assert ck["arch"] == arch and ck["meta"] == {"step": 3} and m2.config == m.config
    s1, v1 = run(m, pack, np.arange(20))
    s2, v2 = run(m2, pack, np.arange(20))
    assert torch.equal(s1, s2) and torch.equal(v1, v2)


def test_config_rejects_unknown():
    with pytest.raises(ValueError):
        M.build("mlp", {"nope": 1})
    with pytest.raises(ValueError):
        M.build("transformer", {"d": 30, "heads": 4})
