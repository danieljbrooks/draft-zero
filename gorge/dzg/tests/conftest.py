import numpy as np
import pytest
import torch

from dzg import packfmt as pf

TINY = {
    "mlp": {"d": 32, "ff": 64, "layers": 2, "d_emb": 16, "card_d": 24, "head_hidden": 32},
    "transformer": {"d": 32, "heads": 4, "layers": 2, "ff": 64, "d_emb": 16, "head_hidden": 32},
    "gnn": {"d": 32, "heads": 4, "global_layers": 1, "ff": 64, "d_emb": 16, "head_hidden": 32},
}


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture(scope="session")
def syn():
    """A small synthetic training pack and its games."""
    return pf.synthetic(400, seed=7)


def edge_records():
    """Records with zero cards, zero options, zero sparse rows, and all three at once."""
    pack, _ = pf.synthetic(6, seed=3, min_opts=1, max_opts=4)
    recs = [pf.get_record(pack, i) for i in range(6)]
    r0 = recs[0]                                   # zero cards (ea/eb must become none)
    r0["card_group"] = r0["card_group"][:0]
    r0["card_raw"] = r0["card_raw"][:0]
    r0["cr_row"], r0["cr_val"] = [], []
    r0["opt_ea"] = np.zeros_like(r0["opt_ea"])
    r0["opt_eb"] = np.zeros_like(r0["opt_eb"])
    r1 = recs[1]                                   # zero options, zero candidates
    for n in ("opt_dense", "opt_bot", "opt_ea", "opt_eb"):
        r1[n] = r1[n][:0]
    r1["os_row"] = r1["os_val"] = r1["oh_row"] = r1["oh_val"] = []
    r1["cands"] = []
    for n in ("cand_visits", "cand_prior", "cand_q"):
        r1[n] = r1[n][:0]
    r2 = recs[2]                                   # zero sparse rows
    r2["sp_row"], r2["sp_val"] = r2["sp_row"][:0], r2["sp_val"][:0]
    r3 = recs[3]                                   # nothing at all
    for k in ("card_group", "card_raw", "sp_row", "sp_val", "opt_dense", "opt_bot", "opt_ea", "opt_eb",
              "cand_visits", "cand_prior", "cand_q"):
        r3[k] = r3[k][:0]
    for k in ("cr_row", "cr_val", "os_row", "os_val", "oh_row", "oh_val", "cands"):
        r3[k] = []
    return recs
