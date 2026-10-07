import json

import numpy as np
import pytest

from dzg import export as E
from dzg import gonet_parity as G
from dzg import models as M

from .conftest import TINY


def test_export_round_trip(tmp_path):
    m = M.build("mlp", TINY["mlp"]).eval()
    h = E.export(m, tmp_path / "m.dzgw", source="x.pt")
    h2, t = E.read(tmp_path / "m.dzgw")
    assert h2 == json.loads(json.dumps(h))
    sd = m.state_dict()
    assert set(t) == set(sd)
    for k, v in sd.items():
        assert np.array_equal(t[k], v.numpy()), k
    assert set(h["layernorm_eps"]) == {"blocks.0.norm", "blocks.1.norm", "norm", "opt_norm"}
    assert all(x["offset"] % E.ALIGN == 0 for x in h["tensors"])


@pytest.mark.parametrize("arch", ["transformer", "gnn"])
def test_export_refuses_other_archs(arch, tmp_path):
    with pytest.raises(ValueError, match="no Go forward pass"):
        E.export(M.build(arch, TINY[arch]), tmp_path / "m.dzgw")


def test_parity_records_match_the_pack(syn):
    pack, _ = syn
    recs = G.records({k: pack[k] for k in pack})
    assert len(recs) == len(pack["dense"])
    i = next(i for i, r in enumerate(recs) if r["cards"] and r["options"])
    r = recs[i]
    assert np.array_equal(np.array(r["dense"], np.float32), pack["dense"][i])
    co = int(pack["card_off"][i])
    assert r["cards"][0]["group"] == int(pack["card_group"][co])
    assert np.array_equal(np.array(r["cards"][0]["raw"], np.float32), pack["card_raw"][co])
    oo = int(pack["opt_off"][i])
    assert r["options"][0]["ea"] == int(pack["opt_ea"][oo])
    assert np.array_equal(np.array(r["options"][0]["dense"], np.float32), pack["opt_dense"][oo])
