import struct

import numpy as np
import pytest

from dzg import packfmt as pf

from .conftest import edge_records


def _same(a: dict, b: dict, names):
    for n in names:
        x, y = np.asarray(a[n]), np.asarray(b[n])
        assert x.dtype == y.dtype, n
        assert x.shape == y.shape, n
        assert np.array_equal(x, y), n


def test_synthetic_valid(syn):
    pack, games = syn
    pf.validate(pack)
    c = pf.counts(pack)
    assert c["B"] == 400 and c["no"] >= 400 and c["nc"] > 0
    assert len(games) == int(pack["game"].max()) + 1
    # candidate 0 is the bot's answer: opt_bot marks exactly its options
    for i in range(50):
        r = pf.get_record(pack, i)
        bot = set(np.nonzero(r["opt_bot"])[0].tolist())
        assert bot == set(r["cands"][0].tolist())


def test_shard_roundtrip(tmp_path, syn):
    pack, games = syn
    pf.write_shard(tmp_path / "s", pack, games)
    for in_ram in (False, True):
        back, meta = pf.read_shard(tmp_path / "s", in_ram=in_ram)
        assert meta == {"format": pf.FORMAT, "n": 400, "games": games, "source": "synthetic"}
        _same(pack, back, pf.STATE_NAMES + pf.TRAIN_NAMES)
        pf.validate(back)
    if True:
        back, _ = pf.read_shard(tmp_path / "s")
        assert isinstance(back["card_raw"], np.memmap)
    assert pf.shard_dirs(tmp_path) == [tmp_path / "s"]


def test_records_roundtrip(syn):
    pack, _ = syn
    back = pf.pack_records([pf.get_record(pack, i) for i in range(len(pack["dense"]))])
    _same(pack, back, pf.STATE_NAMES + pf.TRAIN_NAMES)
    edge = pf.pack_records(edge_records())
    pf.validate(edge)


def test_wire_request_roundtrip(syn):
    pack, _ = syn
    sub = pf.pack_records([pf.get_record(pack, i) for i in range(37)] + edge_records())
    want = (np.arange(len(sub["dense"])) % 3 != 0).astype(np.uint8)
    msg = pf.encode_request(sub, want)
    (n,) = struct.unpack_from("<I", msg)
    assert n == len(msg) - 4 and n % 8 == 0
    c = pf.counts(sub)
    assert n == pf.request_size(c)
    hdr = struct.unpack_from("<8I", msg, 4)
    assert hdr == (pf.REQ_MAGIC, c["B"], c["nsp"], c["nc"], c["ncr"], c["no"], c["nos"], c["noh"])
    back, w = pf.decode_request(bytearray(msg[4:]))
    assert np.array_equal(w, want)
    _same(sub, back, pf.STATE_NAMES)
    pf.validate(back, train=False)
    # every array starts 8-byte aligned in the body; the last array's padding may be omitted
    body = msg[4:]
    last = c["noh"] * 4
    unpadded = body[: len(body) - (pf.pad8(last) - last)]
    back2, _ = pf.decode_request(unpadded)
    _same(sub, back2, pf.STATE_NAMES)
    with pytest.raises(ValueError):
        pf.decode_request(body[:-12])
    with pytest.raises(ValueError):
        pf.decode_request(b"\0" * 32 + body[32:])


def test_wire_layout_by_hand():
    """A one-state request laid out by hand from SPEC §3."""
    rec = {"dense": np.arange(68, dtype=np.float32), "sp_row": np.array([5, 9], np.uint16),
           "sp_val": np.array([1.0, 2.0], np.float32), "card_group": np.array([0], np.uint8),
           "card_raw": np.ones((1, 68), np.float32), "cr_row": [np.array([7], np.uint16)],
           "cr_val": [np.array([1.0], np.float32)], "opt_dense": np.zeros((1, 24), np.float32),
           "opt_bot": np.array([1], np.uint8), "opt_ea": np.array([1], np.int32), "opt_eb": np.array([0], np.int32),
           "os_row": [np.array([3, 4], np.uint16)], "os_val": [np.array([1.0, 1.0], np.float32)],
           "oh_row": [np.array([11], np.uint16)], "oh_val": [np.array([1.0], np.float32)]}
    msg = pf.encode_request(pf.pack_records([rec]), np.array([1], np.uint8))
    parts = [struct.pack("<8I", pf.REQ_MAGIC, 1, 2, 1, 1, 1, 2, 1),
             b"\x01" + b"\0" * 7, np.arange(68, dtype="<f4").tobytes(),
             struct.pack("<2i", 0, 2), struct.pack("<2H", 5, 9) + b"\0" * 4, struct.pack("<2f", 1, 2),
             struct.pack("<2i", 0, 1), b"\0" * 8, np.ones(68, "<f4").tobytes(), struct.pack("<2i", 0, 1),
             struct.pack("<H", 7) + b"\0" * 6, struct.pack("<f", 1) + b"\0" * 4,
             struct.pack("<2i", 0, 1), np.zeros(24, "<f4").tobytes(), b"\x01" + b"\0" * 7,
             struct.pack("<i", 1) + b"\0" * 4, struct.pack("<i", 0) + b"\0" * 4,
             struct.pack("<2i", 0, 2), struct.pack("<2H", 3, 4) + b"\0" * 4, struct.pack("<2f", 1, 1),
             struct.pack("<2i", 0, 1), struct.pack("<H", 11) + b"\0" * 6, struct.pack("<f", 1) + b"\0" * 4]
    body = b"".join(parts)
    assert msg == struct.pack("<I", len(body)) + body


def test_wire_response_roundtrip():
    for B, no in ((1, 0), (3, 5), (4, 7), (0, 0)):
        v = np.random.rand(B).astype(np.float32)
        s = np.random.randn(no).astype(np.float32)
        msg = pf.encode_response(v, s)
        (n,) = struct.unpack_from("<I", msg)
        assert n == len(msg) - 4 == 16 + pf.pad8(4 * B) + 4 * no
        assert struct.unpack_from("<4I", msg, 4) == (pf.RESP_MAGIC, B, no, 0)
        v2, s2 = pf.decode_response(msg[4:])
        assert np.array_equal(v, v2) and np.array_equal(s, s2)


def test_validate_rejects():
    pack, _ = pf.synthetic(5, seed=1)
    bad = dict(pack)
    bad["opt_ea"] = pack["opt_ea"].copy()
    bad["opt_ea"][0] = 1000
    with pytest.raises(ValueError):
        pf.validate(bad)
    bad = dict(pack)
    bad["os_row"] = pack["os_row"].copy()
    bad["os_row"][0] = 200
    with pytest.raises(ValueError):
        pf.validate(bad)


def test_holdout_split_rate():
    ids = [f"g{i}" for i in range(20000)]
    frac = np.mean([pf.holdout_game(g) for g in ids])
    assert 0.04 < frac < 0.06
