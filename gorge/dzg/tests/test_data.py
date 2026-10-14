import numpy as np
import torch

from dzg import data as D
from dzg import packfmt as pf

from .conftest import edge_records


def naive(pack, sel):
    """The Batch's arrays built record by record with Python loops (the reference)."""
    out = {k: [] for k in ("dense", "sp_row", "sp_val", "sp_state", "card_raw", "card_group", "card_state", "card_pos",
                           "cr_row", "cr_val", "cr_card", "opt_dense", "opt_bot", "opt_state", "opt_pos", "opt_ea",
                           "opt_eb", "os_row", "os_val", "oh_row", "oh_val", "cand_state", "cand_visits",
                           "cand_prior", "cand_q", "co_cand", "co_opt", "root_value", "outcome", "subset")}
    offs = {k: [0] for k in ("sp_off", "cr_off", "os_off", "oh_off")}
    ncards, nopts = [], []
    nc = no = ncand = 0
    for s, i in enumerate(sel):
        r = pf.get_record(pack, int(i))
        out["dense"].append(r["dense"])
        for row, val in zip(r["sp_row"], r["sp_val"]):
            out["sp_row"].append(int(row)); out["sp_val"].append(float(val)); out["sp_state"].append(s)
        offs["sp_off"].append(len(out["sp_row"]))
        c = len(r["card_group"])
        for j in range(c):
            out["card_raw"].append(r["card_raw"][j]); out["card_group"].append(int(r["card_group"][j]))
            out["card_state"].append(s); out["card_pos"].append(j)
            for row, val in zip(r["cr_row"][j], r["cr_val"][j]):
                out["cr_row"].append(int(row)); out["cr_val"].append(float(val)); out["cr_card"].append(nc + j)
            offs["cr_off"].append(len(out["cr_row"]))
        o = len(r["opt_bot"])
        for j in range(o):
            out["opt_dense"].append(r["opt_dense"][j]); out["opt_bot"].append(float(r["opt_bot"][j]))
            out["opt_state"].append(s); out["opt_pos"].append(j)
            for n in ("opt_ea", "opt_eb"):
                e = int(r[n][j])
                out[n].append(nc + e - 1 if 1 <= e <= c else -1)
            for q in ("os", "oh"):
                for row, val in zip(r[f"{q}_row"][j], r[f"{q}_val"][j]):
                    out[f"{q}_row"].append(int(row)); out[f"{q}_val"].append(float(val))
                offs[f"{q}_off"].append(len(out[f"{q}_row"]))
        for k, members in enumerate(r["cands"]):
            out["cand_state"].append(s)
            out["cand_visits"].append(r["cand_visits"][k]); out["cand_prior"].append(r["cand_prior"][k])
            out["cand_q"].append(r["cand_q"][k])
            for m in members:
                out["co_cand"].append(ncand + k); out["co_opt"].append(no + int(m))
        for n in ("root_value", "outcome", "subset"):
            out[n].append(r[n])
        ncards.append(c); nopts.append(o)
        nc += c; no += o; ncand += len(r["cands"])
    res = {}
    for k, v in out.items():
        if k in ("dense", "card_raw", "opt_dense"):
            w = {"dense": 68, "card_raw": 68, "opt_dense": 24}[k]
            res[k] = np.array(v, np.float32).reshape(-1, w)
        else:
            res[k] = np.array(v, np.float64)
    res.update({k: np.array(v) for k, v in offs.items()})
    Cmax = max(ncards) if ncards else 0
    res["card_mask"] = np.array([[j < c for j in range(Cmax)] for c in ncards], bool).reshape(len(sel), Cmax)
    res["Cmax"], res["Omax"], res["ncand"] = Cmax, max(nopts) if nopts else 0, ncand
    return res


def check_batch(b: D.Batch, ref: dict, targets: bool = True):
    names = [k for k in ref if k not in ("Cmax", "Omax", "ncand")]
    if not targets:
        names = [k for k in names if k not in ("cand_state", "cand_visits", "cand_prior", "cand_q", "co_cand", "co_opt",
                                               "root_value", "outcome", "subset")]
    for k in names:
        got = getattr(b, k)
        assert isinstance(got, torch.Tensor), k
        g = got.numpy()
        assert g.shape == ref[k].shape, (k, g.shape, ref[k].shape)
        assert np.array_equal(g.astype(np.float64), ref[k].astype(np.float64)), k
    assert b.Cmax == ref["Cmax"] and b.Omax == ref["Omax"]
    assert b.B == len(ref["dense"])
    if targets:
        assert b.ncand == ref["ncand"]
    # dtypes the models rely on
    for k in ("sp_row", "cr_row", "os_row", "oh_row", "card_state", "card_group", "opt_state", "opt_ea", "sp_off"):
        assert getattr(b, k).dtype == torch.int64, k
    for k in ("dense", "card_raw", "opt_dense", "sp_val", "opt_bot"):
        assert getattr(b, k).dtype == torch.float32, k


def test_collate_matches_naive(syn):
    pack, _ = syn
    rng = np.random.default_rng(0)
    for sel in (np.arange(400), rng.choice(400, 64, replace=False), np.array([5, 5, 3, 399, 0]), np.array([17])):
        check_batch(D.collate(pack, sel), naive(pack, sel))


def test_collate_edge_records():
    pack = pf.pack_records(edge_records())
    sel = np.array([3, 0, 1, 2, 4, 5, 3])
    check_batch(D.collate(pack, sel), naive(pack, sel))
    b = D.collate(pack, np.array([3]))
    assert b.Cmax == 0 and b.Omax == 0 and b.ncand == 0 and b.card_mask.shape == (1, 0)


def test_wire_batch_matches_collate(syn):
    pack, _ = syn
    sel = np.array([9, 4, 100, 7, 350])
    sub = D.take(pack, sel, targets=False)
    sub = {k: np.ascontiguousarray(v, dtype=pf.SPECS[k][0]) for k, v in sub.items()}
    wp, want = pf.decode_request(bytearray(pf.encode_request(sub)[4:]))
    b = D.batch_from_wire(wp, want)
    check_batch(b, naive(pack, sel), targets=False)
    assert b.cand_state is None and bool(b.want.all())


def test_shardset_split_and_multishard(tmp_path):
    dirs = pf.write_synthetic(tmp_path / "pk", 500, seed=2, shard_size=200, game_len=7)
    assert [d.name for d in dirs] == ["00000", "00001", "00002"]
    for in_ram in (False, True):
        ds = D.ShardSet([tmp_path / "pk"], in_ram=in_ram)
        assert len(ds) == 500
        tr, ho = ds.split()
        assert len(tr) + len(ho) == 500 and len(np.intersect1d(tr, ho)) == 0
        games = []
        for p, m in zip(ds.packs, ds.metas):
            games += [m["games"][g] for g in np.asarray(p["game"])]
        games = np.array(games)
        assert all(pf.holdout_game(g) for g in games[ho])
        assert not any(pf.holdout_game(g) for g in games[tr])
        assert not set(games[ho]) & set(games[tr])
        tr2, ho2 = ds.split(limit=100)
        assert len(tr2) + len(ho2) == 100
        # a batch across shards: memory-mapped, states grouped by shard in the given order within each
        # shard; in RAM (one merged pack), in the given order
        sel = np.array([450, 10, 210, 5, 199, 200])
        b = ds.collate(sel)
        order = list(sel) if in_ram else [10, 5, 199, 210, 200, 450]
        recs = []
        for g in order:
            s = int(np.searchsorted(ds.starts, g, side="right") - 1)
            recs.append(pf.get_record(ds.packs[s], g - int(ds.starts[s])))
        ref_pack = pf.pack_records(recs)
        check_batch(b, naive(ref_pack, np.arange(len(order))))


def _same_record(a: dict, b: dict) -> bool:
    """Same record, apart from `game` (an index into each pack's own games list)."""
    for k in a:
        if k == "game":
            continue
        x, y = a[k], b[k]
        if isinstance(x, list):
            if len(x) != len(y) or not all(np.array_equal(np.asarray(u), np.asarray(v)) for u, v in zip(x, y)):
                return False
        elif not np.array_equal(np.asarray(x), np.asarray(y)):
            return False
    return True


def test_in_ram_merges_shards(tmp_path):
    pf.write_synthetic(tmp_path / "pk", 530, seed=4, shard_size=100, game_len=9)   # 6 shards, the last partial
    mm = D.ShardSet(tmp_path / "pk")
    ram = D.ShardSet(tmp_path / "pk", in_ram=True)
    assert len(mm.packs) == 6 and len(ram.packs) == 1 and len(mm) == len(ram) == 530
    assert all(not isinstance(a, np.memmap) for a in ram.packs[0].values())
    assert ram.packs[0]["co_off"].dtype == np.int64 and ram.metas[0]["n"] == 530
    # every record, its game id and the split are the same as the shards'
    for g in range(530):
        s = int(np.searchsorted(mm.starts, g, side="right") - 1)
        r_mm = pf.get_record(mm.packs[s], g - int(mm.starts[s]))
        r_ram = pf.get_record(ram.packs[0], g)
        assert _same_record(r_mm, r_ram), g
        assert mm.metas[s]["games"][r_mm["game"]] == ram.metas[0]["games"][r_ram["game"]]
    assert np.array_equal(mm.holdout_mask(), ram.holdout_mask())
    for tr_a, tr_b in zip(mm.split(), ram.split()):
        assert np.array_equal(tr_a, tr_b)
    assert np.array_equal(mm.array("outcome"), ram.array("outcome"))
    # a batch spanning every shard, in the given order, as the naive path builds it
    sel = np.array([529, 3, 250, 101, 99, 400, 0, 350])
    check_batch(ram.collate(sel), naive(ram.packs[0], sel))
    recs = []
    for g in sel:
        s = int(np.searchsorted(mm.starts, g, side="right") - 1)
        recs.append(pf.get_record(mm.packs[s], g - int(mm.starts[s])))
    check_batch(ram.collate(sel), naive(pf.pack_records(recs), np.arange(len(sel))))


def test_shardset_pickles_by_path(tmp_path):
    import pickle
    pf.write_synthetic(tmp_path / "pk", 230, seed=6, shard_size=100)
    for in_ram in (False, True):
        ds = D.ShardSet(tmp_path / "pk", in_ram=in_ram)
        blob = pickle.dumps(ds)
        assert len(blob) < 10000   # paths, not data
        back = pickle.loads(blob)
        sel = np.array([5, 150, 229, 0])
        a, b = ds.collate(sel), back.collate(sel)
        for k in ("dense", "sp_row", "card_raw", "opt_ea", "co_opt", "cand_visits"):
            assert torch.equal(getattr(a, k), getattr(b, k)), k


def _batch_equal(a: D.Batch, b: D.Batch) -> None:
    from dataclasses import fields
    for f in fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, torch.Tensor):
            assert isinstance(y, torch.Tensor) and x.dtype == y.dtype and x.shape == y.shape, f.name
            assert torch.equal(x, y), f.name
        else:
            assert x == y, f.name


def test_flatten_roundtrip(syn):
    pack, _ = syn
    b = D.collate(pack, np.array([4, 77, 3, 300]))
    fb = b.flatten()
    assert set(fb.bufs) == {torch.float32, torch.int64, torch.bool, torch.uint8}
    _batch_equal(b, fb.unflatten())
    w = D.batch_from_wire(D.take(pack, np.arange(7), targets=False), want=np.array([1, 0, 1, 1, 0, 0, 1]))
    _batch_equal(w, w.flatten().unflatten())   # None target fields stay None
    e = D.collate(pf.pack_records(edge_records()), np.array([3]))   # zero cards, options, candidates
    _batch_equal(e, e.flatten().unflatten())


def test_loader_procs(syn, tmp_path):
    import multiprocessing as mp
    pack, games = syn
    pf.write_shard(tmp_path / "s", pack, games)
    ds = D.ShardSet(tmp_path / "s")
    bl = D.batches(np.arange(400), 48, shuffle=True, seed=5)
    # fork where the platform has it: spawn on macOS works too but each worker takes ~5 s to stop
    method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    got = list(D.Loader(ds, bl, prefetch=4, procs=2, start_method=method))
    assert len(got) == len(bl)
    for b, sel in zip(got, bl):
        _batch_equal(b, ds.collate(sel))


def test_loader_order_and_threads(syn, tmp_path):
    pack, games = syn
    pf.write_shard(tmp_path / "s", pack, games)
    ds = D.ShardSet(tmp_path / "s")
    bl = D.batches(np.arange(400), 64, shuffle=True, seed=3)
    assert sorted(np.concatenate(bl).tolist()) == list(range(400))
    for workers in (1, 3):
        got = list(D.Loader(ds, bl, prefetch=2, workers=workers))
        assert len(got) == len(bl)
        for b, sel in zip(got, bl):
            assert torch.equal(b.dense, torch.from_numpy(np.asarray(pack["dense"][sel])))
    # breaking early stops the threads cleanly
    for i, _ in enumerate(D.Loader(ds, bl, prefetch=2, workers=2)):
        if i == 1:
            break


def test_drop_unwanted_options(syn):
    pack, _ = syn
    sub = D.take(pack, np.arange(10), targets=False)
    want = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 1], np.uint8)
    out, kept = D.drop_unwanted_options(sub, want)
    nopt = np.diff(sub["opt_off"])
    assert np.array_equal(np.diff(out["opt_off"]), np.where(want == 1, nopt, 0))
    exp = np.concatenate([np.arange(sub["opt_off"][i], sub["opt_off"][i + 1]) for i in range(10) if want[i]])
    assert np.array_equal(kept, exp)
    assert np.array_equal(out["opt_dense"], sub["opt_dense"][exp])
    for q in ("os", "oh"):
        rows = [sub[f"{q}_row"][sub[f"{q}_off"][o]:sub[f"{q}_off"][o + 1]] for o in exp]
        assert np.array_equal(out[f"{q}_row"], np.concatenate(rows))
    same, none = D.drop_unwanted_options(sub, np.ones(10, np.uint8))
    assert none is None and same is sub
