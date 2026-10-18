import numpy as np
import torch

from draftzero.nets import data as D
from draftzero.nets import packfmt as pf

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


def test_record_weights_reach_the_batch(tmp_path):
    """The optional weight (SPEC §2) follows its records through take, collate, the multi-shard and in-RAM paths,
    flatten and the loader: shards without weight.npy weigh 1, a bare pack without "weight" gives weight None."""
    wpack, games = pf.synthetic(150, seed=21, weighted=0.4)
    pf.write_shard(tmp_path / "pk" / "00000", wpack, games)                     # with weight.npy
    plain, games2 = pf.synthetic(120, seed=22)
    pf.write_shard(tmp_path / "pk" / "00001", plain, games2)                    # without
    assert (tmp_path / "pk" / "00000" / "weight.npy").exists()
    assert not (tmp_path / "pk" / "00001" / "weight.npy").exists()
    want = np.concatenate([wpack["weight"], np.ones(120, np.float32)])
    sel = np.array([3, 140, 149, 150, 0, 269, 77])
    for in_ram in (False, True):
        ds = D.ShardSet(tmp_path / "pk", in_ram=in_ram)
        assert np.array_equal(ds.array("weight"), want)
        b = ds.collate(sel)
        # memory-mapped, a batch's states come grouped by shard (in the given order within each shard)
        order = sel if in_ram else np.array([3, 140, 149, 0, 77, 150, 269])
        assert b.weight.dtype == torch.float32 and torch.equal(b.weight, torch.from_numpy(want[order]))
        _batch_equal(b, b.flatten().unflatten())
        (lb,) = list(D.Loader(ds, [sel], workers=1))
        assert torch.equal(lb.weight, b.weight)
    assert D.collate(plain, np.arange(5)).weight is None                       # a bare pack: no weights
    assert torch.equal(D.collate(wpack, np.array([7, 2])).weight, torch.from_numpy(wpack["weight"][[7, 2]]))
    assert D.batch_from_wire(D.take(wpack, np.arange(3), targets=False)).weight is None


# ------------------------------------------------------------------------------------------------ next_rv

def test_next_own_turn_by_hand():
    """The seat's next own turn: t + 2 from its own turn, t + 1 from the opponent's, t + 1 after an extra turn; its
    first record there; other seats' and other games' records never count; none left: the outcome (-1 unknown)."""
    rows = [  # game, seat, turn, own, root value, outcome
        (0, 0, 1, 1, 0.50, 1.0),   # 0  seat 0's turn 1 -> its turn 3: record 6
        (0, 1, 1, 0, 0.40, 0.0),   # 1  seat 1 on seat 0's turn 1 -> its turn 2: record 3 (not seat 0's record 4)
        (0, 0, 1, 1, 0.52, 1.0),   # 2  -> 6
        (0, 1, 2, 1, 0.45, 0.0),   # 3  seat 1's turn 2 -> its turn 4: none in this game (record 9 is game 1's)
        (0, 0, 2, 0, 0.55, 1.0),   # 4  seat 0 on seat 1's turn 2 -> its turn 3: record 6
        (1, 0, 3, 1, 0.99, 0.5),   # 5  game 1, interleaved with game 0: never game 0's
        (0, 0, 3, 1, 0.60, 1.0),   # 6  the first record of seat 0's turn 3 -> its turn 5 (extra turn 6 after it)
        (0, 0, 3, 1, 0.62, 1.0),   # 7  -> 10
        (0, 1, 3, 0, 0.35, 0.0),   # 8  seat 1's next own turn would be 4: none -> outcome 0
        (1, 1, 4, 1, 0.20, 0.5),   # 9  game 1, seat 1: none later -> outcome 0.5
        (0, 0, 5, 1, 0.70, 1.0),   # 10 seat 0's turn 5 -> its extra turn 6: record 11
        (0, 0, 6, 1, 0.80, 1.0),   # 11 none later -> outcome 1
        (2, 0, 1, 1, 0.30, -1.0),  # 12 game 2, outcome unknown, nothing later: -1 (the value mask drops it)
        (2, 0, 2, 0, 0.31, -1.0),  # 13 -> nothing: -1
    ]
    g, s, t, own, rv, oc = (np.array(c) for c in zip(*rows))
    nxt = D.next_own_turn(g, s, t, own)
    assert nxt.tolist() == [6, 3, 6, -1, 6, -1, 10, 10, -1, -1, 11, -1, -1, -1]
    nrv = D.next_root_value(g, s, t, own, rv, oc)
    want = [0.60, 0.45, 0.60, 0.0, 0.60, 0.5, 0.70, 0.70, 0.0, 0.5, 0.80, 1.0, -1.0, -1.0]
    assert nrv.dtype == np.float32 and np.allclose(nrv, want)
    # games numbered differently, records of other games anywhere: the same answer
    perm = np.array([5, 9, 12, 13, 0, 1, 2, 3, 4, 6, 7, 8, 10, 11])
    back = D.next_own_turn(g[perm] + 7, s[perm], t[perm], own[perm])
    assert [perm[j] if j >= 0 else -1 for j in back] == nxt[perm].tolist()
    assert D.next_own_turn([], [], [], []).tolist() == [] and D.next_own_turn([0], [0], [1], [0]).tolist() == [-1]


def _turn_shards(root):
    """Two shards whose games and turns are set by hand; game g1 spans both (its turn 3 is in the second)."""
    first = [("g0", 0, 1, 1, 0.5, 1.0), ("g0", 1, 1, 0, 0.4, 0.0), ("g1", 0, 1, 1, 0.3, 0.0),
             ("g0", 1, 2, 1, 0.45, 0.0), ("g1", 1, 2, 1, 0.7, 1.0), ("g0", 0, 3, 1, 0.6, 1.0)]
    second = [("g1", 0, 3, 1, 0.35, 0.0), ("g2", 0, 1, 1, 0.9, -1.0), ("g1", 1, 3, 0, 0.66, 1.0)]
    dirs = []
    for k, rows in enumerate((first, second)):
        pack, _ = pf.synthetic(len(rows), seed=30 + k)
        games = list(dict.fromkeys(r[0] for r in rows))
        pack["game"] = np.array([games.index(r[0]) for r in rows], np.int32)
        pack["seat"] = np.array([r[1] for r in rows], np.uint8)
        pack["turn"] = np.array([r[2] for r in rows], np.int32)
        pack["dense"][:, D.ACTIVE] = [r[3] for r in rows]
        pack["root_value"] = np.array([r[4] for r in rows], np.float32)
        pack["outcome"] = np.array([r[5] for r in rows], np.float32)
        d = root / f"{k:05d}"
        pf.write_shard(d, pack, games)
        dirs.append(d)
    # g0: 0 -> 5, 1 -> 3, 3 -> none (0.0), 5 -> none (1.0); g1 across shards: 2 -> 6, 4 -> none, 8 -> none (1.0)
    want = np.array([0.6, 0.45, 0.35, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0], np.float32)
    return dirs, want


def test_shardset_next_rv_reaches_the_batch(tmp_path):
    """ShardSet(next_rv=True): next_rv over every record, a game spanning shards matched by id, carried by take and
    collate on the memory-mapped and in-RAM paths, flatten, the loader and a spawned worker's reopened set; without it,
    no next_rv anywhere and the batch is otherwise identical. It is never written to a shard or sent on the wire."""
    import pickle
    _, want = _turn_shards(tmp_path / "pk")
    sel = np.array([8, 0, 2, 6, 3])
    for in_ram in (False, True):
        ds = D.ShardSet(tmp_path / "pk", in_ram=in_ram, next_rv=True)
        assert np.array_equal(ds.array("next_rv"), want)
        b = ds.collate(sel)
        order = sel if in_ram else np.array([0, 2, 3, 8, 6])       # memory-mapped: grouped by shard
        assert b.next_rv.dtype == torch.float32 and torch.equal(b.next_rv, torch.from_numpy(want[order]))
        _batch_equal(b, b.flatten().unflatten())
        (lb,) = list(D.Loader(ds, [sel], workers=1))
        assert torch.equal(lb.next_rv, b.next_rv)
        back = pickle.loads(pickle.dumps(ds))                    # what a spawned loader process gets
        assert back.next_rv and torch.equal(back.collate(sel).next_rv, b.next_rv)
        plain = D.ShardSet(tmp_path / "pk", in_ram=in_ram)
        assert all("next_rv" not in p for p in plain.packs)
        pb = plain.collate(sel)
        assert pb.next_rv is None
        pb.next_rv = b.next_rv
        _batch_equal(pb, b)                                      # next_rv is the only difference
        sub = ds.take(sel)
        pf.write_shard(tmp_path / f"out{int(in_ram)}", sub, [m for meta in ds.metas for m in meta["games"]])
        assert not (tmp_path / f"out{int(in_ram)}" / "next_rv.npy").exists()
        assert D.batch_from_wire(ds.take(sel, targets=False)).next_rv is None


def test_loader_procs_carry_next_rv(tmp_path):
    import multiprocessing as mp
    _, want = _turn_shards(tmp_path / "pk")
    ds = D.ShardSet(tmp_path / "pk", in_ram=True, next_rv=True)
    bl = [np.array([0, 4, 8]), np.array([1, 2, 3, 5, 6, 7])]
    method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    got = list(D.Loader(ds, bl, procs=1, start_method=method))
    for b, sel in zip(got, bl):
        assert torch.equal(b.next_rv, torch.from_numpy(want[sel]))
