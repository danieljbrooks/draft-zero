"""Tests for draftzero.gameplay.supervised: the losses (set NLL, value cross-entropy from the tanh
logit), per-game value subsampling, TD(lambda) targets, the streaming feature vocab, the sampler,
the non-Pass measure, checkpoints (MageZero format, non-default architectures) and an exact
save / resume round trip, on small synthetic tables in imitation.write_h5's layout."""
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from draftzero.gameplay import imitation as im
from draftzero.gameplay import supervised as sv

TINY = {"layers": 1, "width": 32, "heads": 2, "head_hidden": 16}
GEN0_2B = Path.home() / (".cache/huggingface/hub/models--danbrooks--draftzero-checkpoints/snapshots/"
                         "dfbb2cd5513cc2c9b804f80edbcc4f6e2d030287/2026-09-27_03-42-21/models/FDN_exp2_imit/ver1/gen0.pt.gz")


# ------------------------------------------------------------------------------ synthetic tables

def _features(rng, n_rows: int) -> list:
    pool = 1_000_000 + np.arange(80) * 7919
    return [np.sort(rng.choice(pool, rng.integers(12, 30), replace=False)).astype(np.int32) for _ in range(n_rows)]


def _priority_table(rng, games: list, kinds: list | None = None) -> dict:
    feats, legal, sets, status, meta = [], [], [], [], {"row": [], "turn": [], "won": []}
    for g, won, n_turns in games:
        for turn in range(1, n_turns + 1):
            opts = [0] + sorted(rng.choice(np.arange(1, 25), rng.integers(1, 4), replace=False).tolist())
            r = rng.random()
            if r < 0.15:
                S, st = [0], 0                                    # Pass label
            elif r < 0.2:
                S, st = [], 2                                     # unreachable
            else:
                S, st = sorted(rng.choice(opts[1:], rng.integers(1, len(opts)), replace=False).tolist()), 1
            legal.append(opts)
            sets.append(S)
            status.append(st)
            meta["row"].append(g)
            meta["turn"].append(turn)
            meta["won"].append(won)
    feats = _features(rng, len(legal))
    ind, off = im.csr(feats)
    li, lp = im.csr([np.asarray(x, np.int32) for x in legal])
    si, sp = im.csr([np.asarray(x, np.int32) for x in sets])
    t = {"indices": ind, "offsets": off, "legal_idx": li, "legal_indptr": lp, "set_idx": si, "set_indptr": sp,
         "meta/label_status": np.asarray(status, np.int32), "meta/row": np.asarray(meta["row"], np.int32),
         "meta/turn": np.asarray(meta["turn"], np.int32), "meta/won": np.asarray(meta["won"], np.int32),
         "labels": [[str(x) for x in o] for o in legal], "lab_idx": legal}
    t["z"] = np.where(t["meta/won"] > 0, 1.0, -1.0).astype(np.float32)
    if kinds is not None:
        t["kind"] = [kinds[i % len(kinds)] for i in range(len(legal))]
    return t


def _attack_table(rng, games: list, path: Path) -> None:
    rows = [(g, won, turn) for g, won, n in games for turn in range(2, n + 1)]
    ind, off = im.csr(_features(rng, len(rows)))
    with h5py.File(path, "w") as f:
        f.create_dataset("indices", data=ind)
        f.create_dataset("offsets", data=off)
        f.create_dataset("y", data=rng.integers(0, 2, len(rows)))
        f.create_dataset("meta/row", data=np.asarray([r[0] for r in rows], np.int32))
        f.create_dataset("meta/turn", data=np.asarray([r[2] for r in rows], np.int32))
        f.create_dataset("meta/won", data=np.asarray([r[1] for r in rows], np.int32))


def make_tables(root: Path, seed: int = 0) -> Path:
    """turnstart / replay_priority / replay_attack, train and val, like #2b's tables (tiny), plus
    experiment #4-style block (target CSR) and timing tables (a chosen_idx column, integer
    label_kind 0 exact / 1 imputed, imputed rows at weight 0.5)."""
    rng = np.random.default_rng(seed)
    for split, g0, n_games in (("train", 0, 40), ("val", 1000, 12)):
        games = [(g0 + g, int(rng.integers(0, 2)), int(rng.integers(4, 9))) for g in range(n_games)]
        im._save_table(_priority_table(rng, games), root / f"turnstart_{split}.h5")
        im._save_table(_priority_table(rng, games, kinds=["exact", "imputed_order"]), root / f"replay_priority_{split}.h5")
        _attack_table(rng, games, root / f"replay_attack_{split}.h5")
        im._save_table(_priority_table(rng, games), root / f"block_{split}.h5")
        t = _priority_table(rng, games)
        im._save_table(t, root / f"timing_{split}.h5")
        n = len(t["z"])
        chosen = np.array([t["set_idx"][t["set_indptr"][r]] if t["set_indptr"][r + 1] > t["set_indptr"][r] else -1
                           for r in range(n)], np.int32)
        kind = (np.arange(n) % 2).astype(np.int8)
        with h5py.File(root / f"timing_{split}.h5", "a") as f:
            f.create_dataset("chosen_idx", data=chosen)
            del f["label_kind"], f["weight"]
            f.create_dataset("label_kind", data=kind)
            f.create_dataset("weight", data=np.where(kind == 1, 0.5, 1.0).astype(np.float32))
    return root


def tiny_cfg(root: Path, **kw) -> dict:
    base = {"tables_dir": str(root), "vocab_k": 1, "val_rows": None, "arch": TINY, "device": "cpu", "prefetch": 2,
            "warmup_steps": 3, "time_budget_s": None, "eval_every_s": None, "ckpt_every_s": None,
            "latest_every_s": None, "batch_tokens": 16 * 256, "max_batch_rows": 16, "eval_batch_rows": 32,
            "patience_epochs": None}
    base.update(kw)
    return base


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> Path:
    return make_tables(tmp_path_factory.mktemp("tables"))


# ------------------------------------------------------------------------------ losses

def test_policy_nll_is_set_nll_over_the_legal_softmax_and_onehot_is_plain_nll():
    logits = torch.tensor([[2.0, 1.0, 0.5, -1.0], [0.0, 3.0, 0.0, 0.0]])
    legal = torch.tensor([[True, True, True, False], [True, True, False, False]])
    sset = torch.tensor([[False, True, True, True], [False, True, False, False]])
    got = sv.policy_nll(logits, legal, sset)
    p0 = torch.softmax(logits[0, :3], -1)
    p1 = torch.softmax(logits[1, :2], -1)
    assert got[0].item() == pytest.approx(-math.log(p0[1] + p0[2]), rel=1e-6)     # the illegal S member is ignored
    assert got[1].item() == pytest.approx(-math.log(p1[1]), rel=1e-6)
    assert got[1].item() == pytest.approx(torch.nn.functional.cross_entropy(logits[1:, :2], torch.tensor([1])).item())
    empty = sv.policy_nll(logits[:1], legal[:1], torch.zeros(1, 4, dtype=torch.bool))
    assert math.isinf(empty.item())                       # no label: masked out by the trainer, never averaged


def test_value_bce_from_tanh_logit_matches_the_probability_form_and_stays_finite():
    x = torch.tensor([-3.0, -0.4, 0.0, 0.7, 2.5])
    z = torch.tensor([1.0, -1.0, 1.0, 1.0, -1.0])
    p = (1 + torch.tanh(x)) / 2
    naive = -(((1 + z) / 2) * torch.log(p) + ((1 - z) / 2) * torch.log(1 - p))
    assert torch.allclose(sv.value_bce_from_logit(x, z), naive, atol=1e-5)
    assert torch.allclose((1 + torch.tanh(x)) / 2, torch.sigmoid(2 * x), atol=1e-6)
    # saturated tanh: the probability form gives inf / nan, the logit form the exact large loss
    big = torch.tensor([30.0, -30.0], requires_grad=True)
    loss = sv.value_bce_from_logit(big, torch.tensor([-1.0, 1.0]))
    assert torch.allclose(loss, torch.tensor([60.0, 60.0]), rtol=1e-4)
    loss.sum().backward()
    assert torch.isfinite(big.grad).all() and torch.allclose(big.grad.abs(), torch.tensor([2.0, 2.0]))
    # soft TD targets: minimised where P(win) equals the target probability
    xs = torch.linspace(-2, 2, 4001)
    best = xs[sv.value_bce_from_logit(xs, torch.full_like(xs, 0.4)).argmin()]
    assert math.tanh(best.item()) == pytest.approx(0.4, abs=2e-3)
    assert sv.value_mse(torch.tensor([0.0]), torch.tensor([1.0])).item() == pytest.approx(1.0)


def test_value_subsample_keeps_k_positions_per_game_each_epoch():
    games = np.repeat(np.arange(50), np.arange(1, 51))          # game g has g + 1 positions
    eligible = np.ones(len(games), bool)
    eligible[::7] = False
    m0 = sv.value_subsample_mask(games, 3, seed=1, epoch=0, eligible=eligible)
    per_game = np.bincount(games[m0], minlength=50)
    n_elig = np.bincount(games[eligible], minlength=50)
    assert (per_game == np.minimum(3, n_elig)).all()
    assert not (m0 & ~eligible).any()
    assert (m0 == sv.value_subsample_mask(games, 3, seed=1, epoch=0, eligible=eligible)).all()   # deterministic
    m1 = sv.value_subsample_mask(games, 3, seed=1, epoch=1, eligible=eligible)
    assert (m1 != m0).any() and (np.bincount(games[m1], minlength=50) == per_game).all()       # fresh draw per epoch


def test_td_lambda_targets_follow_each_game_in_turn_order():
    games = np.array([5, 5, 5, 9, 9])
    turns = np.array([3, 1, 2, 2, 1])
    phase = np.zeros(5, int)
    v = np.array([0.3, 0.1, 0.2, -0.5, 0.4])
    z = np.array([1.0, 1.0, 1.0, -1.0, -1.0])
    assert np.allclose(sv.td_lambda_targets(games, turns, phase, v, z, 1.0), z)
    # lambda 0: each position's target is the next position's value; the last position's is z
    assert np.allclose(sv.td_lambda_targets(games, turns, phase, v, z, 0.0), [1.0, 0.2, 0.3, -1.0, -0.5])
    lam = 0.5
    g = sv.td_lambda_targets(games, turns, phase, v, z, lam)
    g2 = 1.0                                  # game 5: turn 3 is last
    g1 = (1 - lam) * 0.3 + lam * g2           # turn 2
    g0 = (1 - lam) * 0.2 + lam * g1           # turn 1
    assert np.allclose(g, [g2, g0, g1, -1.0, (1 - lam) * -0.5 + lam * -1.0])


def test_streaming_feature_stats_match_magezero_kept_feature_ids():
    from magezero.vocab import kept_feature_ids
    rng = np.random.default_rng(3)
    bags = []
    for i in range(600):
        f = set(rng.choice(np.arange(10, 400), rng.integers(5, 40), replace=False).tolist())
        if 50 in f:
            f |= {7, 2_000_000_011}            # always together with 50: one pattern, smallest id kept
        if i < 30:
            f.add(5000 + i % 10)               # rare: in 3 states each
        bags.append(np.asarray(sorted(f), np.int64))
    bags[3] = np.asarray([12, 12, 40, 41], np.int64)   # a repeat inside a state counts once
    ind = np.concatenate(bags)
    ptr = np.r_[0, np.cumsum([len(b) for b in bags])]
    for k in (0, 10, 40):
        want = kept_feature_ids(ind, ptr, k=k)
        st = sv.FeatureStats(seed=k)
        for a in range(0, len(bags), 97):               # streamed in blocks, merged as it goes
            st.add(np.concatenate(bags[a:a + 97]), np.asarray([len(b) for b in bags[a:a + 97]]))
        got = st.kept(k)
        assert np.array_equal(got, want), k
        if k == 0:
            assert 5003 in want
        if k == 10:
            assert 7 in want and 50 not in want and 2_000_000_011 not in want and 5003 not in want


def test_config_layers_and_ff_follows_width():
    base = sv.resolve_config({"arch": {"layers": 2}, "lr": 1e-3})
    cfg = sv.resolve_config(base, {"arch": {"width": 256}})             # a sweep run over a resolved base
    assert sv.full_arch(cfg["arch"])["ff"] == 512 and cfg["arch"]["layers"] == 2 and cfg["lr"] == 1e-3
    assert sv.full_arch(sv.resolve_config(base, {"arch": {"width": 256, "ff": 300}})["arch"])["ff"] == 300
    assert sv.full_arch(base["arch"]) == {**sv.ARCH_DEFAULT, "ff": 1024}
    assert sv.parse_overrides(["arch.layers=4", "lr=1e-3", "aux_targets=[turns_left]", "max_steps=null"]) == \
        {"arch": {"layers": 4}, "lr": 1e-3, "aux_targets": ["turns_left"], "max_steps": None}
    with pytest.raises(KeyError):
        sv.resolve_config({"learning_rate": 1e-3})                       # typos are errors
    with pytest.raises(KeyError):
        sv.resolve_config({"arch": {"depth": 3}})


# ------------------------------------------------------------------------------ data and sampler

def test_load_data_fraction_by_game_and_val_whole_games(tables):
    cfg = sv.resolve_config(tiny_cfg(tables, fraction=0.5, val_rows=20))
    d = sv.load_data(cfg, log=lambda *_: None)
    assert d.info["train_games"] == 20 and d.info["train_games_all"] == 40
    games = {int(g) for t in d.train for g in t.game}
    assert len(games) == 20                                        # the same games in every table
    full = sv.load_data(sv.resolve_config(tiny_cfg(tables)), log=lambda *_: None)
    for t, f in zip(d.train, full.train):
        assert set(np.unique(t.game)) <= set(np.unique(f.game))
        assert t.n == int(np.isin(f.game, t.game).sum())         # every row of a chosen game
    for t in d.val:
        with h5py.File(tables / f"{t.name}_val.h5") as f:
            all_g = f["meta/row"][:]
        assert t.n >= min(20, len(all_g))
        assert t.n == int(np.isin(all_g, np.unique(t.game)).sum())    # whole games
    ts = d.train[0]
    with h5py.File(tables / "turnstart_train.h5") as f:
        status = f["meta/label_status"][:][ts.file_rows]
    assert (ts.w[status == 2] == 0).all() and (ts.w[status != 2] == 1).all()   # unreachable: value only
    rp = d.train[1]
    assert rp.lk_names == ["exact", "imputed_order"] and rp.lk is not None


def test_sampler_covers_each_table_once_per_epoch_in_its_share(tables):
    cfg = sv.resolve_config(tiny_cfg(tables, tables=[{"name": "turnstart", "kind": "priority_set", "weight": 1.0},
                                                     {"name": "replay_attack", "kind": "binary", "weight": 2.0}]))
    d = sv.load_data(cfg, log=lambda *_: None)
    s = sv.Sampler(d, cfg)
    seen = [np.zeros(t.n, int) for t in d.train]
    while min(s.state["epoch"]) < 1:
        ti, sel = s.next()
        if s.state["epoch"][ti] == 0 or s.state["cursor"][ti] == 0:
            seen[ti][sel] += 1
        s.state["drawn"][ti] += len(sel)
    assert (seen[0] >= 1).all() and (seen[1] >= 1).all()
    share = np.asarray(s.state["drawn"]) / sum(s.state["drawn"])
    want = np.array([d.train[0].n, 2 * d.train[1].n]) / (d.train[0].n + 2 * d.train[1].n)
    assert np.allclose(share, want, atol=0.08)
    st = s.state_dict()
    a = [s.make_batch(*s.next()) for _ in range(3)]
    s.load_state_dict(st)
    b = [s.make_batch(*s.next()) for _ in range(3)]
    for x, y in zip(a, b):                      # same rows, same token dropout after a restore
        assert x["ti"] == y["ti"] and torch.equal(x["idx"], y["idx"]) and torch.equal(x["off"], y["off"])


def test_default_batch_sizes_are_pretrains(tables):
    cfg = sv.resolve_config(tiny_cfg(tables, batch_tokens=64 * 768, batch_attn=64 * 768 ** 2, max_batch_rows=64))
    s = sv.Sampler(sv.load_data(cfg, log=lambda *_: None), cfg)
    for L in im.BUCKETS:
        assert s.rows_for(L) == im.batch_for(L, 64), L          # 64 up to 768 tokens, then ~(768 / L)^2


def test_nonpass_measure_drops_pass_and_needs_two_plays():
    # row 0: human played option 3; network's best non-Pass is 3 (Pass scored higher: ignored)
    # row 1: only one non-Pass option legal: not scored; row 2: Pass label: not scored
    # row 3: tie between 5 (in S) and 6: expected 0.5
    t = sv.Table(name="x", kind="priority_set", split="val", spec={}, file_rows=np.arange(4), rows=np.zeros(0),
                 ptr=np.zeros(5, np.int64), game=np.arange(4), turn=np.ones(4, np.int32), z=np.ones(4, np.float32),
                 w=np.ones(4, np.float32),
                 legal_indptr=np.array([0, 3, 5, 7, 10]), legal_idx=np.array([0, 3, 4, 0, 9, 0, 2, 0, 5, 6]),
                 set_indptr=np.array([0, 1, 2, 3, 4]), set_idx=np.array([3, 9, 0, 5]))
    scores = [np.array([9.0, 2.0, 1.0]), np.array([0.0, 1.0]), np.array([3.0, 1.0]), np.array([0.0, 1.0, 1.0])]
    m = sv.priority_metrics(t, {"scores": scores})
    assert m["n_nonpass"] == 2 and m["top1_nonpass"] == pytest.approx(0.75)
    assert m["chance_nonpass"] == pytest.approx(0.5)
    assert m["top1"] == pytest.approx((0 + 1 + 1 + 0.5) / 4)       # raw top-1 counts Pass
    assert m["n_pass_rows"] == 1 and m["top1_pass_rows"] == 1.0


# ------------------------------------------------------------------------------ checkpoints

def _vocab(n: int):
    from magezero.vocab import FeatureVocab
    return FeatureVocab(1_000_000 + np.arange(n) * 3, feature_hash_bins=sv.GLOBAL_MAX)


def test_default_checkpoint_is_magezero_format_and_loads_with_imitation(tmp_path):
    from magezero.model import NetTransformer
    vocab = _vocab(40)
    torch.manual_seed(0)
    m = sv.build_model(sv.ARCH_DEFAULT, len(vocab), vocab=vocab)
    assert type(m) is NetTransformer and sv.magezero_loadable(sv.ARCH_DEFAULT)
    p = tmp_path / "model.pt.gz"
    sv.save_weights(p, m, vocab, sv.ARCH_DEFAULT, {"step": 3})
    m2, v2 = im.load_checkpoint(p)
    assert np.array_equal(v2.ids, vocab.ids)
    for k, v in m.state_dict().items():
        assert torch.equal(v, m2.state_dict()[k]), k
    # what MageZero's server does: a NetTransformer of the vocab's size, strict load
    server = im.new_net(len(v2))
    server.load_state_dict(_raw(p)["model_state_dict"])
    m3, v3, meta = sv.load_any_checkpoint(p)
    assert meta["magezero_loadable"] and meta["info"]["step"] == 3
    idx, off = torch.tensor([0, 5, 7, 1, 2]), torch.tensor([0, 3])
    m.eval()
    assert torch.allclose(m(idx, off)[4], m3(idx, off)[4], atol=1e-6)


def _raw(p: Path) -> dict:
    from magezero.model import load_model
    return load_model(str(p))


@pytest.mark.parametrize("arch", [{"layers": 1, "width": 64}, {"layers": 4, "width": 256},
                                  {"type": "mlp", "layers": 2, "width": 64}, {"layers": 2, "width": 64, "norm_first": True},
                                  {"layers": 1, "width": 64, "ffn": "swiglu"},
                                  {"layers": 1, "width": 64, "pool": "attn", "norm_first": True},
                                  {"layers": 1, "width": 64, "value_tower": True, "value_layers": 2}])
def test_non_default_architectures_rebuild_from_the_saved_arch(tmp_path, arch):
    vocab = _vocab(30)
    torch.manual_seed(1)
    m = sv.build_model(arch, len(vocab), vocab=vocab).eval()
    a = sv.full_arch(arch)
    assert not sv.magezero_loadable(a) and a["ff"] == 2 * a["width"]
    p = tmp_path / "x.pt.gz"
    sv.save_weights(p, m, vocab, a, aux=sv.AuxHeads(sv.emb_width(a), ["turns_left"]))
    m2, _, meta = sv.load_any_checkpoint(p)
    assert meta["arch"] == a and meta["aux"] is not None and not meta["magezero_loadable"]
    idx, off = torch.tensor([0, 5, 7, 1, 2, 9]), torch.tensor([0, 2, 4])
    for o1, o2 in zip(m(idx, off), m2(idx, off)):
        assert torch.allclose(o1, o2, atol=1e-6)
    with pytest.raises((RuntimeError, KeyError)):    # MageZero's own loader builds the default network
        im.load_checkpoint(p)


def test_pre_ln_and_embedding_init_options(tables, tmp_path):
    vocab = _vocab(50)
    # pre-LN: a final LayerNorm, not MageZero's shape, and the weights say so
    m = sv.build_model({"norm_first": True}, len(vocab), vocab=vocab)
    assert isinstance(m, sv.TransformerNet) and m.transformer.layers[0].norm_first and m.transformer.norm is not None
    assert not sv.magezero_loadable({"norm_first": True})
    assert sv.infer_arch(m.state_dict())["norm_first"] and not sv.infer_arch(sv.build_model({}, 50).state_dict())["norm_first"]
    # the embedding init: MageZero's N(0, 1) keyed rows, scaled; the default shape stays MageZero's class
    from magezero.model import NetTransformer
    a = sv.build_model(sv.ARCH_DEFAULT, len(vocab), vocab=vocab)
    b = sv.build_model(sv.ARCH_DEFAULT, len(vocab), vocab=vocab, emb_std=0.02)
    assert type(b) is NetTransformer and torch.allclose(b.embedding.weight, a.embedding.weight * 0.02)
    # and a short run with the "modern" recipe trains and checkpoints
    log = lambda *_: None                                           # noqa: E731
    s = sv.train(tiny_cfg(tables, max_steps=12, emb_init_std=0.02, warmup_steps=6,
                          arch={**TINY, "norm_first": True}), tmp_path / "m", log=log)
    assert s["step"] == 12
    m2, _, meta = sv.load_any_checkpoint(tmp_path / "m" / "final.pt.gz")
    assert meta["arch"]["norm_first"] and not meta["magezero_loadable"]


def test_extended_transformer_trains_with_a_value_tower(tables, tmp_path):
    vocab = _vocab(40)
    arch = {**TINY, "ffn": "swiglu", "pool": "attn", "norm_first": True, "value_tower": True}
    m = sv.build_model(arch, len(vocab), vocab=vocab, emb_std=0.02)
    assert isinstance(m, sv.TransformerNetX) and not sv.magezero_loadable(arch)
    assert torch.equal(m.embedding.weight, m.value_tower.embedding.weight)      # both towers start from the same rows
    idx, off = torch.tensor([0, 5, 7, 1, 2, 9]), torch.tensor([0, 2, 4])
    emb = sv.encode(m.eval(), idx, off)
    assert emb.shape == (3, 2 * sv.full_arch(arch)["width"]) == (3, sv.emb_width(arch))
    # the policy heads read the policy tower's half only: a value-tower change leaves them as they are
    pa = m.player_priority_head(emb)
    emb2 = emb.clone()
    emb2[:, sv.full_arch(arch)["width"]:] += 1.0
    assert torch.equal(m.player_priority_head(emb2), pa) and not torch.equal(m.value_head(emb2), m.value_head(emb))
    log = lambda *_: None                                           # noqa: E731
    s = sv.train(tiny_cfg(tables, max_steps=10, emb_init_std=0.02, arch=arch, aux_targets=["turns_left"]),
                 tmp_path / "x", log=log)
    assert s["step"] == 10
    m2, _, meta = sv.load_any_checkpoint(tmp_path / "x" / "final.pt.gz")
    assert isinstance(m2, sv.TransformerNetX) and meta["aux"] is not None and meta["arch"]["value_tower"]


def test_value_detach_keeps_the_value_loss_out_of_the_shared_features(tables, tmp_path):
    arch = {**TINY, "norm_first": True, "value_detach": True}
    m = sv.build_model(arch, 40)
    assert isinstance(m, sv.TransformerNet) and not sv.magezero_loadable(arch)
    idx, off = torch.tensor([0, 5, 7, 1, 2, 9]), torch.tensor([0, 2, 4])
    sv.value_logit(m, sv.encode(m, idx, off)).sum().backward()
    assert m.embedding.weight.grad is None or not m.embedding.weight.grad.any()   # no gradient into the trunk
    assert m.value_head[1].weight.grad.abs().sum() > 0                             # ... but the head learns
    m.zero_grad()
    m.player_priority_head(sv.encode(m, idx, off)).sum().backward()
    assert m.embedding.weight.grad.abs().sum() > 0
    s = sv.train(tiny_cfg(tables, max_steps=10, arch=arch), tmp_path / "d", log=lambda *_: None)
    assert s["step"] == 10
    m2, _, meta = sv.load_any_checkpoint(tmp_path / "d" / "final.pt.gz")
    assert meta["arch"]["value_detach"] and isinstance(m2.value_head[0], sv._Detach)


def test_mlp_value_tower_beside_a_transformer_policy_tower(tables, tmp_path):
    vocab = _vocab(40)
    arch = {**TINY, "norm_first": True, "value_tower": True, "value_tower_type": "mlp", "value_layers": 1}
    m = sv.build_model(arch, len(vocab), vocab=vocab, emb_std=0.02)
    assert isinstance(m, sv.TransformerNetX) and isinstance(m.value_tower, sv._BagTower)
    assert torch.equal(m.embedding.weight, m.value_tower.embedding.weight)
    idx, off = torch.tensor([0, 5, 7, 1, 2, 9]), torch.tensor([0, 2, 4])
    assert sv.encode(m.eval(), idx, off).shape == (3, sv.emb_width(arch))
    s = sv.train(tiny_cfg(tables, max_steps=10, emb_init_std=0.02, arch=arch), tmp_path / "v", log=lambda *_: None)
    assert s["step"] == 10
    m2, _, meta = sv.load_any_checkpoint(tmp_path / "v" / "final.pt.gz")
    assert isinstance(m2.value_tower, sv._BagTower) and meta["arch"]["value_tower_type"] == "mlp"


@pytest.mark.parametrize("ffn,norm,bag", [("gelu", "batch", "sum"), ("swiglu", "none", "max"), ("relu", "layer", "mean")])
def test_mlp_activation_norm_and_pooling_options(tables, tmp_path, ffn, norm, bag):
    arch = {**TINY, "type": "mlp", "ffn": ffn, "mlp_norm": norm, "bag_mode": bag}
    m = sv.build_model(arch, 40)
    assert isinstance(m, sv.BagMLPNet) and m.embedding.mode == bag
    assert isinstance(m.blocks[0].norm, {"layer": torch.nn.LayerNorm, "batch": torch.nn.BatchNorm1d,
                                         "none": torch.nn.Identity}[norm])
    idx, off = torch.tensor([0, 5, 7, 1, 2, 9]), torch.tensor([0, 2, 4])
    assert sv.encode(m.eval(), idx, off).shape == (3, sv.full_arch(arch)["width"])
    s = sv.train(tiny_cfg(tables, max_steps=6, arch=arch), tmp_path / "m", log=lambda *_: None)
    assert s["step"] == 6
    m2, _, meta = sv.load_any_checkpoint(tmp_path / "m" / "final.pt.gz")
    assert meta["arch"]["ffn"] == ffn and meta["arch"]["mlp_norm"] == norm and meta["arch"]["bag_mode"] == bag
    with pytest.raises(ValueError):
        sv.full_arch({"type": "transformer", "ffn": "gelu"})
    m.train()
    assert sv.encode(m, torch.tensor([0, 5]), torch.tensor([0])).shape[0] == 1    # a one-row batch trains too


def test_mlp_embedding_dimension_separate_from_the_width(tables, tmp_path):
    """emb_dim: features embedded (and pooled) at size e, then one linear layer to the width."""
    arch = {**TINY, "type": "mlp", "bag_mode": "max", "ffn": "swiglu", "emb_dim": 8}
    m = sv.build_model(arch, 40)
    assert m.embedding.weight.shape == (40, 8) and m.emb_proj.in_features == 8 and m.emb_proj.out_features == 32
    idx, off = torch.tensor([0, 5, 7, 1, 2, 9]), torch.tensor([0, 2, 4])
    assert sv.encode(m.eval(), idx, off).shape == (3, 32)
    assert sv.build_model({**arch, "emb_dim": 32}, 40).emb_proj is None          # e = width: no projection
    assert sv.build_model({**TINY, "type": "mlp"}, 40).emb_proj is None           # the default: e = width
    s = sv.train(tiny_cfg(tables, max_steps=6, emb_init_std=0.02, arch=arch), tmp_path / "e", log=lambda *_: None)
    assert s["step"] == 6
    m2, _, meta = sv.load_any_checkpoint(tmp_path / "e" / "final.pt.gz")
    assert meta["arch"]["emb_dim"] == 8 and m2.embedding.weight.shape[1] == 8
    with pytest.raises(ValueError):
        sv.full_arch({"type": "transformer", "emb_dim": 128})


@pytest.mark.parametrize("opt", ["adamw", "adagrad"])
def test_embedding_learning_rate_optimizer_and_maxmean_pooling(tables, tmp_path, opt):
    """emb_lr_mult / emb_optimizer: the table trains at its own rate (and optimizer); bag_mode maxmean adds the mean
    pool to the max pool; the run resumes with both optimizers' state; emb_init_from copies rows by feature id."""
    arch = {**TINY, "type": "mlp", "bag_mode": "maxmean", "ffn": "swiglu"}
    m = sv.build_model(arch, 40)
    idx, off = torch.tensor([0, 5, 7, 1, 2, 9]), torch.tensor([0, 2, 4])
    want = m.embedding(idx, off) + torch.nn.functional.embedding_bag(idx, m.embedding.weight, off, mode="mean")
    assert m.add_mean and m.embedding.mode == "max"
    x = m.embedding(idx, off) + torch.nn.functional.embedding_bag(idx, m.embedding.weight, off, mode="mean")
    assert torch.allclose(x, want)
    cfg = tiny_cfg(tables, max_steps=6, emb_init_std=0.02, arch=arch, emb_lr_mult=30.0, emb_optimizer=opt)
    tr = sv.Trainer(cfg, tmp_path / "a", log=lambda *_: None)
    lrs = {id(p): g["lr_mult"] for o in tr.opts for g in o.param_groups for p in g["params"]}
    assert lrs[id(tr.model.embedding.weight)] == 30.0 and lrs[id(tr.model.blocks[0].fc1.weight)] == 1.0
    assert (tr.opt_emb is not None) == (opt == "adagrad")
    s = sv.train(cfg, tmp_path / "b", log=lambda *_: None)
    assert s["step"] == 6
    s = sv.train({**cfg, "max_steps": 8}, tmp_path / "b", resume=True, log=lambda *_: None)   # both optimizers resume
    assert s["step"] == 8
    m2, _, meta = sv.load_any_checkpoint(tmp_path / "b" / "final.pt.gz")
    assert meta["arch"]["bag_mode"] == "maxmean"
    c = sv.Trainer({**cfg, "emb_init_from": str(tmp_path / "b" / "final.pt.gz")}, tmp_path / "c", log=lambda *_: None)
    assert torch.equal(c.model.embedding.weight.cpu(), m2.embedding.weight.cpu())
    with pytest.raises(ValueError):
        sv.full_arch({"type": "transformer", "bag_mode": "maxmean"})


def test_feature_stats_caps_the_vocab_at_the_most_frequent_ids():
    st = sv.FeatureStats(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(3):
        lens = np.full(50, 4)
        idx = np.concatenate([np.sort(rng.choice([10, 20, 30, 40, 50, 60], 4, replace=False, p=[.3, .3, .2, .1, .05, .05]))
                              for _ in lens])
        st.add(idx, lens)
    full = st.kept(0)
    cnt = {i: 0 for i in full}
    assert len(full) >= 5
    top = st.kept(0, max_n=3)
    assert len(top) == 3 and set(top) <= set(full.tolist()) and list(top) == sorted(top)
    assert list(st.kept(0, max_n=100)) == list(full)


def test_rowstore_slices_back_the_true_rows():
    r = np.array([3, 70000, 5, 65535, 65534, 90000, 1], np.int64)
    st = sv.RowStore.from_parts([r[:3], r[3:]])
    assert len(st) == 7 and st.low.dtype == np.uint16 and list(st.over_pos) == [1, 3, 5]
    for a in range(7):
        for b in range(a, 8):
            assert list(st[a:b]) == list(r[a:b])
    with pytest.raises(TypeError):
        st[np.array([0, 1])]


def test_a_vocab_past_16_bits_keeps_every_feature(tables, tmp_path, monkeypatch):
    """Past U16_ROWS rows the vocab is ordered by frequency and the tables are RowStores: the same
    features in every row as the plain 16-bit path, from a fresh load and from the cache, and it trains."""
    base = sv.load_data(sv.resolve_config(tiny_cfg(tables)), log=lambda *_: None)
    monkeypatch.setattr(sv, "U16_ROWS", 20)                  # the tiny vocab (~80 features) is "wide"
    for cache in (None, tmp_path / "cache", tmp_path / "cache"):    # the second cached load reads the .npz
        wide = sv.load_data(sv.resolve_config(tiny_cfg(tables, data_cache=str(cache) if cache else None)),
                            log=lambda *_: None)
        assert len(wide.vocab) == len(base.vocab) > 20 and sorted(wide.vocab.ids) == sorted(base.vocab.ids)
        for tb, tw in zip(base.train + base.val, wide.train + wide.val):
            assert isinstance(tw.rows, sv.RowStore) and len(tw.rows.over_pos) > 0
            for i in range(tb.n):
                a, b = tb.ptr[i], tb.ptr[i + 1]
                assert list(base.vocab.ids[tb.rows[a:b]]) == list(wide.vocab.ids[tw.rows[a:b]])
    s = sv.train(tiny_cfg(tables, max_steps=8, value_target="td", td_start_epochs=0.01), tmp_path / "w",
                 log=lambda *_: None)
    assert s["step"] == 8


def test_imitation_net_loads_a_checkpoint_and_evaluates_states(tables, tmp_path):
    from draftzero.gameplay.imitation_net import ImitationNet, resolve
    arch = {**TINY, "norm_first": True}
    sv.train(tiny_cfg(tables, max_steps=6, arch=arch, emb_init_std=0.02), tmp_path / "r", log=lambda *_: None)
    net = ImitationNet.load(tmp_path / "r" / "final.pt.gz", device="cpu")
    assert "TransformerNet" in net.describe() and net.info["config"]["arch"]["norm_first"]
    pool = 1_000_000 + np.arange(80) * 7919
    states = [pool[:20].tolist(), pool[30:55].tolist(), [5, 6, 7]]          # the last: no known feature
    out = net.evaluate(states, batch=2)
    assert len(out) == 3 and out[2]["value"] == 0.0 and not out[2]["policy_player"].any()
    w = net.arch["policy_width"]
    for o in out[:2]:
        assert o["policy_player"].shape == (w,) and o["policy_target"].shape == (w,) and o["policy_binary"].shape == (2,)
        assert -1.0 <= o["value"] <= 1.0
    # the same numbers as the network on the mapped rows, one state at a time
    rows, _ = net.vocab.map_bags(states[1], [0])
    with torch.no_grad():
        pa, *_, v = net.model(torch.as_tensor(np.asarray(rows, np.int64)), torch.tensor([0]))
    assert np.allclose(out[1]["policy_player"], pa[0].numpy(), atol=1e-5) and abs(out[1]["value"] - float(v.reshape(-1)[0])) < 1e-5
    p = ImitationNet.policy_over(out[0]["policy_player"], [0, 5, 9])
    assert p.shape == (3,) and abs(p.sum() - 1) < 1e-9
    assert ImitationNet.policy_over(np.array([0.0, 2.0, 1.0]), [0, 1, 2], temperature=0).tolist() == [0, 1, 0]
    assert ImitationNet.win_probability(0.5) == 0.75
    assert resolve(tmp_path / "x.pt.gz") == tmp_path / "x.pt.gz"
    with pytest.raises(ValueError):
        resolve("hf://owner-only")


@pytest.mark.skipif(not GEN0_2B.exists(), reason="experiment #2b's checkpoint is not in the HF cache")
def test_load_any_checkpoint_reads_pretrain_checkpoints():
    m, vocab, meta = sv.load_any_checkpoint(GEN0_2B)
    assert meta["magezero_loadable"] and len(vocab) == 14496 and meta["info"] is None
    m2, _ = im.load_checkpoint(GEN0_2B)
    assert all(torch.equal(v, m2.state_dict()[k]) for k, v in m.state_dict().items())


# ------------------------------------------------------------------------------ training: save / resume

@pytest.mark.parametrize("extra", [{}, {"value_per_game": 2, "value_target": "td", "td_start_epochs": 0.05,
                                        "td_refresh_epochs": 0.1, "aux_targets": ["turns_left"],
                                        "value_loss": "mse"}])
def test_resume_continues_the_same_run_exactly(tables, tmp_path, extra):
    log = lambda *_: None                                           # noqa: E731
    cfg = tiny_cfg(tables, eval_every_steps=5, **extra)
    data = sv.load_data(sv.resolve_config(cfg), log=log)
    straight = sv.train({**cfg, "max_steps": 14}, tmp_path / "a", data=data, log=log)
    sv.train({**cfg, "max_steps": 8}, tmp_path / "b", data=data, log=log)
    with pytest.raises(FileExistsError):
        sv.train({**cfg, "max_steps": 8}, tmp_path / "b", data=data, log=log)
    with pytest.raises(ValueError):                                 # a model change is not a resume
        sv.train({**cfg, "max_steps": 14, "arch": {**TINY, "width": 48}}, tmp_path / "b", resume=True, data=data,
                 log=log)
    resumed = sv.train({**cfg, "max_steps": 14}, tmp_path / "b", resume=True, data=data, log=log)
    assert straight["step"] == resumed["step"] == 14 and straight["seen"] == resumed["seen"]
    assert resumed["sessions"] == 2
    if extra.get("value_target") == "td":
        assert straight["td_refreshes"] == resumed["td_refreshes"] >= 2
    a = sv.load_any_checkpoint(tmp_path / "a" / "final.pt.gz")[0].state_dict()
    b = sv.load_any_checkpoint(tmp_path / "b" / "final.pt.gz")[0].state_dict()
    for k in a:
        assert torch.equal(a[k], b[k]), k
    ea, eb = sv.read_evals(tmp_path / "a" / "evals.jsonl"), sv.read_evals(tmp_path / "b" / "evals.jsonl")
    assert [e["step"] for e in ea] == [0, 5, 10, 14]
    assert [e["step"] for e in eb] == [0, 5, 8, 10, 14]            # + the first session's closing evaluation
    eb = [e for e in eb if e["step"] != 8]
    for x, y in zip(ea, eb):
        assert x["select/combined"] == pytest.approx(y["select/combined"], abs=1e-6)
        assert x["policy/top1_nonpass"] == y["policy/top1_nonpass"]
    for name in ("best_policy.pt.gz", "best_value.pt.gz", "best.pt.gz", "latest.pt", "summary.json"):
        assert (tmp_path / "b" / name).exists(), name
    s = json.loads((tmp_path / "b" / "summary.json").read_text())
    assert s["stop"] == "max steps" and s["best"]["policy"][1] is not None


def test_target_and_onehot_tables_train_and_score(tables, tmp_path):
    specs = [{"name": "turnstart", "kind": "priority_set"},
             {"name": "block", "kind": "target", "value": False},
             {"name": "timing", "kind": "priority_onehot", "weight": 0.5,
              "label_kind_names": {0: "exact", 1: "imputed"}}]
    cfg = tiny_cfg(tables, tables=specs, max_steps=12, eval_every_steps=6)
    data = sv.load_data(sv.resolve_config(cfg), log=lambda *_: None)
    tm = data.train[2]
    assert (np.diff(tm.set_indptr) <= 1).all()                          # chosen_idx, not the set, is the label
    assert tm.lk_names == ["exact", "imputed"] and set(np.unique(tm.w)) <= {0.0, 0.5, 1.0}
    assert not data.g_value_ok[data.goff[1]:data.goff[2]].any()          # block rows: no value loss
    s = sv.train(cfg, tmp_path / "r", data=data, log=lambda *_: None)
    ev = s["last_eval"]
    assert ev["block/top1"] is not None and ev["block/set_nll"] is not None and "block/value_auc" not in ev
    assert ev["timing/exact/top1"] is not None and ev["timing/imputed/top1"] is not None
    assert ev["policy/n"] == ev["turnstart/n"] + ev["timing/n"]          # pooled priority = priority tables


def test_plateau_stop_and_cli_train(tables, tmp_path):
    out = tmp_path / "run"
    rc = sv.main(["train", "--out", str(out), "--tables-dir", str(tables), "--set", "arch.layers=1",
                  "--set", "arch.width=32", "--set", "arch.heads=2", "--set", "arch.head_hidden=16",
                  "--set", "vocab_k=1", "--set", "device=cpu", "--set", "max_steps=40", "--set", "eval_every_steps=4",
                  "--set", "patience_epochs=0.01", "--set", "min_delta=10", "--set", "time_budget_s=null",
                  "--set", "batch_tokens=4096", "--set", "max_batch_rows=16"])
    assert rc == 0
    s = json.loads((out / "summary.json").read_text())
    assert s["stop"] == "plateau" and s["step"] < 40
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["arch"]["width"] == 32 and cfg["max_steps"] == 40


# ------------------------------------------------------------------------------ stage 6 (docs/017 §6.6)

def _selfplay_tool():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "tools" / "imitation_scale" / "selfplay_tables.py"
    spec = importlib.util.spec_from_file_location("selfplay_tables", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _selfplay_tables(root: Path, seed: int = 3) -> None:
    """selfplay_{train,val}.h5 in selfplay_tables.py's layout: play.py-style records of searched
    decisions (priority, target and yes/no rows, visit counts with zeros), features from the human
    tables' pool."""
    st = _selfplay_tool()
    rng = np.random.default_rng(seed)
    for split, pairs in (("train", range(0, 24)), ("val", range(100, 108))):
        rows = []
        for p in pairs:
            for seat, res in (("A", 1), ("B", -1)):
                recs = []
                for i in range(int(rng.integers(4, 9))):
                    kind = ("PRIORITY", "PRIORITY", "CHOOSE_TARGET", "CHOOSE_USE")[i % 4]
                    legal = ([0] + sorted(rng.choice(np.arange(1, 25), 3, replace=False).tolist()) if kind == "PRIORITY"
                             else sorted(rng.choice(np.arange(1, 25), 2, replace=False).tolist()) if kind == "CHOOSE_TARGET"
                             else [0, 1])
                    vis = rng.integers(0, 6, len(legal))
                    vis[0] += 1
                    recs.append({"features": _features(rng, 1)[0].tolist(), "type": kind, "turn": 1 + i,
                                 "legal": legal, "visits": vis.tolist(), "q": float(rng.uniform(-1, 1))})
                rows.extend(st.rows_of({"pair": p, "swap": False, "seat": seat, "result": res, "records": recs}, 0.95))
        st.write_table(rows, root / f"selfplay_{split}.h5")


@pytest.fixture(scope="module")
def stage6(tables, tmp_path_factory) -> dict:
    """The human tables plus soft self-play tables, and a tiny network trained on the human tables
    (the stand-in for the IL+BC network)."""
    _selfplay_tables(tables)
    out = tmp_path_factory.mktemp("init")
    sv.train(tiny_cfg(tables, max_steps=6, eval_at_start=False), out, log=lambda *_: None)
    return {"tables": tables, "init": out / "final.pt.gz"}


def stage6_cfg(s: dict, **kw) -> dict:
    specs = [{"name": "selfplay", "kind": "soft", "value_column": "z_td"},
             {"name": "turnstart", "kind": "priority_set", "group": "human"},
             {"name": "replay_attack", "kind": "binary", "group": "human"}]
    return tiny_cfg(s["tables"], **{"tables": specs, "group_shares": {"human": 0.25}, "init_checkpoint": str(s["init"]),
                                    **kw})


def test_soft_rows_use_their_heads_and_cross_entropy_against_visit_shares():
    torch.manual_seed(0)
    m = sv.build_model({**TINY, "policy_width": 8}, 50)
    emb = torch.randn(3, 32)
    at = torch.tensor([0, 3, 5])
    lg = sv.soft_logits(m, emb, at)
    assert torch.equal(lg[0], m.player_priority_head(emb)[0]) and torch.equal(lg[1], m.target_head(emb)[1])
    assert torch.equal(lg[2, :2], m.binary_head(emb)[2]) and torch.isinf(lg[2, 2:]).all()
    L = torch.zeros(3, 8, dtype=torch.bool)
    L[0, [0, 2, 5]] = L[1, [1, 4]] = L[2, [0, 1]] = True
    P = torch.zeros(3, 8)
    P[0, 0], P[0, 5], P[1, 4], P[2, 1] = 0.25, 0.75, 1.0, 1.0
    ce, lsm = sv.soft_ce(lg, L, P)
    for r in range(3):
        logp = torch.log_softmax(lg[r][L[r]], -1)
        want = -(P[r][L[r]] * logp).sum()
        assert ce[r].item() == pytest.approx(want.item(), rel=1e-5)
    assert torch.isfinite(ce).all()
    ce.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_table_shares_and_epoch_with_a_fixed_human_share(stage6):
    cfg = sv.resolve_config(stage6_cfg(stage6))
    data = sv.load_data(cfg, log=lambda *_: None)
    assert np.array_equal(data.vocab.ids, sv.checkpoint_vocab(stage6["init"]).ids)    # the init network's vocab
    share = sv.table_shares(data.train, cfg)
    assert share[0] == pytest.approx(0.75) and share[1:].sum() == pytest.approx(0.25)
    assert share[1] / share[2] == pytest.approx(data.train[1].n / data.train[2].n)
    assert sv.epoch_length(data.train, share) == round(data.train[0].n / 0.75)        # one pass over the self-play rows
    soft = data.train[0]
    assert soft.kind == "soft" and set(np.unique(soft.atype)) == {0, 3, 5}
    assert np.allclose(np.add.reduceat(soft.set_p, soft.set_indptr[:-1]), 1, atol=1e-6)
    assert not np.array_equal(data.g_z[:soft.n], soft.z)                              # the value trains on z_td
    with pytest.raises(ValueError):
        sv.resolve_config(stage6_cfg(stage6, group_shares={"human": 1.0}))
    with pytest.raises(ValueError):
        sv.resolve_config(stage6_cfg(stage6, value_target="td"))
    with pytest.raises(ValueError):
        sv.resolve_config(stage6_cfg(stage6, freeze=["trunk", "policy", "value"]))


def test_stage6_starts_from_the_init_network_anneals_kl_and_resumes_exactly(stage6, tmp_path):
    log = lambda *_: None                                           # noqa: E731
    cfg = stage6_cfg(stage6, kl_weight=0.5, eval_every_steps=4)
    data = sv.load_data(sv.resolve_config(cfg), log=log)
    # a constant KL weight: an annealed one follows max_steps, which a stop-and-resume test changes
    straight = sv.train({**cfg, "max_steps": 12}, tmp_path / "a", data=data, log=log)
    sv.train({**cfg, "max_steps": 6}, tmp_path / "b", data=data, log=log)
    resumed = sv.train({**cfg, "max_steps": 12}, tmp_path / "b", resume=True, data=data, log=log)
    assert straight["seen"] == resumed["seen"]
    a = sv.load_any_checkpoint(tmp_path / "a" / "final.pt.gz")[0].state_dict()
    b = sv.load_any_checkpoint(tmp_path / "b" / "final.pt.gz")[0].state_dict()
    for k in a:
        assert torch.equal(a[k], b[k]), k
    sv.train({**cfg, "kl_weight": 1.0, "kl_weight_end": 0.0, "max_steps": 12}, tmp_path / "c", data=data, log=log)
    ev = sv.read_evals(tmp_path / "c" / "evals.jsonl")
    first, last = ev[0], ev[-1]
    assert first["step"] == 0 and first["selfplay/kl_ref"] == pytest.approx(0, abs=1e-5)   # it starts as the reference
    assert last["selfplay/kl_ref"] > 0 and [e["kl_weight"] for e in ev] == sorted((e["kl_weight"] for e in ev), reverse=True)
    assert ev[0]["kl_weight"] == 1.0 and last["kl_weight"] == 0.0
    for k in ("selfplay/ce", "selfplay/top1_search", "selfplay/entropy", "selfplay/search_entropy",
              "selfplay/value_auc", "selfplay/priority/top1_search", "policy/pass_top1", "policy/pass_human",
              "policy/entropy", "value/ece"):
        assert last[k] is not None, k
    assert "train/kl" in last and "train/soft" in last and "train/priority" in last


def test_freezing_trunk_and_policy_trains_the_value_head_alone(stage6, tmp_path):
    cfg = stage6_cfg(stage6, freeze=["trunk", "policy"], max_steps=8, lr=1e-2)
    sv.train(cfg, tmp_path / "f", log=lambda *_: None)
    init = sv.load_any_checkpoint(stage6["init"])[0].state_dict()
    now = sv.load_any_checkpoint(tmp_path / "f" / "final.pt.gz")[0].state_dict()
    changed = {k for k in init if not torch.equal(init[k], now[k])}
    assert changed and all(k.startswith("value_head.") for k in changed)


def test_evaluate_cli_scores_a_network_on_the_selfplay_split(stage6, tmp_path, capsys):
    import yaml
    cfgp = tmp_path / "stage6.yml"
    cfgp.write_text(yaml.safe_dump(stage6_cfg(stage6)))
    out = tmp_path / "ev.json"
    assert sv.main(["evaluate", "--checkpoint", str(stage6["init"]), "--config", str(cfgp), "--split", "val",
                    "--json", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["selfplay/kl_ref"] == pytest.approx(0, abs=1e-5) and r["selfplay/top1_search"] is not None


def test_sweep_follow_picks_up_runs_added_while_it_runs(tables, tmp_path):
    import threading
    import time as _t
    import yaml
    base = {k: v for k, v in tiny_cfg(tables, max_steps=4, eval_every_steps=4).items()}
    spec_path = tmp_path / "spec.yml"
    spec = {"title": "t", "base": base, "runs": [{"name": "a", "seed": 0}]}
    spec_path.write_text(yaml.safe_dump(spec))
    out = tmp_path / "sweep"
    res = {}
    th = threading.Thread(target=lambda: res.update(rows=sv.run_sweep(spec_path, out, follow=True, follow_idle_s=120,
                                                                       bench_seconds=0.05, log=lambda *_: None)))
    th.start()
    for _ in range(600):                      # run a finishes ...
        if (out / "runs" / "a" / "speed.json").exists():
            break
        _t.sleep(0.1)
    spec["runs"].insert(0, {"name": "b", "seed": 1, "lr": 1e-3})      # ... then a run is added (ahead of a)
    spec_path.write_text(yaml.safe_dump(spec))
    for _ in range(600):
        if (out / "runs" / "b" / "speed.json").exists():
            break
        _t.sleep(0.1)
    (out / "STOP").write_text("")
    th.join(timeout=120)
    assert not th.is_alive()
    assert {r["name"] for r in res["rows"]} == {"a", "b"}
    log = (out / "sweep.log").read_text()
    assert log.count("data loaded") == 1 and "STOP file found" in log
    assert [r["name"] for r in json.loads((out / "sweep.json").read_text())["runs"]] == ["b", "a"]   # the spec's order


def test_result_aux_head_trains_on_every_row(tables, tmp_path):
    cfg = tiny_cfg(tables, max_steps=6, aux_targets=["result"], aux_weight=0.5, value_per_game=1)
    data = sv.load_data(sv.resolve_config(cfg), log=lambda *_: None)
    t = data.train[0]
    assert np.array_equal(t.aux["result"], t.z)                 # every row carries its game's result
    s = sv.train(cfg, tmp_path / "r", data=data, log=lambda *_: None)
    assert s["step"] == 6
    assert sv.load_any_checkpoint(tmp_path / "r" / "final.pt.gz")[2]["aux"].names == ["result"]


def test_a_runs_own_aux_targets_train_on_data_loaded_without_them(tables, tmp_path):
    """A sweep loads its data once with the base config: a run that adds aux targets must still get them
    (they used to stay unset, so the aux head trained with no loss)."""
    base = tiny_cfg(tables, max_steps=8, eval_every_steps=4)
    data = sv.load_data(sv.resolve_config(base), log=lambda *_: None)
    assert not any("result" in t.aux or "turns_left" in t.aux for t in data.train)
    sv.train({**base, "aux_targets": ["result", "turns_left"], "aux_weight": 0.5}, tmp_path / "a", data=data,
             log=lambda *_: None)
    ev = sv.read_evals(tmp_path / "a" / "evals.jsonl")
    assert ev[-1]["train/aux"] > 0
    assert all("result" in t.aux and "turns_left" in t.aux for t in data.train)


def test_act_weights_upweight_acted_rows_in_a_per_run_copy(tables, tmp_path):
    base = tiny_cfg(tables, max_steps=4)
    data = sv.load_data(sv.resolve_config(base), log=lambda *_: None)
    t0 = next(t for t in data.train if t.kind in sv.POLICY_KINDS and t.set_indptr is not None)
    acted = sv.acted_rows(t0)
    assert acted.any() and not acted.all()
    w_before = t0.w.copy()
    d2 = sv.with_act_weights(data, {t0.name: 3.0}, log=lambda *_: None)
    t2 = next(t for t in d2.train if t.name == t0.name)
    assert np.allclose(t2.w[acted], 3 * w_before[acted]) and np.allclose(t2.w[~acted], w_before[~acted])
    assert np.array_equal(t0.w, w_before) and d2.val is data.val              # the shared data are untouched
    s = sv.train({**base, "act_weights": {t0.name: 3.0}}, tmp_path / "a", data=data, log=lambda *_: None)
    assert s["step"] == 4 and np.array_equal(t0.w, w_before)
    with pytest.raises(ValueError):
        sv.with_act_weights(data, {"no_such_table": 2.0})
